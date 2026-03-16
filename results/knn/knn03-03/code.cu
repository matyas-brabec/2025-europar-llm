#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace knn_detail {

// Brute-force 2D k-NN specialized for modern NVIDIA data-center GPUs.
//
// Design summary:
// - One warp owns one query point.
// - A whole thread block streams the data set through shared memory in large tiles.
// - Each warp keeps its current top-k set as a private max-heap in shared memory.
//   This is the key choice: maintaining a heap makes accepted-candidate updates
//   O(log k) instead of O(k), which matters when k can be as large as 1024.
// - The heap is sorted only once, after the full scan, before writing results.
// - No extra device memory is allocated; only dynamic shared memory is used.
//
// Synchronization:
// - __syncthreads() protects the block-wide shared-memory data tile.
// - __syncwarp() protects the warp-private heap when one lane updates it and the
//   whole warp subsequently reads it.
// - __shfl_sync()/__ballot_sync() are used for warp-level broadcast/compaction.

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// `std::pair<int,float>` is the required public ABI, but its constructors / assignment
// operators are not guaranteed to be device-callable across all host standard library
// implementations. We therefore write the object representation of an equivalent trivial
// POD directly into the output buffer.
struct ResultPairPod {
    int first;
    float second;
};

static_assert(sizeof(ResultPairPod) == sizeof(std::pair<int, float>),
              "std::pair<int,float> must be an 8-byte {int,float} pair on this ABI.");
static_assert(std::is_trivially_copyable<std::pair<int, float>>::value,
              "std::pair<int,float> must be trivially copyable for raw byte stores.");

// Tuning choices for A100/H100-class GPUs:
// - For k <= 256, use 16 warps/block (512 threads) to maximize reuse of each shared
//   data tile across many simultaneous queries.
// - For k >= 512, use 8 warps/block (256 threads) to keep per-block shared memory
//   low enough to preserve good residency.
// - Use 4096 cached points per block except for k=1024, where the warp-private heaps
//   already consume 64 KiB and the tile is reduced to 2048 points to keep the total
//   dynamic shared memory at 80 KiB.
template <int K>
struct KnnConfig {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0),
                  "K must be a power of two in [32, 1024].");

    static constexpr int BLOCK_THREADS   = (K <= 256) ? 512 : 256;
    static constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;
    static constexpr int BATCH_SIZE      = (K == 1024) ? 2048 : 4096;

    static_assert((BLOCK_THREADS % kWarpSize) == 0, "Block size must be warp-aligned.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");
    static_assert((BATCH_SIZE % kWarpSize) == 0, "Batch size must be warp-aligned.");
    static_assert((BATCH_SIZE % 2) == 0, "Batch size must stay even for 16-byte vector loads.");
};

template <int K>
constexpr std::size_t shared_memory_bytes() {
    return static_cast<std::size_t>(KnnConfig<K>::WARPS_PER_BLOCK) * static_cast<std::size_t>(K) *
               (sizeof(float) + sizeof(int)) +
           static_cast<std::size_t>(KnnConfig<K>::BATCH_SIZE) * sizeof(float2);
}

// Deterministic tie-breaks are used only to give the heap / final sort a total order.
// The API itself does not require any specific tie resolution.
__device__ __forceinline__ bool pair_better(float a_dist, int a_idx, float b_dist, int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

__device__ __forceinline__ bool pair_worse(float a_dist, int a_idx, float b_dist, int b_idx) {
    return (a_dist > b_dist) || ((a_dist == b_dist) && (a_idx > b_idx));
}

__device__ __forceinline__ float squared_l2(float qx, float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return __fmaf_rn(dy, dy, dx * dx);
}

__device__ __forceinline__ void swap_entry(float* dist, int* idx, int a, int b) {
    const float td = dist[a];
    dist[a] = dist[b];
    dist[b] = td;

    const int ti = idx[a];
    idx[a] = idx[b];
    idx[b] = ti;
}

// Classical sift-down for a max-heap ordered by (distance, index).
// Only lane 0 calls this helper.
__device__ __forceinline__ void sift_down_max_heap(float* dist, int* idx, int root, int heap_size) {
    int current = root;

    while (true) {
        const int left = (current << 1) + 1;
        if (left >= heap_size) {
            break;
        }

        const int right = left + 1;
        int child = left;

        if (right < heap_size && pair_worse(dist[right], idx[right], dist[left], idx[left])) {
            child = right;
        }

        if (!pair_worse(dist[child], idx[child], dist[current], idx[current])) {
            break;
        }

        swap_entry(dist, idx, current, child);
        current = child;
    }
}

template <int K>
__device__ __forceinline__ void build_max_heap(float* dist, int* idx) {
    const int lane = threadIdx.x & (kWarpSize - 1);

    if (lane == 0) {
#pragma unroll 1
        for (int root = (K >> 1) - 1; root >= 0; --root) {
            sift_down_max_heap(dist, idx, root, K);
        }
    }

    __syncwarp(kFullMask);
}

template <int K>
__device__ __forceinline__ void maybe_replace_root(float* dist, int* idx,
                                                   float cand_dist, int cand_idx) {
    // Only lane 0 calls this helper.
    if (pair_better(cand_dist, cand_idx, dist[0], idx[0])) {
        dist[0] = cand_dist;
        idx[0] = cand_idx;
        sift_down_max_heap(dist, idx, 0, K);
    }
}

// Heapsort from a max-heap yields ascending order, which matches the required result layout.
// Only lane 0 performs the serial heap sort; the whole warp then writes the sorted output.
template <int K>
__device__ __forceinline__ void heap_sort_ascending(float* dist, int* idx) {
    const int lane = threadIdx.x & (kWarpSize - 1);

    if (lane == 0) {
#pragma unroll 1
        for (int end = K - 1; end > 0; --end) {
            swap_entry(dist, idx, 0, end);
            sift_down_max_heap(dist, idx, 0, end);
        }
    }

    __syncwarp(kFullMask);
}

// Vectorized block-wide batch load.
// Each int4 copy transfers two consecutive float2 points (16 bytes).
template <int BLOCK_THREADS>
__device__ __forceinline__ void load_data_batch(const float2* __restrict__ data,
                                                int base,
                                                int batch_count,
                                                float2* __restrict__ batch_points) {
    const int tid = threadIdx.x;

    const int vec_count = batch_count >> 1;
    const int4* __restrict__ gvec = reinterpret_cast<const int4*>(data + base);
    int4* __restrict__ svec = reinterpret_cast<int4*>(batch_points);

    for (int i = tid; i < vec_count; i += BLOCK_THREADS) {
        svec[i] = gvec[i];
    }

    if ((batch_count & 1) && (tid == 0)) {
        batch_points[batch_count - 1] = data[base + batch_count - 1];
    }
}

template <int K>
__device__ __forceinline__ void initialize_heap_from_first_k(float qx, float qy,
                                                             const float2* batch_points,
                                                             float* heap_dist,
                                                             int* heap_idx) {
    const int lane = threadIdx.x & (kWarpSize - 1);

#pragma unroll
    for (int i = lane; i < K; i += kWarpSize) {
        heap_dist[i] = squared_l2(qx, qy, batch_points[i]);
        heap_idx[i]  = i;  // First batch starts at global index 0.
    }

    __syncwarp(kFullMask);
    build_max_heap<K>(heap_dist, heap_idx);
}

// Scan a cached shared-memory tile.
// The dominant work (distance evaluation) is fully warp-parallel.
// Heap maintenance is intentionally serialized in lane 0: once the heap is primed,
// very few candidates beat the current kth-best threshold, so serial O(log k)
// updates are typically much cheaper than keeping the whole heap warp-distributed.
template <int K>
__device__ __forceinline__ void process_cached_batch(float qx, float qy,
                                                     const float2* batch_points,
                                                     int base_index,
                                                     int start_local,
                                                     int batch_count,
                                                     float* heap_dist,
                                                     int* heap_idx) {
    const int lane = threadIdx.x & (kWarpSize - 1);

    for (int tile = start_local; tile < batch_count; tile += kWarpSize) {
        const int local = tile + lane;

        float cand_dist = CUDART_INF_F;
        int cand_index = -1;

        if (local < batch_count) {
            cand_dist = squared_l2(qx, qy, batch_points[local]);
            cand_index = base_index + local;
        }

        const float threshold =
            __shfl_sync(kFullMask, (lane == 0 ? heap_dist[0] : 0.0f), 0);

        unsigned pending =
            __ballot_sync(kFullMask, (local < batch_count) && (cand_dist < threshold));

        if (pending != 0u) {
            while (pending != 0u) {
                const int src_lane = __ffs(pending) - 1;

                const float selected_dist = __shfl_sync(kFullMask, cand_dist, src_lane);
                const int selected_index = __shfl_sync(kFullMask, cand_index, src_lane);

                if (lane == 0) {
                    maybe_replace_root<K>(heap_dist, heap_idx, selected_dist, selected_index);
                }

                pending &= (pending - 1);
            }

            __syncwarp(kFullMask);
        }
    }
}

__device__ __forceinline__ void store_result_pair(std::pair<int, float>* out,
                                                  std::size_t pos,
                                                  int index,
                                                  float distance) {
    const ResultPairPod pod{index, distance};

    unsigned char* dst =
        reinterpret_cast<unsigned char*>(out) + pos * sizeof(std::pair<int, float>);
    const unsigned char* src = reinterpret_cast<const unsigned char*>(&pod);

#pragma unroll
    for (int i = 0; i < static_cast<int>(sizeof(ResultPairPod)); ++i) {
        dst[i] = src[i];
    }
}

template <int K>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result) {
    using Config = KnnConfig<K>;

    constexpr int BLOCK_THREADS   = Config::BLOCK_THREADS;
    constexpr int WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    constexpr int BATCH_SIZE      = Config::BATCH_SIZE;
    constexpr int TOTAL_TOPK      = WARPS_PER_BLOCK * K;

    // Declaring the dynamic shared segment as int4 guarantees 16-byte base alignment.
    extern __shared__ int4 shared_storage[];
    unsigned char* smem = reinterpret_cast<unsigned char*>(shared_storage);

    // Layout:
    //   [ warp0 heap distances | warp1 heap distances | ... ]
    //   [ warp0 heap indices   | warp1 heap indices   | ... ]
    //   [ block-wide cached data tile                         ]
    float* all_heap_dist = reinterpret_cast<float*>(smem);
    int* all_heap_idx = reinterpret_cast<int*>(all_heap_dist + TOTAL_TOPK);
    float2* batch_points = reinterpret_cast<float2*>(all_heap_idx + TOTAL_TOPK);

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid >> 5;

    const int query_index = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active = (query_index < query_count);

    float qx = 0.0f;
    float qy = 0.0f;

    if (active && (lane == 0)) {
        const float2 q = query[query_index];
        qx = q.x;
        qy = q.y;
    }

    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    float* heap_dist = all_heap_dist + warp_id * K;
    int* heap_idx = all_heap_idx + warp_id * K;

    for (int base = 0; base < data_count; base += BATCH_SIZE) {
        int batch_count = data_count - base;
        if (batch_count > BATCH_SIZE) {
            batch_count = BATCH_SIZE;
        }

        load_data_batch<BLOCK_THREADS>(data, base, batch_count, batch_points);
        __syncthreads();

        if (active) {
            if (base == 0) {
                initialize_heap_from_first_k<K>(qx, qy, batch_points, heap_dist, heap_idx);
                process_cached_batch<K>(qx, qy, batch_points, 0, K, batch_count, heap_dist, heap_idx);
            } else {
                process_cached_batch<K>(qx, qy, batch_points, base, 0, batch_count, heap_dist, heap_idx);
            }
        }

        // Ensure every warp has finished reading the current tile before the block overwrites it.
        __syncthreads();
    }

    if (active) {
        heap_sort_ascending<K>(heap_dist, heap_idx);

        const std::size_t out_base =
            static_cast<std::size_t>(query_index) * static_cast<std::size_t>(K);

#pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            store_result_pair(result, out_base + static_cast<std::size_t>(i),
                              heap_idx[i], heap_dist[i]);
        }
    }
}

template <int K>
inline void launch_knn_specialized(const float2* query,
                                   int query_count,
                                   const float2* data,
                                   int data_count,
                                   std::pair<int, float>* result) {
    using Config = KnnConfig<K>;
    constexpr std::size_t smem_bytes = shared_memory_bytes<K>();

    static_assert((smem_bytes % sizeof(int4)) == 0,
                  "The shared-memory layout relies on 16-byte alignment.");
    static_assert(smem_bytes <= (96u * 1024u),
                  "Chosen specialization exceeds the intended per-block shared-memory budget.");

    const dim3 block(Config::BLOCK_THREADS, 1, 1);
    const dim3 grid((query_count + Config::WARPS_PER_BLOCK - 1) / Config::WARPS_PER_BLOCK, 1, 1);

    // Opt in to the larger dynamic shared-memory footprint and strongly prefer SMEM
    // over L1 for this kernel. The target hardware is A100/H100-class, where this
    // configuration is appropriate.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    // Default-stream launch; the caller may synchronize externally as needed.
    knn_kernel<K><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result);
}

}  // namespace knn_detail

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:
            knn_detail::launch_knn_specialized<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_detail::launch_knn_specialized<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_detail::launch_knn_specialized<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_detail::launch_knn_specialized<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_detail::launch_knn_specialized<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_detail::launch_knn_specialized<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // Inputs are documented to be valid; keep a defensive default anyway.
            return;
    }
}