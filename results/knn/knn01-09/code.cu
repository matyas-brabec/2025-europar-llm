#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace {

// Exact brute-force k-NN for 2D points.
//
// Main design choices:
//
// 1) One warp owns one query.
//    - 256 threads / CTA => 8 warps / CTA => 8 queries processed together.
//
// 2) The CTA streams the data set through shared memory.
//    - Each loaded data tile is reused by all 8 queries in the CTA.
//    - This is the dominant optimization because the kernel is bandwidth-bound.
//
// 3) Each query keeps an exact top-k as a max-heap in shared memory.
//    - Only lane 0 mutates the heap.
//    - All 32 lanes compute distances and forward only promising candidates
//      to lane 0 via ballot+shuffle.
//
// 4) Heap entries are stored as a single 64-bit key:
//        high 32 bits = float distance bit pattern
//        low  32 bits = integer data index
//    For non-negative floats, unsigned ordering of the high 32 bits matches
//    numeric ordering of the distance, so unsigned 64-bit ordering matches
//    lexicographic ordering by (distance, index).
//    This makes heap comparisons cheap and also lets the final write be a
//    single 64-bit store directly into the pair<int,float> output buffer on
//    the standard little-endian CUDA ABI.
//
// 5) Shared memory budget is fixed at 80 KiB / CTA.
//    - On A100, 2 CTAs/SM fit exactly (163,840 B / SM).
//    - H100 also handles this comfortably.
//    - Tile size is specialized by K so that heap bytes + tile bytes = 80 KiB.

using PackedKnnWord = unsigned long long;

constexpr int KNN_THREADS          = 256;
constexpr int KNN_WARPS_PER_BLOCK  = KNN_THREADS / 32;
constexpr int KNN_SHARED_BYTES     = 80 * 1024;
constexpr int KNN_BATCH_GROUPS     = 4;   // refresh cutoff every 128 points / warp

static_assert(sizeof(int) == 4, "This implementation assumes 32-bit int.");
static_assert(sizeof(float) == 4, "This implementation assumes IEEE-754 float.");
static_assert(sizeof(PackedKnnWord) == 8, "PackedKnnWord must be 64-bit.");
static_assert(sizeof(std::pair<int, float>) == sizeof(PackedKnnWord),
              "This implementation assumes std::pair<int,float> occupies 8 bytes.");

template <int K>
struct KnnConfig {
    static constexpr int HEAP_BYTES  = KNN_WARPS_PER_BLOCK * K * sizeof(PackedKnnWord);
    static constexpr int TILE_POINTS = (KNN_SHARED_BYTES - HEAP_BYTES) / (2 * sizeof(float));

    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0),
                  "k must be a power of two in [32, 1024].");
    static_assert(TILE_POINTS >= K, "The first tile must be large enough to seed the heap.");
    static_assert((TILE_POINTS % 32) == 0, "Tile size must be warp-aligned.");
    static_assert(2 * TILE_POINTS * static_cast<int>(sizeof(float)) + HEAP_BYTES == KNN_SHARED_BYTES,
                  "Shared-memory layout must exactly match the 80 KiB budget.");
};

__device__ __forceinline__ float sq_l2(const float qx, const float qy,
                                       const float px, const float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return fmaf(dx, dx, dy * dy);
}

__device__ __forceinline__ PackedKnnWord pack_key(const float dist, const int idx) {
    return (static_cast<PackedKnnWord>(__float_as_uint(dist)) << 32) |
           static_cast<unsigned int>(idx);
}

__device__ __forceinline__ void sift_down_max(PackedKnnWord* heap, int root, const int count) {
    PackedKnnWord value = heap[root];

    for (int child = (root << 1) + 1; child < count; child = (root << 1) + 1) {
        int worst_child = child;
        const int right = child + 1;
        if (right < count && heap[right] > heap[child]) {
            worst_child = right;
        }
        if (heap[worst_child] <= value) {
            break;
        }
        heap[root] = heap[worst_child];
        root = worst_child;
    }

    heap[root] = value;
}

__device__ __forceinline__ void heapify_max(PackedKnnWord* heap, const int count) {
    for (int root = (count >> 1) - 1; root >= 0; --root) {
        sift_down_max(heap, root, count);
    }
}

__device__ __forceinline__ void replace_root_if_better(PackedKnnWord* heap,
                                                       const PackedKnnWord cand,
                                                       const int count) {
    if (cand < heap[0]) {
        heap[0] = cand;
        sift_down_max(heap, 0, count);
    }
}

__device__ __forceinline__ void heapsort_ascending(PackedKnnWord* heap, const int count) {
    for (int end = count - 1; end > 0; --end) {
        const PackedKnnWord tmp = heap[0];
        heap[0] = heap[end];
        heap[end] = tmp;
        sift_down_max(heap, 0, end);
    }
}

__device__ __forceinline__ void load_tile_soa(const float2* __restrict__ data,
                                              const int base,
                                              const int count,
                                              float* __restrict__ tile_x,
                                              float* __restrict__ tile_y) {
    // Shared-memory tile is stored as SoA instead of AoS to avoid 64-bit shared
    // memory bank conflicts when a warp reads consecutive points.
    for (int t = threadIdx.x; t < count; t += KNN_THREADS) {
        const float2 p = data[base + t];
        tile_x[t] = p.x;
        tile_y[t] = p.y;
    }
}

template <int K>
__device__ __forceinline__ void process_tile_for_query(
    const float qx, const float qy,
    const float* __restrict__ tile_x,
    const float* __restrict__ tile_y,
    const int tile_base,
    const int tile_count,
    const int local_start,
    PackedKnnWord* __restrict__ heap) {

    constexpr unsigned FULL_MASK = 0xffffffffu;
    constexpr int GROUP_POINTS = 32 * KNN_BATCH_GROUPS;

    const int lane = threadIdx.x & 31;
    const PackedKnnWord invalid_key = ~PackedKnnWord(0);

    PackedKnnWord cutoff_key =
        __shfl_sync(FULL_MASK, (lane == 0 ? heap[0] : PackedKnnWord(0)), 0);

    for (int group_base = local_start; group_base < tile_count; group_base += GROUP_POINTS) {
        bool heap_changed = false;

        #pragma unroll
        for (int sub = 0; sub < KNN_BATCH_GROUPS; ++sub) {
            const int loc = group_base + (sub << 5) + lane;

            PackedKnnWord cand = invalid_key;
            if (loc < tile_count) {
                const float d = sq_l2(qx, qy, tile_x[loc], tile_y[loc]);
                cand = pack_key(d, tile_base + loc);
            }

            // Using a slightly stale cutoff inside this short group is safe:
            // the heap root only ever decreases, so this can create extra
            // candidate traffic but cannot miss a true top-k element. Lane 0
            // always rechecks against the current heap root before insertion.
            unsigned mask = __ballot_sync(FULL_MASK, loc < tile_count && cand < cutoff_key);

            if (mask) {
                heap_changed = true;
                while (mask) {
                    const int src = __ffs(mask) - 1;
                    const PackedKnnWord candidate = __shfl_sync(FULL_MASK, cand, src);
                    if (lane == 0) {
                        replace_root_if_better(heap, candidate, K);
                    }
                    mask &= mask - 1;
                }
            }
        }

        if (heap_changed) {
            __syncwarp();
            cutoff_key = __shfl_sync(FULL_MASK,
                                     (lane == 0 ? heap[0] : PackedKnnWord(0)),
                                     0);
        }
    }
}

template <int K>
__global__ __launch_bounds__(256, 2)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                PackedKnnWord* __restrict__ result_words) {
    using C = KnnConfig<K>;
    constexpr unsigned FULL_MASK = 0xffffffffu;

    extern __shared__ __align__(8) unsigned char smem[];

    float* const tile_x = reinterpret_cast<float*>(smem);
    float* const tile_y = tile_x + C::TILE_POINTS;
    PackedKnnWord* const heaps = reinterpret_cast<PackedKnnWord*>(tile_y + C::TILE_POINTS);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int query_id = static_cast<int>(blockIdx.x) * KNN_WARPS_PER_BLOCK + warp;
    const bool active = query_id < query_count;

    PackedKnnWord* const my_heap = heaps + warp * K;

    float qx_local = 0.0f;
    float qy_local = 0.0f;
    if (lane == 0 && active) {
        const float2 q = query[query_id];
        qx_local = q.x;
        qy_local = q.y;
    }

    const float qx = __shfl_sync(FULL_MASK, qx_local, 0);
    const float qy = __shfl_sync(FULL_MASK, qy_local, 0);

    // First tile: also seeds the heap. By construction TILE_POINTS >= K.
    int tile_count = (data_count < C::TILE_POINTS) ? data_count : C::TILE_POINTS;
    load_tile_soa(data, 0, tile_count, tile_x, tile_y);
    __syncthreads();

    if (active && lane == 0) {
        for (int i = 0; i < K; ++i) {
            my_heap[i] = pack_key(sq_l2(qx, qy, tile_x[i], tile_y[i]), i);
        }
        heapify_max(my_heap, K);
    }
    __syncwarp();

    if (active && tile_count > K) {
        process_tile_for_query<K>(qx, qy, tile_x, tile_y, 0, tile_count, K, my_heap);
    }
    __syncthreads();

    // Remaining tiles.
    for (int tile_base = C::TILE_POINTS; tile_base < data_count; tile_base += C::TILE_POINTS) {
        tile_count = data_count - tile_base;
        if (tile_count > C::TILE_POINTS) {
            tile_count = C::TILE_POINTS;
        }

        load_tile_soa(data, tile_base, tile_count, tile_x, tile_y);
        __syncthreads();

        if (active) {
            process_tile_for_query<K>(qx, qy, tile_x, tile_y, tile_base, tile_count, 0, my_heap);
        }

        __syncthreads();
    }

    // Convert the max-heap into ascending nearest-neighbor order.
    if (active && lane == 0) {
        heapsort_ascending(my_heap, K);
    }
    __syncwarp();

    if (active) {
        const std::size_t out_base = static_cast<std::size_t>(query_id) * K;
        for (int j = lane; j < K; j += 32) {
            // The packed key stores:
            //   low  32 bits = index
            //   high 32 bits = float distance bits
            // This matches the in-memory layout of pair<int,float> on the
            // little-endian CUDA ABI, so a single 64-bit store writes the
            // required {index, distance} pair.
            result_words[out_base + j] = my_heap[j];
        }
    }
}

template <int K>
inline void launch_knn_impl(const float2* query,
                            const int query_count,
                            const float2* data,
                            const int data_count,
                            PackedKnnWord* result_words) {
    using C = KnnConfig<K>;

    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               KNN_SHARED_BYTES);
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const dim3 block(KNN_THREADS);
    const dim3 grid((query_count + KNN_WARPS_PER_BLOCK - 1) / KNN_WARPS_PER_BLOCK);

    knn_kernel<K><<<grid, block, KNN_SHARED_BYTES>>>(
        query, query_count, data, data_count, result_words);
}

} // anonymous namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    PackedKnnWord* const result_words = reinterpret_cast<PackedKnnWord*>(result);

    switch (k) {
        case 32:   launch_knn_impl<32>(query, query_count, data, data_count, result_words); break;
        case 64:   launch_knn_impl<64>(query, query_count, data, data_count, result_words); break;
        case 128:  launch_knn_impl<128>(query, query_count, data, data_count, result_words); break;
        case 256:  launch_knn_impl<256>(query, query_count, data, data_count, result_words); break;
        case 512:  launch_knn_impl<512>(query, query_count, data, data_count, result_words); break;
        case 1024: launch_knn_impl<1024>(query, query_count, data, data_count, result_words); break;
        default:   break; // k is guaranteed valid by the caller contract.
    }
}