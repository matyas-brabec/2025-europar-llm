#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace knn_cuda_detail {

// Exact brute-force 2D k-NN specialized for modern NVIDIA data-center GPUs.
//
// Design summary:
//  - One warp computes one query.
//  - The whole block stages a batch of input data points into shared memory.
//  - The staged batch is stored in shared memory as SoA (x[] / y[]) rather than AoS
//    to avoid the 64-bit shared-memory bank conflicts that float2 would cause in
//    the hot distance-computation loop.
//  - Each warp owns two disjoint on-chip arrays:
//      * topk[] : the current intermediate k nearest neighbors for that query.
//      * cand[] : a k-entry candidate buffer.
//    Both arrays live in shared memory, but each warp has a private, non-overlapping
//    slice, which satisfies the "private copy per query" requirement without using
//    any extra global/device memory.
//  - Distances and indices are packed into one 64-bit key:
//        high 32 bits = IEEE-754 bits of the non-negative squared distance
//        low  32 bits = data-point index
//    Because squared distances are non-negative, unsigned integer ordering of the
//    float bit pattern matches numeric ordering, so one integer compare orders
//    (distance, index) lexicographically.
//  - When the candidate buffer fills, it is sorted cooperatively by the warp with a
//    bitonic network, then merged with topk[] using a warp-parallel merge-path merge.
//    The merge is stable with A-before-B tie handling so equal sentinels remain correct.
//
// Tuning summary (chosen to fit A100's 160 KiB/block opt-in shared-memory limit while
// maximizing reuse of each loaded data batch):
//   k=32   -> 32 warps/block, batch=8192  (80 KiB/block, enables 2 resident blocks/SM)
//   k=64   -> 32 warps/block, batch=6144  (80 KiB/block, enables 2 resident blocks/SM)
//   k=128  -> 32 warps/block, batch=12288 (160 KiB/block, maximizes reuse per batch)
//   k=256  -> 32 warps/block, batch=4096  (160 KiB/block)
//   k=512  -> 16 warps/block, batch=4096  (160 KiB/block)
//   k=1024 ->  8 warps/block, batch=4096  (160 KiB/block)

constexpr int kWarpSize = 32;
constexpr std::size_t kMaxOptinDynamicSharedBytesA100 = 163840;

using KeyT = std::uint64_t;

// We write into device memory allocated for std::pair<int,float>. To avoid relying on
// std::pair being device-constructible, the kernel uses an ABI-compatible POD overlay.
// Mainstream CUDA host compilers lay out std::pair<int,float> as two adjacent 32-bit
// scalars; the size check below enforces that assumption at compile time.
struct ResultPairABI {
    int first;
    float second;
};
static_assert(sizeof(ResultPairABI) == sizeof(std::pair<int, float>),
              "std::pair<int,float> is expected to occupy 8 bytes.");

constexpr unsigned kInfBits = 0x7f800000u;
constexpr KeyT kInfKey =
    (static_cast<KeyT>(kInfBits) << 32) | static_cast<KeyT>(0xffffffffu);

template <int K>
struct KnnConfig;

template <>
struct KnnConfig<32> {
    static constexpr int kWarpsPerBlock = 32;
    static constexpr int kBatchPoints = 8192;
};

template <>
struct KnnConfig<64> {
    static constexpr int kWarpsPerBlock = 32;
    static constexpr int kBatchPoints = 6144;
};

template <>
struct KnnConfig<128> {
    static constexpr int kWarpsPerBlock = 32;
    static constexpr int kBatchPoints = 12288;
};

template <>
struct KnnConfig<256> {
    static constexpr int kWarpsPerBlock = 32;
    static constexpr int kBatchPoints = 4096;
};

template <>
struct KnnConfig<512> {
    static constexpr int kWarpsPerBlock = 16;
    static constexpr int kBatchPoints = 4096;
};

template <>
struct KnnConfig<1024> {
    static constexpr int kWarpsPerBlock = 8;
    static constexpr int kBatchPoints = 4096;
};

template <int K, int WARPS_PER_BLOCK, int BATCH_POINTS>
constexpr std::size_t shared_storage_bytes() {
    return static_cast<std::size_t>(BATCH_POINTS) * sizeof(float) * 2u +  // x[] + y[]
           static_cast<std::size_t>(WARPS_PER_BLOCK) * static_cast<std::size_t>(K) *
               sizeof(KeyT) * 2u;  // topk[] + cand[]
}

__device__ __forceinline__ KeyT pack_key(const unsigned dist_bits, const unsigned idx) {
    return (static_cast<KeyT>(dist_bits) << 32) | static_cast<KeyT>(idx);
}

__device__ __forceinline__ unsigned key_dist_bits(const KeyT key) {
    return static_cast<unsigned>(key >> 32);
}

__device__ __forceinline__ float key_dist(const KeyT key) {
    return __uint_as_float(key_dist_bits(key));
}

__device__ __forceinline__ int key_index(const KeyT key) {
    return static_cast<int>(static_cast<unsigned>(key));
}

// Warp-cooperative bitonic sort in shared memory for one warp-private array.
// One warp handles one array; after every stage we use __syncwarp() because Volta+
// allows independent thread scheduling.
template <int N>
__device__ __forceinline__ void warp_bitonic_sort(KeyT* arr) {
    static_assert((N & (N - 1)) == 0, "N must be a power of two.");
    const int lane = threadIdx.x & (kWarpSize - 1);

    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int i = lane; i < N; i += kWarpSize) {
                const int partner = i ^ stride;
                if (partner > i) {
                    const KeyT a = arr[i];
                    const KeyT b = arr[partner];
                    const bool ascending = ((i & size) == 0);
                    if ((ascending && (b < a)) || (!ascending && (a < b))) {
                        arr[i] = b;
                        arr[partner] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge-path partition for the first "diag" outputs of a stable merge of A and B.
// Stability is A-before-B on equal keys. That detail matters because padding uses
// identical +inf sentinels on both inputs.
template <int K>
__device__ __forceinline__ int merge_path_partition(const KeyT* a,
                                                    const KeyT* b,
                                                    const int diag) {
    int low = (diag > K) ? (diag - K) : 0;
    int high = (diag < K) ? diag : K;

    while (low <= high) {
        const int a_count = (low + high) >> 1;
        const int b_count = diag - a_count;

        const bool move_left =
            (a_count > 0) && (b_count < K) && (b[b_count] < a[a_count - 1]);
        const bool move_right =
            (b_count > 0) && (a_count < K) && !(b[b_count - 1] < a[a_count]);

        if (move_left) {
            high = a_count - 1;
        } else if (move_right) {
            low = a_count + 1;
        } else {
            return a_count;
        }
    }
    return low;
}

// Merge two sorted K-element arrays (topk and cand) and keep only the first K outputs.
// Each lane computes K/32 contiguous outputs into registers, then writes them back.
template <int K>
__device__ __forceinline__ void merge_sorted_topk(KeyT* topk, const KeyT* cand) {
    constexpr int ITEMS_PER_LANE = K / kWarpSize;
    static_assert(ITEMS_PER_LANE >= 1, "K must be at least warp size.");
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int out_begin = lane * ITEMS_PER_LANE;

    KeyT local_out[ITEMS_PER_LANE];

    int a_pos = merge_path_partition<K>(topk, cand, out_begin);
    int b_pos = out_begin - a_pos;

#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const bool take_a =
            (b_pos >= K) || ((a_pos < K) && !(cand[b_pos] < topk[a_pos]));
        local_out[t] = take_a ? topk[a_pos++] : cand[b_pos++];
    }

    // Ensure all lanes finished reading the old topk[] before any lane overwrites it.
    __syncwarp();

#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        topk[out_begin + t] = local_out[t];
    }

    __syncwarp();
}

// Pad the candidate buffer to K with +inf, sort it, merge it into topk[], and return
// the new kth-neighbor distance bits (topk[K-1].distance).
template <int K>
__device__ __forceinline__ unsigned merge_buffer_into_topk(KeyT* topk,
                                                           KeyT* cand,
                                                           const int cand_count) {
    const int lane = threadIdx.x & (kWarpSize - 1);

    if (cand_count < K) {
#pragma unroll
        for (int i = cand_count + lane; i < K; i += kWarpSize) {
            cand[i] = kInfKey;
        }
    }

    __syncwarp();
    warp_bitonic_sort<K>(cand);
    merge_sorted_topk<K>(topk, cand);
    return key_dist_bits(topk[K - 1]);
}

template <int K, int WARPS_PER_BLOCK, int BATCH_POINTS>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                ResultPairABI* __restrict__ result) {
    static_assert((K & (K - 1)) == 0 && K >= 32 && K <= 1024,
                  "Supported K values are powers of two in [32, 1024].");
    static_assert((K % kWarpSize) == 0, "K must be divisible by warp size.");
    static_assert((BATCH_POINTS % kWarpSize) == 0,
                  "Batch size must be a multiple of warp size.");
    static_assert(WARPS_PER_BLOCK * kWarpSize <= 1024,
                  "Block size exceeds CUDA limit.");
    static_assert(shared_storage_bytes<K, WARPS_PER_BLOCK, BATCH_POINTS>() <=
                      kMaxOptinDynamicSharedBytesA100,
                  "Shared-memory footprint exceeds the A100-compatible limit.");

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;

    // One dynamically allocated shared-memory slab:
    //   [ batch_x[BATCH_POINTS] | batch_y[BATCH_POINTS] | topk[warps][K] | cand[warps][K] ]
    extern __shared__ unsigned long long smem_ull[];

    float* const batch_x = reinterpret_cast<float*>(smem_ull);
    float* const batch_y = batch_x + BATCH_POINTS;
    KeyT* const topk_all = reinterpret_cast<KeyT*>(batch_y + BATCH_POINTS);
    KeyT* const cand_all = topk_all + (WARPS_PER_BLOCK * K);

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid >> 5;
    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active_query = (query_idx < query_count);

    KeyT* const topk = topk_all + warp_id * K;
    KeyT* const cand = cand_all + warp_id * K;

    // Warp-private state for this query.
    // topk[] is kept globally sorted in ascending distance order.
#pragma unroll
    for (int i = lane; i < K; i += kWarpSize) {
        topk[i] = kInfKey;
    }
    __syncwarp();

    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && active_query) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xffffffffu, qx, 0);
    qy = __shfl_sync(0xffffffffu, qy, 0);

    const unsigned lower_lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);

    unsigned kth_bits = kInfBits;
    int cand_count = 0;
    bool done = false;

    for (int batch_begin = 0; batch_begin < data_count; batch_begin += BATCH_POINTS) {
        const int remaining = data_count - batch_begin;
        const int batch_count = (remaining < BATCH_POINTS) ? remaining : BATCH_POINTS;

        // Cooperative block-wide load from global memory into shared memory.
        // AoS in global memory -> SoA in shared memory for conflict-free hot reads.
        for (int i = tid; i < batch_count; i += THREADS_PER_BLOCK) {
            const float2 p = data[batch_begin + i];
            batch_x[i] = p.x;
            batch_y[i] = p.y;
        }

        __syncthreads();

        if (active_query && !done) {
            // Iterate in warp-sized chunks so every lane participates in every ballot.
            for (int base = 0; base < batch_count; base += kWarpSize) {
                const int local_i = base + lane;
                const bool valid = (local_i < batch_count);

                float dist = 0.0f;
                unsigned dist_bits = 0u;

                if (valid) {
                    const float dx = qx - batch_x[local_i];
                    const float dy = qy - batch_y[local_i];
                    dist = fmaf(dx, dx, dy * dy);
                    dist_bits = __float_as_uint(dist);
                }

                // Strictly closer only: the prompt explicitly allows arbitrary tie handling.
                bool accept = valid && (dist_bits < kth_bits);
                unsigned mask = __ballot_sync(0xffffffffu, accept);
                int accepted = __popc(mask);

                if (accepted) {
                    // If the current ballot would overflow the candidate buffer, merge the
                    // existing buffer first. Then re-test the current ballot against the
                    // tighter kth threshold produced by the merge.
                    if (cand_count + accepted > K) {
                        kth_bits = merge_buffer_into_topk<K>(topk, cand, cand_count);
                        cand_count = 0;

                        if (kth_bits == 0u) {
                            // Squared distances are non-negative, so once kth == 0, no future
                            // point can be strictly closer.
                            done = true;
                            break;
                        }

                        accept = valid && (dist_bits < kth_bits);
                        mask = __ballot_sync(0xffffffffu, accept);
                        accepted = __popc(mask);
                    }

                    if (accepted) {
                        if (accept) {
                            const int rank = __popc(mask & lower_lane_mask);
                            cand[cand_count + rank] =
                                pack_key(dist_bits, static_cast<unsigned>(batch_begin + local_i));
                        }

                        cand_count += accepted;
                        __syncwarp();

                        if (cand_count == K) {
                            kth_bits = merge_buffer_into_topk<K>(topk, cand, cand_count);
                            cand_count = 0;

                            if (kth_bits == 0u) {
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    if (active_query) {
        if (!done && cand_count > 0) {
            kth_bits = merge_buffer_into_topk<K>(topk, cand, cand_count);
            (void)kth_bits;
        }

        ResultPairABI* const out = result + static_cast<std::size_t>(query_idx) * K;

#pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            const KeyT key = topk[i];
            out[i].first = key_index(key);
            out[i].second = key_dist(key);
        }
    }
}

template <int K>
inline void launch_specialized(const float2* query,
                               const int query_count,
                               const float2* data,
                               const int data_count,
                               std::pair<int, float>* result) {
    constexpr int WARPS_PER_BLOCK = KnnConfig<K>::kWarpsPerBlock;
    constexpr int BATCH_POINTS = KnnConfig<K>::kBatchPoints;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * kWarpSize;
    constexpr std::size_t SHARED_BYTES =
        shared_storage_bytes<K, WARPS_PER_BLOCK, BATCH_POINTS>();

    static_assert(SHARED_BYTES <= kMaxOptinDynamicSharedBytesA100,
                  "Launch configuration exceeds the A100-compatible shared-memory budget.");

    // Opt in to large dynamic shared memory. The wrapper intentionally remains
    // asynchronous; callers retain full control over stream synchronization and
    // error handling.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK, BATCH_POINTS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SHARED_BYTES));
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK, BATCH_POINTS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    knn_kernel<K, WARPS_PER_BLOCK, BATCH_POINTS>
        <<<blocks, THREADS_PER_BLOCK, SHARED_BYTES>>>(
            query,
            query_count,
            data,
            data_count,
            reinterpret_cast<ResultPairABI*>(result));
}

}  // namespace knn_cuda_detail

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
            knn_cuda_detail::launch_specialized<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_cuda_detail::launch_specialized<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_cuda_detail::launch_specialized<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_cuda_detail::launch_specialized<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_cuda_detail::launch_specialized<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_cuda_detail::launch_specialized<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // The prompt guarantees valid k (power of two in [32, 1024]).
            break;
    }
}