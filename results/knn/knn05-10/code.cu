#include <cuda_runtime.h>
#include <utility>
#include <cstdint>
#include <cstddef>

namespace knn_detail {

// The interface requires std::pair<int, float> in device memory.
// Writing fields individually avoids depending on any host/device annotation on pair assignment.
using result_pair_t = std::pair<int, float>;

// A packed key stores [distance_bits | index] in one 64-bit word.
// Distances are squared Euclidean distances, hence non-negative; for non-negative IEEE-754 floats,
// unsigned integer order matches numeric order, so a plain integer compare is enough.
using packed_key_t = std::uint64_t;

constexpr int      kWarpSize      = 32;
constexpr unsigned kFullMask      = 0xFFFFFFFFu;
constexpr int      kBlockThreads  = 256;   // 8 warps/block => 8 queries/block.
constexpr int      kBatchPoints   = 2048;  // 16 KiB tile of float2 points.
constexpr int      kSentinelIndex = 0x7FFFFFFF;

// Tuning rationale for A100/H100-class GPUs:
// * One warp computes one query, so block size is chosen in warps, not arbitrary threads.
// * 256 threads/block keeps grid granularity high for query counts in the low thousands.
// * 2048 cached data points amortize global memory loads well.
// * Worst-case shared memory footprint at K=1024 is:
//       2048 * sizeof(float2) + 8 warps * (2 * 1024 * sizeof(uint64_t))
//     = 16 KiB + 128 KiB = 144 KiB/block,
//   which fits comfortably under the opt-in per-block shared-memory budget on A100/H100.

template <int K, int BLOCK_THREADS, int BATCH_POINTS>
struct KernelConfig {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    static_assert(BLOCK_THREADS % kWarpSize == 0, "BLOCK_THREADS must be a multiple of 32.");
    static_assert(BATCH_POINTS % kWarpSize == 0, "BATCH_POINTS must be a multiple of 32.");
    static_assert(BATCH_POINTS % BLOCK_THREADS == 0, "BATCH_POINTS should be divisible by BLOCK_THREADS for fully unrolled cooperative loads.");

    static constexpr int kWarpsPerBlock = BLOCK_THREADS / kWarpSize;
    static constexpr std::size_t kBatchBytes =
        static_cast<std::size_t>(BATCH_POINTS) * sizeof(float2);
    static constexpr std::size_t kPerWarpStateBytes =
        static_cast<std::size_t>(2) * K * sizeof(packed_key_t); // [0, K): top-K, [K, 2K): candidate buffer
    static constexpr std::size_t kSharedBytes =
        kBatchBytes + static_cast<std::size_t>(kWarpsPerBlock) * kPerWarpStateBytes;

    static_assert((kBatchBytes % alignof(packed_key_t)) == 0, "Shared-memory layout must preserve packed_key_t alignment.");
};

__device__ __forceinline__ packed_key_t pack_key(const float dist, const int idx) {
    return (static_cast<packed_key_t>(__float_as_uint(dist)) << 32) |
           static_cast<std::uint32_t>(idx);
}

__device__ __forceinline__ float unpack_dist(const packed_key_t key) {
    return __uint_as_float(static_cast<std::uint32_t>(key >> 32));
}

__device__ __forceinline__ int unpack_idx(const packed_key_t key) {
    return static_cast<int>(static_cast<std::uint32_t>(key));
}

__device__ __forceinline__ void store_result_pair(result_pair_t* dst, const int idx, const float dist) {
    dst->first  = idx;
    dst->second = dist;
}

// Full ascending bitonic sort of a power-of-two segment in shared memory.
// Only one warp touches the segment, so __syncwarp is sufficient.
template <int N>
__device__ __forceinline__ void bitonic_sort_segment(packed_key_t* base, const int lane) {
#pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
#pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    const packed_key_t ai = base[i];
                    const packed_key_t aj = base[j];
                    const bool up = ((i & size) == 0);
                    const bool do_swap = up ? (ai > aj) : (ai < aj);
                    if (do_swap) {
                        base[i] = aj;
                        base[j] = ai;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

// Ascending bitonic merge of a segment that is already known to be bitonic.
template <int N>
__device__ __forceinline__ void bitonic_merge_segment(packed_key_t* base, const int lane) {
#pragma unroll 1
    for (int stride = N >> 1; stride > 0; stride >>= 1) {
        for (int i = lane; i < N; i += kWarpSize) {
            const int j = i ^ stride;
            if (j > i) {
                const packed_key_t ai = base[i];
                const packed_key_t aj = base[j];
                if (ai > aj) {
                    base[i] = aj;
                    base[j] = ai;
                }
            }
        }
        __syncwarp(kFullMask);
    }
}

// Merge the candidate buffer into the already-sorted top-K array.
// Storage layout for one warp/query:
//   storage[0 .. K-1]     : current top-K, sorted ascending
//   storage[K .. 2K-1]    : candidate buffer, compacted at the front, unsorted
//
// To avoid needing an extra 3rd K-sized buffer, the merge is done as:
//   1) pad unused candidate slots with +INF,
//   2) sort the candidate buffer ascending,
//   3) bitonic split top[i] against cand[K-1-i], which moves the smallest K elements into top
//      as a bitonic sequence,
//   4) bitonic-merge only top back to ascending order.
// This is cheaper than sorting the full 2K union and satisfies the "no extra device memory" rule.
template <int K>
__device__ __forceinline__ float merge_buffer_into_top(
    packed_key_t* storage,
    const int lane,
    const int valid_count,
    const packed_key_t inf_key)
{
    packed_key_t* const top  = storage;
    packed_key_t* const cand = storage + K;

    if (valid_count < K) {
        for (int i = lane; i < K; i += kWarpSize) {
            if (i >= valid_count) {
                cand[i] = inf_key;
            }
        }
    }

    // Make all candidate writes/padding visible before sorting.
    __syncwarp(kFullMask);

    bitonic_sort_segment<K>(cand, lane);

    // Bitonic split: compare the ascending top half with the descending candidate half.
    // We only need the smaller side; the larger side can be discarded because the next merge
    // overwrites/re-pads the candidate area anyway.
    for (int i = lane; i < K; i += kWarpSize) {
        const int j = K - 1 - i;
        const packed_key_t a = top[i];
        const packed_key_t b = cand[j];
        if (a > b) {
            top[i] = b;
        }
    }

    __syncwarp(kFullMask);

    bitonic_merge_segment<K>(top, lane);

    float kth = 0.0f;
    if (lane == 0) {
        kth = unpack_dist(top[K - 1]);
    }
    return __shfl_sync(kFullMask, kth, 0);
}

template <int K, int BLOCK_THREADS, int BATCH_POINTS>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    const int query_count,
    const float2* __restrict__ data,
    const int data_count,
    result_pair_t* __restrict__ result)
{
    using Cfg = KernelConfig<K, BLOCK_THREADS, BATCH_POINTS>;

    const int lane          = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int query_idx     = static_cast<int>(blockIdx.x) * Cfg::kWarpsPerBlock + warp_in_block;
    const bool active_query = (query_idx < query_count);

    // Shared-memory layout:
    //   [batch tile of BATCH_POINTS float2]
    //   [per-warp 2*K packed_key_t state]
    extern __shared__ packed_key_t smem_u64[];
    unsigned char* const smem      = reinterpret_cast<unsigned char*>(smem_u64);
    float2* const batch            = reinterpret_cast<float2*>(smem);
    packed_key_t* const warp_state = reinterpret_cast<packed_key_t*>(smem + Cfg::kBatchBytes);

    packed_key_t* const my_state =
        warp_state + static_cast<std::size_t>(warp_in_block) * (2 * K);

    const unsigned lane_mask_lt = (1u << lane) - 1u;
    const packed_key_t inf_key  = pack_key(CUDART_INF_F, kSentinelIndex);

    float qx = 0.0f;
    float qy = 0.0f;
    float kth = CUDART_INF_F;
    int   buf_count = 0;

    if (active_query) {
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);

        // Top-K lives in the warp-private slice of shared memory.
        // K can be as large as 1024, so a full register-resident copy would be too large;
        // shared memory keeps it private to the warp without spilling to local/global memory.
        for (int i = lane; i < K; i += kWarpSize) {
            my_state[i] = inf_key;
        }

        __syncwarp(kFullMask);
    }

    for (int batch_base = 0; batch_base < data_count; batch_base += BATCH_POINTS) {
        const int remaining = data_count - batch_base;
        const int batch_n   = (remaining < BATCH_POINTS) ? remaining : BATCH_POINTS;
        const float2* const tile = data + batch_base;

        // Whole-block cooperative load of the current data tile into shared memory.
        // With the chosen tuning parameters this loop has exactly 8 iterations/thread.
#pragma unroll
        for (int i = threadIdx.x; i < BATCH_POINTS; i += BLOCK_THREADS) {
            if (i < batch_n) {
                batch[i] = tile[i];
            }
        }

        __syncthreads();

        if (active_query) {
            // Process the shared-memory tile in warp-wide stripes of 32 points.
            // This keeps all warp-wide ballots/shuffles structurally aligned even for tail tiles.
#pragma unroll 1
            for (int j_base = 0; j_base < BATCH_POINTS; j_base += kWarpSize) {
                if (j_base >= batch_n) {
                    break;
                }

                const int j = j_base + lane;

                float dist = 0.0f;
                int   idx  = 0;
                bool  accept = false;

                if (j < batch_n) {
                    const float2 p = batch[j];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);
                    idx = batch_base + j;

                    // Skip candidates that are not closer than the current K-th neighbor.
                    accept = (dist < kth);
                }

                // Warp-wide compact append into the candidate buffer without atomics:
                // ballot -> prefix rank -> dense write.
                unsigned active_mask = __ballot_sync(kFullMask, accept);

                while (active_mask != 0u) {
                    const int rank    = accept ? __popc(active_mask & lane_mask_lt) : 0;
                    const int pending = __popc(active_mask);
                    const int space   = K - buf_count;
                    const int take    = (pending < space) ? pending : space;

                    if (accept && rank < take) {
                        my_state[K + buf_count + rank] = pack_key(dist, idx);
                    }

                    // Candidate writes must be visible before a possible merge.
                    __syncwarp(kFullMask);

                    buf_count += take;

                    if (accept && rank < take) {
                        accept = false;
                    }

                    if (buf_count == K) {
                        kth = merge_buffer_into_top<K>(my_state, lane, K, inf_key);
                        buf_count = 0;
                    }

                    // If the previous append overflowed the buffer, the remaining accepted lanes
                    // stay live and immediately continue with the now-empty buffer.
                    active_mask = __ballot_sync(kFullMask, accept);
                }
            }
        }

        // All warps must be done reading the current tile before the block overwrites it.
        __syncthreads();
    }

    if (active_query) {
        if (buf_count != 0) {
            kth = merge_buffer_into_top<K>(my_state, lane, buf_count, inf_key);
            (void)kth;
        }

        result_pair_t* const out =
            result + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        for (int i = lane; i < K; i += kWarpSize) {
            const packed_key_t key = my_state[i];
            store_result_pair(out + i, unpack_idx(key), unpack_dist(key));
        }
    }
}

template <int K>
static inline void launch_knn_impl(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    result_pair_t* result)
{
    using Cfg = KernelConfig<K, kBlockThreads, kBatchPoints>;

    static_assert(Cfg::kSharedBytes <= (144u * 1024u),
                  "Chosen launch policy must stay within the intended A100/H100 shared-memory budget.");

    const dim3 block(kBlockThreads);
    const dim3 grid((query_count + Cfg::kWarpsPerBlock - 1) / Cfg::kWarpsPerBlock);

    // Opt in to the large dynamic shared-memory footprint and bias the carveout toward shared memory.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, kBlockThreads, kBatchPoints>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Cfg::kSharedBytes));

    (void)cudaFuncSetAttribute(
        knn_kernel<K, kBlockThreads, kBatchPoints>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    // Standard CUDA launch semantics: asynchronous with respect to the host.
    knn_kernel<K, kBlockThreads, kBatchPoints>
        <<<grid, block, Cfg::kSharedBytes>>>(query, query_count, data, data_count, result);
}

} // namespace knn_detail

void run_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k)
{
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:
            knn_detail::launch_knn_impl<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_detail::launch_knn_impl<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_detail::launch_knn_impl<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_detail::launch_knn_impl<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_detail::launch_knn_impl<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_detail::launch_knn_impl<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem statement guarantees valid K, so this path is unreachable.
            break;
    }
}