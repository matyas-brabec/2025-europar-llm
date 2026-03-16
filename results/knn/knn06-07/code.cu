#include <cuda_runtime.h>
#include <utility>
#include <cstdint>
#include <cstddef>

namespace knn_cuda_detail {

// Warp-per-query k-NN for 2D points.
//
// Design summary:
//   * One warp processes one query end-to-end.
//   * The whole block stages a batch of data points in shared memory; every warp reuses it.
//   * Per query, both the current top-k and the candidate buffer live in a warp-private
//     slice of shared memory. This avoids register/local-memory pressure for k up to 1024.
//   * Each logical (distance, index) pair is packed into one sortable 64-bit key:
//       high 32 bits = IEEE-754 bits of the non-negative squared distance
//       low  32 bits = data index
//     Since squared distances are non-negative, ordering by the high 32 bits matches
//     numeric distance ordering. The low 32 bits provide a deterministic tie-break.
//   * Candidate insertion uses one shared-memory atomicAdd per warp-round to reserve a
//     contiguous slot range, then warp ballot/prefix logic scatters accepting lanes into
//     that range. This satisfies the required atomicAdd usage while minimizing atomic traffic.
//   * When the candidate buffer is full (or would overflow), it is merged with the current
//     top-k by an in-place bitonic sort over 2k keys; the first k outputs become the new
//     intermediate result.
//
// Hyper-parameters:
//   * 8 warps/block (256 threads) gives good data reuse for the staged batch.
//   * 2016 staged points/batch = 63 warp-rounds. This is large enough to amortize the
//     block-wide synchronization cost while keeping shared-memory usage favorable across k.
//     For reference, the total shared footprint is about:
//       - k = 512  -> ~79.8 KiB / CTA
//       - k = 1024 -> ~143.8 KiB / CTA
//     which is appropriate for A100/H100-class GPUs with opt-in dynamic shared memory.

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;
constexpr int kDataBatchPoints = 2016;

using key_t = unsigned long long;

constexpr unsigned int kInvalidIndex = 0xffffffffu;
constexpr key_t kInvalidKey =
    (static_cast<key_t>(0x7f800000u) << 32) | static_cast<key_t>(kInvalidIndex);

static_assert(kDataBatchPoints % kWarpSize == 0, "Batch size must be a multiple of 32.");

__host__ __device__ constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

template <int K, int WARPS_PER_BLOCK = kWarpsPerBlock, int DATA_BATCH_POINTS = kDataBatchPoints>
__host__ __device__ constexpr std::size_t shared_bytes() {
    std::size_t bytes = 0;
    bytes = align_up(bytes, alignof(float2));
    bytes += static_cast<std::size_t>(DATA_BATCH_POINTS) * sizeof(float2);
    bytes = align_up(bytes, alignof(int));
    bytes += static_cast<std::size_t>(WARPS_PER_BLOCK) * sizeof(int);
    bytes = align_up(bytes, alignof(key_t));
    bytes += static_cast<std::size_t>(WARPS_PER_BLOCK) * static_cast<std::size_t>(2 * K) * sizeof(key_t);
    return bytes;
}

__device__ __forceinline__ key_t pack_key(const float distance, const int index) {
    return (static_cast<key_t>(__float_as_uint(distance)) << 32) |
           static_cast<key_t>(static_cast<unsigned int>(index));
}

__device__ __forceinline__ float unpack_distance(const key_t key) {
    return __uint_as_float(static_cast<unsigned int>(key >> 32));
}

__device__ __forceinline__ int unpack_index(const key_t key) {
    return static_cast<int>(static_cast<unsigned int>(key));
}

// Merge the warp-private candidate buffer with the warp-private current top-k.
// Layout of warp_keys:
//   [0, K)   : current intermediate top-k, sorted ascending by (distance, index)
//   [K, 2K)  : candidate buffer, candidate_count valid entries followed by garbage
//
// The merge is done in place by padding the unused candidate slots with +inf keys and
// running a bitonic sort over the full 2K-key segment. Because 2K is also a power of two,
// the first K entries after sorting are exactly the updated top-k. This keeps the merge
// entirely in shared memory and needs no extra device allocation.
template <int K>
__device__ __forceinline__ float merge_candidate_buffer(key_t* warp_keys,
                                                        int* candidate_count_ptr,
                                                        const int candidate_count) {
    constexpr int N = 2 * K;
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & (kWarpSize - 1);

    if (candidate_count == 0) {
        return unpack_distance(warp_keys[K - 1]);
    }

    // Pad unused candidate slots with +inf so the 2K-key sort behaves as a merge with
    // invalid elements automatically sinking to the end.
    #pragma unroll
    for (int i = lane + candidate_count; i < K; i += kWarpSize) {
        warp_keys[K + i] = kInvalidKey;
    }
    __syncwarp(mask);

    // Full in-place bitonic sort. K is bounded (<= 1024) and merges are much less frequent
    // than distance evaluations on the intended problem sizes, so this simple no-temp merge
    // strategy performs well while staying compact and robust.
    #pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll 1
            for (int i = lane; i < N; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    const key_t a = warp_keys[i];
                    const key_t b = warp_keys[j];
                    const bool up = ((i & size) == 0);
                    if (up ? (a > b) : (a < b)) {
                        warp_keys[i] = b;
                        warp_keys[j] = a;
                    }
                }
            }
            __syncwarp(mask);
        }
    }

    // The buffer has been fully consumed by the merge.
    if (lane == 0) {
        *candidate_count_ptr = 0;
    }
    __syncwarp(mask);

    return unpack_distance(warp_keys[K - 1]);
}

// One warp owns one query from start to finish. The whole block repeatedly stages batches
// of data points into shared memory so that each batch load is reused by every warp in the
// block. The shared-memory state per query is:
//
//   * best_keys[0:K]  : the current intermediate top-k
//   * cand_keys[0:K]  : the candidate buffer
//   * candidate_count : number of currently stored candidates (shared, updated by atomicAdd)
//
// A register mirror of candidate_count is kept in every lane; because reservations are done
// warp-wide with a single atomicAdd, this mirror stays exact and avoids rereading shared
// memory in the hot loop.
template <int K, int WARPS_PER_BLOCK = kWarpsPerBlock, int DATA_BATCH_POINTS = kDataBatchPoints>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 1)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                std::pair<int, float>* __restrict__ result) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");

    // Dynamic shared-memory layout:
    //   float2 batch[DATA_BATCH_POINTS]
    //   int    candidate_count[WARPS_PER_BLOCK]
    //   key_t  packed_entries[WARPS_PER_BLOCK][2*K]
    extern __shared__ key_t smem_base[];
    unsigned char* const smem = reinterpret_cast<unsigned char*>(smem_base);

    std::size_t offset = 0;
    offset = align_up(offset, alignof(float2));
    float2* const sh_data = reinterpret_cast<float2*>(smem + offset);
    offset += static_cast<std::size_t>(DATA_BATCH_POINTS) * sizeof(float2);

    offset = align_up(offset, alignof(int));
    int* const sh_candidate_count = reinterpret_cast<int*>(smem + offset);
    offset += static_cast<std::size_t>(WARPS_PER_BLOCK) * sizeof(int);

    offset = align_up(offset, alignof(key_t));
    key_t* const sh_keys = reinterpret_cast<key_t*>(smem + offset);

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool active = (query_idx < query_count);
    const unsigned mask = 0xffffffffu;

    key_t* const warp_keys = sh_keys + static_cast<std::size_t>(warp) * static_cast<std::size_t>(2 * K);
    key_t* const best_keys = warp_keys;
    key_t* const cand_keys = warp_keys + K;
    int* const candidate_count_ptr = sh_candidate_count + warp;

    float qx = 0.0f;
    float qy = 0.0f;
    float max_distance = CUDART_INF_F;
    int candidate_count = 0;  // Register mirror of *candidate_count_ptr.

    if (active) {
        float qx_local = 0.0f;
        float qy_local = 0.0f;
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx_local = q.x;
            qy_local = q.y;
        }
        qx = __shfl_sync(mask, qx_local, 0);
        qy = __shfl_sync(mask, qy_local, 0);

        // Start with an all-+inf top-k. max_distance therefore stays +inf until at least
        // K real points have been merged in.
        #pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            best_keys[i] = kInvalidKey;
        }

        if (lane == 0) {
            *candidate_count_ptr = 0;
        }
    }

    // Even inactive warps in the final partial block must participate in the block-level
    // synchronization for shared batch loading.
    __syncthreads();

    for (int data_base = 0; data_base < data_count; data_base += DATA_BATCH_POINTS) {
        int batch_count = data_count - data_base;
        if (batch_count > DATA_BATCH_POINTS) {
            batch_count = DATA_BATCH_POINTS;
        }

        // Cooperative, fully coalesced load of the current data batch into shared memory.
        for (int i = threadIdx.x; i < batch_count; i += blockDim.x) {
            sh_data[i] = data[data_base + i];
        }
        __syncthreads();

        if (active) {
            // Process the batch one warp-sized round at a time. Each lane handles at most
            // one cached point per round.
            for (int round_base = 0; round_base < batch_count; round_base += kWarpSize) {
                const int local_idx = round_base + lane;
                const bool valid = (local_idx < batch_count);
                const int global_idx = data_base + local_idx;

                float dist = 0.0f;
                if (valid) {
                    const float2 p = sh_data[local_idx];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);
                }

                // Use the current kth distance as the rejection threshold. Strict '<' is
                // intentional and allowed by the specification.
                bool accept = valid && (dist < max_distance);
                unsigned accept_mask = __ballot_sync(mask, accept);
                int num_accept = __popc(accept_mask);

                // If the whole round would overflow the candidate buffer, merge first so the
                // threshold can shrink before any of the round's candidates are committed.
                if (candidate_count + num_accept > K) {
                    max_distance = merge_candidate_buffer<K>(warp_keys, candidate_count_ptr, candidate_count);
                    candidate_count = 0;

                    accept = valid && (dist < max_distance);
                    accept_mask = __ballot_sync(mask, accept);
                    num_accept = __popc(accept_mask);
                }

                if (num_accept != 0) {
                    int base_pos = 0;
                    if (lane == 0) {
                        // One warp-wide reservation atomic per round. This updates the
                        // required shared candidate count and returns the base slot for
                        // all new candidates in the round.
                        base_pos = atomicAdd(candidate_count_ptr, num_accept);
                    }
                    base_pos = __shfl_sync(mask, base_pos, 0);

                    if (accept) {
                        const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);
                        const int rank = __popc(accept_mask & lane_mask_lt);
                        cand_keys[base_pos + rank] = pack_key(dist, global_idx);
                    }

                    candidate_count += num_accept;
                }

                __syncwarp(mask);

                // Merge immediately when the buffer becomes full so max_distance tracks the
                // current kth neighbor as closely as possible.
                if (candidate_count == K) {
                    max_distance = merge_candidate_buffer<K>(warp_keys, candidate_count_ptr, candidate_count);
                    candidate_count = 0;
                }
            }
        }

        // The cached batch is reused by every warp in the block, so no warp may overwrite
        // it with the next batch until all warps are done with the current one.
        __syncthreads();
    }

    if (active) {
        // The problem statement explicitly requires a final merge if the candidate buffer
        // still contains unmerged entries.
        if (candidate_count != 0) {
            max_distance = merge_candidate_buffer<K>(warp_keys, candidate_count_ptr, candidate_count);
            candidate_count = 0;
        }

        (void)max_distance;

        // best_keys is already sorted ascending, so it is already in the required output
        // order. Write std::pair member-wise so no device-side std::pair constructor is needed.
        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            const key_t key = best_keys[i];
            result[out_base + static_cast<std::size_t>(i)].first = unpack_index(key);
            result[out_base + static_cast<std::size_t>(i)].second = unpack_distance(key);
        }
    }
}

template <int K>
inline void launch_knn_impl(const float2* query,
                            int query_count,
                            const float2* data,
                            int data_count,
                            std::pair<int, float>* result) {
    constexpr std::size_t smem_bytes =
        shared_bytes<K, kWarpsPerBlock, kDataBatchPoints>();

    // Opt in to the larger dynamic shared-memory allocation needed by the larger k cases.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, kWarpsPerBlock, kDataBatchPoints>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(
        knn_kernel<K, kWarpsPerBlock, kDataBatchPoints>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
    knn_kernel<K, kWarpsPerBlock, kDataBatchPoints>
        <<<blocks, kThreadsPerBlock, smem_bytes>>>(query, query_count, data, data_count, result);
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

    // k is runtime-valued but comes from a small power-of-two set, so dispatch to
    // compile-time-specialized kernels and keep all k-dependent loop bounds/static layout
    // constants known to the compiler.
    switch (k) {
        case 32:
            knn_cuda_detail::launch_knn_impl<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_cuda_detail::launch_knn_impl<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_cuda_detail::launch_knn_impl<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_cuda_detail::launch_knn_impl<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_cuda_detail::launch_knn_impl<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_cuda_detail::launch_knn_impl<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // The caller guarantees k is valid; this is only a defensive fallback.
            break;
    }
}