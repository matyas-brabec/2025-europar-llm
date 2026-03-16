#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace detail {

// Tuned for modern data-center GPUs (A100/H100):
// - 256 threads/block = 8 warps/block, so each staged data tile is reused by 8 queries.
// - DATA_TILE = 2048 keeps the staged point batch large enough to amortize __syncthreads()
//   while still allowing two resident blocks/SM even for K=1024:
//     shared bytes = 2*2048*sizeof(float) + 8*K*(sizeof(float)+sizeof(int))
//                  = 16 KiB + 64 KiB = 80 KiB at K=1024.
constexpr int BLOCK_THREADS    = 256;
constexpr int WARPS_PER_BLOCK  = BLOCK_THREADS / 32;
constexpr int DATA_TILE        = 2048;
constexpr unsigned FULL_MASK   = 0xFFFFFFFFu;

using result_pair = std::pair<int, float>;

__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ bool pair_greater(float da, int ia, float db, int ib) {
    return pair_less(db, ib, da, ia);
}

__device__ __forceinline__ void swap_pair(float &da, int &ia, float &db, int &ib) {
    const float td = da;
    const int   ti = ia;
    da = db;
    ia = ib;
    db = td;
    ib = ti;
}

// Warp-distributed bitonic sort on K elements.
// Layout invariant:
//   thread t owns elements [t*(K/32), ..., t*(K/32) + (K/32 - 1)].
// Because K/32 is a power of two, whenever bitonic partner distance exceeds the per-thread
// chunk size, the partner element always has the same register index inside the partner thread.
// This lets us use:
//   - plain register swaps for intra-thread exchanges
//   - warp shuffles for inter-thread exchanges
template <int K>
__device__ __forceinline__ void bitonic_sort_warp(
    float (&dist)[K / 32],
    int   (&idx)[K / 32],
    int lane_base)
{
    constexpr int E = K / 32;

    #pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if constexpr (E > 1) {
                if (stride < E) {
                    // Intra-thread compare/exchange: both elements reside in this thread.
                    #pragma unroll
                    for (int r = 0; r < E; ++r) {
                        const int partner = r ^ stride;
                        if (partner > r) {
                            const int g = lane_base + r;
                            const bool ascending = ((g & size) == 0);

                            const bool out_of_order = ascending
                                ? pair_greater(dist[r], idx[r], dist[partner], idx[partner])
                                : pair_less(dist[r], idx[r], dist[partner], idx[partner]);

                            if (out_of_order) {
                                swap_pair(dist[r], idx[r], dist[partner], idx[partner]);
                            }
                        }
                    }
                } else {
                    // Inter-thread compare/exchange: partner is in another lane, same register slot.
                    const int lane_xor = stride / E;

                    #pragma unroll
                    for (int r = 0; r < E; ++r) {
                        const float other_d = __shfl_xor_sync(FULL_MASK, dist[r], lane_xor);
                        const int   other_i = __shfl_xor_sync(FULL_MASK, idx[r],  lane_xor);

                        const int g = lane_base + r;
                        const bool ascending       = ((g & size) == 0);
                        const bool smaller_partner = ((g & stride) == 0);
                        const bool keep_min        = (ascending == smaller_partner);

                        const bool take_other = keep_min
                            ? pair_greater(dist[r], idx[r], other_d, other_i)
                            : pair_less(dist[r], idx[r], other_d, other_i);

                        if (take_other) {
                            dist[r] = other_d;
                            idx[r]  = other_i;
                        }
                    }
                }
            } else {
                // Special case K=32 -> one element per lane.
                const float other_d = __shfl_xor_sync(FULL_MASK, dist[0], stride);
                const int   other_i = __shfl_xor_sync(FULL_MASK, idx[0],  stride);

                const int g = lane_base;
                const bool ascending       = ((g & size) == 0);
                const bool smaller_partner = ((g & stride) == 0);
                const bool keep_min        = (ascending == smaller_partner);

                const bool take_other = keep_min
                    ? pair_greater(dist[0], idx[0], other_d, other_i)
                    : pair_less(dist[0], idx[0], other_d, other_i);

                if (take_other) {
                    dist[0] = other_d;
                    idx[0]  = other_i;
                }
            }
        }
    }
}

template <int K>
__device__ __forceinline__ float kth_distance(const float (&dist)[K / 32]) {
    constexpr int E = K / 32;
    return __shfl_sync(FULL_MASK, dist[E - 1], 31);
}

// Flushes the shared-memory candidate buffer into the warp-private intermediate result.
// This follows the requested merge procedure exactly:
//   0. intermediate result is already sorted ascending
//   1. swap shared buffer <-> register intermediate, so buffer is now in registers
//   2. bitonic sort the buffer in ascending order
//   3. form a bitonic top-K merge against the reversed intermediate held in shared memory
//   4. bitonic sort again, yielding the updated sorted intermediate result
template <int K>
__device__ __forceinline__ void flush_candidate_buffer(
    float (&result_dist)[K / 32],
    int   (&result_idx)[K / 32],
    float *warp_buf_dist,
    int   *warp_buf_idx,
    int   &buffer_count,
    float &max_distance,
    int lane_base)
{
    constexpr int E = K / 32;

    // Pad the unused tail with +inf so that the final partial buffer can be merged with
    // the exact same code path as a full buffer.
    #pragma unroll
    for (int r = 0; r < E; ++r) {
        const int g = lane_base + r;
        if (g >= buffer_count) {
            warp_buf_dist[g] = CUDART_INF_F;
            warp_buf_idx[g]  = -1;
        }
    }

    // Step 1: swap shared candidate buffer with the register-resident intermediate result.
    // Each thread swaps only its own contiguous chunk, so no temporary arrays are needed.
    #pragma unroll
    for (int r = 0; r < E; ++r) {
        const int g = lane_base + r;

        const float tmp_d = warp_buf_dist[g];
        const int   tmp_i = warp_buf_idx[g];

        warp_buf_dist[g] = result_dist[r];
        warp_buf_idx[g]  = result_idx[r];

        result_dist[r]   = tmp_d;
        result_idx[r]    = tmp_i;
    }

    // Shared memory now contains the old sorted intermediate result, which will be read in
    // reverse order during the merge step.
    __syncwarp(FULL_MASK);

    // Step 2: sort the buffer in registers.
    bitonic_sort_warp<K>(result_dist, result_idx, lane_base);

    // Step 3: top-K bitonic merge with reversed old intermediate result in shared memory.
    #pragma unroll
    for (int r = 0; r < E; ++r) {
        const int g   = lane_base + r;
        const int rev = K - 1 - g;

        const float other_d = warp_buf_dist[rev];
        const int   other_i = warp_buf_idx[rev];

        if (pair_less(other_d, other_i, result_dist[r], result_idx[r])) {
            result_dist[r] = other_d;
            result_idx[r]  = other_i;
        }
    }

    // Step 4: sort the merged bitonic sequence to recover the new sorted intermediate result.
    bitonic_sort_warp<K>(result_dist, result_idx, lane_base);

    max_distance = kth_distance<K>(result_dist);
    buffer_count = 0;
}

template <int K>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void knn_kernel(
    const float2 *__restrict__ query,
    int query_count,
    const float2 *__restrict__ data,
    int data_count,
    result_pair *__restrict__ result)
{
    static_assert(K >= 32 && K <= 1024, "K out of supported range");
    static_assert((K & (K - 1)) == 0, "K must be a power of two");

    constexpr int E = K / 32;

    extern __shared__ unsigned char shared_raw[];

    // Shared layout:
    //   [sm_x DATA_TILE floats][sm_y DATA_TILE floats]
    //   [candidate distances WARPS_PER_BLOCK*K floats]
    //   [candidate indices   WARPS_PER_BLOCK*K ints]
    float * const sm_x         = reinterpret_cast<float *>(shared_raw);
    float * const sm_y         = sm_x + DATA_TILE;
    float * const sm_buf_dist  = sm_y + DATA_TILE;
    int   * const sm_buf_idx   = reinterpret_cast<int *>(sm_buf_dist + WARPS_PER_BLOCK * K);

    const int tid       = threadIdx.x;
    const int lane      = tid & 31;
    const int warp_id   = tid >> 5;
    const int lane_base = lane * E;

    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    const int query_idx   = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool warp_active = (query_idx < query_count);

    // One query per warp. Load once in lane 0 and broadcast.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active) {
        float qx0 = 0.0f;
        float qy0 = 0.0f;
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx0 = q.x;
            qy0 = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx0, 0);
        qy = __shfl_sync(FULL_MASK, qy0, 0);
    }

    // Warp-private intermediate result in registers: each thread owns E consecutive neighbors.
    float result_dist[E];
    int   result_idx[E];

    #pragma unroll
    for (int r = 0; r < E; ++r) {
        result_dist[r] = CUDART_INF_F;
        result_idx[r]  = -1;
    }

    // Kept redundantly in all lanes of the warp so that updates are warp-synchronous and cheap.
    float max_distance = CUDART_INF_F;
    int   buffer_count = 0;

    float * const warp_buf_dist = sm_buf_dist + warp_id * K;
    int   * const warp_buf_idx  = sm_buf_idx  + warp_id * K;

    // Iterate over the database in block-cooperative shared-memory tiles.
    for (int base = 0; base < data_count; base += DATA_TILE) {
        const int remaining  = data_count - base;
        const int tile_count = (remaining < DATA_TILE) ? remaining : DATA_TILE;

        // Whole block stages the next tile.
        for (int i = tid; i < tile_count; i += BLOCK_THREADS) {
            const float2 p = data[base + i];
            sm_x[i] = p.x;
            sm_y[i] = p.y;
        }

        __syncthreads();

        if (warp_active) {
            // Each warp walks the cached tile in lane-strided order.
            // One distance per lane per iteration, then warp-ballot compaction into the buffer.
            #pragma unroll 1
            for (int t = lane; t < tile_count; t += 32) {
                const float dx = qx - sm_x[t];
                const float dy = qy - sm_y[t];
                const float d  = fmaf(dx, dx, dy * dy);
                const int   di = base + t;

                bool pending = (d < max_distance);

                // If the current ballot overflows the buffer, insert what fits, flush, then
                // continue with the leftover candidates from the same ballot.
                while (true) {
                    const unsigned mask = __ballot_sync(FULL_MASK, pending);
                    if (mask == 0u) {
                        break;
                    }

                    const int pending_count = __popc(mask);
                    const int rank          = pending ? __popc(mask & lane_mask_lt) : -1;
                    const int space         = K - buffer_count;
                    const int stored        = (space < pending_count) ? space : pending_count;

                    if (pending && rank < stored) {
                        const int pos = buffer_count + rank;
                        warp_buf_dist[pos] = d;
                        warp_buf_idx[pos]  = di;
                    }

                    __syncwarp(FULL_MASK);

                    buffer_count += stored;

                    if (buffer_count == K) {
                        flush_candidate_buffer<K>(
                            result_dist, result_idx,
                            warp_buf_dist, warp_buf_idx,
                            buffer_count, max_distance,
                            lane_base);
                    }

                    // Leftover candidate from this ballot iteration, if any.
                    pending = pending && (rank >= stored) && (d < max_distance);
                }
            }
        }

        // Ensure no warp still consumes the tile before the next one is staged.
        __syncthreads();
    }

    // Final partial buffer flush, if needed.
    if (warp_active) {
        if (buffer_count > 0) {
            flush_candidate_buffer<K>(
                result_dist, result_idx,
                warp_buf_dist, warp_buf_idx,
                buffer_count, max_distance,
                lane_base);
        }

        const std::size_t out_base = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int r = 0; r < E; ++r) {
            const std::size_t out_pos = out_base + static_cast<std::size_t>(lane_base + r);
            result[out_pos].first  = result_idx[r];
            result[out_pos].second = result_dist[r];
        }
    }
}

template <int K>
inline void launch_knn_impl(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    result_pair *result)
{
    constexpr std::size_t SHARED_BYTES =
        2ull * DATA_TILE * sizeof(float) +
        static_cast<std::size_t>(WARPS_PER_BLOCK) * K * (sizeof(float) + sizeof(int));

    // Opt in to the required dynamic shared memory size and bias the SM partition toward shared
    // memory because the kernel deliberately uses shared memory as both a block-wide data cache
    // and per-warp candidate storage.
    cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SHARED_BYTES));

    cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    cudaFuncSetCacheConfig(knn_kernel<K>, cudaFuncCachePreferShared);

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn_kernel<K><<<grid, block, SHARED_BYTES>>>(
        query, query_count, data, data_count, result);
}

} // namespace detail

void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k)
{
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:   detail::launch_knn_impl<32>(query, query_count, data, data_count, result);   break;
        case 64:   detail::launch_knn_impl<64>(query, query_count, data, data_count, result);   break;
        case 128:  detail::launch_knn_impl<128>(query, query_count, data, data_count, result);  break;
        case 256:  detail::launch_knn_impl<256>(query, query_count, data, data_count, result);  break;
        case 512:  detail::launch_knn_impl<512>(query, query_count, data, data_count, result);  break;
        case 1024: detail::launch_knn_impl<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Inputs are specified to be valid; this keeps the function total.
    }
}