#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <utility>

namespace knn_detail {

// Warp-per-query k-NN for 2D points.
// Design summary:
//   * one warp owns one query;
//   * the intermediate top-k result is warp-private and kept in registers;
//   * each warp also owns a shared-memory candidate buffer of k (index, distance) pairs
//     plus one shared counter updated via atomicAdd;
//   * the whole block cooperatively stages the input data in shared memory batches;
//   * merges are warp-private and use only __syncwarp(), while batch loads use __syncthreads().
//
// Hyper-parameters chosen for A100/H100-class GPUs:
//   * 256 threads/block = 8 warps/block = 8 queries/block;
//   * 2048 data points per shared-memory batch.
//
// With k = 1024, the kernel uses:
//   * 16 KiB static shared memory for the staged data batch,
//   * 64 KiB dynamic shared memory for 8 candidate buffers,
//   * 32 B for the 8 candidate counters.
// Total = 81,952 B/block, which is small enough for the target GPUs while still
// giving 8-way reuse of each staged data point across queries.

constexpr int WARP_LANES       = 32;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / WARP_LANES;
constexpr int BATCH_POINTS      = 2048;
constexpr int LOADS_PER_THREAD  = BATCH_POINTS / THREADS_PER_BLOCK;
constexpr unsigned FULL_MASK    = 0xFFFFFFFFu;
constexpr int INVALID_INDEX     = INT_MAX;

static_assert(THREADS_PER_BLOCK % WARP_LANES == 0, "Block size must be a whole number of warps.");
static_assert(BATCH_POINTS % THREADS_PER_BLOCK == 0, "Batch size is chosen to allow a fully unrolled full-batch load path.");

// Device code does not rely on std::pair ABI details directly. Instead it writes a POD with
// the same expected layout. The static_asserts enforce the only properties we need.
struct alignas(alignof(std::pair<int, float>)) PairIF {
    int   first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "std::pair<int,float> must be layout-compatible with {int,float}.");
static_assert(alignof(PairIF) == alignof(std::pair<int, float>), "std::pair<int,float> alignment mismatch.");

template <int K>
constexpr int dynamic_smem_bytes() {
    // Dynamic shared layout:
    //   int   cand_count[WARPS_PER_BLOCK]
    //   int   cand_idx  [WARPS_PER_BLOCK][K]
    //   float cand_dist [WARPS_PER_BLOCK][K]
    return WARPS_PER_BLOCK * static_cast<int>(sizeof(int)) +
           WARPS_PER_BLOCK * K * static_cast<int>(sizeof(int)) +
           WARPS_PER_BLOCK * K * static_cast<int>(sizeof(float));
}

__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    // Distances are compared first; index is only a deterministic tie-breaker.
    // The problem statement does not constrain tie handling, so any deterministic
    // policy is acceptable.
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ void compare_swap_shared(float* dist, int* idx, int a, int b, bool ascending) {
    const float da = dist[a];
    const float db = dist[b];
    const int   ia = idx[a];
    const int   ib = idx[b];

    const bool do_swap = ascending ? pair_less(db, ib, da, ia)
                                   : pair_less(da, ia, db, ib);

    if (do_swap) {
        dist[a] = db;
        idx[a]  = ib;
        dist[b] = da;
        idx[b]  = ia;
    }
}

template <int K>
__device__ __forceinline__ void bitonic_sort_shared(float* dist, int* idx, int lane) {
    // Full in-place bitonic sort of K shared-memory pairs.
    // This is only executed when the candidate buffer is merged, i.e. relatively infrequently.
    #pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < K; i += WARP_LANES) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = ((i & size) == 0);
                    compare_swap_shared(dist, idx, i, j, ascending);
                }
            }
            __syncwarp();
        }
    }
}

template <int K>
__device__ __forceinline__ void bitonic_merge_shared(float* dist, int* idx, int lane) {
    // The first phase of the merge creates the low half as a bitonic sequence.
    // This function performs only the bitonic-merge half, not a full sort.
    #pragma unroll 1
    for (int stride = K >> 1; stride > 0; stride >>= 1) {
        for (int i = lane; i < K; i += WARP_LANES) {
            const int j = i ^ stride;
            if (j > i) {
                compare_swap_shared(dist, idx, i, j, true);
            }
        }
        __syncwarp();
    }
}

template <int K>
__device__ __forceinline__ void merge_candidates(
    float* cand_dist,
    int* cand_idx,
    int* cand_count_ptr,
    int lane,
    int count,
    float (&local_dist)[K / WARP_LANES],
    int (&local_idx)[K / WARP_LANES],
    float& max_distance)
{
    constexpr int LOCAL_K = K / WARP_LANES;

    // Pad the inactive part of the candidate buffer with (+inf, INVALID_INDEX) so that the
    // full K-wide bitonic sort can be reused even when the buffer is only partially filled.
    for (int pos = lane; pos < K; pos += WARP_LANES) {
        if (pos >= count) {
            cand_dist[pos] = CUDART_INF_F;
            cand_idx[pos]  = INVALID_INDEX;
        }
    }
    __syncwarp();

    // Candidate buffer -> ascending order.
    bitonic_sort_shared<K>(cand_dist, cand_idx, lane);

    // The current top-k and the candidate list are both ascending.
    // Compare the current top-k against the reversed candidate list:
    //
    //   low[i] = min(topk[i], cand[K-1-i])
    //
    // Those K minima are exactly the K smallest elements of the union, but only in bitonic
    // order. We therefore write that low half back to shared memory and finish with an
    // in-place bitonic merge.
    //
    // The register-resident top-k is stored in an interleaved layout:
    //   lane l owns global positions l, l+32, l+64, ...
    // This layout keeps the shared-memory copies and the final global writes naturally
    // warp-coalesced / bank-friendly.
    int pos = lane;
    #pragma unroll
    for (int t = 0; t < LOCAL_K; ++t, pos += WARP_LANES) {
        const int rev = K - 1 - pos;
        const float bd = cand_dist[rev];
        const int   bi = cand_idx[rev];

        if (pair_less(bd, bi, local_dist[t], local_idx[t])) {
            local_dist[t] = bd;
            local_idx[t]  = bi;
        }
    }

    // All reads from the sorted candidate buffer must complete before any lane overwrites it.
    __syncwarp();

    pos = lane;
    #pragma unroll
    for (int t = 0; t < LOCAL_K; ++t, pos += WARP_LANES) {
        cand_dist[pos] = local_dist[t];
        cand_idx[pos]  = local_idx[t];
    }
    __syncwarp();

    // Bitonic low half -> sorted ascending new top-k.
    bitonic_merge_shared<K>(cand_dist, cand_idx, lane);

    pos = lane;
    #pragma unroll
    for (int t = 0; t < LOCAL_K; ++t, pos += WARP_LANES) {
        local_dist[t] = cand_dist[pos];
        local_idx[t]  = cand_idx[pos];
    }

    // The k-th nearest neighbor lives at global position K-1, i.e. lane 31, local slot LOCAL_K-1.
    const float lane_max = local_dist[LOCAL_K - 1];
    max_distance = __shfl_sync(FULL_MASK, lane_max, WARP_LANES - 1);

    if (lane == 0) {
        *cand_count_ptr = 0;
    }
    __syncwarp();
}

template <int K>
__global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    PairIF* __restrict__ result)
{
    constexpr int LOCAL_K = K / WARP_LANES;

    const int lane          = threadIdx.x & (WARP_LANES - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int query_idx     = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    const bool active       = (query_idx < query_count);

    // Static shared memory for the staged data batch.
    __shared__ float2 sh_data[BATCH_POINTS];

    // Dynamic shared memory for the per-warp candidate buffers.
    extern __shared__ unsigned char smem_raw[];
    int*   cand_count    = reinterpret_cast<int*>(smem_raw);
    int*   cand_idx_all  = cand_count + WARPS_PER_BLOCK;
    float* cand_dist_all = reinterpret_cast<float*>(cand_idx_all + WARPS_PER_BLOCK * K);

    int*   my_count = cand_count + warp_in_block;
    int*   my_idx   = cand_idx_all + warp_in_block * K;
    float* my_dist  = cand_dist_all + warp_in_block * K;

    if (lane == 0) {
        *my_count = 0;
    }

    // Warp-private intermediate top-k in registers. All entries start as +inf.
    float local_dist[LOCAL_K];
    int   local_idx[LOCAL_K];

    #pragma unroll
    for (int t = 0; t < LOCAL_K; ++t) {
        local_dist[t] = CUDART_INF_F;
        local_idx[t]  = INVALID_INDEX;
    }

    // Broadcast the query point from lane 0 to the whole warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Warp-local shadow of the shared candidate count.
    // The shared variable still exists exactly as requested and is updated via atomicAdd;
    // this mirror just avoids a shared-memory reload at every 32-point tile.
    int count = 0;

    // "filled" tracks how many real neighbors are already present in the intermediate top-k.
    // It is only needed during bootstrap, while max_distance is still +inf.
    int filled = 0;

    float max_distance = CUDART_INF_F;

    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_POINTS) {
        const int remaining = data_count - batch_start;
        int batch_size = 0;

        // Fast path for the common case: a full 2048-point batch.
        if (remaining >= BATCH_POINTS) {
            batch_size = BATCH_POINTS;
            #pragma unroll
            for (int i = 0; i < LOADS_PER_THREAD; ++i) {
                const int local_offset = threadIdx.x + i * THREADS_PER_BLOCK;
                sh_data[local_offset] = data[batch_start + local_offset];
            }
        } else {
            batch_size = remaining;
            for (int local_offset = threadIdx.x; local_offset < batch_size; local_offset += THREADS_PER_BLOCK) {
                sh_data[local_offset] = data[batch_start + local_offset];
            }
        }

        __syncthreads();

        if (active) {
            // Process the shared batch in warp-sized tiles so that at most 32 new candidates
            // can be appended before the next opportunity to flush/merge.
            for (int tile = 0; tile < batch_size; tile += WARP_LANES) {
                int points_in_tile = batch_size - tile;
                if (points_in_tile > WARP_LANES) {
                    points_in_tile = WARP_LANES;
                }

                // Flush before the next tile if the private candidate buffer cannot safely accept
                // all points from that tile.
                if (count > K - points_in_tile) {
                    merge_candidates<K>(my_dist, my_idx, my_count, lane, count, local_dist, local_idx, max_distance);
                    filled += count;
                    if (filled > K) {
                        filled = K;
                    }
                    count = 0;
                }

                float dist = 0.0f;
                int data_idx = 0;
                bool accept = false;

                if (lane < points_in_tile) {
                    data_idx = batch_start + tile + lane;
                    const float2 p = sh_data[tile + lane];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);
                    accept = (dist < max_distance);
                }

                // Warp-aggregated candidate insertion:
                // one atomicAdd reserves a contiguous range, and each accepting lane computes
                // its own position via a prefix within the ballot mask.
                const unsigned mask = __ballot_sync(FULL_MASK, accept);
                const int n = __popc(mask);

                int base = 0;
                if (lane == 0 && n > 0) {
                    base = atomicAdd(my_count, n);
                }
                base = __shfl_sync(FULL_MASK, base, 0);

                if (accept) {
                    const unsigned lower_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
                    const int offset = __popc(mask & lower_mask);
                    const int pos = base + offset;

                    // The pre-flush condition guarantees pos < K. The guard is defensive.
                    if (pos < K) {
                        my_idx[pos] = data_idx;
                        my_dist[pos] = dist;
                    }
                }

                // Ensure all candidate writes complete before a possible merge in the next step.
                __syncwarp();

                if (n > 0) {
                    count = base + n;
                }

                // Bootstrap optimization:
                // as soon as the union of the current top-k and the current candidate buffer
                // contains at least K real points, merge immediately so that max_distance
                // becomes finite and aggressive filtering can begin.
                if (filled < K && (filled + count) >= K) {
                    merge_candidates<K>(my_dist, my_idx, my_count, lane, count, local_dist, local_idx, max_distance);
                    filled += count;
                    if (filled > K) {
                        filled = K;
                    }
                    count = 0;
                }
            }
        }

        // The staged data batch is block-shared, so the whole block must finish consuming it
        // before it can be overwritten by the next global-memory batch.
        __syncthreads();
    }

    if (active) {
        // Final leftover candidates, if any.
        if (count > 0) {
            merge_candidates<K>(my_dist, my_idx, my_count, lane, count, local_dist, local_idx, max_distance);
        }

        PairIF* out = result + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        // The register layout is already in global sorted order:
        // positions lane, lane+32, lane+64, ...
        int pos = lane;
        #pragma unroll
        for (int t = 0; t < LOCAL_K; ++t, pos += WARP_LANES) {
            out[pos].first  = local_idx[t];
            out[pos].second = local_dist[t];
        }
    }
}

template <int K>
inline void launch_knn_typed(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result)
{
    const int grid = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int smem = dynamic_smem_bytes<K>();

    // Request the large dynamic shared-memory budget needed by the worst-case K values.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem);

    // The kernel is intentionally shared-memory heavy; prefer shared memory over L1.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    knn_kernel<K><<<grid, THREADS_PER_BLOCK, smem>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<PairIF*>(result));
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
    // Standard CUDA behavior: asynchronous launch on the current stream, with no extra device
    // allocations. All inputs are assumed valid as stated by the problem.
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:   knn_detail::launch_knn_typed<32>(query, query_count, data, data_count, result);   break;
        case 64:   knn_detail::launch_knn_typed<64>(query, query_count, data, data_count, result);   break;
        case 128:  knn_detail::launch_knn_typed<128>(query, query_count, data, data_count, result);  break;
        case 256:  knn_detail::launch_knn_typed<256>(query, query_count, data, data_count, result);  break;
        case 512:  knn_detail::launch_knn_typed<512>(query, query_count, data, data_count, result);  break;
        case 1024: knn_detail::launch_knn_typed<1024>(query, query_count, data, data_count, result); break;
        default:   break; // k is guaranteed valid by the problem statement.
    }
}