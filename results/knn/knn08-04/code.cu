#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <utility>

namespace knn_detail {

// Warp-per-query k-NN for 2D points with squared Euclidean distance.
//
// Tuning choices:
// - 256 threads/block = 8 warps/block = 8 queries/block.
// - 2048 staged data points per batch.
//   For the worst supported case K=1024, dynamic shared memory is
//   2048*sizeof(float2) + 8*1024*(sizeof(float)+sizeof(int)) + 8*sizeof(int)
//   = 81,952 bytes, which fits the intended A100/H100 target while still
//   allowing two resident blocks/SM in the common configuration.
//
// The algorithm follows the requested structure exactly:
// - one warp computes one query;
// - the current top-k is kept distributed in registers, with each thread owning
//   K/32 consecutive entries;
// - each warp has a shared-memory candidate buffer of size K (indices+distances);
// - block-wide batches of data points are cached in shared memory;
// - candidates are collected with warp ballots;
// - whenever the candidate buffer is full, it is merged into the register top-k
//   via: swap -> bitonic sort -> bitonic merge -> bitonic sort.
//
// No auxiliary device memory is allocated; shared memory is reused both for the
// staged data batch and for the per-warp candidate buffers.

constexpr unsigned FULL_MASK      = 0xffffffffu;
constexpr int      WARP_SIZE      = 32;
constexpr int      LAST_LANE      = WARP_SIZE - 1;
constexpr int      BLOCK_THREADS  = 256;
constexpr int      WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;
constexpr int      BATCH_POINTS   = 2048;
constexpr int      LOAD_ITERS     = BATCH_POINTS / BLOCK_THREADS;
constexpr int      INVALID_INDEX  = -1;

static_assert(BLOCK_THREADS % WARP_SIZE == 0, "BLOCK_THREADS must be a multiple of 32.");
static_assert(BATCH_POINTS % WARP_SIZE == 0, "BATCH_POINTS must be a multiple of 32.");
static_assert(BATCH_POINTS % BLOCK_THREADS == 0, "BATCH_POINTS must be divisible by BLOCK_THREADS.");

template <int K>
constexpr size_t shared_bytes() {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0, "K must be a power of two in [32, 1024].");
    return sizeof(float2) * BATCH_POINTS
         + sizeof(float)  * WARPS_PER_BLOCK * K
         + sizeof(int)    * WARPS_PER_BLOCK * K
         + sizeof(int)    * WARPS_PER_BLOCK;
}

// Invalid placeholders use index = -1.
// When distances tie, valid entries must rank before placeholders; otherwise a
// real +inf distance could incorrectly lose to a sentinel +inf.
__device__ __forceinline__ bool pair_less(const float da, const int ia,
                                          const float db, const int ib) {
    if (da < db) return true;
    if (da > db) return false;

    const bool a_valid = (ia >= 0);
    const bool b_valid = (ib >= 0);
    if (a_valid != b_valid) return a_valid;  // valid < invalid

    return ia < ib;
}

__device__ __forceinline__ bool pair_greater(const float da, const int ia,
                                             const float db, const int ib) {
    return pair_less(db, ib, da, ia);
}

__device__ __forceinline__ void swap_pair(float &da, int &ia, float &db, int &ib) {
    const float td = da;
    da = db;
    db = td;

    const int ti = ia;
    ia = ib;
    ib = ti;
}

// Distributed bitonic sort over K elements stored as K/32 consecutive registers
// per thread. For strides smaller than K/32, both elements of a compare-swap
// pair live in the same thread and are swapped locally. For larger strides, the
// partner lives in another lane, and the exchange uses warp shuffles. Because
// the distribution is by consecutive chunks, cross-lane exchanges always touch
// the same register index within each lane.
template <int K>
__device__ __forceinline__ void bitonic_sort_warp(float (&dist)[K / WARP_SIZE],
                                                  int   (&idx )[K / WARP_SIZE],
                                                  const int lane) {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0, "Invalid K.");
    constexpr int REGS_PER_THREAD = K / WARP_SIZE;
    const int lane_base = lane * REGS_PER_THREAD;

    #pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if constexpr (REGS_PER_THREAD > 1) {
                if (stride < REGS_PER_THREAD) {
                    #pragma unroll
                    for (int r = 0; r < REGS_PER_THREAD; ++r) {
                        const int rp = r ^ stride;
                        if (rp > r) {
                            const int  i         = lane_base + r;
                            const bool ascending = ((i & size) == 0);
                            const bool do_swap   = ascending
                                ? pair_greater(dist[r], idx[r], dist[rp], idx[rp])
                                : pair_less   (dist[r], idx[r], dist[rp], idx[rp]);

                            if (do_swap) {
                                swap_pair(dist[r], idx[r], dist[rp], idx[rp]);
                            }
                        }
                    }
                    continue;
                }
            }

            const int lane_mask = stride / REGS_PER_THREAD;

            #pragma unroll
            for (int r = 0; r < REGS_PER_THREAD; ++r) {
                const float other_d = __shfl_xor_sync(FULL_MASK, dist[r], lane_mask);
                const int   other_i = __shfl_xor_sync(FULL_MASK, idx [r], lane_mask);

                const int  i         = lane_base + r;
                const bool ascending = ((i & size) == 0);
                const bool lower     = ((lane & lane_mask) == 0);
                const bool keep_min  = (ascending == lower);

                const bool take_other = keep_min
                    ? pair_greater(dist[r], idx[r], other_d, other_i)
                    : pair_less   (dist[r], idx[r], other_d, other_i);

                if (take_other) {
                    dist[r] = other_d;
                    idx [r] = other_i;
                }
            }
        }
    }
}

// Merge the shared-memory candidate buffer into the register-resident top-k.
// This is the exact requested sequence:
//   1) swap buffer <-> top-k so the buffer moves to registers;
//   2) sort that register buffer with bitonic sort;
//   3) form a bitonic sequence by taking min(buffer[i], old_topk[K-1-i]);
//   4) sort the bitonic sequence with bitonic sort.
//
// The shared candidate buffer is reused as scratch space for the old top-k, so
// the kernel never allocates extra storage.
//
// topk_full is intentionally tracked separately from max_distance:
// before the first K valid elements exist, max_distance is only a sentinel,
// so filtering must be disabled to avoid dropping true +inf distances.
template <int K>
__device__ __forceinline__ void flush_candidates(float (&topk_dist)[K / WARP_SIZE],
                                                 int   (&topk_idx )[K / WARP_SIZE],
                                                 float *warp_buf_dist,
                                                 int   *warp_buf_idx,
                                                 int   *warp_buf_count,
                                                 const int lane,
                                                 int   &candidate_count,
                                                 float &max_distance,
                                                 bool  &topk_full) {
    constexpr int REGS_PER_THREAD = K / WARP_SIZE;
    const int lane_base = lane * REGS_PER_THREAD;

    // Order all previous shared-memory writes by this warp before consuming the buffer.
    __syncwarp();

    // Pad the partially filled candidate buffer with sentinels so the merge always
    // operates on exactly K elements.
    #pragma unroll
    for (int r = 0; r < REGS_PER_THREAD; ++r) {
        const int pos = lane_base + r;
        if (pos >= candidate_count) {
            warp_buf_dist[pos] = CUDART_INF_F;
            warp_buf_idx [pos] = INVALID_INDEX;
        }
    }

    // Swap shared candidate buffer with the register top-k:
    // - registers receive the (unsorted) candidate buffer,
    // - shared memory receives the old sorted top-k, which becomes merge scratch.
    #pragma unroll
    for (int r = 0; r < REGS_PER_THREAD; ++r) {
        const int pos = lane_base + r;

        const float buf_d = warp_buf_dist[pos];
        const int   buf_i = warp_buf_idx [pos];

        warp_buf_dist[pos] = topk_dist[r];
        warp_buf_idx [pos] = topk_idx [r];

        topk_dist[r] = buf_d;
        topk_idx [r] = buf_i;
    }

    // Every lane will now read arbitrary positions from the old top-k in shared memory.
    __syncwarp();

    // Step 2: sort the former candidate buffer now living in registers.
    bitonic_sort_warp<K>(topk_dist, topk_idx, lane);

    // Step 3: build the bitonic merge input by pairing ascending buffer[i] with
    // descending old_topk[K-1-i] and keeping only the smaller of the two.
    #pragma unroll
    for (int r = 0; r < REGS_PER_THREAD; ++r) {
        const int i       = lane_base + r;
        const int rev_i   = K - 1 - i;
        const float old_d = warp_buf_dist[rev_i];
        const int   old_i = warp_buf_idx [rev_i];

        if (pair_greater(topk_dist[r], topk_idx[r], old_d, old_i)) {
            topk_dist[r] = old_d;
            topk_idx [r] = old_i;
        }
    }

    // Step 4: sort the bitonic sequence back into ascending order.
    bitonic_sort_warp<K>(topk_dist, topk_idx, lane);

    // Update the pruning threshold and whether the top-k is fully initialized.
    const float kth_dist_local = topk_dist[REGS_PER_THREAD - 1];
    const int   kth_idx_local  = topk_idx [REGS_PER_THREAD - 1];
    max_distance = __shfl_sync(FULL_MASK, kth_dist_local, LAST_LANE);
    topk_full    = (__shfl_sync(FULL_MASK, kth_idx_local,  LAST_LANE) >= 0);

    candidate_count = 0;
    if (lane == 0) {
        warp_buf_count[0] = 0;
    }
}

using result_pair_t = std::pair<int, float>;

template <int K>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void knn_kernel(const float2 * __restrict__ query,
                const int query_count,
                const float2 * __restrict__ data,
                const int data_count,
                result_pair_t * __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0, "Invalid K.");
    constexpr int REGS_PER_THREAD = K / WARP_SIZE;

    // Shared memory layout:
    //   [0, BATCH_POINTS)                          : staged float2 data batch
    //   [0, WARPS_PER_BLOCK*K)                    : per-warp candidate distances
    //   [0, WARPS_PER_BLOCK*K)                    : per-warp candidate indices
    //   [0, WARPS_PER_BLOCK)                      : per-warp candidate counts
    extern __shared__ unsigned char smem_raw[];
    float2 *const sh_batch      = reinterpret_cast<float2*>(smem_raw);
    float  *const sh_buf_dist   = reinterpret_cast<float*>(sh_batch + BATCH_POINTS);
    int    *const sh_buf_idx    = reinterpret_cast<int*>(sh_buf_dist + WARPS_PER_BLOCK * K);
    int    *const sh_buf_count  = reinterpret_cast<int*>(sh_buf_idx  + WARPS_PER_BLOCK * K);

    const int lane     = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id  = threadIdx.x >> 5;
    const int query_id = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active_query = (query_id < query_count);

    float *const warp_buf_dist  = sh_buf_dist  + warp_id * K;
    int   *const warp_buf_idx   = sh_buf_idx   + warp_id * K;
    int   *const warp_buf_count = sh_buf_count + warp_id;

    // Shared-memory bookkeeping variable requested by the specification.
    if (lane == 0) {
        warp_buf_count[0] = 0;
    }

    // Load one query point per warp and broadcast it.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active_query && lane == 0) {
        const float2 q = query[query_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Register-resident distributed top-k.
    float topk_dist[REGS_PER_THREAD];
    int   topk_idx [REGS_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < REGS_PER_THREAD; ++r) {
        topk_dist[r] = CUDART_INF_F;
        topk_idx [r] = INVALID_INDEX;
    }

    int   candidate_count = 0;
    float max_distance    = CUDART_INF_F;
    bool  topk_full       = false;

    // Precomputed lane mask for compacting ballot-selected candidates.
    const unsigned lane_lt_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);

    for (int base = 0; base < data_count; base += BATCH_POINTS) {
        const int remaining   = data_count - base;
        const int batch_count = (remaining < BATCH_POINTS) ? remaining : BATCH_POINTS;

        // Whole-block staging of a large data batch into shared memory.
        #pragma unroll
        for (int it = 0; it < LOAD_ITERS; ++it) {
            const int t = threadIdx.x + it * BLOCK_THREADS;
            if (t < batch_count) {
                sh_batch[t] = data[base + t];
            }
        }
        __syncthreads();

        if (active_query) {
            // Each warp scans the cached batch in tiles of 32 points so that
            // one ballot handles exactly one point per lane.
            for (int tile = 0; tile < batch_count; tile += WARP_SIZE) {
                const int local_idx  = tile + lane;
                const bool valid_pt  = (local_idx < batch_count);

                float dist = CUDART_INF_F;
                int   data_idx = base + local_idx;

                if (valid_pt) {
                    const float2 p  = sh_batch[local_idx];
                    const float  dx = qx - p.x;
                    const float  dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);
                }

                // Before the first full top-k exists, max_distance is only a
                // sentinel, so filtering is disabled to avoid dropping +inf values.
                bool keep = valid_pt && (!topk_full || (dist < max_distance));
                unsigned keep_mask = __ballot_sync(FULL_MASK, keep);
                int new_candidates = __popc(keep_mask);

                if (new_candidates != 0) {
                    // If the current tile would overflow the candidate buffer,
                    // first flush the existing buffer, then re-evaluate this tile
                    // against the tighter threshold.
                    if (candidate_count + new_candidates > K) {
                        flush_candidates<K>(topk_dist, topk_idx,
                                            warp_buf_dist, warp_buf_idx, warp_buf_count,
                                            lane, candidate_count, max_distance, topk_full);

                        keep = valid_pt && (!topk_full || (dist < max_distance));
                        keep_mask = __ballot_sync(FULL_MASK, keep);
                        new_candidates = __popc(keep_mask);
                    }

                    if (new_candidates != 0) {
                        if (keep) {
                            const int offset = __popc(keep_mask & lane_lt_mask);
                            const int pos    = candidate_count + offset;
                            warp_buf_dist[pos] = dist;
                            warp_buf_idx [pos] = data_idx;
                        }

                        candidate_count += new_candidates;
                        if (lane == 0) {
                            warp_buf_count[0] = candidate_count;
                        }

                        if (candidate_count == K) {
                            flush_candidates<K>(topk_dist, topk_idx,
                                                warp_buf_dist, warp_buf_idx, warp_buf_count,
                                                lane, candidate_count, max_distance, topk_full);
                        }
                    }
                }
            }
        }

        // All warps must finish reading the staged batch before the block overwrites it.
        __syncthreads();
    }

    if (active_query && candidate_count > 0) {
        flush_candidates<K>(topk_dist, topk_idx,
                            warp_buf_dist, warp_buf_idx, warp_buf_count,
                            lane, candidate_count, max_distance, topk_full);
    }

    if (active_query) {
        // Write back the sorted top-k in row-major order:
        // result[query_id * K + j] = {index, squared_distance}.
        const int lane_base = lane * REGS_PER_THREAD;
        result_pair_t *out = result
                           + static_cast<size_t>(query_id) * static_cast<size_t>(K)
                           + static_cast<size_t>(lane_base);

        #pragma unroll
        for (int r = 0; r < REGS_PER_THREAD; ++r) {
            out[r].first  = topk_idx[r];
            out[r].second = topk_dist[r];
        }
    }
}

template <int K>
inline void launch_knn_impl(const float2 *query,
                            const int query_count,
                            const float2 *data,
                            const int data_count,
                            result_pair_t *result) {
    const size_t smem = shared_bytes<K>();

    // The target devices support this amount of dynamic shared memory.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem));
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               static_cast<int>(cudaSharedmemCarveoutMaxShared));

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn_kernel<K><<<grid, block, smem>>>(query, query_count, data, data_count, result);
}

} // namespace knn_detail

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:   knn_detail::launch_knn_impl<32>(query, query_count, data, data_count, result);   return;
        case 64:   knn_detail::launch_knn_impl<64>(query, query_count, data, data_count, result);   return;
        case 128:  knn_detail::launch_knn_impl<128>(query, query_count, data, data_count, result);  return;
        case 256:  knn_detail::launch_knn_impl<256>(query, query_count, data, data_count, result);  return;
        case 512:  knn_detail::launch_knn_impl<512>(query, query_count, data, data_count, result);  return;
        case 1024: knn_detail::launch_knn_impl<1024>(query, query_count, data, data_count, result); return;
        default:   return; // Inputs are guaranteed valid, so this path is unreachable in normal use.
    }
}