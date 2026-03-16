#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace knn_detail {

// Warp-per-query design:
// - 1 warp computes the full k-NN result for 1 query.
// - The current top-k ("intermediate result") is private to the warp and kept in registers, distributed across lanes.
// - Each warp also owns a shared-memory candidate buffer of size k plus a shared candidate counter.
// - The whole block stages a batch of data points in shared memory so multiple query-warps reuse each global load.
//
// Chosen launch parameters:
// - 8 warps/block (256 threads): good balance between data reuse and enough blocks for the stated "thousands of queries" regime.
// - 1024 staged data points/block iteration: exactly 4 coalesced float2 loads/thread, large enough to amortize barriers,
//   and still below the A100 opt-in shared-memory limit even for k = 1024.
//
// The staged batch is stored as structure-of-arrays (x[] and y[]) rather than float2[] to avoid the avoidable
// shared-memory bank conflicts that consecutive float2 reads would create on 4-byte banks.

using ResultPair = std::pair<int, float>;

constexpr int WARP_SIZE        = 32;
constexpr int LAST_LANE        = WARP_SIZE - 1;
constexpr int WARPS_PER_BLOCK  = 8;
constexpr int BLOCK_THREADS    = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int BATCH_POINTS     = 1024;
constexpr int LOAD_ITERS       = BATCH_POINTS / BLOCK_THREADS;
constexpr unsigned FULL_MASK   = 0xffffffffu;

static_assert(BATCH_POINTS % WARP_SIZE == 0, "BATCH_POINTS must be a multiple of the warp size.");
static_assert(BATCH_POINTS % BLOCK_THREADS == 0, "BATCH_POINTS is chosen so the block load loop is fully unrolled.");

template <int K>
constexpr std::size_t shared_bytes() {
    return
        sizeof(float) * BATCH_POINTS +                     // batch_x
        sizeof(float) * BATCH_POINTS +                     // batch_y
        sizeof(int)   * WARPS_PER_BLOCK +                  // per-warp candidate counts
        sizeof(int)   * WARPS_PER_BLOCK * K +              // candidate indices
        sizeof(float) * WARPS_PER_BLOCK * K +              // candidate distances
        sizeof(int)   * WARPS_PER_BLOCK * K +              // scratch indices for current top-k
        sizeof(float) * WARPS_PER_BLOCK * K;               // scratch distances for current top-k
}

template <int K>
__device__ __forceinline__ int co_rank(int diag, const float* a, const float* b) {
    // Merge-path / co-rank partition for two sorted arrays a[0:K) and b[0:K).
    // It returns how many elements of the first 'diag' outputs of the stable merge
    // (ties broken in favor of 'a') come from 'a'.
    int low  = (diag > K) ? (diag - K) : 0;
    int high = (diag < K) ? diag : K;

    while (low <= high) {
        const int i = (low + high) >> 1;
        const int j = diag - i;

        const float a_im1 = (i > 0) ? a[i - 1] : -CUDART_INF_F;
        const float a_i   = (i < K) ? a[i]     :  CUDART_INF_F;
        const float b_jm1 = (j > 0) ? b[j - 1] : -CUDART_INF_F;
        const float b_j   = (j < K) ? b[j]     :  CUDART_INF_F;

        if (a_im1 > b_j) {
            high = i - 1;
        } else if (b_jm1 >= a_i) {
            low = i + 1;
        } else {
            return i;
        }
    }

    return low;
}

template <int K>
__device__ __forceinline__ void bitonic_sort_shared(int lane, float* dist, int* idx) {
    // K is guaranteed to be a power of two, so a shared-memory bitonic sort is a natural fit.
    // Sorting happens only when the candidate buffer is flushed/merged, which is infrequent once max_distance stabilizes.
    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < K; i += WARP_SIZE) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = ((i & size) == 0);

                    const float di = dist[i];
                    const float dj = dist[j];
                    const int   ii = idx[i];
                    const int   ij = idx[j];

                    const bool do_swap = ascending ? (di > dj) : (di < dj);
                    if (do_swap) {
                        dist[i] = dj; dist[j] = di;
                        idx[i]  = ij; idx[j]  = ii;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

template <int K>
__device__ __forceinline__ void merge_candidates(
    int lane,
    int count,
    int   (&top_idx)[K / WARP_SIZE],
    float (&top_dist)[K / WARP_SIZE],
    float& max_distance,
    int*   cand_count_ptr,
    int*   cand_idx,
    float* cand_dist,
    int*   scratch_idx,
    float* scratch_dist)
{
    constexpr int CHUNK = K / WARP_SIZE;

    if (count == 0) {
        return;
    }

    // Spill the warp-private current top-k from registers to shared memory so the warp can access it randomly
    // during the merge-path partition/merge.
    #pragma unroll
    for (int t = 0; t < CHUNK; ++t) {
        const int pos = lane * CHUNK + t;
        scratch_idx[pos]  = top_idx[t];
        scratch_dist[pos] = top_dist[t];
    }

    // Pad the candidate buffer to exactly K entries with sentinels so the sort/merge logic stays branch-light.
    for (int pos = count + lane; pos < K; pos += WARP_SIZE) {
        cand_idx[pos]  = -1;
        cand_dist[pos] = CUDART_INF_F;
    }
    __syncwarp(FULL_MASK);

    bitonic_sort_shared<K>(lane, cand_dist, cand_idx);
    __syncwarp(FULL_MASK);

    // Each lane owns a contiguous CHUNK-sized segment of the merged top-k. This makes merge-path partitioning simple:
    // lane l produces outputs [l*CHUNK, (l+1)*CHUNK).
    const int out_begin = lane * CHUNK;
    const int out_end   = out_begin + CHUNK;

    int a_begin = co_rank<K>(out_begin, scratch_dist, cand_dist);
    int a_end   = __shfl_down_sync(FULL_MASK, a_begin, 1);
    if (lane == LAST_LANE) {
        a_end = co_rank<K>(K, scratch_dist, cand_dist);
    }

    int b_begin = out_begin - a_begin;
    int b_end   = out_end   - a_end;

    int a_pos = a_begin;
    int b_pos = b_begin;

    #pragma unroll
    for (int t = 0; t < CHUNK; ++t) {
        const bool take_a =
            (b_pos >= b_end) ||
            ((a_pos < a_end) && (scratch_dist[a_pos] <= cand_dist[b_pos]));

        if (take_a) {
            top_dist[t] = scratch_dist[a_pos];
            top_idx[t]  = scratch_idx[a_pos];
            ++a_pos;
        } else {
            top_dist[t] = cand_dist[b_pos];
            top_idx[t]  = cand_idx[b_pos];
            ++b_pos;
        }
    }

    __syncwarp(FULL_MASK);

    // Reset the shared candidate counter for the next accumulation round.
    if (lane == 0) {
        *cand_count_ptr = 0;
    }

    // The k-th nearest neighbor is the last element of the sorted top-k, owned by the last lane.
    const float lane_last = (lane == LAST_LANE) ? top_dist[CHUNK - 1] : 0.0f;
    max_distance = __shfl_sync(FULL_MASK, lane_last, LAST_LANE);

    __syncwarp(FULL_MASK);
}

__device__ __forceinline__ void store_result(ResultPair* dst, int idx, float dist) {
    // Writing fields individually avoids relying on any device-side std::pair helper functions.
    dst->first  = idx;
    dst->second = dist;
}

template <int K>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    ResultPair* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024, "K must be within the supported range.");
    static_assert((K & (K - 1)) == 0,   "K must be a power of two.");
    static_assert((K % WARP_SIZE) == 0, "K must be divisible by the warp size.");

    constexpr int CHUNK = K / WARP_SIZE;

    // Dynamic shared-memory layout:
    // [batch_x | batch_y | cand_count[warps] | cand_idx[warps][K] | cand_dist[warps][K]
    //  | scratch_idx[warps][K] | scratch_dist[warps][K]]
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem = smem_raw;

    float* batch_x = reinterpret_cast<float*>(smem);
    smem += sizeof(float) * BATCH_POINTS;

    float* batch_y = reinterpret_cast<float*>(smem);
    smem += sizeof(float) * BATCH_POINTS;

    int* cand_count_all = reinterpret_cast<int*>(smem);
    smem += sizeof(int) * WARPS_PER_BLOCK;

    int* cand_idx_all = reinterpret_cast<int*>(smem);
    smem += sizeof(int) * WARPS_PER_BLOCK * K;

    float* cand_dist_all = reinterpret_cast<float*>(smem);
    smem += sizeof(float) * WARPS_PER_BLOCK * K;

    int* scratch_idx_all = reinterpret_cast<int*>(smem);
    smem += sizeof(int) * WARPS_PER_BLOCK * K;

    float* scratch_dist_all = reinterpret_cast<float*>(smem);

    const int warp_id     = threadIdx.x >> 5;
    const int lane        = threadIdx.x & (WARP_SIZE - 1);
    const int query_index = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active     = (query_index < query_count);

    int*   cand_count_ptr = cand_count_all + warp_id;
    int*   cand_idx       = cand_idx_all + warp_id * K;
    float* cand_dist      = cand_dist_all + warp_id * K;
    int*   scratch_idx    = scratch_idx_all + warp_id * K;
    float* scratch_dist   = scratch_dist_all + warp_id * K;

    // Warp-private intermediate top-k, distributed across lanes.
    int   top_idx[CHUNK];
    float top_dist[CHUNK];

    float qx = 0.0f;
    float qy = 0.0f;
    float max_distance = CUDART_INF_F;
    int buffer_count = 0;  // Register shadow of the per-warp shared candidate counter.

    if (active) {
        // One lane loads the query point, then broadcasts it to the whole warp.
        float qx_lane0 = 0.0f;
        float qy_lane0 = 0.0f;
        if (lane == 0) {
            const float2 q = query[query_index];
            qx_lane0 = q.x;
            qy_lane0 = q.y;
            *cand_count_ptr = 0;
        }
        qx = __shfl_sync(FULL_MASK, qx_lane0, 0);
        qy = __shfl_sync(FULL_MASK, qy_lane0, 0);

        #pragma unroll
        for (int t = 0; t < CHUNK; ++t) {
            top_idx[t]  = -1;
            top_dist[t] = CUDART_INF_F;
        }

        __syncwarp(FULL_MASK);
    }

    for (int batch_base = 0; batch_base < data_count; batch_base += BATCH_POINTS) {
        int batch_n = data_count - batch_base;
        if (batch_n > BATCH_POINTS) {
            batch_n = BATCH_POINTS;
        }

        // Cooperative block-wide load of the next data batch into shared memory.
        #pragma unroll
        for (int it = 0; it < LOAD_ITERS; ++it) {
            const int i = it * BLOCK_THREADS + threadIdx.x;
            if (i < batch_n) {
                const float2 p = data[batch_base + i];
                batch_x[i] = p.x;
                batch_y[i] = p.y;
            }
        }

        __syncthreads();

        if (active) {
            // Warp-synchronous scan of the cached batch. All 32 lanes participate every iteration,
            // even on the last partial tile, so warp collectives always use FULL_MASK safely.
            for (int tile = 0; tile < batch_n; tile += WARP_SIZE) {
                const int off   = tile + lane;
                const bool valid = (off < batch_n);

                float dist = CUDART_INF_F;
                int data_index = -1;

                if (valid) {
                    const float dx = qx - batch_x[off];
                    const float dy = qy - batch_y[off];
                    dist = fmaf(dx, dx, dy * dy);  // squared Euclidean distance
                    data_index = batch_base + off;
                }

                // Retry loop: if the current mini-tile would overflow the candidate buffer,
                // merge first, tighten max_distance, then re-test the very same distance.
                while (true) {
                    const bool pass = valid && (dist < max_distance);
                    const unsigned pass_mask = __ballot_sync(FULL_MASK, pass);
                    const int num_pass = __popc(pass_mask);

                    if (num_pass == 0) {
                        break;
                    }

                    if (buffer_count + num_pass > K) {
                        merge_candidates<K>(
                            lane, buffer_count,
                            top_idx, top_dist, max_distance,
                            cand_count_ptr, cand_idx, cand_dist,
                            scratch_idx, scratch_dist);
                        buffer_count = 0;
                        continue;
                    }

                    // Warp-aggregated reservation:
                    // one shared atomicAdd reserves a contiguous range for all passing lanes.
                    int base_pos = 0;
                    if (lane == 0) {
                        base_pos = atomicAdd(cand_count_ptr, num_pass);
                    }
                    base_pos = __shfl_sync(FULL_MASK, base_pos, 0);

                    if (pass) {
                        const unsigned lower_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
                        const int rank = __popc(pass_mask & lower_mask);
                        const int pos = base_pos + rank;
                        cand_idx[pos]  = data_index;
                        cand_dist[pos] = dist;
                    }

                    __syncwarp(FULL_MASK);

                    buffer_count += num_pass;

                    if (buffer_count == K) {
                        merge_candidates<K>(
                            lane, buffer_count,
                            top_idx, top_dist, max_distance,
                            cand_count_ptr, cand_idx, cand_dist,
                            scratch_idx, scratch_dist);
                        buffer_count = 0;
                    }

                    break;
                }
            }
        }

        // The next iteration overwrites the staged batch, so all warps in the block must be done with it.
        __syncthreads();
    }

    if (active) {
        if (buffer_count > 0) {
            merge_candidates<K>(
                lane, buffer_count,
                top_idx, top_dist, max_distance,
                cand_count_ptr, cand_idx, cand_dist,
                scratch_idx, scratch_dist);
        }

        // Output is already globally sorted by ascending distance.
        const std::size_t out_base =
            static_cast<std::size_t>(query_index) * static_cast<std::size_t>(K) +
            static_cast<std::size_t>(lane) * static_cast<std::size_t>(CHUNK);

        ResultPair* out = result + out_base;

        #pragma unroll
        for (int t = 0; t < CHUNK; ++t) {
            store_result(out + t, top_idx[t], top_dist[t]);
        }
    }
}

template <int K>
void launch_case(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    ResultPair* result)
{
    static_assert(shared_bytes<K>() <= 163840,
                  "Kernel configuration exceeds the A100 opt-in shared-memory limit.");

    constexpr std::size_t smem_bytes = shared_bytes<K>();

    // The public API returns void, so launch/error handling is intentionally left to the caller.
    // We still request the shared-memory carveout required by the chosen configuration.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    (void)cudaFuncSetCacheConfig(
        knn_kernel<K>,
        cudaFuncCachePreferShared);

    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    knn_kernel<K><<<blocks, BLOCK_THREADS, smem_bytes>>>(
        query, query_count, data, data_count, result);
}

}  // namespace knn_detail

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

    // k is guaranteed by contract to be one of these values.
    switch (k) {
        case 32:   knn_detail::launch_case<32>(query, query_count, data, data_count, result);   break;
        case 64:   knn_detail::launch_case<64>(query, query_count, data, data_count, result);   break;
        case 128:  knn_detail::launch_case<128>(query, query_count, data, data_count, result);  break;
        case 256:  knn_detail::launch_case<256>(query, query_count, data, data_count, result);  break;
        case 512:  knn_detail::launch_case<512>(query, query_count, data, data_count, result);  break;
        case 1024: knn_detail::launch_case<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Unreachable under the stated preconditions.
    }
}