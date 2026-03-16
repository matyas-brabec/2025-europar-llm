#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace {

// One warp computes one query. Eight warps per block gives strong reuse of each
// loaded data batch while still fitting the worst-case k=1024 state in shared
// memory on A100/H100-class parts.
constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr unsigned FULL_MASK    = 0xffffffffu;

// Total order on (distance, index). The index tie-break is only used to make the
// sorting / merging networks deterministic; tie resolution was otherwise unspecified.
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

// Shared-memory compare/swap for a single K-element buffer.
__device__ __forceinline__ void compare_swap_shared(
    float *dist, int *idx, int a, int b, bool ascending)
{
    float da = dist[a];
    int   ia = idx[a];
    float db = dist[b];
    int   ib = idx[b];

    const bool do_swap = ascending ? pair_greater(da, ia, db, ib)
                                   : pair_less   (da, ia, db, ib);
    if (do_swap) {
        dist[a] = db;
        idx[a]  = ib;
        dist[b] = da;
        idx[b]  = ia;
    }
}

// Standard bitonic sort for a single K-element shared-memory buffer.
template <int N>
__device__ __forceinline__ void bitonic_sort_shared(float *dist, int *idx, int lane)
{
#pragma unroll
    for (int size = 2; size <= N; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += WARP_SIZE) {
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

// Helpers to address the combined [best | candidate] 2K-element sequence.
template <int K>
__device__ __forceinline__ void load_combined(
    const float *best_dist, const int *best_idx,
    const float *cand_dist, const int *cand_idx,
    int pos, float &d, int &i)
{
    if (pos < K) {
        d = best_dist[pos];
        i = best_idx[pos];
    } else {
        pos -= K;
        d = cand_dist[pos];
        i = cand_idx[pos];
    }
}

template <int K>
__device__ __forceinline__ void store_combined(
    float *best_dist, int *best_idx,
    float *cand_dist, int *cand_idx,
    int pos, float d, int i)
{
    if (pos < K) {
        best_dist[pos] = d;
        best_idx[pos]  = i;
    } else {
        pos -= K;
        cand_dist[pos] = d;
        cand_idx[pos]  = i;
    }
}

// Reverse the candidate buffer in-place. If best is sorted ascending and candidate
// is sorted ascending, then [best, reverse(candidate)] is bitonic and can be
// merged with an O(K log K) bitonic-merge network instead of a full O(K log^2 K) sort.
template <int K>
__device__ __forceinline__ void reverse_shared(float *dist, int *idx, int lane)
{
    for (int i = lane; i < (K >> 1); i += WARP_SIZE) {
        const int j = K - 1 - i;
        float da = dist[i];
        int   ia = idx[i];
        float db = dist[j];
        int   ib = idx[j];
        dist[i] = db;
        idx[i]  = ib;
        dist[j] = da;
        idx[j]  = ia;
    }
    __syncwarp();
}

// Merge two already-sorted K-element buffers into sorted [best | candidate].
// After completion, the best K elements reside in best_*[0..K-1].
template <int K>
__device__ __forceinline__ void bitonic_merge_two_sorted_shared(
    float *best_dist, int *best_idx,
    float *cand_dist, int *cand_idx,
    int lane)
{
    reverse_shared<K>(cand_dist, cand_idx, lane);

    constexpr int N = 2 * K;
#pragma unroll
    for (int stride = N >> 1; stride > 0; stride >>= 1) {
        for (int i = lane; i < N; i += WARP_SIZE) {
            const int j = i ^ stride;
            if (j > i) {
                float di, dj;
                int   ii, ij;
                load_combined<K>(best_dist, best_idx, cand_dist, cand_idx, i, di, ii);
                load_combined<K>(best_dist, best_idx, cand_dist, cand_idx, j, dj, ij);

                if (pair_greater(di, ii, dj, ij)) {
                    store_combined<K>(best_dist, best_idx, cand_dist, cand_idx, i, dj, ij);
                    store_combined<K>(best_dist, best_idx, cand_dist, cand_idx, j, di, ii);
                }
            }
        }
        __syncwarp();
    }
}

// Merge the shared candidate buffer into the current top-k state.
// The candidate buffer is padded with (+inf, INT_MAX), sorted, and then merged
// with the current best buffer.
template <int K>
__device__ __forceinline__ void merge_candidate_buffer(
    float *best_dist, int *best_idx,
    float *cand_dist, int *cand_idx,
    int *cand_count_ptr,
    float &max_distance,
    int lane)
{
    int count = 0;
    if (lane == 0) count = *cand_count_ptr;
    count = __shfl_sync(FULL_MASK, count, 0);
    if (count == 0) return;

    for (int pos = lane; pos < K; pos += WARP_SIZE) {
        if (pos >= count) {
            cand_dist[pos] = CUDART_INF_F;
            cand_idx[pos]  = INT_MAX;
        }
    }
    __syncwarp();

    bitonic_sort_shared<K>(cand_dist, cand_idx, lane);
    bitonic_merge_two_sorted_shared<K>(best_dist, best_idx, cand_dist, cand_idx, lane);

    if (lane == 0) {
        *cand_count_ptr = 0;
        max_distance = best_dist[K - 1];
    }
    max_distance = __shfl_sync(FULL_MASK, max_distance, 0);
    __syncwarp();
}

// Scan one shared-memory data batch. The candidate slots are reserved with a
// warp-aggregated atomicAdd (still using atomicAdd exactly as requested, but only
// once per 32-point tile instead of once per candidate).
template <int K>
__device__ __forceinline__ void process_batch_points(
    float qx, float qy,
    int batch_global_start,
    int local_begin,
    int local_end,
    const float *batch_x,
    const float *batch_y,
    float *best_dist,
    int *best_idx,
    float *cand_dist,
    int *cand_idx,
    int *cand_count_ptr,
    float &max_distance,
    int lane)
{
    for (int base = local_begin; base < local_end; base += WARP_SIZE) {
        if (max_distance == 0.0f) break;  // squared distances are non-negative

        const int local_pos = base + lane;

        float d = CUDART_INF_F;
        int   idx = -1;
        bool  pass = false;

        if (local_pos < local_end) {
            const float dx = qx - batch_x[local_pos];
            const float dy = qy - batch_y[local_pos];
            d   = fmaf(dx, dx, dy * dy);
            idx = batch_global_start + local_pos;
            pass = (d < max_distance);
        }

        unsigned pass_mask = __ballot_sync(FULL_MASK, pass);
        int hits = __popc(pass_mask);
        if (hits == 0) continue;

        int current_count = 0;
        if (lane == 0) current_count = *cand_count_ptr;
        current_count = __shfl_sync(FULL_MASK, current_count, 0);

        // If the current tile would overflow the candidate buffer, flush the
        // existing candidates first, then re-evaluate the current tile against the
        // tightened max_distance.
        if (current_count + hits > K) {
            merge_candidate_buffer<K>(
                best_dist, best_idx, cand_dist, cand_idx, cand_count_ptr,
                max_distance, lane);

            pass = (local_pos < local_end) && (d < max_distance);
            pass_mask = __ballot_sync(FULL_MASK, pass);
            hits = __popc(pass_mask);
            if (hits == 0) continue;
        }

        int base_slot = 0;
        if (lane == 0) base_slot = atomicAdd(cand_count_ptr, hits);
        base_slot = __shfl_sync(FULL_MASK, base_slot, 0);

        if (pass) {
            const unsigned prior_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
            const int local_rank = __popc(pass_mask & prior_mask);
            const int slot = base_slot + local_rank;
            cand_dist[slot] = d;
            cand_idx[slot]  = idx;
        }

        // Publish the writes before a possible merge on the next line or in the
        // next tile iteration.
        __syncwarp();

        if (base_slot + hits == K) {
            merge_candidate_buffer<K>(
                best_dist, best_idx, cand_dist, cand_idx, cand_count_ptr,
                max_distance, lane);
        }
    }
}

// Kernel layout in shared memory:
// [best_idx][best_dist][cand_idx][cand_dist][cand_count][batch_x][batch_y]
template <int K>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void knn_kernel(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    std::pair<int, float> * __restrict__ result,
    int batch_capacity)
{
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0),
                  "K must be a power of two in [32, 1024]");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    unsigned char *ptr = smem_raw;

    int   *best_idx_all  = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * K * sizeof(int);

    float *best_dist_all = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * K * sizeof(float);

    int   *cand_idx_all  = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * K * sizeof(int);

    float *cand_dist_all = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * K * sizeof(float);

    int   *cand_count    = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * sizeof(int);

    // Shared batch as SoA to avoid 64-bit shared-memory bank effects from float2.
    float *batch_x       = reinterpret_cast<float*>(ptr);
    float *batch_y       = batch_x + batch_capacity;

    const int tid      = threadIdx.x;
    const int lane     = tid & (WARP_SIZE - 1);
    const int warp_id  = tid >> 5;
    const int query_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active  = (query_id < query_count);

    float *best_dist = best_dist_all + warp_id * K;
    int   *best_idx  = best_idx_all  + warp_id * K;
    float *cand_dist = cand_dist_all + warp_id * K;
    int   *cand_idx  = cand_idx_all  + warp_id * K;
    int   *cand_count_ptr = cand_count + warp_id;

    if (tid < WARPS_PER_BLOCK) {
        cand_count[tid] = 0;
    }
    __syncthreads();

    float qx = 0.0f;
    float qy = 0.0f;
    float max_distance = CUDART_INF_F;

    if (active) {
        if (lane == 0) {
            const float2 q = query[query_id];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);
    }

    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;

    for (int batch_start = 0; batch_start < data_count; batch_start += batch_capacity) {
        int batch_count = data_count - batch_start;
        if (batch_count > batch_capacity) batch_count = batch_capacity;

        for (int i = tid; i < batch_count; i += THREADS_PER_BLOCK) {
            const float2 p = data[batch_start + i];
            batch_x[i] = p.x;
            batch_y[i] = p.y;
        }
        __syncthreads();

        if (active) {
            if (batch_start == 0) {
                // Initialize the private top-k state from the first K points.
#pragma unroll
                for (int t = 0, pos = lane; t < ITEMS_PER_LANE; ++t, pos += WARP_SIZE) {
                    const float dx = qx - batch_x[pos];
                    const float dy = qy - batch_y[pos];
                    best_dist[pos] = fmaf(dx, dx, dy * dy);
                    best_idx[pos]  = pos;
                }
                __syncwarp();

                bitonic_sort_shared<K>(best_dist, best_idx, lane);

                if (lane == 0) {
                    *cand_count_ptr = 0;
                    max_distance = best_dist[K - 1];
                }
                max_distance = __shfl_sync(FULL_MASK, max_distance, 0);

                if (batch_count > K && max_distance > 0.0f) {
                    process_batch_points<K>(
                        qx, qy,
                        batch_start,
                        K,
                        batch_count,
                        batch_x,
                        batch_y,
                        best_dist,
                        best_idx,
                        cand_dist,
                        cand_idx,
                        cand_count_ptr,
                        max_distance,
                        lane);
                }
            } else if (max_distance > 0.0f) {
                process_batch_points<K>(
                    qx, qy,
                    batch_start,
                    0,
                    batch_count,
                    batch_x,
                    batch_y,
                    best_dist,
                    best_idx,
                    cand_dist,
                    cand_idx,
                    cand_count_ptr,
                    max_distance,
                    lane);
            }
        }

        __syncthreads();
    }

    if (active) {
        // Final flush of the candidate buffer.
        merge_candidate_buffer<K>(
            best_dist, best_idx, cand_dist, cand_idx, cand_count_ptr,
            max_distance, lane);

        const std::size_t out_base = static_cast<std::size_t>(query_id) * static_cast<std::size_t>(K);

#pragma unroll
        for (int t = 0, pos = lane; t < ITEMS_PER_LANE; ++t, pos += WARP_SIZE) {
            result[out_base + pos].first  = best_idx[pos];
            result[out_base + pos].second = best_dist[pos];
        }
    }
}

// Shared memory excluding the [batch_x][batch_y] tail.
template <int K>
constexpr std::size_t shared_base_bytes()
{
    return static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(int)   +  // best_idx
           static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(float) +  // best_dist
           static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(int)   +  // cand_idx
           static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(float) +  // cand_dist
           static_cast<std::size_t>(WARPS_PER_BLOCK) * sizeof(int);         // cand_count
}

// Heuristic:
// - If two resident blocks/SM are feasible while still holding at least K batch points,
//   cap the per-block shared memory to ~1/2 of the opt-in maximum.
// - Otherwise, use the full opt-in maximum because occupancy is already limited to 1 block/SM.
template <int K>
inline int choose_batch_capacity(int optin_max_shared_per_block)
{
    const std::size_t optin_max = static_cast<std::size_t>(optin_max_shared_per_block);
    const std::size_t base      = shared_base_bytes<K>();
    const std::size_t min_total = base + static_cast<std::size_t>(K) * 2u * sizeof(float);

    std::size_t total_budget = (2u * min_total <= optin_max) ? (optin_max / 2u) : optin_max;
    if (total_budget < min_total) total_budget = min_total;

    return static_cast<int>((total_budget - base) / (2u * sizeof(float)));
}

template <int K>
inline void launch_knn_impl(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result)
{
    int device = 0;
    cudaGetDevice(&device);

    int optin_max_shared = 0;
    cudaDeviceGetAttribute(&optin_max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    const int batch_capacity = choose_batch_capacity<K>(optin_max_shared);
    const std::size_t dynamic_smem_bytes =
        shared_base_bytes<K>() + static_cast<std::size_t>(batch_capacity) * 2u * sizeof(float);

    cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem_bytes));

    // This kernel is shared-memory heavy and has little use for L1 compared to its
    // block-wide data reuse in shared memory.
    cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn_kernel<K><<<grid, block, dynamic_smem_bytes>>>(
        query, query_count, data, data_count, result, batch_capacity);
}

} // namespace

void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k)
{
    if (query_count <= 0) return;

    switch (k) {
        case 32:   launch_knn_impl<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_impl<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_impl<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_impl<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_impl<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_impl<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Inputs were stated to be valid.
    }
}