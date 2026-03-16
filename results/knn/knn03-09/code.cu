#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace knn_detail {

/*
Exact exhaustive 2D k-NN on modern NVIDIA GPUs.

Design choices:
- One warp computes one query end-to-end.
- The thread block cooperatively stages a batch of data points into shared memory.
- Shared memory uses a structure-of-arrays layout (x[] and y[]) even though the input is float2:
  this avoids the shared-memory bank conflict pattern that a direct float2 cache would create.
- Each warp keeps its current top-K entirely on-chip, distributed across warp lanes:
    lane l owns the entries at global positions l, l+32, l+64, ...
  so each lane stores K/32 (distance, index) pairs in registers.
- The distributed top-K buffer is kept globally sorted at all times.
- For every new candidate that beats the current worst entry, insertion is warp-cooperative:
  only the first affected 32-element "slot" needs a ballot/shuffle based insertion; all later
  slots are just warp-wide right shifts carrying displaced boundary elements forward.
- K is specialized at compile time (32,64,128,256,512,1024), which gives the compiler a fixed
  register footprint and fully unrolled slot-manipulation loops.

The code intentionally does not allocate any additional device memory.
*/

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kWarpSize = 32;

/*
A 4096-point batch occupies 32 KiB in shared memory in SoA form:
    4096 * 2 * sizeof(float) = 32768 bytes
This stays below the default 48 KiB dynamic shared-memory limit, so no extra function
attributes are needed, while still being large enough to amortize block-wide synchronization.
*/
constexpr int kSharedBatchPoints = 4096;
constexpr std::size_t kSharedBytes =
    static_cast<std::size_t>(2) * static_cast<std::size_t>(kSharedBatchPoints) * sizeof(float);

constexpr int kSmallBlockThreads = 256;  // 8 queries/block
constexpr int kLargeBlockThreads = 512;  // 16 queries/block
constexpr int kSmallWarpsPerBlock = kSmallBlockThreads / kWarpSize;
constexpr int kLargeWarpsPerBlock = kLargeBlockThreads / kWarpSize;

static_assert((kSharedBatchPoints % kWarpSize) == 0, "Shared batch size must be a multiple of 32.");
static_assert((kSmallBlockThreads % kWarpSize) == 0, "Block size must be a multiple of 32.");
static_assert((kLargeBlockThreads % kWarpSize) == 0, "Block size must be a multiple of 32.");
static_assert(kSharedBytes <= (48u * 1024u), "Shared batch intentionally fits under 48 KiB.");

/*
Strict total order for (distance, index). The problem allows arbitrary tie resolution;
using the index as a secondary key simply makes the order deterministic and keeps the
warp insertion logic well-defined.
*/
__device__ __forceinline__ bool pair_less(float a_dist, int a_idx, float b_dist, int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

/*
Write field-wise so the kernel does not depend on a device-side std::pair constructor or
assignment operator. This matches the requested output layout.
*/
__device__ __forceinline__ void store_result(std::pair<int, float>* result, int out, int idx, float dist) {
    result[out].first = idx;
    result[out].second = dist;
}

/*
Warp-cooperative insertion into the distributed sorted top-K buffer.

Layout:
- Each lane stores ITEMS_PER_LANE = K/32 entries in registers.
- Slot s consists of top_dist[s] / top_idx[s] across all 32 lanes.
- The full K-entry sequence is:
    slot 0 (lanes 0..31), slot 1 (lanes 0..31), ...

Important optimization:
- Lane 31 owns the last element of every slot, i.e. the slot boundaries.
- We first find the earliest slot whose boundary is larger than the candidate.
- The candidate is inserted into that one slot using ballot/shuffle operations.
- The displaced boundary element from that slot is guaranteed to be <= every element
  of the next slot, so every later slot update is just a right shift with lane 0
  receiving the carry value.
*/
template <int K>
__device__ __forceinline__ void insert_candidate(
    float (&top_dist)[K / kWarpSize],
    int (&top_idx)[K / kWarpSize],
    float cand_dist,
    int cand_idx,
    int lane)
{
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "Unsupported K specialization.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of 32.");

    constexpr int ITEMS_PER_LANE = K / kWarpSize;

    // Find the first affected slot by scanning slot boundaries in lane 31.
    int start_slot = ITEMS_PER_LANE - 1;
    if (lane == (kWarpSize - 1)) {
        #pragma unroll
        for (int s = 0; s < ITEMS_PER_LANE - 1; ++s) {
            if ((s < start_slot) && pair_less(cand_dist, cand_idx, top_dist[s], top_idx[s])) {
                start_slot = s;
            }
        }
    }
    start_slot = __shfl_sync(kFullMask, start_slot, kWarpSize - 1);

    float carry_dist = cand_dist;
    int carry_idx = cand_idx;

    #pragma unroll
    for (int s = 0; s < ITEMS_PER_LANE; ++s) {
        if (s == start_slot) {
            const float cur_dist = top_dist[s];
            const int cur_idx = top_idx[s];

            const float slot_last_dist = __shfl_sync(kFullMask, cur_dist, kWarpSize - 1);
            const int slot_last_idx = __shfl_sync(kFullMask, cur_idx, kWarpSize - 1);

            const unsigned less_mask =
                __ballot_sync(kFullMask, pair_less(cur_dist, cur_idx, carry_dist, carry_idx));
            const int insert_pos = __popc(less_mask);

            const float prev_dist = __shfl_up_sync(kFullMask, cur_dist, 1);
            const int prev_idx = __shfl_up_sync(kFullMask, cur_idx, 1);

            // insert_pos is guaranteed to be < 32 because start_slot was chosen so that
            // the candidate is better than the boundary element of this slot.
            if (lane == insert_pos) {
                top_dist[s] = carry_dist;
                top_idx[s] = carry_idx;
            } else if (lane > insert_pos) {
                top_dist[s] = prev_dist;
                top_idx[s] = prev_idx;
            }

            carry_dist = slot_last_dist;
            carry_idx = slot_last_idx;
        } else if (s > start_slot) {
            const float cur_dist = top_dist[s];
            const int cur_idx = top_idx[s];

            const float slot_last_dist = __shfl_sync(kFullMask, cur_dist, kWarpSize - 1);
            const int slot_last_idx = __shfl_sync(kFullMask, cur_idx, kWarpSize - 1);

            const float prev_dist = __shfl_up_sync(kFullMask, cur_dist, 1);
            const int prev_idx = __shfl_up_sync(kFullMask, cur_idx, 1);

            // Later slots are always affected at position 0.
            if (lane == 0) {
                top_dist[s] = carry_dist;
                top_idx[s] = carry_idx;
            } else {
                top_dist[s] = prev_dist;
                top_idx[s] = prev_idx;
            }

            carry_dist = slot_last_dist;
            carry_idx = slot_last_idx;
        }
    }
}

template <int K, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS)
void knn_kernel(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result)
{
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "Unsupported K specialization.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of 32.");
    static_assert((BLOCK_THREADS % kWarpSize) == 0, "Block size must be a multiple of 32.");

    constexpr int ITEMS_PER_LANE = K / kWarpSize;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;

    extern __shared__ float shared_mem[];
    float* const sh_x = shared_mem;
    float* const sh_y = shared_mem + kSharedBatchPoints;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    const int query_index = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const bool query_active = (query_index < query_count);

    // Broadcast the query point from lane 0 of each warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (query_active && lane == 0) {
        const float2 q = query[query_index];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Warp-private top-K buffer, distributed across lanes and kept globally sorted.
    float top_dist[ITEMS_PER_LANE];
    int top_idx[ITEMS_PER_LANE];

    #pragma unroll
    for (int s = 0; s < ITEMS_PER_LANE; ++s) {
        top_dist[s] = CUDART_INF_F;
        top_idx[s] = INT_MAX;
    }

    float worst_dist = CUDART_INF_F;
    int worst_idx = INT_MAX;

    // Process the data set in block-cached batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += kSharedBatchPoints) {
        int current_batch = data_count - batch_start;
        if (current_batch > kSharedBatchPoints) {
            current_batch = kSharedBatchPoints;
        }

        // Cooperative global->shared staging for the whole block.
        for (int i = tid; i < current_batch; i += BLOCK_THREADS) {
            const float2 p = data[batch_start + i];
            sh_x[i] = p.x;
            sh_y[i] = p.y;
        }
        __syncthreads();

        if (query_active) {
            /*
            Iterate by warp-wide 32-point strips, not by "for (j = lane; j < ...; j += 32)" with
            per-lane trip counts. This guarantees that every lane executes the same number of
            warp-synchronous iterations, and tail lanes simply contribute an inactive +inf candidate.
            */
            const int strip_count = (current_batch + kWarpSize - 1) / kWarpSize;

            #pragma unroll 1
            for (int strip = 0; strip < strip_count; ++strip) {
                const int point_in_batch = strip * kWarpSize + lane;

                float cand_dist = CUDART_INF_F;
                int cand_idx = INT_MAX;

                if (point_in_batch < current_batch) {
                    const float dx = qx - sh_x[point_in_batch];
                    const float dy = qy - sh_y[point_in_batch];
                    cand_dist = __fmaf_rn(dx, dx, dy * dy);  // squared L2 distance
                    cand_idx = batch_start + point_in_batch;
                }

                /*
                The acceptance threshold is the current worst element in our total order.
                That threshold is monotone non-increasing over time, so lanes that are inactive
                now can never become active later in the scan.
                */
                unsigned active_mask =
                    __ballot_sync(kFullMask, pair_less(cand_dist, cand_idx, worst_dist, worst_idx));

                // Process active candidates in increasing data index order (lane order within the strip).
                while (active_mask) {
                    const int src_lane = __ffs(active_mask) - 1;
                    const float ins_dist = __shfl_sync(kFullMask, cand_dist, src_lane);
                    const int ins_idx = __shfl_sync(kFullMask, cand_idx, src_lane);

                    // Re-check against the current threshold because earlier insertions in this strip
                    // may have tightened it.
                    if (pair_less(ins_dist, ins_idx, worst_dist, worst_idx)) {
                        insert_candidate<K>(top_dist, top_idx, ins_dist, ins_idx, lane);
                        worst_dist = __shfl_sync(kFullMask, top_dist[ITEMS_PER_LANE - 1], kWarpSize - 1);
                        worst_idx = __shfl_sync(kFullMask, top_idx[ITEMS_PER_LANE - 1], kWarpSize - 1);
                    }

                    active_mask &= (active_mask - 1);
                }
            }
        }

        // All warps must finish consuming the cached batch before the next batch overwrites it.
        __syncthreads();
    }

    if (query_active) {
        const int base = query_index * K;

        #pragma unroll
        for (int s = 0; s < ITEMS_PER_LANE; ++s) {
            const int out = base + s * kWarpSize + lane;
            store_result(result, out, top_idx[s], top_dist[s]);
        }
    }
}

template <int K, int BLOCK_THREADS>
inline void launch_kernel(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result)
{
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;
    const dim3 block(BLOCK_THREADS);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn_kernel<K, BLOCK_THREADS><<<grid, block, kSharedBytes>>>(
        query, query_count, data, data_count, result);
}

inline void dispatch_small_block(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k)
{
    // K is specialized so the register-resident top-K buffer size is known at compile time.
    switch (k) {
        case 32:   launch_kernel<32,   kSmallBlockThreads>(query, query_count, data, data_count, result); break;
        case 64:   launch_kernel<64,   kSmallBlockThreads>(query, query_count, data, data_count, result); break;
        case 128:  launch_kernel<128,  kSmallBlockThreads>(query, query_count, data, data_count, result); break;
        case 256:  launch_kernel<256,  kSmallBlockThreads>(query, query_count, data, data_count, result); break;
        case 512:  launch_kernel<512,  kSmallBlockThreads>(query, query_count, data, data_count, result); break;
        case 1024: launch_kernel<1024, kSmallBlockThreads>(query, query_count, data, data_count, result); break;
        default: break;  // Inputs are guaranteed valid; this is only a defensive fallback.
    }
}

inline void dispatch_large_block(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k)
{
    /*
    512-thread blocks double shared-batch reuse (16 queries/block instead of 8), but they also
    produce fewer blocks. We only use this path when the host-side heuristic says there are
    enough queries to keep the GPU populated, and we intentionally leave K=1024 on the smaller
    block size to keep register pressure conservative.
    */
    switch (k) {
        case 32:  launch_kernel<32,  kLargeBlockThreads>(query, query_count, data, data_count, result); break;
        case 64:  launch_kernel<64,  kLargeBlockThreads>(query, query_count, data, data_count, result); break;
        case 128: launch_kernel<128, kLargeBlockThreads>(query, query_count, data, data_count, result); break;
        case 256: launch_kernel<256, kLargeBlockThreads>(query, query_count, data, data_count, result); break;
        case 512: launch_kernel<512, kLargeBlockThreads>(query, query_count, data, data_count, result); break;
        default:  dispatch_small_block(query, query_count, data, data_count, result, k); break;
    }
}

inline bool use_large_block_heuristic(int query_count, int k) {
    // Keep the largest register footprint (K=1024) on the smaller block size.
    if (k > 512) {
        return false;
    }

    int device = 0;
    int sm_count = 0;

    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return false;
    }

    /*
    Use 512-thread blocks once they still provide at least one block per SM.
    In the intended "thousands of queries" regime this is a good tradeoff:
    we increase data reuse substantially without under-filling the GPU.
    */
    return query_count >= (sm_count * kLargeWarpsPerBlock);
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
    if (query_count <= 0 || data_count <= 0) {
        return;
    }

    if (knn_detail::use_large_block_heuristic(query_count, k)) {
        knn_detail::dispatch_large_block(query, query_count, data, data_count, result, k);
    } else {
        knn_detail::dispatch_small_block(query, query_count, data, data_count, result, k);
    }
}