#include <cuda_runtime.h>
#include <cuda.h>
#include <utility>

// Optimized CUDA k-NN (squared L2) for 2D points.
// - One warp (32 threads) processes one query.
// - Data points are processed in shared-memory tiles loaded by the whole block.
// - Each warp keeps a distributed top-k structure across lanes (k/32 slots per lane),
//   maintained in descending order (slot 0 = worst/largest distance).
// - For each tile, each warp evaluates candidates in micro-batches of 32 points.
//   The warp cooperatively inserts multiple qualifying candidates into its distributed top-k
//   using warp-wide reductions (argmin/argmax) and shuffles, replacing the current global worst
//   until no remaining candidate can improve the set.
// - At the end, the warp emits the k neighbors in ascending order by repeatedly selecting
//   the current best among lanes using warp-wide argmin.

// Tunable parameters selected for modern data center GPUs (A100/H100).
#ifndef KNN_WARP_SIZE
#define KNN_WARP_SIZE 32
#endif

#ifndef KNN_BLOCK_THREADS
#define KNN_BLOCK_THREADS 256  // 8 warps per block
#endif

#ifndef KNN_TILE_POINTS
#define KNN_TILE_POINTS 4096   // 4096 float2 = 32KB shared memory; fits well on A100/H100
#endif

// Full warp mask constant for shuffles
#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xFFFFFFFFu
#endif

// Device utility: get lane id (0..31)
__device__ __forceinline__ int lane_id() {
    int lid;
#if __CUDA_ARCH__ >= 300
    asm volatile ("mov.s32 %0, %laneid;" : "=r"(lid));
#else
    lid = threadIdx.x & (KNN_WARP_SIZE - 1);
#endif
    return lid;
}

// Device utility: argmax across warp (value, lane_id).
// Ties are resolved in favor of smaller lane_id.
struct WarpArgMax {
    float val;
    int lane;
};

__device__ __forceinline__ WarpArgMax warp_argmax(float v) {
    WarpArgMax p{v, lane_id()};
    // Always use full-warp mask, set v = -INF for inactive lanes (if any) beforehand.
    #pragma unroll
    for (int offset = KNN_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float v2 = __shfl_down_sync(FULL_WARP_MASK, p.val, offset);
        int   l2 = __shfl_down_sync(FULL_WARP_MASK, p.lane, offset);
        if (v2 > p.val || (v2 == p.val && l2 < p.lane)) {
            p.val = v2;
            p.lane = l2;
        }
    }
    // Broadcast winner to all lanes
    float max_val = __shfl_sync(FULL_WARP_MASK, p.val, 0);
    int   max_lane = __shfl_sync(FULL_WARP_MASK, p.lane, 0);
    return WarpArgMax{max_val, max_lane};
}

// Device utility: argmin across warp (value, lane_id).
// Ties are resolved in favor of smaller lane_id.
struct WarpArgMin {
    float val;
    int lane;
};

__device__ __forceinline__ WarpArgMin warp_argmin(float v) {
    WarpArgMin p{v, lane_id()};
    // Always use full-warp mask, set v = +INF for inactive lanes (if any) beforehand.
    #pragma unroll
    for (int offset = KNN_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float v2 = __shfl_down_sync(FULL_WARP_MASK, p.val, offset);
        int   l2 = __shfl_down_sync(FULL_WARP_MASK, p.lane, offset);
        if (v2 < p.val || (v2 == p.val && l2 < p.lane)) {
            p.val = v2;
            p.lane = l2;
        }
    }
    // Broadcast winner to all lanes
    float min_val = __shfl_sync(FULL_WARP_MASK, p.val, 0);
    int   min_lane = __shfl_sync(FULL_WARP_MASK, p.lane, 0);
    return WarpArgMin{min_val, min_lane};
}

// Replace the current worst (slot 0) in a descending-ordered array of length S
// with a new (better/smaller) value and sift it down to restore descending order.
// Descending means: d[0] >= d[1] >= ... >= d[S-1]
// After insertion (new value is smaller), it bubbles right until order holds.
__device__ __forceinline__ void replace_worst_and_sift(float d[], int idx[], int S, float new_d, int new_i) {
    d[0] = new_d;
    idx[0] = new_i;
    // Bubble down until d[j] >= d[j+1] holds again
    #pragma unroll
    for (int j = 0; j < KNN_WARP_SIZE; ++j) { // upper bound unrolled; loop exits early when S reached
        if (j + 1 >= S) break;
        if (d[j] < d[j + 1]) {
            float td = d[j + 1];
            d[j + 1] = d[j];
            d[j] = td;
            int ti = idx[j + 1];
            idx[j + 1] = idx[j];
            idx[j] = ti;
        } else {
            break;
        }
    }
}

// Kernel implementing k-NN for 2D points, squared L2 distances.
// Each warp processes one query; block-level shared memory caches tiles of data points.
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k)
{
    extern __shared__ float2 sh_data[]; // Shared memory tile for data points

    const int lane = lane_id();
    const int warp_in_block = threadIdx.x >> 5; // threadIdx.x / 32
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global = blockIdx.x * warps_per_block + warp_in_block;

    const bool warp_active = (warp_global < query_count);

    // Per-warp query coordinates (broadcast from lane 0).
    float qx = 0.0f, qy = 0.0f;
    if (warp_active && lane == 0) {
        float2 q = query[warp_global];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_WARP_MASK, qx, 0);
    qy = __shfl_sync(FULL_WARP_MASK, qy, 0);

    // Per-lane top-k storage (distributed across the warp).
    // S = k / 32; since k is guaranteed to be power of two between 32 and 1024,
    // S is an integer in [1, 32].
    const int S = k >> 5; // k / 32
    float best_d[1024 / KNN_WARP_SIZE]; // up to 32
    int   best_i[1024 / KNN_WARP_SIZE];

    // Initialize per-lane arrays as descending with +INF (worst) to the left.
    // After first replacements, values bubble toward the right (smaller indices carry larger values).
    #pragma unroll
    for (int t = 0; t < 1024 / KNN_WARP_SIZE; ++t) {
        if (t < S) {
            best_d[t] = CUDART_INF_F;
            best_i[t] = -1;
        }
    }

    // Iterate through data in tiles loaded into shared memory by the entire block.
    for (int tile_base = 0; tile_base < data_count; tile_base += KNN_TILE_POINTS) {
        const int tile_count = min(KNN_TILE_POINTS, data_count - tile_base);

        // Block-wide cooperative load into shared memory
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            sh_data[i] = data[tile_base + i];
        }

        __syncthreads(); // Ensure tile loaded

        if (warp_active) {
            // Process the tile in micro-batches of 32 candidates (one per lane).
            for (int base = 0; base < tile_count; base += KNN_WARP_SIZE) {
                const int idx_in_tile = base + lane;
                const bool has_cand = (idx_in_tile < tile_count);

                // Compute candidate distance for this lane (or set to +INF if no candidate in this micro-batch)
                float cand_dist = CUDART_INF_F;
                int   cand_idx  = -1;
                if (has_cand) {
                    float2 p = sh_data[idx_in_tile];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    // Squared L2 distance using FMA
                    cand_dist = fmaf(dx, dx, dy * dy);
                    cand_idx = tile_base + idx_in_tile;
                }

                // Cooperative insertion: attempt to insert as many candidates (<=32) into the distributed top-k.
                // At each iteration:
                // - Find the current global worst among all lanes' worsts (best_d[0])
                // - Among the 32 candidates, find the best (minimum) whose distance is smaller than the global worst
                // - Replace the global worst with that candidate (in its owner lane), then mark the candidate consumed
                // Repeat until no candidates improve the set.
                #pragma unroll
                for (int iter = 0; iter < KNN_WARP_SIZE; ++iter) {
                    // Find current global worst in the distributed top-k across lanes (each lane contributes best_d[0])
                    float my_worst = best_d[0];
                    // All lanes participate; my_worst is meaningful for all lanes
                    WarpArgMax worst = warp_argmax(my_worst);
                    float worst_val = worst.val;

                    // Compute the best candidate among lanes that can improve (cand_dist < worst_val)
                    float cand_for_min = (cand_dist < worst_val) ? cand_dist : CUDART_INF_F;
                    WarpArgMin best_cand = warp_argmin(cand_for_min);

                    // If no candidate improves (min >= worst), we are done with this micro-batch
                    if (!(best_cand.val < worst_val)) {
                        break;
                    }

                    // Broadcast the winning candidate's index to all lanes
                    int winning_cand_idx = __shfl_sync(FULL_WARP_MASK, cand_idx, best_cand.lane);

                    // Lane owning the global worst performs the replacement and sifts locally
                    if (lane == worst.lane) {
                        replace_worst_and_sift(best_d, best_i, S, best_cand.val, winning_cand_idx);
                    }

                    // Mark the winning candidate as consumed so it won't be selected again
                    if (lane == best_cand.lane) {
                        cand_dist = CUDART_INF_F;
                        // cand_idx can remain unchanged; distance sentinel prevents re-selection
                    }

                    // Implicit warp-synchronous progress; no explicit sync needed within warp
                }
            }
        }

        __syncthreads(); // Ensure all warps finished using this tile before loading the next one
    }

    // Emit per-query results in ascending order: result[q * k + j] is the j-th nearest neighbor.
    if (warp_active) {
        // For each lane, we will pop from the right (smallest) of its descending array.
        int ptr = S - 1; // points to current smallest in this lane (best)
        float head_dist = (ptr >= 0) ? best_d[ptr] : CUDART_INF_F;
        int   head_idx  = (ptr >= 0) ? best_i[ptr] : -1;

        // Base offset in the global result array for this query
        const int out_base = warp_global * k;

        // Repeatedly select the current best among lanes and write to output
        for (int j = 0; j < k; ++j) {
            // Find the global minimum among head_dist of all lanes
            WarpArgMin m = warp_argmin(head_dist);

            // Broadcast the winning index and distance
            int   w_idx = __shfl_sync(FULL_WARP_MASK, head_idx, m.lane);
            float w_dst = m.val;

            // Write only from lane 0 (or any one lane); to avoid inter-thread race, let the owner of "minimum" write.
            if (lane == m.lane) {
                // Advance this lane's pointer for next round
                --ptr;
                head_dist = (ptr >= 0) ? best_d[ptr] : CUDART_INF_F;
                head_idx  = (ptr >= 0) ? best_i[ptr] : -1;
            }

            // Broadcast final values to a single chosen lane to perform the store; use lane 0 for uniformity
            int   out_idx = __shfl_sync(FULL_WARP_MASK, w_idx, m.lane);
            float out_dst = __shfl_sync(FULL_WARP_MASK, w_dst, m.lane);

            if (lane == 0) {
                result[out_base + j].first  = out_idx;
                result[out_base + j].second = out_dst;
            }
        }
    }
}

// Host function: launch the kernel.
// Assumes:
// - query, data, and result are device pointers allocated by cudaMalloc.
// - data_count >= k, and k is a power of two between 32 and 1024 inclusive.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Derived launch configuration
    const dim3 block(KNN_BLOCK_THREADS);
    const int warps_per_block = KNN_BLOCK_THREADS / KNN_WARP_SIZE;
    const int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Shared memory size for one tile
    const size_t shmem_size = static_cast<size_t>(KNN_TILE_POINTS) * sizeof(float2);

    // Launch kernel
    knn_kernel<<<num_blocks, block, shmem_size>>>(query, query_count, data, data_count, result, k);

    // The caller is responsible for error checking and synchronization if desired.
}