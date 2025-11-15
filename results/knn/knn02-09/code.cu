#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// -------------------------------------------------------------------------------------------------
// Optimized CUDA k-NN for 2D points
//
// Each query is processed by a single warp (32 threads). The warp cooperatively maintains a
// per-query top-k neighbor list distributed across threads. Distances are computed in tiles of
// the data set cached in shared memory.
//
// - Data dimensionality: 2D (float2).
// - Distance: squared Euclidean (no sqrt).
// - k is a power of two, 32 <= k <= 1024, and data_count >= k.
// - query, data, and result pointers are device pointers (allocated via cudaMalloc).
// - One warp handles one query. Warps inside a block share a tile of data in shared memory.
// - No additional device memory (cudaMalloc) is used; only static shared memory and registers.
// -------------------------------------------------------------------------------------------------

// Configuration constants (tuned for modern datacenter GPUs like A100/H100)
constexpr int WARP_SIZE          = 32;
constexpr int MAX_K              = 1024;  // maximum supported k
constexpr int MAX_K_PER_LANE     = MAX_K / WARP_SIZE; // 1024 / 32 = 32
constexpr int WARPS_PER_BLOCK    = 4;    // 4 warps per block => 128 threads per block
constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int TILE_SIZE          = 1024; // number of data points cached per tile in shared memory

// Convenience typedef for the result type
using PairIF = std::pair<int, float>;

// -------------------------------------------------------------------------------------------------
// Device kernel
// -------------------------------------------------------------------------------------------------
__global__ void knn_kernel_2d(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    PairIF* __restrict__ result,
    int k
)
{
    // Shared memory:
    //  - sh_data:    tile of data points shared by all warps in the block.
    //  - sh_sort_*:  per-warp buffers used only at the end to sort the k best neighbors.
    __shared__ float2 sh_data[TILE_SIZE];
    __shared__ float  sh_sort_dist[WARPS_PER_BLOCK][MAX_K];
    __shared__ int    sh_sort_idx [WARPS_PER_BLOCK][MAX_K];

    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Warp and thread identification
    const int lane_id          = threadIdx.x & (WARP_SIZE - 1);   // thread index within warp [0,31]
    const int warp_id_in_block = threadIdx.x >> 5;                // warp index within block
    const int warps_per_block  = blockDim.x >> 5;                 // should equal WARPS_PER_BLOCK
    const int global_warp_id   = blockIdx.x * warps_per_block + warp_id_in_block;

    // Determine if this warp has an assigned query
    const bool warp_active = (global_warp_id < query_count);

    // Each warp processes one query (index == global_warp_id)
    const int query_idx = global_warp_id;

    // k values per lane (we rely on k being a multiple of WARP_SIZE)
    const int k_per_lane = k / WARP_SIZE; // 1 <= k_per_lane <= 32 for k in [32,1024]

    // Per-lane top-k buffer (distributed representation of the warp's top-k set):
    // Each lane maintains k_per_lane elements; collectively the warp maintains k entries.
    float best_dist[MAX_K_PER_LANE];
    int   best_idx [MAX_K_PER_LANE];

    float local_worst_dist = 0.0f; // largest distance among this lane's local top list
    int   local_worst_pos  = 0;    // position of local_worst_dist in best_dist[]

    // Initialize per-lane top-k buffers for active warps
    if (warp_active) {
        // Initialize all used entries to +inf distance and invalid index
        for (int i = 0; i < k_per_lane; ++i) {
            best_dist[i] = FLT_MAX;
            best_idx[i]  = -1;
        }
        // Initially, the local worst (maximum) distance is +inf
        local_worst_dist = FLT_MAX;
        local_worst_pos  = 0;
    }

    // Load the query point for this warp and broadcast it to all lanes using shuffles
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active) {
        if (lane_id == 0) {
            float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);
    }

    // Process the data set in tiles cached in shared memory
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_SIZE) tile_size = TILE_SIZE;

        // Block-cooperative loading of data tile into shared memory
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            sh_data[i] = data[tile_start + i];
        }

        // Synchronize to make sure the tile is fully loaded before any warp uses it
        __syncthreads();

        if (warp_active) {
            // Each thread in the warp processes a subset of points in the tile
            for (int j = lane_id; j < tile_size; j += WARP_SIZE) {
                float2 p = sh_data[j];

                // Squared Euclidean distance in 2D: (qx - px)^2 + (qy - py)^2
                float dx   = qx - p.x;
                float dy   = qy - p.y;
                float dist = dx * dx + dy * dy;
                int   idx  = tile_start + j;

                // Compute the current global worst distance among all k candidates in the warp.
                // Each lane has its local_worst_dist, and we perform a warp-wide max reduction.
                float global_worst = local_worst_dist;
                for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
                    float other = __shfl_xor_sync(FULL_MASK, global_worst, offset);
                    global_worst = fmaxf(global_worst, other);
                }

                // If the new distance is not better than the current global worst, skip it
                if (dist >= global_worst)
                    continue;

                // Identify the lane that currently holds the global worst distance.
                // Multiple lanes may have the same worst value (especially during initialization);
                // we pick the first lane (lowest ID) among them as the leader.
                unsigned worst_mask = __ballot_sync(FULL_MASK, local_worst_dist == global_worst);
                int leader_lane = __ffs(worst_mask) - 1; // __ffs returns 1-based index

                // Only the leader lane updates its local top-k list by replacing its worst entry
                if (lane_id == leader_lane) {
                    // Insert the new (better) neighbor, replacing the local worst
                    best_dist[local_worst_pos] = dist;
                    best_idx[local_worst_pos]  = idx;

                    // Recompute the local worst (maximum) distance and its position
                    float lw = best_dist[0];
                    int   lp = 0;
                    for (int t = 1; t < k_per_lane; ++t) {
                        if (best_dist[t] > lw) {
                            lw = best_dist[t];
                            lp = t;
                        }
                    }
                    local_worst_dist = lw;
                    local_worst_pos  = lp;
                }
                // All other lanes leave their local top-k lists unchanged.
            }
        }

        // Synchronize before loading the next tile into shared memory
        __syncthreads();
    }

    // At this point, for each active warp:
    // - The distributed buffers best_dist/best_idx across all 32 lanes together hold k candidates
    //   that approximate the k nearest neighbors for this query. We now gather these candidates
    //   into a contiguous per-warp buffer in shared memory and sort them by distance.

    if (warp_active) {
        float* warp_sort_dist = sh_sort_dist[warp_id_in_block];
        int*   warp_sort_idx  = sh_sort_idx [warp_id_in_block];

        // Gather each lane's local candidates into the per-warp shared buffers contiguously
        for (int i = 0; i < k_per_lane; ++i) {
            int pos = lane_id * k_per_lane + i; // global position [0, k-1] inside this warp
            warp_sort_dist[pos] = best_dist[i];
            warp_sort_idx [pos] = best_idx[i];
        }

        // Ensure that all lanes have written their portion before sorting
        __syncwarp();

        // Use a simple insertion sort in lane 0 to sort the k candidates by ascending distance.
        // k <= 1024, so O(k^2) is acceptable compared to the cost of scanning millions of points.
        if (lane_id == 0) {
            for (int i = 1; i < k; ++i) {
                float key_dist = warp_sort_dist[i];
                int   key_idx  = warp_sort_idx[i];
                int j = i - 1;
                // Move elements greater than key_dist one position ahead
                while (j >= 0 && warp_sort_dist[j] > key_dist) {
                    warp_sort_dist[j + 1] = warp_sort_dist[j];
                    warp_sort_idx [j + 1] = warp_sort_idx [j];
                    --j;
                }
                warp_sort_dist[j + 1] = key_dist;
                warp_sort_idx [j + 1] = key_idx;
            }
        }

        // Synchronize the warp to make sure sorting is complete
        __syncwarp();

        // Write the k nearest neighbors to the result array in parallel across the warp.
        // result is laid out as result[query_idx * k + neighbor_rank].
        int base = query_idx * k;
        for (int i = lane_id; i < k; i += WARP_SIZE) {
            result[base + i].first  = warp_sort_idx [i];
            result[base + i].second = warp_sort_dist[i];
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Host interface
// -------------------------------------------------------------------------------------------------
void run_knn(
    const float2* query,
    int           query_count,
    const float2* data,
    int           data_count,
    PairIF*       result,
    int           k
)
{
    // Assumes:
    //  - query, data, result are device pointers allocated by cudaMalloc.
    //  - k is a power of two, 32 <= k <= 1024.
    //  - data_count >= k.
    // No explicit parameter validation is performed here to avoid overhead.

    // Grid configuration: one warp per query, WARPS_PER_BLOCK warps per block
    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = THREADS_PER_BLOCK;
    const int num_blocks =
        (query_count + warps_per_block - 1) / warps_per_block;

    knn_kernel_2d<<<num_blocks, threads_per_block>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k
    );

    // Optional: synchronize or error-check can be done by the caller if desired.
    // cudaDeviceSynchronize();
}