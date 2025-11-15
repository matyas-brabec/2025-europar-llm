#include <cuda_runtime.h>
#include <utility>

// ------------------------------------------------------------
// Constants and helper types
// ------------------------------------------------------------

static constexpr int KNN_WARP_SIZE    = 32;
static constexpr int KNN_MAX_K        = 1024;  // maximum k supported
static constexpr int KNN_TILE_POINTS  = 2048;  // number of data points cached per block in shared memory

// Device-side representation of std::pair<int,float>.
// Layout must match exactly so we can reinterpret_cast between them.
struct PairIF {
    int   first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>),
              "PairIF must have the same size as std::pair<int,float>");

// ------------------------------------------------------------
// CUDA kernel: each warp processes one query point
// ------------------------------------------------------------

__global__ void knn_kernel_2d(const float2* __restrict__ d_query,
                              int query_count,
                              const float2* __restrict__ d_data,
                              int data_count,
                              PairIF* __restrict__ d_result,
                              int k)
{
    extern __shared__ unsigned char shared_raw[];

    // Shared memory layout:
    // [0 ... KNN_TILE_POINTS-1]           : float2 data tile
    // [KNN_TILE_POINTS ...]               : per-warp best_dist arrays (float)
    // [after best_dist ...]               : per-warp best_idx arrays (int)
    float2* shared_points = reinterpret_cast<float2*>(shared_raw);

    const int warps_per_block = blockDim.x / KNN_WARP_SIZE;

    float* shared_best_dist = reinterpret_cast<float*>(
        shared_points + KNN_TILE_POINTS
    );
    int* shared_best_idx = reinterpret_cast<int*>(
        shared_best_dist + warps_per_block * KNN_MAX_K
    );

    const int lane          = threadIdx.x & (KNN_WARP_SIZE - 1); // lane id in warp
    const int warp_in_block = threadIdx.x / KNN_WARP_SIZE;       // warp id in block
    const int global_warp_id = blockIdx.x * warps_per_block + warp_in_block;

    const bool warp_active = (global_warp_id < query_count);
    const unsigned int full_mask = 0xffffffffu;

    // Each warp gets its own private slice of the top-k buffers
    float* best_dist = shared_best_dist + warp_in_block * KNN_MAX_K;
    int*   best_idx  = shared_best_idx  + warp_in_block * KNN_MAX_K;

    // Per-warp bookkeeping variables (replicated across lanes)
    float2 q;                       // query point
    int    count   = 0;             // number of neighbors currently stored (<= k)
    int    max_pos = 0;             // position of current worst (largest distance) neighbor
    float  max_dist = CUDART_INF_F; // distance of current worst neighbor

    // Initialize warp-specific data for active warps
    if (warp_active) {
        // Load the query point into lane 0, then broadcast to entire warp
        if (lane == 0) {
            q = d_query[global_warp_id];

            // Initialize the top-k buffers with dummy values.
            // Only entries [0, k) will be used.
            for (int i = 0; i < k; ++i) {
                best_dist[i] = CUDART_INF_F;
                best_idx[i]  = -1;
            }

            count    = 0;
            max_pos  = 0;
            max_dist = CUDART_INF_F;
        }

        // Broadcast query and bookkeeping to all lanes in the warp
        q.x      = __shfl_sync(full_mask, q.x, 0);
        q.y      = __shfl_sync(full_mask, q.y, 0);
        count    = __shfl_sync(full_mask, count, 0);
        max_pos  = __shfl_sync(full_mask, max_pos, 0);
        max_dist = __shfl_sync(full_mask, max_dist, 0);
    }

    // --------------------------------------------------------------------
    // Main loop: iterate over the data points in tiles cached in shared mem
    // --------------------------------------------------------------------
    for (int tile_start = 0; tile_start < data_count; tile_start += KNN_TILE_POINTS) {
        int remaining = data_count - tile_start;
        int tile_size = (remaining < KNN_TILE_POINTS) ? remaining : KNN_TILE_POINTS;

        // Cooperative load of data tile into shared memory (all threads)
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
            shared_points[idx] = d_data[tile_start + idx];
        }
        __syncthreads(); // ensure tile is fully loaded

        if (warp_active) {
            // Process this tile: each warp computes distances w.r.t its query point.
            // The tile is processed in batches of 32 points so that each lane
            // handles (at most) one point per inner iteration.
            for (int base = 0; base < tile_size; base += KNN_WARP_SIZE) {
                int local_idx  = base + lane;
                bool is_valid  = (local_idx < tile_size);
                float dist     = 0.0f;
                int   global_idx = 0;

                if (is_valid) {
                    float2 p = shared_points[local_idx];
                    float dx = p.x - q.x;
                    float dy = p.y - q.y;
                    dist      = dx * dx + dy * dy;      // squared Euclidean distance
                    global_idx = tile_start + local_idx;
                }

                // Mask of lanes with valid data points in this batch
                unsigned int valid_mask = __ballot_sync(full_mask, is_valid);
                if (valid_mask == 0u) {
                    continue; // no valid candidates in this batch
                }

                // Snapshot of current global threshold and count, identical on all lanes
                int   cur_count    = __shfl_sync(full_mask, count, 0);
                float cur_max_dist = __shfl_sync(full_mask, max_dist, 0);

                // Pre-filter candidates using the snapshot threshold. This avoids
                // invoking the (sequential) insertion logic when no candidate can
                // possibly enter the current top-k set.
                bool is_candidate = is_valid && (cur_count < k || dist < cur_max_dist);
                unsigned int cand_mask = __ballot_sync(full_mask, is_candidate);
                if (cand_mask == 0u) {
                    continue; // none of the distances in this batch can improve the result
                }

                // Sequentially handle all candidate lanes for this batch.
                // For each candidate, lane 0 performs the insertion into the
                // warp-private top-k buffer. All lanes keep a consistent copy
                // of the bookkeeping variables using warp shuffles.
                while (cand_mask) {
                    // Select the lowest-numbered lane that has a candidate
                    int leader = __ffs(cand_mask) - 1; // lane index in [0,31]

                    // Gather that lane's distance/index to all threads
                    float cand_dist = __shfl_sync(full_mask, dist, leader);
                    int   cand_idx  = __shfl_sync(full_mask, global_idx, leader);

                    if (lane == 0) {
                        if (count < k) {
                            // Still filling initial buffer: always insert
                            best_dist[count] = cand_dist;
                            best_idx[count]  = cand_idx;
                            ++count;

                            // Once we've filled k entries, compute the current worst neighbor
                            if (count == k) {
                                max_pos  = 0;
                                max_dist = best_dist[0];
                                for (int t = 1; t < k; ++t) {
                                    float v = best_dist[t];
                                    if (v > max_dist) {
                                        max_dist = v;
                                        max_pos  = t;
                                    }
                                }
                            }
                        } else if (cand_dist < max_dist) {
                            // Candidate improves over current worst neighbor: replace it
                            best_dist[max_pos] = cand_dist;
                            best_idx[max_pos]  = cand_idx;

                            // Recompute worst neighbor position and distance
                            max_pos  = 0;
                            max_dist = best_dist[0];
                            for (int t = 1; t < k; ++t) {
                                float v = best_dist[t];
                                if (v > max_dist) {
                                    max_dist = v;
                                    max_pos  = t;
                                }
                            }
                        }
                        // Else: candidate does not improve the current top-k, ignore it
                    }

                    // Broadcast updated bookkeeping values from lane 0 to all lanes
                    count    = __shfl_sync(full_mask, count, 0);
                    max_pos  = __shfl_sync(full_mask, max_pos, 0);
                    max_dist = __shfl_sync(full_mask, max_dist, 0);

                    // Clear the processed candidate's bit and continue
                    cand_mask &= (cand_mask - 1u);
                }
            }
        }

        __syncthreads(); // synchronize before loading the next tile
    }

    // --------------------------------------------------------------------
    // Finalization: sort each warp's top-k neighbors and write out results
    // --------------------------------------------------------------------
    if (warp_active && lane == 0) {
        // At this point, best_dist/best_idx hold the k best candidates found,
        // but not necessarily sorted. Sort them by distance (ascending) using
        // insertion sort, which is efficient enough for k up to 1024 and
        // executed only once per query.
        for (int i = 1; i < k; ++i) {
            float key_dist = best_dist[i];
            int   key_idx  = best_idx[i];
            int j = i - 1;
            while (j >= 0 && best_dist[j] > key_dist) {
                best_dist[j + 1] = best_dist[j];
                best_idx[j + 1]  = best_idx[j];
                --j;
            }
            best_dist[j + 1] = key_dist;
            best_idx[j + 1]  = key_idx;
        }

        // Write sorted results to global memory. For query q, its k nearest
        // neighbors reside at result[q * k + j].
        int out_base = global_warp_id * k;
        for (int i = 0; i < k; ++i) {
            d_result[out_base + i].first  = best_idx[i];
            d_result[out_base + i].second = best_dist[i];
        }
    }
}

// ------------------------------------------------------------
// Host interface
// ------------------------------------------------------------

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Guard against invalid inputs; user contract guarantees valid values,
    // but these checks make the function robust.
    if (query_count <= 0 || data_count <= 0 || k <= 0 || k > KNN_MAX_K) {
        return;
    }

    // Configure CUDA kernel launch parameters.
    // Each warp handles one query; choose 256 threads (8 warps) per block to
    // balance occupancy and shared memory usage.
    const int threads_per_block = 256;
    const int warps_per_block   = threads_per_block / KNN_WARP_SIZE;

    // Number of warps needed equals number of queries
    int num_warps = query_count;
    int blocks    = (num_warps + warps_per_block - 1) / warps_per_block;

    // Shared memory size per block:
    //  - KNN_TILE_POINTS float2 for the data tile
    //  - warps_per_block * KNN_MAX_K floats for best_dist
    //  - warps_per_block * KNN_MAX_K ints   for best_idx
    size_t shared_bytes =
        static_cast<size_t>(KNN_TILE_POINTS) * sizeof(float2) +
        static_cast<size_t>(warps_per_block) * KNN_MAX_K * sizeof(float) +
        static_cast<size_t>(warps_per_block) * KNN_MAX_K * sizeof(int);

    // Launch the kernel. result is passed as PairIF* to match the device struct.
    PairIF* d_result_pair = reinterpret_cast<PairIF*>(result);

    knn_kernel_2d<<<blocks, threads_per_block, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        d_result_pair,
        k
    );

    // Ensure completion before returning to the caller.
    cudaDeviceSynchronize();
}