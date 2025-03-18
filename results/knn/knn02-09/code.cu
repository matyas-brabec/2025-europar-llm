#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// -----------------------------------------------------------------------------
// This CUDA code implements an optimized k-nearest neighbors (k-NN)
// algorithm for 2D points. For each query point, a warp (32 threads)
// computes its k nearest neighbors from a large dataset of 2D points.
// Each thread in the warp is responsible for maintaining a private partial
// list (of size k/32) of candidate neighbors. The candidates from all 32 threads
// are later merged and sorted to produce the final output (sorted in
// increasing order of squared Euclidean distance).
//
// The algorithm processes the large 'data' array iteratively in batches,
// which are cached in shared memory to improve global-memory throughput.
// The kernel assumes that k is a power-of-two between 32 and 1024 inclusive,
// and that 'data_count' is at least k.
//
// The host interface to launch the kernel is provided in the function
// "run_knn". It chooses a block configuration (128 threads per block) so that
// a number of warps (each warp handling one query) are launched.
// -----------------------------------------------------------------------------

// Define the tile size for caching data in shared memory.
#define TILE_SIZE 1024

// The CUDA kernel for k-NN computation.
// Each warp processes one query point.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Determine the warp id among all warps in the grid.
    // Each warp (32 threads) processes one query.
    int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane_id = threadIdx.x & 31; // thread index within warp (0-31)

    // If the warp id exceeds the number of queries, exit.
    if (warp_id >= query_count)
        return;

    // Each warp needs the same query point.
    // Use lane 0 to load the query point from global memory and broadcast it.
    float2 query_point;
    if (lane_id == 0) {
        query_point = query[warp_id];
    }
    // Broadcast both components from lane 0 to all lanes in the warp.
    query_point.x = __shfl_sync(0xFFFFFFFF, query_point.x, 0);
    query_point.y = __shfl_sync(0xFFFFFFFF, query_point.y, 0);

    // Each thread will maintain a private candidate list of size (k / 32).
    int candidateCount = k / 32; // guaranteed to be an integer power-of-two (1 to 32)
    // Declare fixed-size arrays (maximum size 32) for candidate distances and indices.
    float local_d[32];
    int local_idx[32];
    // Initialize candidate distances to a very high value and indices to -1.
    for (int i = 0; i < candidateCount; i++) {
        local_d[i] = FLT_MAX;
        local_idx[i] = -1;
    }

    // Declare shared memory for caching a batch (tile) of data points.
    // The tile size is chosen to be TILE_SIZE.
    __shared__ float2 sdata[TILE_SIZE];

    // Process the dataset in tiles to leverage reuse of global memory loads.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        // Determine actual number of data points in this tile.
        int tile_size = TILE_SIZE;
        if (tile_start + TILE_SIZE > data_count)
            tile_size = data_count - tile_start;

        // Load the current tile of data points from global memory into shared memory.
        // Multiple threads in the block participate in the load.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            sdata[i] = data[tile_start + i];
        }
        // Synchronize the block to ensure the tile is fully loaded.
        __syncthreads();

        // Each warp processes the tile: each thread in the warp takes a strided subset.
        for (int j = lane_id; j < tile_size; j += 32) {
            // Load a data point from shared memory.
            float2 dpt = sdata[j];
            // Compute squared Euclidean distance to the query point.
            float dx = dpt.x - query_point.x;
            float dy = dpt.y - query_point.y;
            float dist = dx * dx + dy * dy;
            // Compute the global index of the data point.
            int global_index = tile_start + j;

            // Update the thread's candidate list.
            // Find the candidate with the maximum (worst) distance.
            float max_val = local_d[0];
            int max_pos = 0;
            for (int p = 1; p < candidateCount; p++) {
                float val = local_d[p];
                if (val > max_val) {
                    max_val = val;
                    max_pos = p;
                }
            }
            // If the current distance is smaller than the worst candidate,
            // replace that candidate.
            if (dist < max_val) {
                local_d[max_pos] = dist;
                local_idx[max_pos] = global_index;
            }
        }
        // Ensure all threads finish processing the current tile before loading the next tile.
        __syncthreads();
    } // end for each tile

    // At this point, each thread in the warp has candidateCount items.
    // Together, the warp has k candidates.
    // Now, perform a warp-level merge: let thread 0 of the warp gather all candidate
    // pairs from its 32 lanes and sort them.
    if (lane_id == 0) {
        const int total_candidates = candidateCount * 32; // equals k
        // Temporarily store the merged candidate indices and distances.
        int combined_idx[1024];    // maximum k is 1024
        float combined_dist[1024];

        int pos = 0;
        // Loop over each warp lane and each candidate in that lane.
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < candidateCount; j++) {
                // Use warp shuffle to obtain the candidate value from lane i.
                float d = __shfl_sync(0xFFFFFFFF, local_d[j], i);
                int idx_val = __shfl_sync(0xFFFFFFFF, local_idx[j], i);
                combined_dist[pos] = d;
                combined_idx[pos] = idx_val;
                pos++;
            }
        }

        // Insertion sort the merged candidates in increasing order of distance.
        for (int i = 1; i < total_candidates; i++) {
            float key_d = combined_dist[i];
            int key_idx = combined_idx[i];
            int j = i - 1;
            while (j >= 0 && combined_dist[j] > key_d) {
                combined_dist[j + 1] = combined_dist[j];
                combined_idx[j + 1] = combined_idx[j];
                j--;
            }
            combined_dist[j + 1] = key_d;
            combined_idx[j + 1] = key_idx;
        }

        // Write sorted results to global memory.
        // For query with index 'warp_id', the k neighbors are stored contiguously.
        int base = warp_id * k;
        for (int i = 0; i < total_candidates; i++) {
            result[base + i] = std::make_pair(combined_idx[i], combined_dist[i]);
        }
    }
    // End of kernel.
}

// Host function to launch the k-NN CUDA kernel.
// The input arrays 'query' and 'data' and the output array 'result' are assumed
// to be allocated on the device (using cudaMalloc).
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose 128 threads per block.
    const int threads_per_block = 128;
    // Each block has (threads_per_block / 32) warps.
    const int warps_per_block = threads_per_block / 32;
    // Each warp processes one query. Compute the required number of blocks.
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Allocate shared memory for caching a tile of data points.
    int sharedMemBytes = TILE_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<blocks, threads_per_block, sharedMemBytes>>>(query, query_count,
                                                              data, data_count,
                                                              result, k);
    // Optional: synchronize and check for errors (omitted for brevity).
}