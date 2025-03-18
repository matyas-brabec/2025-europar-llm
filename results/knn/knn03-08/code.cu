#include <cuda_runtime.h>
#include <cfloat>        // for FLT_MAX
#include <utility>       // for std::pair

// This kernel implements k-NN for 2D points using squared Euclidean distance.
// Each warp (32 threads) is assigned one query point and jointly computes its k nearest neighbors.
// Each thread in the warp maintains a private candidate list of size (k / 32) in registers.
// The kernel processes the huge “data” array in batches (tiles), each tile being loaded into shared memory.
// After processing all tiles, lane 0 of each warp gathers the candidate lists from all lanes,
// then sorts them in ascending order (nearest first) and writes out the results.
/// @FIXED
/// extern "C" __global__
__global__
void knn_kernel(const float2 *query, int query_count,
                const float2 *data, int data_count,
                std::pair<int, float> *result, int k)
{
    // In our design, each warp processes one query.
    // The following variables compute the warp’s global id and its lane index.
    const unsigned warpSize = 32;
    int warp_id_in_block = threadIdx.x / warpSize; // warp index within the block
    int lane = threadIdx.x % warpSize;             // lane index within the warp
    int warpsPerBlock = blockDim.x / warpSize;
    int global_warp_id = blockIdx.x * warpsPerBlock + warp_id_in_block;
    if (global_warp_id >= query_count)
        return; // no query assigned to this warp

    // Load the query point for this warp.
    // Let lane 0 load the query from global memory and then broadcast to all lanes using __shfl_sync.
    float2 q;
    if (lane == 0)
        q = query[global_warp_id];
    q.x = __shfl_sync(0xffffffff, q.x, 0);
    q.y = __shfl_sync(0xffffffff, q.y, 0);

    // Each warp must compute k nearest neighbors.
    // We partition the candidate list among the 32 lanes.
    // Each lane will hold local_k = k/32 candidate pairs.
    int local_k = k / warpSize; // since k is a power of 2 and at least 32, this divides evenly.
    // Maximum allowed local_k is 1024/32 = 32.
    float local_dists[32];    // private candidate distances for this lane
    int   local_indices[32];  // corresponding candidate indices
    int local_count = 0;      // number of valid candidates currently stored

    // Initialize candidate list slots to FLT_MAX and invalid index.
#pragma unroll
    for (int i = 0; i < 32; i++) {
        if (i < local_k) {
            local_dists[i] = FLT_MAX;
            local_indices[i] = -1;
        }
    }

    // Process the dataset iteratively in batches.
    // We use a tile size of 1024 data points per batch,
    // which are loaded into shared memory by all threads in the block.
    const int TILE_SIZE = 1024;
    extern __shared__ float2 sh_data[]; // dynamic shared memory for one data tile

    // Loop over data tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE)
    {
        // Each thread in the block cooperatively loads elements into shared memory.
        // A loop is used in case blockDim.x is less than TILE_SIZE.
        int idx = threadIdx.x;
        for (int i = idx; i < TILE_SIZE; i += blockDim.x) {
            int global_index = tile_start + i;
            if (global_index < data_count) {
                sh_data[i] = data[global_index];
            } else {
                // Out-of-bound threads load a dummy value.
                sh_data[i].x = 0.0f;
                sh_data[i].y = 0.0f;
            }
        }
        __syncthreads(); // ensure the entire tile is loaded

        // Each warp processes the tile.
        // The 32 lanes cooperate by each processing data points with a stride equal to warpSize.
        for (int i = lane; i < TILE_SIZE; i += warpSize) {
            int global_index = tile_start + i;
            if (global_index < data_count) {
                // Access the data point from shared memory.
                float2 d = sh_data[i];
                // Compute squared Euclidean distance.
                float dx = q.x - d.x;
                float dy = q.y - d.y;
                float dist = dx * dx + dy * dy;

                // Each lane inserts this candidate into its private list if it qualifies.
                // We maintain the list sorted in descending order,
                // so that the worst (largest) distance is in index 0.
                if (local_count < local_k) {
                    // If the candidate list is not yet full, simply add the candidate.
                    int pos = local_count;
                    local_dists[pos] = dist;
                    local_indices[pos] = global_index;
                    local_count++;
                    // Insertion sort: bubble the new candidate toward the end
                    // if it is smaller than already inserted candidates.
                    int j = pos;
                    while (j > 0 && local_dists[j] < local_dists[j - 1]) {
                        // Swap to maintain descending order: larger (worse) first.
                        float tmp = local_dists[j];
                        local_dists[j] = local_dists[j - 1];
                        local_dists[j - 1] = tmp;
                        int tmpi = local_indices[j];
                        local_indices[j] = local_indices[j - 1];
                        local_indices[j - 1] = tmpi;
                        j--;
                    }
                } else if (dist < local_dists[0]) {
                    // The candidate list is full.
                    // Check if this candidate is better than the current worst candidate.
                    // If yes, replace the worst candidate and re-insert to maintain order.
                    local_dists[0] = dist;
                    local_indices[0] = global_index;
                    int j = 0;
                    // Bubble the replaced candidate down to the correct position.
                    while (j < local_k - 1 && local_dists[j] < local_dists[j + 1]) {
                        float tmp = local_dists[j];
                        local_dists[j] = local_dists[j + 1];
                        local_dists[j + 1] = tmp;
                        int tmpi = local_indices[j];
                        local_indices[j] = local_indices[j + 1];
                        local_indices[j + 1] = tmpi;
                        j++;
                    }
                }
            }
        }
        // Synchronize before loading the next tile.
        __syncthreads();
    } // end for tile

    // If a lane's candidate list is not fully occupied, fill remaining slots with FLT_MAX.
    for (int i = local_count; i < local_k; i++) {
        local_dists[i] = FLT_MAX;
        local_indices[i] = -1;
    }

    // Merge the candidate lists from all lanes in the warp.
    // The total number of candidates from the warp is: total = local_k * warpSize = k.
    // We use warp shuffles so that a single thread (lane 0) can gather candidates from all lanes.
    if (lane == 0) {
        const int total = local_k * warpSize;
        int final_idx[1024];    // max k is 1024
        float final_dist[1024];
        int pos = 0;
        // Loop over each lane in the warp and each candidate index in the lane.
        for (int src = 0; src < warpSize; src++) {
#pragma unroll
            for (int i = 0; i < 32; i++) {
                if (i < local_k) {
                    // Use __shfl_sync to read the candidate from lane 'src'
                    float d_val = __shfl_sync(0xffffffff, local_dists[i], src);
                    int d_idx   = __shfl_sync(0xffffffff, local_indices[i], src);
                    final_dist[pos] = d_val;
                    final_idx[pos] = d_idx;
                    pos++;
                }
            }
        }
        // Sort the gathered candidate array in ascending order of distance
        // (nearest first). We use a simple insertion sort.
        for (int i = 1; i < total; i++) {
            float key_val = final_dist[i];
            int key_idx = final_idx[i];
            int j = i - 1;
            while (j >= 0 && final_dist[j] > key_val) {
                final_dist[j + 1] = final_dist[j];
                final_idx[j + 1] = final_idx[j];
                j--;
            }
            final_dist[j + 1] = key_val;
            final_idx[j + 1] = key_idx;
        }
        // Write the sorted k nearest neighbors for the query to global memory.
        // For query[global_warp_id], result[global_warp_id * k + j] corresponds to the j-th nearest neighbor.
        int base = global_warp_id * k;
        for (int i = 0; i < total; i++) {
            result[base + i] = std::make_pair(final_idx[i], final_dist[i]);
        }
    }
}

// Host interface function for k-NN.
// query: pointer to query points on device (float2 array of length query_count).
// data: pointer to data points on device (float2 array of length data_count).
// result: pointer to result pairs (index, distance) on device (array of length query_count * k).
// k: number of nearest neighbors to compute per query (a power of two between 32 and 1024).
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                          const float2 *data, int data_count,
                          std::pair<int, float> *result, int k)
{
    // Choose block size with a multiple of warp size.
    // For example, we choose 128 threads per block so that each block processes 128/32 = 4 queries.
    const int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query, so number of warps needed equals query_count.
    int numWarps = query_count;
    int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Shared memory per block: TILE_SIZE * sizeof(float2) where TILE_SIZE is 1024.
    int sharedMemSize = 1024 * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize(); // synchronize to ensure kernel completion (error checking can be added)
}
