#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>

// -----------------------------------------------------------------------------
// This CUDA code implements an optimized k-nearest neighbors (k-NN) algorithm
// for 2D points in Euclidean space. For each query point, the k closest data
// points are found using squared Euclidean distance. Each query is processed
// by a single warp (32 threads). Each thread in the warp holds a private
// candidate list of size (k/32) elements in registers. After iterating over the
// entire data set (loaded in shared memory in tiles), the 32 sorted candidate
// lists are merged by lane 0 of the warp using dynamic shared memory.
// The final merged list (of size k) is then written to global memory.
//
// The dynamic shared memory per block is allocated for merging the candidate
// arrays. Its layout is as follows (per block):
//   - A float array for distances: (warpsPerBlock * k) floats.
//   - An int   array for data indices: (warpsPerBlock * k) ints.
// The tile of data points is stored in a statically allocated shared memory
// array of size TILE_SIZE.
// -----------------------------------------------------------------------------


// Define the tile size used for caching portions of the data array in shared memory.
#define TILE_SIZE 1024

// CUDA kernel that computes the k-nearest neighbors for 2D points.
// Each warp processes one query point, and within the warp each of the 32 threads
// maintains (k/32) candidate nearest neighbors (distance and index).
// The entire data set is processed in tiles that are loaded into shared memory.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result,
                           int k)
{
    // Each warp (32 threads) will process one query.
    int warp_id_in_block = threadIdx.x / 32;  // warp id within the block
    int lane = threadIdx.x % 32;               // lane (thread) id within the warp
    int warpsPerBlock = blockDim.x / 32;
    int global_warp_id = blockIdx.x * warpsPerBlock + warp_id_in_block;
    
    // If the current warp does not correspond to a query, exit.
    if (global_warp_id >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[global_warp_id];

    // Each query must output k nearest neighbors.
    // We divide k equally among the 32 threads, so each thread holds:
    int cand_per_thread = k >> 5;  // equivalent to k / 32
    
    // Define a very large value to represent "infinity".
    const float INF = 1e30f;

    // Each thread maintains a private candidate list in registers.
    // Maximum cand_per_thread is 32 when k == 1024.
    int local_idx[32];
    float local_dist[32];
#pragma unroll
    for (int i = 0; i < cand_per_thread; i++) {
        local_dist[i] = INF;
        local_idx[i] = -1;
    }

    // Shared memory used for loading a tile of data points.
    __shared__ float2 s_tile[TILE_SIZE];

    // Process the data points in tiles, looping over the entire data array.
    for (int tile_offset = 0; tile_offset < data_count; tile_offset += TILE_SIZE)
    {
        // Determine the number of elements in this tile.
        int tile_elems = (tile_offset + TILE_SIZE <= data_count) ? TILE_SIZE : (data_count - tile_offset);

        // Cooperative loading: all threads in the block load parts of the tile.
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x)
        {
            s_tile[i] = data[tile_offset + i];
        }
        __syncthreads();  // ensure tile is loaded into shared memory

        // Each warp now processes the tile.
        // Each thread in the warp processes a subset of the tile (stride = 32).
        for (int j = lane; j < tile_elems; j += 32)
        {
            float2 pt = s_tile[j];
            // Compute squared Euclidean distance.
            float dx = pt.x - q.x;
            float dy = pt.y - q.y;
            float d = dx * dx + dy * dy;
            int data_idx = tile_offset + j;

            // Update the local candidate list if d is smaller than the worst candidate.
            // First, find the maximum (worst) candidate in the local list.
            float max_val = local_dist[0];
            int max_pos = 0;
#pragma unroll
            for (int i = 1; i < cand_per_thread; i++)
            {
                if (local_dist[i] > max_val)
                {
                    max_val = local_dist[i];
                    max_pos = i;
                }
            }
            // If the new distance is better than the worst candidate, replace it.
            if (d < max_val) {
                local_dist[max_pos] = d;
                local_idx[max_pos] = data_idx;
            }
        }
        __syncthreads();   // ensure all threads have completed processing this tile
    }

    // At this point, each thread has a private candidate list (of cand_per_thread elements).
    // Sort the candidate list in ascending order (nearest first) using insertion sort.
    for (int i = 1; i < cand_per_thread; i++)
    {
        float key_dist = local_dist[i];
        int key_idx = local_idx[i];
        int j = i - 1;
        while (j >= 0 && local_dist[j] > key_dist)
        {
            local_dist[j + 1] = local_dist[j];
            local_idx[j + 1] = local_idx[j];
            j--;
        }
        local_dist[j + 1] = key_dist;
        local_idx[j + 1] = key_idx;
    }

    // Now merge the 32 sorted candidate lists (one per thread in the warp)
    // into one final sorted list of k nearest neighbors.
    // To enable inter-thread communication, each thread copies its candidate list
    // into dynamically allocated shared memory.
    // Dynamic shared memory layout per block:
    //  [0, warpsPerBlock*k) floats for distances,
    //  [warpsPerBlock*k, 2*warpsPerBlock*k) ints for data indices.
    extern __shared__ char shared_buffer[];
    float *merge_dist = (float *) shared_buffer;
    int *merge_idx = (int *)(merge_dist + warpsPerBlock * k);

    // Each warp writes its candidate list contiguously.
    int warp_base = warp_id_in_block * k;       // starting index in merge arrays for this warp
    int offset = lane * cand_per_thread;          // each thread writes into its own slot
#pragma unroll
    for (int i = 0; i < cand_per_thread; i++) {
        merge_dist[warp_base + offset + i] = local_dist[i];
        merge_idx[warp_base + offset + i] = local_idx[i];
    }
    // Synchronize only the warp since we are merging only among its 32 threads.
    __syncwarp();

    // Let lane 0 of the warp perform a k-way merge.
    if (lane == 0)
    {
        // Final merged candidate list arrays (k elements maximum).
        int final_idx[1024];    // k is at most 1024.
        float final_dist[1024];
        // Pointers to track the current position in each thread's candidate list.
        int pos[32];
#pragma unroll
        for (int i = 0; i < 32; i++)
            pos[i] = 0;

        // Merge the 32 sorted arrays.
        // Each thread contributed cand_per_thread elements, so total candidates = 32*cand_per_thread = k.
        for (int m = 0; m < k; m++)
        {
            float best_val = INF;
            int best_lane = -1;
            // Among the current heads of each candidate list, find the smallest distance.
            for (int r = 0; r < 32; r++)
            {
                if (pos[r] < cand_per_thread)
                {
                    float candidate_val = merge_dist[warp_base + r * cand_per_thread + pos[r]];
                    if (candidate_val < best_val)
                    {
                        best_val = candidate_val;
                        best_lane = r;
                    }
                }
            }
            // Record the best candidate from best_lane.
            final_idx[m] = merge_idx[warp_base + best_lane * cand_per_thread + pos[best_lane]];
            final_dist[m] = best_val;
            pos[best_lane]++;
        }

        // Write the final sorted list for this query to global memory.
        int out_base = global_warp_id * k;
        for (int m = 0; m < k; m++)
        {
            result[out_base + m] = std::pair<int, float>(final_idx[m], final_dist[m]);
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface: run_knn
// Launches the k-NN kernel. It assumes that the input arrays (query, data, result)
// have been allocated on the device (via cudaMalloc) and contain valid data.
// query: pointer to an array of float2, one per query (query_count total)
// data: pointer to an array of float2, one per data point (data_count total)
// result: pointer to an array of std::pair<int,float> with room for query_count*k elements
// k: the number of nearest neighbors (power-of-two, between 32 and 1024)
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose a thread block size. Here, 256 threads per block (i.e. 8 warps) is used.
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query.
    int numQueries = query_count;
    int numBlocks = (numQueries + warpsPerBlock - 1) / warpsPerBlock;
    
    // Compute dynamic shared memory size per block:
    // For each block, we need space for merging candidate lists:
    //   (warpsPerBlock * k) floats for distances and the same number of ints for indices.
    size_t sharedMemSize = warpsPerBlock * k * (sizeof(float) + sizeof(int));

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count,
                                                              data, data_count,
                                                              result, k);

    // (Optional) Synchronize to ensure kernel completion.
    cudaDeviceSynchronize();
}