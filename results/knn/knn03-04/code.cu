#include <cuda_runtime.h>
#include <utility>
#include <float.h>   // for FLT_MAX

// --------------------------------------------------------------------------
// This CUDA implementation computes the k-nearest neighbors (k-NN)
// for 2D points using squared Euclidean distance.
// Each warp (32 threads) processes one query point.
// Each thread in the warp holds a private candidate list (of size k/32)
// and processes a portion of each tile loaded into shared memory.
// After processing all data tiles, lane 0 in each warp collects and sorts
// the merged candidate list and writes the final k results to global memory.
//
// The design parameters are chosen for modern data–center GPUs (e.g. A100/H100).
// Block size is 256 threads (8 warps per block). The data points are processed
// iteratively in tiles loaded into shared memory (tile size defined by TILE_SIZE).
// --------------------------------------------------------------------------

#define TILE_SIZE 1024        // Number of data points loaded per tile into shared memory.
#define WARP_SIZE 32          // Warp size (hard-coded to 32).
#define MAX_CAND_PER_THREAD 32  // Maximum candidates per thread; note: k must be in {32,64,...,1024} so k/32 <= 32.

__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp processes one query.
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    
    // Shared memory tile for data points.
    __shared__ float2 s_data[TILE_SIZE];

    // Determine number of candidate entries that each thread will hold.
    int cand_per_thread = k / WARP_SIZE;  // k is guaranteed to be a power-of-two within [32,1024]
    
    // Each thread in the warp maintains its private candidate list.
    // We allocate arrays of fixed maximum size (MAX_CAND_PER_THREAD) for simplicity.
    int localIndices[MAX_CAND_PER_THREAD];
    float localDists[MAX_CAND_PER_THREAD];
    for (int i = 0; i < cand_per_thread; i++) {
        localDists[i] = FLT_MAX;
        localIndices[i] = -1;
    }
    
    // Only active warps (global_warp_id < query_count) process a query.
    bool valid_query = (global_warp_id < query_count);
    float2 queryPoint;
    if (valid_query) {
        queryPoint = query[global_warp_id];  // All lanes in this warp use the same query point.
    }
    
    // Process all data points by iterating over tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (tile_start + TILE_SIZE > data_count)
            tile_size = data_count - tile_start;
        
        // Load the current tile from global memory into shared memory.
        // Each block thread cooperatively loads one or more data points.
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
            s_data[idx] = data[tile_start + idx];
        }
        __syncthreads();  // Ensure the tile is fully loaded.
        
        // If this warp processes a valid query, each lane processes a subset of the tile.
        if (valid_query) {
            for (int j = lane; j < tile_size; j += WARP_SIZE) {
                float2 d = s_data[j];
                float dx = queryPoint.x - d.x;
                float dy = queryPoint.y - d.y;
                float dist = dx * dx + dy * dy;
                int data_index = tile_start + j;
                // Update the local candidate list:
                // Find the candidate with the worst (largest) distance.
                float worst = localDists[0];
                int worst_idx = 0;
                for (int i = 1; i < cand_per_thread; i++) {
                    if (localDists[i] > worst) {
                        worst = localDists[i];
                        worst_idx = i;
                    }
                }
                // If the computed distance is lower than the worst candidate, update it.
                if (dist < worst) {
                    localDists[worst_idx] = dist;
                    localIndices[worst_idx] = data_index;
                }
            }
        }
        __syncthreads();  // Ensure all warps finish processing the tile before loading the next.
    }
    
    // At this point, each lane in the warp holds its private candidate list.
    // Now merge the candidate lists from all 32 lanes in the warp.
    if (valid_query) {
        // Only lane 0 in each warp will perform the merge.
        if (lane == 0) {
            // Allocate final candidate arrays (size k) in local memory.
            int finalIndices[1024];  // k is at most 1024.
            float finalDists[1024];
            int pos = 0;
            unsigned full_mask = 0xffffffff;  // full warp mask
            // Gather candidates from each lane.
            for (int src = 0; src < WARP_SIZE; src++) {
                for (int i = 0; i < cand_per_thread; i++) {
                    int cand_idx = __shfl_sync(full_mask, localIndices[i], src);
                    float cand_dist = __shfl_sync(full_mask, localDists[i], src);
                    finalIndices[pos] = cand_idx;
                    finalDists[pos] = cand_dist;
                    pos++;
                }
            }
            // At this point, pos == k.
            // Sort the final candidate list by ascending distance.
            // A simple insertion sort is used (k is relatively small: <=1024).
            for (int i = 1; i < k; i++) {
                float key_dist = finalDists[i];
                int key_index = finalIndices[i];
                int j = i - 1;
                while (j >= 0 && finalDists[j] > key_dist) {
                    finalDists[j + 1] = finalDists[j];
                    finalIndices[j + 1] = finalIndices[j];
                    j--;
                }
                finalDists[j + 1] = key_dist;
                finalIndices[j + 1] = key_index;
            }
            // Write the sorted k nearest neighbors into the global result array.
            int out_base = global_warp_id * k;
            for (int i = 0; i < k; i++) {
                result[out_base + i] = std::pair<int, float>(finalIndices[i], finalDists[i]);
            }
        }
    }
}

// Host function that launches the k-NN kernel.
// query           : pointer to query points (float2)
// query_count     : number of query points
// data            : pointer to data points (float2)
// data_count      : number of data points (>= k)
// result          : pointer to output array where each query's k neighbors are stored in row-major order
// k               : number of nearest neighbors to compute (power-of-two, between 32 and 1024)
// --------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose a block size of 256 threads (8 warps per block).
    int threadsPerBlock = 256;
    int warps_per_block = threadsPerBlock / WARP_SIZE;  // should be 8
    // Each warp computes one query.
    // Total number of warps needed equals query_count.
    // Calculate grid dimension accordingly.
    int numWarpsNeeded = query_count;
    int numBlocks = (numWarpsNeeded + warps_per_block - 1) / warps_per_block;
    
    // Shared memory size per block.
    size_t sharedMemSize = TILE_SIZE * sizeof(float2);
    
    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}