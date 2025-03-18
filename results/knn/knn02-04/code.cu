#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>        // for FLT_MAX
#include <utility>       // for std::pair

// Constant tile size for loading data points into shared memory.
// A tile size of 256 ensures reasonable occupancy and reuse.
#define TILE_SIZE 256

//---------------------------------------------------------------------------
// Optimized CUDA kernel for k-nearest neighbors (k-NN) in 2D Euclidean space.
// Each query is processed by one warp (32 threads), which collaborates
// to compute the squared distances to all data points (processed in tiles)
// and keeps its own private candidate list of k nearest neighbors.
// The candidate list is distributed across the 32 threads of the warp:
// each thread holds k/32 candidate pairs (index and squared-distance) in its registers.
// After processing all data in batches (tiles loaded into shared memory),
// the warp gathers the candidate lists using warp shuffle intrinsics,
// sorts the combined list (of length k) using a bitonic sort (run by lane 0),
// and writes the sorted results (sorted in ascending order of distance)
// into the global results array.
//
// The kernel assumes that k is a power-of-two between 32 and 1024 inclusive.
//---------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count, 
                           const float2 *data, int data_count, 
                           std::pair<int, float> *result, int k)
{
    // Each warp processes one query.
    // Compute warp ID: each warp is 32 threads.
    int warpId_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;  // lane index in warp [0,31]
    int warpsPerBlock = blockDim.x / 32;
    
    // Global warp (query) index:
    int queryId = blockIdx.x * warpsPerBlock + warpId_in_block;
    if (queryId >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[queryId];

    // Each warp will hold a distributed candidate list.
    // Each thread holds a private sub-array of size local_k.
    // Since k is a power-of-two and in [32,1024], we have:
    int local_k = k / 32;  // e.g., if k==1024 then local_k==32, if k==32 then local_k==1.
    // In registers, each thread holds its candidate distances and indices.
    // Initialize all candidate distances to FLT_MAX and index to -1.
    // Maximum candidate distance means "worse" than any real distance.
    float cand_dist[32];  // maximum size needed is 32 (when k==1024)
    int   cand_idx[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (i < local_k) {
            cand_dist[i] = FLT_MAX;
            cand_idx[i]  = -1;
        }
    }

    // Iterate over all data points in batches loaded into shared memory.
    // Declare shared memory tile for data points.
    __shared__ float2 tile[TILE_SIZE];
    
    // Loop over tiles.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE)
    {
        // Compute number of points in current tile.
        int tileSize = (data_count - tileStart < TILE_SIZE) ? (data_count - tileStart) : TILE_SIZE;
        
        // Load tile elements from global memory into shared memory.
        // All threads in the block participate.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x)
        {
            tile[i] = data[tileStart + i];
        }
        __syncthreads();  // Ensure tile is fully loaded before processing.

        // Each warp processes the tile.
        // Distribute the tile elements among the 32 threads: each thread processes 
        // indices j in [lane, tileSize) with a stride of 32.
        for (int j = lane; j < tileSize; j += 32)
        {
            float2 d = tile[j];
            // Compute squared Euclidean distance.
            float dx = q.x - d.x;
            float dy = q.y - d.y;
            float dist = dx * dx + dy * dy;
            
            // Compare with the worst candidate in the thread's local list.
            // Find index of maximum distance in the local candidate array.
            float maxVal = cand_dist[0];
            int   maxPos = 0;
            #pragma unroll
            for (int c = 1; c < 32; c++) {
                if (c < local_k) {
                    if (cand_dist[c] > maxVal) {
                        maxVal = cand_dist[c];
                        maxPos = c;
                    }
                }
            }
            // If the current distance is smaller than the worst so far, update.
            if (dist < maxVal) {
                cand_dist[maxPos] = dist;
                cand_idx[maxPos]  = tileStart + j; // global data index
            }
        }
        __syncthreads();  // Ensure all warps finish processing the tile before next tile load.
    }

    // At this point, each warp has 32 sub-candidate lists (one per thread) of size local_k.
    // Together they represent k candidates (possibly unsorted).
    // We now merge them into one list of k candidates.
    // To do this, we use warp shuffle intrinsics.
    // All threads in the warp participate in the __shfl_sync calls (even though only lane 0 will store the gathered result).

    // Only lane 0 will gather all candidate pairs into a local array for sorting.
    // Declare temporary arrays with maximum size of 1024 elements (k <= 1024).
    int all_idx[1024];
    float all_dist[1024];

    // Gather candidates from all 32 lanes:
    for (int srcLane = 0; srcLane < 32; srcLane++) {
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            if (j < local_k) {
                // All threads call __shfl_sync so that lane 0 can gather the value.
                int cand = __shfl_sync(0xFFFFFFFF, cand_idx[j], srcLane);
                float dCand = __shfl_sync(0xFFFFFFFF, cand_dist[j], srcLane);
                if (lane == 0) {
                    all_idx[srcLane * local_k + j] = cand;
                    all_dist[srcLane * local_k + j] = dCand;
                }
            }
        }
    }

    // Now, lane 0 of the warp holds the full candidate list of size k.
    if (lane == 0) {
        // In-place bitonic sort of the candidate list.
        // k is guaranteed to be a power-of-two.
        int N = k; // total number of candidates
        // Bitonic sort: sort in ascending order of distance.
        for (int size = 2; size <= N; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = 0; i < N; i++) {
                    int ixj = i ^ stride;
                    if (ixj > i) {
                        // Determine sort order; for ascending order, if the bit corresponding to "size" is 0 then compare normally.
                        bool ascending = ((i & size) == 0);
                        if (ascending) {
                            if (all_dist[i] > all_dist[ixj]) {
                                // Swap candidate pairs.
                                float tmp_d = all_dist[i];
                                all_dist[i] = all_dist[ixj];
                                all_dist[ixj] = tmp_d;
                                int tmp_i = all_idx[i];
                                all_idx[i] = all_idx[ixj];
                                all_idx[ixj] = tmp_i;
                            }
                        } else {
                            if (all_dist[i] < all_dist[ixj]) {
                                float tmp_d = all_dist[i];
                                all_dist[i] = all_dist[ixj];
                                all_dist[ixj] = tmp_d;
                                int tmp_i = all_idx[i];
                                all_idx[i] = all_idx[ixj];
                                all_idx[ixj] = tmp_i;
                            }
                        }
                    }
                }
            }
        }

        // After sorting, the candidate list is in ascending order.
        // Write the sorted results for this query into global memory.
        std::pair<int, float>* queryRes = result + queryId * k;
        for (int i = 0; i < N; i++) {
            queryRes[i] = std::pair<int,float>(all_idx[i], all_dist[i]);
        }
    }
}

//---------------------------------------------------------------------------
// Host interface function to launch the k-NN kernel.
//
// The function receives:
//  - query: pointer to query points (each of type float2) of length query_count.
//  - data: pointer to data points (each of type float2) of length data_count.
//  - result: pointer to an array of std::pair<int, float> for output; for each query,
//            k neighbors are stored consecutively in row-major order.
//  - k: number of nearest neighbors to compute per query (a power-of-two between 32 and 1024).
//---------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count, 
             const float2 *data, int data_count, 
             std::pair<int, float> *result, int k)
{
    // Each warp (32 threads) processes one query.
    // Choose block size of 256 threads => 256/32 = 8 warps per block.
    const int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    // Compute required number of blocks.
    int blockCount = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    
    // Launch the kernel.
    knn_kernel<<<blockCount, threadsPerBlock>>>(query, query_count, data, data_count, result, k);
    
    // Optionally, one might call cudaDeviceSynchronize() here.
    cudaDeviceSynchronize();
}