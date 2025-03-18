// This is an optimized CUDA implementation of k‐Nearest Neighbors (k-NN)
// for 2D points using a warp‐per‐query strategy. Each warp (32 threads)
// processes one query point. All data points are processed in batches
// (tiles) that are loaded into shared memory by the thread block. Each warp
// keeps a "private" intermediate result (of k candidates) and a candidate
// buffer (of up to k candidates) in shared memory. When the candidate buffer
// is full, it is merged with the intermediate result using a Bitonic Sort–based
// merge procedure. The final k sorted neighbors are then written to global memory.

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Define TILE_SIZE for data batching and MAX_K as the maximum allowed k.
#define TILE_SIZE 1024
#define MAX_K 1024

// Struct to hold a candidate neighbor: index and squared distance.
struct Candidate {
    int idx;
    float dist;
};

//---------------------------------------------------------------------
// Device function: Serial Bitonic Sort for an array of Candidate (ascending by dist).
// This implementation follows the provided pseudocode.
// It is intended to be run by a single thread (e.g., the warp leader)
// on a buffer of size n (where n is equal to k).
//---------------------------------------------------------------------
__device__ void bitonic_sort_serial(Candidate* arr, int n) {
    // k_val is the size of the bitonic sequence.
    for (int k_val = 2; k_val <= n; k_val *= 2) {
        for (int j = k_val / 2; j > 0; j /= 2) {
            for (int i = 0; i < n; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    // Decide whether to swap based on the bitonic ordering property.
                    if (((i & k_val) == 0 && arr[i].dist > arr[ixj].dist) ||
                        ((i & k_val) != 0 && arr[i].dist < arr[ixj].dist)) {
                        Candidate temp = arr[i];
                        arr[i] = arr[ixj];
                        arr[ixj] = temp;
                    }
                }
            }
        }
    }
}

//---------------------------------------------------------------------
// Device function: Merge the candidate buffer with the intermediate result.
// The procedure is as follows:
//  1. Sort the candidate buffer (of size k) via Bitonic Sort.
//  2. Merge with the intermediate result: for each i in [0,k),
//     select the minimum between buffer[i] and inter[k-i-1].
//  3. Sort the merged k elements to produce an updated intermediate result.
// This function is meant to be called by the warp leader (lane 0)
// and uses a temporary local array allocated in registers.
//---------------------------------------------------------------------
__device__ void merge_buffer(Candidate* buffer, Candidate* inter, int k) {
    // Step 1: Sort the candidate buffer.
    bitonic_sort_serial(buffer, k);

    // Step 2: Merge buffer and intermediate result.
    Candidate merged[MAX_K];  // MAX_K is guaranteed to be at least k.
    for (int i = 0; i < k; i++) {
        // Note: inter is sorted in ascending order; its last element is the current max.
        Candidate candBuf = buffer[i];
        Candidate candInter = inter[k - i - 1];
        merged[i] = (candBuf.dist < candInter.dist) ? candBuf : candInter;
    }

    // Step 3: Sort the merged array to update the intermediate result.
    bitonic_sort_serial(merged, k);
    for (int i = 0; i < k; i++) {
        inter[i] = merged[i];
    }
}

//---------------------------------------------------------------------
// Kernel: knn_kernel
// Each warp (32 threads) processes one query point. Shared memory is
// used to store:
//   - A candidate buffer for each warp (size: (warpCount * k))
//   - A candidate count (per warp)
//   - An intermediate result array for each warp (size: (warpCount * k))
//   - A tile (batch) of data points (size: TILE_SIZE)
//---------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Determine the warp id within the block and global warp id.
    int warpInBlock = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpInBlock;
    if (globalWarpId >= query_count)
        return;  // Out of range.

    // Declare shared memory layout.
    extern __shared__ char smem[];
    // Candidate buffer for new candidates (per warp, size: k)
    Candidate* candBuffer = (Candidate*) smem;
    // Candidate count per warp.
    int* candCount = (int*)(candBuffer + warpsPerBlock * k);
    // Intermediate result buffer per warp (sorted; size: k)
    Candidate* interResult = (Candidate*)(candCount + warpsPerBlock);
    // Tile (batch) of data points (shared among whole block)
    float2* tileBuffer = (float2*)(interResult + warpsPerBlock * k);

    // Initialize candidate count for this warp.
    if (lane == 0)
        candCount[warpInBlock] = 0;
    __syncwarp();

    // Initialize the intermediate result for this warp.
    // Each warp's intermediate result is located at: interResult[warpInBlock * k ... warpInBlock * k + k - 1]
    for (int i = lane; i < k; i += 32) {
        interResult[warpInBlock * k + i].dist = FLT_MAX;
        interResult[warpInBlock * k + i].idx = -1;
    }
    __syncwarp();

    // Load the query point for this warp.
    float2 q;
    if (lane == 0)
        q = query[globalWarpId];
    // Broadcast the query point across the warp.
    q.x = __shfl_sync(0xffffffff, q.x, 0);
    q.y = __shfl_sync(0xffffffff, q.y, 0);

    // Local variable to hold the current maximum distance from the intermediate result.
    // Initially, it is FLT_MAX.
    float local_max = FLT_MAX;

    // Process data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        // Calculate the number of data points to load in this batch.
        int tileSize = (batch_start + TILE_SIZE <= data_count) ? TILE_SIZE : (data_count - batch_start);

        // Load the batch of data points into shared memory.
        // Use the whole block, stride = blockDim.x.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            tileBuffer[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the tile is fully loaded.

        // Each warp processes the tile.
        for (int i = lane; i < tileSize; i += 32) {
            float2 d = tileBuffer[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            // If the point is closer than the current k-th nearest (local_max),
            // add it to the candidate buffer.
            if (dist < local_max) {
                int pos = atomicAdd(&candCount[warpInBlock], 1);
                if (pos < k) {
                    int index = batch_start + i;  // Global data index.
                    candBuffer[warpInBlock * k + pos].dist = dist;
                    candBuffer[warpInBlock * k + pos].idx = index;
                }
            }
        }
        __syncwarp();

        // If the candidate buffer is full (or overfull), merge it with the intermediate result.
        if (candCount[warpInBlock] >= k) {
            if (lane == 0) {
                // Merge the candidate buffer (first k entries) with the intermediate result.
                merge_buffer(&candBuffer[warpInBlock * k],
                             &interResult[warpInBlock * k],
                             k);
                // Update the local maximum distance from the updated intermediate result.
                local_max = interResult[warpInBlock * k + k - 1].dist;
                // Reset the candidate count for this warp.
                candCount[warpInBlock] = 0;
            }
            __syncwarp();
            // Broadcast the updated local_max to all lanes.
            local_max = __shfl_sync(0xffffffff, local_max, 0);
        }
        __syncthreads();  // Ensure the tile is not overwritten before next batch load.
    }

    // After processing all batches, merge any remaining candidates if present.
    if (candCount[warpInBlock] > 0) {
        if (lane == 0) {
            int cnt = candCount[warpInBlock];
            // Fill the remaining slots with dummy candidates.
            for (int i = cnt; i < k; i++) {
                candBuffer[warpInBlock * k + i].dist = FLT_MAX;
                candBuffer[warpInBlock * k + i].idx = -1;
            }
            merge_buffer(&candBuffer[warpInBlock * k],
                         &interResult[warpInBlock * k],
                         k);
            // Reset candidate count.
            candCount[warpInBlock] = 0;
            local_max = interResult[warpInBlock * k + k - 1].dist;
        }
        __syncwarp();
        local_max = __shfl_sync(0xffffffff, local_max, 0);
    }

    // Write the final sorted k nearest neighbors (from the intermediate result)
    // to global memory for this query.
    for (int i = lane; i < k; i += 32) {
        Candidate cand = interResult[warpInBlock * k + i];
        result[globalWarpId * k + i] = std::make_pair(cand.idx, cand.dist);
    }
}

//---------------------------------------------------------------------
// Host function: run_knn
// Launches the knn_kernel with an appropriate grid, block, and shared memory
// configuration. It assumes that query, data, and result have been allocated using cudaMalloc, 
// and that data_count >= k and k is a power of two between 32 and 1024.
//---------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose 128 threads per block (i.e., 4 warps per block).
    const int threadsPerBlock = 128;
    const int warpsPerBlock = threadsPerBlock / 32;
    // Total number of warps (queries processed per block) must cover query_count.
    int totalWarps = (query_count + 0 - 1) / 1; // one warp per query.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    
    // Compute shared memory size needed per block.
    // Layout:
    //  - Candidate buffer: warpsPerBlock * k * sizeof(Candidate)
    //  - Candidate count: warpsPerBlock * sizeof(int)
    //  - Intermediate result: warpsPerBlock * k * sizeof(Candidate)
    //  - Tile buffer: TILE_SIZE * sizeof(float2)
    size_t shmemSize = warpsPerBlock * k * sizeof(Candidate) +
                       warpsPerBlock * sizeof(int) +
                       warpsPerBlock * k * sizeof(Candidate) +
                       TILE_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, shmemSize>>>(query, query_count,
                                                       data, data_count,
                                                       result, k);
    // Optionally, check for kernel launch errors (omitted here for brevity).
    cudaDeviceSynchronize();
}