// This complete .cu file contains an optimized CUDA implementation of the k‐nearest neighbors (k‐NN)
// for 2D points in Euclidean space using a single warp (32 threads) per query. Each warp maintains
// a private intermediate sorted list (distributed across its 32 lanes) of k nearest neighbors, and it also
// uses a per‐warp candidate buffer in shared memory for new candidates. When the candidate buffer becomes
// full (i.e. holds k candidates), the warp “flushes” it by:
//   1. Sorting the candidate buffer in ascending order (using a parallel Bitonic Sort algorithm executed by the warp).
//   2. Merging the candidate buffer with its intermediate result by taking, for each global index i in [0,k),
//      the minimum (by squared Euclidean distance) of candidateBuffer[i] and intermediate_result[k-i-1].
//   3. Sorting the merged array (again by Bitonic Sort) to yield an updated intermediate result.
// After processing all batches of data points (each batch preloaded cooperatively into shared memory),
// any remaining candidates are flushed and the final results written in global memory.
// The host function run_knn launches the kernel with an appropriate grid/block configuration and computes the required shared memory size.
//
// NOTE: This code assumes that k is a power of two (between 32 and 1024) and that k is divisible by 32.
//       The squared Euclidean distance is computed (no square root is applied).
//

#include <cuda_runtime.h>
#include <float.h>
#include <utility>

//
// Custom pair structure to store a neighbor candidate (data index and squared distance).
//
struct Pair {
    int index;
    float dist;
};

//
// __device__ function: Bitonic sort a shared array of Pair elements.
// The array 'arr' of length 'n' is sorted in ascending order (by .dist).
// This implementation uses a warp of 32 threads. Each thread processes multiple indices in steps of 32.
// The algorithm follows the pseudocode provided.
//
__device__ inline void bitonicSortShared(Pair *arr, int n) {
    // Assume warp-synchronous execution (all 32 threads of the warp participate).
    unsigned int mask = 0xffffffff;
    int lane = threadIdx.x & 31;
    // Outer loop: sequence size doubles from 2 to n.
    for (int size = 2; size <= n; size *= 2) {
        // Inner loop: step decreases from size/2 to 1.
        for (int stride = size / 2; stride > 0; stride /= 2) {
            // Each thread loops over indices i in its stride.
            for (int i = lane; i < n; i += 32) {
                int j = i ^ stride;
                if (j > i && j < n) {
                    // Determine sorting direction: ascending if (i & size)==0, descending otherwise.
                    bool ascending = ((i & size) == 0);
                    // Compare and swap if out of order according to the desired direction.
                    if ( (ascending && arr[i].dist > arr[j].dist) ||
                         (!ascending && arr[i].dist < arr[j].dist) ) {
                        Pair temp = arr[i];
                        arr[i] = arr[j];
                        arr[j] = temp;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

//
// __device__ function: Retrieve an element from the warp-distributed intermediate result.
// The intermediate result is distributed such that each warp lane holds m = k/32 elements in its local array.
// The global index "globalIdx" (in [0, k)) is located in lane = globalIdx % 32 at local index = globalIdx / 32.
// This function uses __shfl_sync to fetch the data from the lane that owns the desired element.
//
__device__ inline Pair warpGetKNN(const Pair localKNN[], int m, int globalIdx, unsigned int mask = 0xffffffff) {
    int owner = globalIdx & 31;         // globalIdx % 32
    int localIdx = globalIdx >> 5;        // globalIdx / 32 (since 32 == 1<<5)
    int knn_index;
    float knn_dist;
    // Only the owner lane has the valid element.
    if ((threadIdx.x & 31) == owner) {
        knn_index = localKNN[localIdx].index;
        knn_dist = localKNN[localIdx].dist;
    }
    knn_index = __shfl_sync(mask, knn_index, owner);
    knn_dist  = __shfl_sync(mask, knn_dist,  owner);
    Pair ret;
    ret.index = knn_index;
    ret.dist  = knn_dist;
    return ret;
}

//
// __device__ function: Flush (merge) the candidate buffer with the intermediate result.
// This function is called when the candidate buffer (of length k) for the current warp is full
// or at the end after all data is processed (if non‐empty).
// Parameters:
//   warpID         - The warp index within the block (i.e. threadIdx.x/32).
//   k              - Total number of nearest neighbors to maintain for the query.
//   m              - Local number of elements per lane (m = k/32).
//   warpCandCount  - Pointer to the array of candidate counts per warp (in shared memory).
//   warpCandBuffer - Pointer to the candidate buffer for this warp (in shared memory), length k.
//   localKNN       - The warp’s private intermediate k-NN result stored in registers (distributed among lanes).
//   fullMask       - The active mask for full warp (usually 0xffffffff).
//
// The merge is performed in three steps:
// 1. Sort the candidate buffer using Bitonic Sort.
// 2. For each global index i in [0, k), compute merged[i] = min(candidateBuffer[i], intermediate_result[k-i-1]).
// 3. Sort the merged array with Bitonic Sort and update the intermediate result accordingly.
// Finally, the candidate count for this warp is reset to zero.
//
/// @FIXED (-2:+2)
/// __device__ inline void flushMerge(int warpID, int k, int m, volatile int *warpCandCount,
///                                   volatile Pair *warpCandBuffer, Pair localKNN[],
__device__ inline void flushMerge(int warpID, int k, int m, int *warpCandCount,
                                  Pair *warpCandBuffer, Pair localKNN[],
                                  unsigned int fullMask = 0xffffffff) {
    // Step 1: Sort the candidate buffer in shared memory.
    bitonicSortShared((Pair *)warpCandBuffer, k);
    __syncwarp(fullMask);

    // Step 2: Merge candidate buffer and intermediate result.
    // For each global index i (0 <= i < k) that this warp processes:
    int lane = threadIdx.x & 31;
    for (int i = lane; i < k; i += 32) {
        int partner = k - 1 - i;  // corresponding index in the intermediate result.
        // Get the intermediate result element at global index 'partner' from the warp-distributed registers.
        Pair interm = warpGetKNN(localKNN, m, partner, fullMask);
        // Take the minimum (by distance) of candidate and intermediate element.
        Pair cand = warpCandBuffer[i];
        Pair merged;
        merged = (cand.dist < interm.dist) ? cand : interm;
        warpCandBuffer[i] = merged;
    }
    __syncwarp(fullMask);

    // Step 3: Sort the merged result in shared memory.
    bitonicSortShared((Pair *)warpCandBuffer, k);
    __syncwarp(fullMask);

    // Step 4: Update the warp's intermediate result registers with the sorted merged result.
    for (int i = lane; i < k; i += 32) {
        int localIdx = i >> 5; // i / 32
        // Each lane writes the element if it owns the corresponding global index.
        localKNN[localIdx] = warpCandBuffer[i];
    }
    __syncwarp(fullMask);

    // Reset candidate buffer count for this warp (only one thread needs to do it).
    if (lane == 0) {
        warpCandCount[warpID] = 0;
    }
    __syncwarp(fullMask);
}

//
// __global__ kernel: Process queries to compute k-nearest neighbors.
// Each warp (32 threads) processes one query point using the following strategy:
// 1. Each warp initializes its private (distributed) intermediate result with k entries (all set to { -1, FLT_MAX }).
// 2. The data points are processed in batches. Each block loads a batch (tile) of data points into shared memory.
// 3. Each warp computes distances from its query point to the data points in the batch. If a distance is smaller
//    than the current maximum distance in its intermediate result, the candidate is inserted into a per‐warp candidate
//    buffer (using an atomic add for position).
// 4. Whenever the candidate buffer is full (i.e. holds k candidates), it is flushed (merged) with the intermediate result.
// 5. After all batches are processed, any remaining candidates in the candidate buffer are merged.
// 6. Finally, the k sorted nearest neighbors (indices and distances) are written to global memory.
//
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           Pair *result, int k) {
    // Each warp (32 threads) processes exactly one query.
    int warpIDInBlock = threadIdx.x >> 5;  // warp index within block
    int lane = threadIdx.x & 31;           // lane index within warp
    int warpsPerBlock = blockDim.x >> 5;
    int globalWarpID = blockIdx.x * warpsPerBlock + warpIDInBlock;
    if (globalWarpID >= query_count)
        return;

    // Load the query point for this warp.
    float2 queryPoint = query[globalWarpID];

    // Each warp will maintain its intermediate k-NN result in registers.
    // We partition k among the 32 threads: each lane stores m = k/32 elements.
    int m = k >> 5;  // since k is a power of two and >= 32, m = k / 32.
    // Use a fixed-size array with maximum size 32 (sufficient when k <= 1024).
    Pair localKNN[32];
    for (int i = 0; i < m; i++) {
        localKNN[i].index = -1;
        localKNN[i].dist  = FLT_MAX;
    }
    // Current maximum distance is the last element in the sorted intermediate result.
    float max_distance = localKNN[m - 1].dist; // Initially FLT_MAX.

    // Shared memory layout per block:
    // [0, warpsPerBlock*sizeof(int))         : candidate buffer counts (one int per warp).
    // [warpsPerBlock*sizeof(int), warpsPerBlock*sizeof(int) + warpsPerBlock*k*sizeof(Pair))
    //                                          : candidate buffers (each warp gets k Pair elements).
    // [rest]                                   : data tile for batching (float2 array).
    extern __shared__ char shared_mem[];
    int *warpCandCount = (int *)shared_mem;
    Pair *warpCandBufferBase = (Pair *)(shared_mem + warpsPerBlock * sizeof(int));
    // Define tile size for data batch.
    const int TILE_SIZE = 1024;
    float2 *dataTile = (float2 *)(shared_mem + warpsPerBlock * sizeof(int) + warpsPerBlock * k * sizeof(Pair));

    // Initialize this warp's candidate count (only lane 0 does it).
    if (lane == 0)
        warpCandCount[warpIDInBlock] = 0;
    __syncwarp(0xffffffff);

    // Get pointer to this warp's candidate buffer in shared memory.
    Pair *warpCandBuffer = &warpCandBufferBase[warpIDInBlock * k];

    // Process the data points in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += TILE_SIZE) {
        int batchSize = TILE_SIZE;
        if (batchStart + batchSize > data_count)
            batchSize = data_count - batchStart;

        // Cooperative load: all threads in the block load data points into shared memory.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            dataTile[i] = data[batchStart + i];
        }
        __syncthreads(); // Ensure batch is loaded.

        // Each warp processes the loaded batch.
        for (int i = lane; i < batchSize; i += 32) {
            float2 dpt = dataTile[i];
            float dx = queryPoint.x - dpt.x;
            float dy = queryPoint.y - dpt.y;
            float dist = dx * dx + dy * dy;
            if (dist < max_distance) {
                // Attempt to insert candidate (global index is batchStart+i).
                int pos = atomicAdd(&warpCandCount[warpIDInBlock], 1);
                if (pos < k) {
                    warpCandBuffer[pos].index = batchStart + i;
                    warpCandBuffer[pos].dist  = dist;
                }
            }
        }
        __syncwarp(0xffffffff);

        // If the candidate buffer is full (or over‐full), flush it.
        if (warpCandCount[warpIDInBlock] >= k) {
            flushMerge(warpIDInBlock, k, m, warpCandCount, warpCandBuffer, localKNN, 0xffffffff);
            // Update the current max_distance from the updated intermediate result.
            int globalIdxMax = k - 1;
            int owner = globalIdxMax & 31;
            int idx = globalIdxMax >> 5;
            float new_max;
            if (lane == owner)
                new_max = localKNN[idx].dist;
            new_max = __shfl_sync(0xffffffff, new_max, owner);
            max_distance = new_max;
        }
        __syncthreads();  // Synchronize block before next batch load.
    }

    // After processing all batches, flush any remaining candidates (if any).
    if (warpCandCount[warpIDInBlock] > 0) {
        flushMerge(warpIDInBlock, k, m, warpCandCount, warpCandBuffer, localKNN, 0xffffffff);
        int globalIdxMax = k - 1;
        int owner = globalIdxMax & 31;
        int idx = globalIdxMax >> 5;
        float new_max;
        if (lane == owner)
            new_max = localKNN[idx].dist;
        new_max = __shfl_sync(0xffffffff, new_max, owner);
        max_distance = new_max;
    }

    // Write out the final k nearest neighbors in sorted (ascending) order.
    // The intermediate result is distributed over the warp's registers.
    // Each thread writes the elements for which (global index mod 32) == lane.
    for (int i = lane; i < k; i += 32) {
        int idx = i >> 5;  // i / 32
        // Global result for query globalWarpID is stored in row-major order.
        /// @FIXED
        /// result[globalWarpID * k + i] = std::make_pair(localKNN[idx].index, localKNN[idx].dist);
        result[globalWarpID * k + i] = Pair(localKNN[idx].index, localKNN[idx].dist);
    }
}

//
// Host function: run_knn
// Sets up grid/block configuration, computes shared memory size, and launches the knn_kernel.
// Inputs:
//   query       - device pointer to query points (float2).
//   query_count - number of queries.
//   data        - device pointer to data points (float2).
//   data_count  - number of data points.
//   result      - device pointer where output pairs (int index, float distance) are stored.
//   k           - number of nearest neighbors to compute (power-of-two between 32 and 1024).
//
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k) {
    // Choose threads per block. Each warp (32 threads) processes one query.
    // For example: 128 threads per block (i.e. 4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock >> 5;  // = threadsPerBlock / 32
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Define tile size (batch size for data points) used in the kernel.
    const int TILE_SIZE = 1024;
    // Shared memory layout per block:
    // Candidate counts: warpsPerBlock * sizeof(int)
    // Candidate buffers: warpsPerBlock * k * sizeof(Pair)
    // Data tile: TILE_SIZE * sizeof(float2)
    size_t sharedMemSize = warpsPerBlock * sizeof(int)
                         + warpsPerBlock * k * sizeof(Pair)
                         + TILE_SIZE * sizeof(float2);

    // Launch the kernel.
    // Since our kernel uses our custom Pair type, we reinterpret the result pointer accordingly.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count,
                                                           reinterpret_cast<Pair *>(result), k);
    // (Error checking omitted for brevity.)
}
