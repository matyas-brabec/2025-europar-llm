#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <utility>

// -----------------------------------------------------------------------------
// This implementation computes k-nearest neighbors (k-NN) for 2D points in
// Euclidean (L2-squared) space. Each query point is processed by one warp (32
// threads) within a block. Each warp maintains two data structures in shared
// memory: a candidate buffer for new potential neighbors and an intermediate
// result buffer holding the current best k neighbors (sorted in ascending order
// by distance). When a new candidate (from processing a batch of data points)
// has a squared distance lower than the worst distance (max_distance) in the
// intermediate result, it is added to the candidate buffer using an atomic
// addition. When the candidate buffer fills up (its count reaches k) or after the
// last batch, the candidate buffer is merged with the intermediate result to
// update the best k neighbors.
// 
// For each block, a batch of data points (of fixed size BATCH_SIZE) is first
// loaded from global memory to shared memory so that all threads in the block
// can re-use them (from a memory‐bandwidth perspective).
// 
// Hyper-parameters:
//   - BLOCK_SIZE: number of threads per block
//   - BATCH_SIZE: number of data points processed per batch (loaded into shared mem)
//   We choose BLOCK_SIZE = 256 (i.e. 8 warps per block) and BATCH_SIZE = 1024.
// -----------------------------------------------------------------------------

// Candidate structure used to hold a data point candidate:
// 'idx' holds the global index in the data array and 'dist' the squared L2 distance.
struct Candidate {
    int idx;
    float dist;
};

// We define a compile-time constant for batch size.
#define BATCH_SIZE 1024

// -----------------------------------------------------------------------------
// Device function: mergeCandidateBuffer
// This function is invoked by lane 0 of each warp when the candidate buffer is
// full (or at the end if non-empty). It merges the warp's candidate buffer (which
// holds new candidate points, unsorted) with the warp's intermediate result (which
// holds the best k candidates sorted in ascending order by distance). The merge is
// performed by copying both arrays into a temporary buffer, sorting it by distance,
// and then copying back the first k elements as the updated sorted intermediate result.
// Note: k is guaranteed to be a power-of-two between 32 and 1024.
// -----------------------------------------------------------------------------
__device__ inline void mergeCandidateBuffer(int warpId, int k,
                                              Candidate *intermediateBuffer, // size: (#warps * k)
                                              Candidate *candidateBuffer,    // size: (#warps * k)
                                              int *candidateCount)           // candidateCount[warp]
{
    // Only lane 0 of the warp should execute this merge.
    int candCount = candidateCount[warpId];
    int total = k + candCount; // total number of items to sort (all k from intermediate + candCount)
    
    // Allocate temporary buffer on the stack.
    // Maximum size is 2*k; since k <= 1024, we allocate fixed 2048 elements.
    Candidate tmp[2048];

    // Copy the current intermediate result (already sorted) into tmp[0..k-1].
    for (int j = 0; j < k; j++) {
        tmp[j] = intermediateBuffer[warpId * k + j];
    }
    // Append the candidate buffer entries (unsorted) into tmp[k...].
    for (int j = 0; j < candCount; j++) {
        tmp[k + j] = candidateBuffer[warpId * k + j];
    }

    // Sort the combined array 'tmp' using insertion sort.
    // (Since total <= 2*k, and k is relatively small, this simple sort is acceptable.)
    for (int i = 1; i < total; i++) {
        Candidate key = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j].dist > key.dist) {
            tmp[j + 1] = tmp[j];
            j--;
        }
        tmp[j + 1] = key;
    }

    // Copy back the first k elements (the k smallest distances) into the intermediate result.
    for (int j = 0; j < k; j++) {
        intermediateBuffer[warpId * k + j] = tmp[j];
    }
    // Reset the candidate buffer counter for this warp.
    candidateCount[warpId] = 0;
}

// -----------------------------------------------------------------------------
// CUDA kernel: knn_kernel
// Each warp processes one query point. The kernel processes the dataset in
// batches. For each batch the following steps are taken:
//   1. A batch of data points is loaded into shared memory by all threads in the block.
//   2. Each warp iterates over the batch (each lane processing different points)
//      and computes the squared distance from its query point.
//   3. If a computed distance is less than the warp's current "max_distance" 
//      (the worst among the current k best), it uses atomicAdd to append the candidate
//      (global data index and computed distance) into its candidate buffer.
//   4. If the candidate buffer becomes full (>= k elements), lane 0 of the warp
//      merges the candidate buffer with the intermediate result buffer.
//   5. After all batches, a final merge (if the candidate buffer is non-empty) is performed.
//   6. Finally, the sorted k nearest neighbors (candidate indices and distances)
//      for the query are written to global memory.
// Shared memory layout (per block):
//   [0, BATCH_SIZE*sizeof(float2))                   --> data batch (float2 array)
//   [BATCH_SIZE*sizeof(float2), BATCH_SIZE*sizeof(float2) + (nw*k*sizeof(Candidate)))  
//                         --> candidate buffers for each warp (nw = number of warps per block)
//   [next, next + (nw*k*sizeof(Candidate)))            --> intermediate result buffers for each warp
//   [next, next + (nw*sizeof(int))]                    --> candidate counter for each warp
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int queryCount,
                           const float2 *data, int dataCount,
                           std::pair<int, float> *result,
                           int k)
{
    // Calculate warp and lane IDs.
    int threadId = threadIdx.x;
    int lane = threadId & 31;           // lane index in warp (0-31)
    int warpIdInBlock = threadId >> 5;  // warp index within block
    int warpsPerBlock = blockDim.x >> 5;

    // Global warp (query) id.
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (globalWarpId >= queryCount)
        return; // No query to process.

    // Partition the dynamic shared memory.
    extern __shared__ char sharedMem[];
    // Pointer to shared memory for current data batch.
    float2 *sData = (float2*)sharedMem;
    // Candidate buffer: each warp gets k Candidate elements.
    Candidate *candBuffer = (Candidate*)(sData + BATCH_SIZE);
    // Next, intermediate result buffers: each warp gets k Candidate elements.
    Candidate *intermediateBuffer = (Candidate*)( (char*)candBuffer + warpsPerBlock * k * sizeof(Candidate) );
    // Finally, candidate count: one int per warp.
    int *candCount = (int*)( (char*)intermediateBuffer + warpsPerBlock * k * sizeof(Candidate) );

    // Each warp processes one query.
    float2 q = query[globalWarpId];
    
    // Initialize the warp's intermediate result buffer and candidate counter.
    if (lane == 0) {
        // Set candidate count to 0.
        candCount[warpIdInBlock] = 0;
        // Initialize the intermediate result with k entries having FLT_MAX distance.
        for (int j = 0; j < k; j++) {
            intermediateBuffer[warpIdInBlock * k + j].dist = FLT_MAX;
            intermediateBuffer[warpIdInBlock * k + j].idx = -1;
        }
    }
    // Synchronize warp (intra-warp sync is enough; here __syncwarp ensures all lanes see init).
    __syncwarp();

    // Load the current "max distance" (distance of k-th nearest neighbor so far) from the intermediate result.
    float curMax = intermediateBuffer[warpIdInBlock * k + (k - 1)].dist;
    // Make sure all lanes in the warp have the same value using warp shfl.
    curMax = __shfl_sync(0xffffffff, curMax, 0);

    // Process data points in batches.
    for (int batchStart = 0; batchStart < dataCount; batchStart += BATCH_SIZE)
    {
        // Determine number of points to load in this batch.
        int batchSize = BATCH_SIZE;
        if (batchStart + batchSize > dataCount)
            batchSize = dataCount - batchStart;

        // --- Step 1: Load the batch of data points from global memory into shared memory.
        // All threads in the block cooperate to load the data.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            sData[i] = data[batchStart + i];
        }
        __syncthreads(); // Ensure entire batch is loaded.

        // --- Step 2: Each warp processes the shared data batch to compute distances.
        // Each lane in the warp processes a subset (stride of warp size) of the batch.
        for (int i = lane; i < batchSize; i += 32)
        {
            float2 d = sData[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            // If the computed distance is smaller than current max, add as a candidate.
            if (dist < curMax)
            {
                // Atomically obtain a slot in the candidate buffer for this warp.
                int pos = atomicAdd(&candCount[warpIdInBlock], 1);
                // Only store if within allowed size.
                if (pos < k) {
                    candBuffer[warpIdInBlock * k + pos].dist = dist;
                    candBuffer[warpIdInBlock * k + pos].idx = batchStart + i; // global index into 'data'
                }
            }
        }
        // Synchronize warp threads so that candidate buffer writes are visible.
        __syncwarp();
        // Also synchronize across block because shared memory sData is used by the whole block.
        __syncthreads();

        // --- Step 3: If candidate buffer is full (or overfull), merge it with the intermediate result.
        if (lane == 0)
        {
            if (candCount[warpIdInBlock] >= k)
            {
                mergeCandidateBuffer(warpIdInBlock, k, intermediateBuffer, candBuffer, candCount);
                // After merging, update the current maximum distance.
                curMax = intermediateBuffer[warpIdInBlock * k + (k - 1)].dist;
            }
        }
        // Broadcast the updated curMax to all lanes in the warp.
        curMax = __shfl_sync(0xffffffff, curMax, 0);
    } // end for each batch

    // --- Step 4: Merge any remaining candidates in the candidate buffer (if non-empty).
    if (lane == 0)
    {
        if (candCount[warpIdInBlock] > 0)
        {
            mergeCandidateBuffer(warpIdInBlock, k, intermediateBuffer, candBuffer, candCount);
        }
    }
    __syncwarp();

    // --- Step 5: Write the final k nearest neighbors (sorted by ascending distance)
    // into the output global memory.
    // Each warp writes its own k results.
    for (int i = lane; i < k; i += 32)
    {
        Candidate cand = intermediateBuffer[warpIdInBlock * k + i];
        result[globalWarpId * k + i] = std::pair<int, float>(cand.idx, cand.dist);
    }
}

// -----------------------------------------------------------------------------
// Host function: run_knn
// This function sets up kernel launch parameters and invokes the knn_kernel.
// It assumes that the input arrays (query, data, result) have been allocated on
// the device (using cudaMalloc) and that the 'data_count' is at least k.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Define block configuration.
    const int BLOCK_SIZE = 256;         // 256 threads per block => 8 warps per block.
    const int warpsPerBlock = BLOCK_SIZE >> 5; // 256/32 = 8

    // Calculate required shared memory size (in bytes) per block.
    // Shared memory layout:
    //   sData: BATCH_SIZE * sizeof(float2)
    //   candBuffer: warpsPerBlock * k * sizeof(Candidate)
    //   intermediateBuffer: warpsPerBlock * k * sizeof(Candidate)
    //   candCount: warpsPerBlock * sizeof(int)
    size_t sharedBytes = BATCH_SIZE * sizeof(float2)
                         + warpsPerBlock * k * sizeof(Candidate)    // candidate buffers
                         + warpsPerBlock * k * sizeof(Candidate)    // intermediate result buffers
                         + warpsPerBlock * sizeof(int);             // candidate counts

    // Compute grid configuration: one warp per query.
    // Total warps required = query_count, so number of blocks = ceil(query_count / warpsPerBlock)
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Launch the kernel.
    knn_kernel<<<numBlocks, BLOCK_SIZE, sharedBytes>>>(query, query_count,
                                                       data, data_count,
                                                       result, k);
    // Synchronize and check for errors.
    cudaDeviceSynchronize();
}