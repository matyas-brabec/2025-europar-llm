#include <cuda_runtime.h>
#include <cfloat>      // for FLT_MAX
#include <utility>     // for std::pair

// We define a custom candidate structure to represent a neighbor with an index and its squared Euclidean distance.
struct Candidate {
    int index;
    float dist;
};

// Maximum allowed k (per problem, k is between 32 and 1024)
#define MAX_K 1024

// Constant for the batch size of data points loaded into shared memory.
// This value is a hyper‐parameter that can be tuned. Here we choose 1024.
#define DATA_BATCH_SIZE 1024

// -----------------------------------------------------------------------------
// Sequential Bitonic Sort on an array of Candidate elements.
// This function is called by a single thread (lane 0) to sort an array in ascending order (by dist).
// The pseudocode follows the provided reference. We assume n is a power-of-two.
// If n is smaller (e.g. when padding is needed) the extra entries are assumed to have FLT_MAX distance.
__device__ void bitonicSortSeq(Candidate *arr, int n) {
    // Outer loop: size of bitonic sequence doubles each iteration.
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            // Loop over all indices in the array.
            for (int i = 0; i < n; i++) {
                int l = i ^ j;
                if (l > i) {
                    bool swapNeeded = false;
                    // Determine if we must swap based on the bitonic condition.
                    // If the bitwise AND of i and k is zero, then we are in the ascending part.
                    if ((i & k) == 0) {
                        if (arr[i].dist > arr[l].dist)
                            swapNeeded = true;
                    } else {
                        if (arr[i].dist < arr[l].dist)
                            swapNeeded = true;
                    }
                    if (swapNeeded) {
                        Candidate tmp = arr[i];
                        arr[i] = arr[l];
                        arr[l] = tmp;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// __global__ kernel implementing k-NN for 2D points.
// Each block processes ONE query using a single warp (32 threads).
// The kernel uses a candidate buffer (in shared memory) to hold potential neighbors
// and an intermediate result stored in registers (distributed across the warp).
// Data points are processed in batches loaded into shared memory.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k,
                           int dataBatchSize)
{
    // Each block processes one query.
    int q_idx = blockIdx.x;
    if (q_idx >= query_count) return;

    // In our block configuration, blockDim.x is 32, so each block is one warp.
    int laneId = threadIdx.x;  // lane indices 0..31

    // Number of candidate elements stored per thread in the intermediate result.
    // Since k is a power-of-two >=32 and we have 32 threads, k/32 is an integer.
    int local_k = k >> 5; // equivalent to k/32

    // -------------------------------------------------------------------------
    // Partition dynamic shared memory:
    // Layout in shared memory (per block):
    //   [ candidateBuffer ]: k Candidate elements (for storing candidates that pass the max_distance test)
    //   [ candidateCount ]: 1 integer (number of candidates stored so far)
    //   [ mergeBuffer ]: k Candidate elements (temporary buffer for merging)
    //   [ sharedData ]: DATA_BATCH_SIZE float2 elements (data batch loaded from global memory)
    extern __shared__ char sdata[];
    Candidate *candidateBuffer = (Candidate*) sdata; 
    int *candidateCount_ptr = (int*) (sdata + k * sizeof(Candidate));
    Candidate *mergeBuffer = (Candidate*) (sdata + k * sizeof(Candidate) + sizeof(int));
    float2 *sharedData = (float2*) (sdata + k * sizeof(Candidate) + sizeof(int) + k * sizeof(Candidate));

    // Only one thread (lane 0) initializes the candidate count to zero.
    if (laneId == 0)
        candidateCount_ptr[0] = 0;
    __syncthreads();

    // -------------------------------------------------------------------------
    // Load the query point corresponding to this block.
    float2 q = query[q_idx];

    // Each thread prepares a private copy (in registers) of a portion of the intermediate result.
    // The overall intermediate result is conceptually sorted in ascending order.
    // We initialize each local element with a sentinel value (FLT_MAX).
    Candidate localInter[32];  // Maximum local_k is at most 32 since k <= MAX_K.
    #pragma unroll
    for (int i = 0; i < local_k; i++) {
        localInter[i].index = -1;
        localInter[i].dist  = FLT_MAX;
    }
    // Initialize the max distance (i.e. distance of the k-th neighbor) as FLT_MAX.
    float current_max = FLT_MAX;

    // Determine the total number of batches needed to process all data points.
    int totalBatches = (data_count + dataBatchSize - 1) / dataBatchSize;

    // Process data points in batches.
    for (int b = 0; b < totalBatches; b++) {
        int batch_start = b * dataBatchSize;
        int batch_size = ((data_count - batch_start) < dataBatchSize) ? (data_count - batch_start) : dataBatchSize;

        // ---------------------------------------------------------------------
        // Load the current batch of data points into shared memory.
        // Each thread loads several points in a strided manner.
        for (int i = laneId; i < batch_size; i += 32) {
            sharedData[i] = data[batch_start + i];
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // Each thread computes distances from the query point to its assigned data points.
        for (int i = laneId; i < batch_size; i += 32) {
            float2 d = sharedData[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            // If the distance is less than the current worst (max) distance in the intermediate result...
            if (dist < current_max) {
                // Atomically add a new candidate into the candidate buffer.
                int pos = atomicAdd(candidateCount_ptr, 1);
                if (pos < k) {
                    candidateBuffer[pos].index = batch_start + i; // global index of the data point
                    candidateBuffer[pos].dist = dist;
                }
            }
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // When the candidate buffer is full, merge it with the intermediate result.
        if (candidateCount_ptr[0] >= k) {
            // (1) Sort the candidate buffer with Bitonic Sort.
            if (laneId == 0) {
                bitonicSortSeq(candidateBuffer, k);
            }
            __syncthreads();

            // (2) Write the current intermediate result from registers into mergeBuffer.
            // The intermediate result is stored distributed: each thread writes its local array.
            for (int j = 0; j < local_k; j++) {
                mergeBuffer[laneId + j * 32] = localInter[j];
            }
            __syncthreads();

            // (3) Merge the candidate buffer and the intermediate result.
            // This is done by lane 0 using the following procedure:
            //    For each index i in [0, k), take the minimum (by distance) of
            //    mergeBuffer[i] (from the intermediate result) and candidateBuffer[k - i - 1].
            //    The resulting array is bitonic, so we sort it to obtain the updated intermediate result.
            if (laneId == 0) {
                // Temporary buffer for merged candidates.
                Candidate temp_merged[MAX_K];
                for (int i = 0; i < k; i++) {
                    Candidate cand1 = mergeBuffer[i];
                    Candidate cand2 = candidateBuffer[k - i - 1];
                    temp_merged[i] = (cand1.dist <= cand2.dist) ? cand1 : cand2;
                }
                // Sort the merged array to restore the invariant (sorted ascending).
                bitonicSortSeq(temp_merged, k);
                // Copy the sorted merged result back to mergeBuffer.
                for (int i = 0; i < k; i++) {
                    mergeBuffer[i] = temp_merged[i];
                }
                // Update the max distance (k-th neighbor's distance).
                current_max = mergeBuffer[k - 1].dist;
            }
            __syncthreads();

            // (4) Broadcast the merged intermediate result back into registers.
            for (int j = 0; j < local_k; j++) {
                localInter[j] = mergeBuffer[laneId + j * 32];
            }
            __syncthreads();

            // (5) Reset the candidate buffer counter.
            if (laneId == 0) {
                candidateCount_ptr[0] = 0;
            }
            __syncthreads();
        }
        __syncthreads();
    } // end for each batch

    // -------------------------------------------------------------------------
    // After all batches, if any candidates remain in the candidate buffer, merge them.
    if (candidateCount_ptr[0] > 0) {
        // Because the candidate buffer may not be completely full,
        // pad the unused entries with sentinel values.
        if (laneId == 0) {
            int cnt = candidateCount_ptr[0];
            for (int i = cnt; i < k; i++) {
                candidateBuffer[i].index = -1;
                candidateBuffer[i].dist = FLT_MAX;
            }
            bitonicSortSeq(candidateBuffer, k);
        }
        __syncthreads();

        // Copy the current intermediate result from registers into mergeBuffer.
        for (int j = 0; j < local_k; j++) {
            mergeBuffer[laneId + j * 32] = localInter[j];
        }
        __syncthreads();
        if (laneId == 0) {
            Candidate temp_merged[MAX_K];
            for (int i = 0; i < k; i++) {
                Candidate cand1 = mergeBuffer[i];
                Candidate cand2 = candidateBuffer[k - i - 1];
                temp_merged[i] = (cand1.dist <= cand2.dist) ? cand1 : cand2;
            }
            bitonicSortSeq(temp_merged, k);
            for (int i = 0; i < k; i++) {
                mergeBuffer[i] = temp_merged[i];
            }
            current_max = mergeBuffer[k - 1].dist;
        }
        __syncthreads();
        for (int j = 0; j < local_k; j++) {
            localInter[j] = mergeBuffer[laneId + j * 32];
        }
        __syncthreads();
        if (laneId == 0) {
            candidateCount_ptr[0] = 0;
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Finally, write the k nearest neighbors (sorted ascending) to the global result array.
    // The result for query q_idx is stored starting at result[q_idx * k].
    for (int j = 0; j < local_k; j++) {
        int pos = q_idx * k + (laneId + j * 32);
        result[pos] = std::pair<int, float>(localInter[j].index, localInter[j].dist);
    }
}

// -----------------------------------------------------------------------------
// Host function interface to run the k-NN CUDA kernel.
// It sets up the kernel configuration and dynamically allocates the required shared memory.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // We choose one warp (32 threads) per block so that each block processes one query.
    int blockSize = 32;
    int gridSize = query_count;  // one block per query

    // Compute shared memory size per block.
    // Layout:
    //   candidateBuffer: k * sizeof(Candidate)
    //   candidateCount: sizeof(int)
    //   mergeBuffer: k * sizeof(Candidate)
    //   sharedData: DATA_BATCH_SIZE * sizeof(float2)
    size_t shmemSize = k * sizeof(Candidate) + sizeof(int) +
                       k * sizeof(Candidate) +
                       DATA_BATCH_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<gridSize, blockSize, shmemSize>>>(query, query_count, data, data_count, result, k, DATA_BATCH_SIZE);
    cudaDeviceSynchronize();
}