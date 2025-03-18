#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <utility>

// This kernel implements the k‐nearest neighbors (k-NN) algorithm for 2D points.
// For each query point, one warp (32 threads) computes its k nearest data points
// (using the squared Euclidean distance) by processing the data in batches.
// Each warp keeps an “intermediate result” of k best candidates in registers and a
// candidate buffer of up to k candidate entries in shared memory. When the buffer is full,
// it is merged with the intermediate result via a two‐phase Bitonic Sort based merge.
// The final sorted candidate list for a query is written to the global results array.

// Define the size of each data batch loaded into shared memory.
#define BATCH_SIZE 1024

// Structure to hold a candidate pair (data index and squared distance).
struct Candidate {
    int index;
    float dist;
};

//------------------------------------------------------------------------------
// Device function: bitonicSort
// Sorts an array of Candidate elements in-place in shared memory in ascending order (by dist)
// using the Bitonic Sort algorithm. This routine is designed to be executed cooperatively
// by one warp (32 threads). Each thread processes multiple indices in a strided loop.
//------------------------------------------------------------------------------
__device__ void bitonicSort(Candidate* arr, int n) {
    // Get the lane id (0-31) within the warp.
    unsigned int lane = threadIdx.x & 31;
    // Outer loop: size of the sorted subsequence grows from 2 to n.
    for (int size = 2; size <= n; size <<= 1) {
        // Inner loop: stride decreases from size/2 to 1.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes multiple indices in the array.
            for (int idx = lane; idx < n; idx += 32) {
                int partner = idx ^ stride;
                if (partner > idx && partner < n) {
                    // Determine the sorting direction.
                    bool ascending = ((idx & size) == 0);
                    // Swap if the order is wrong.
                    if ((ascending && (arr[idx].dist > arr[partner].dist)) ||
                        (!ascending && (arr[idx].dist < arr[partner].dist))) {
                        Candidate temp = arr[idx];
                        arr[idx] = arr[partner];
                        arr[partner] = temp;
                    }
                }
            }
            // Synchronize threads within the warp.
            __syncwarp();
        }
    }
}

//------------------------------------------------------------------------------
// Kernel: knn_kernel
// Each warp processes one query point, iterating over the data points in batches.
// Data points for each batch are loaded into shared memory (sharedData) for faster access.
// Each warp maintains an intermediate result of k nearest candidates (kept in registers)
// and a candidate buffer (in shared memory) that temporarily accumulates new candidates.
// When the candidate buffer is full, it is merged with the intermediate result using two
// rounds of Bitonic Sort and a merging operation as specified.
//------------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float>* result, int k)
{
    // Each warp (32 threads) processes one query.
    const int warpIdInBlock = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int warpsPerBlock = blockDim.x / 32;
    // Global query index for this warp.
    const int queryIdx = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (queryIdx >= query_count) return;

    // Load the query point from global memory.
    float2 q = query[queryIdx];

    // Dynamically allocated shared memory layout:
    // [ candidateBuffers | candidateCounts | sharedData | scratchBuffer ]
    // - candidateBuffers: per-warp buffer for candidates, size = (warpsPerBlock * k) Candidate elements.
    // - candidateCounts: per-warp counter, size = (warpsPerBlock) ints.
    // - sharedData: for data batch, size = BATCH_SIZE float2 elements.
    // - scratchBuffer: per-warp scratch for merging, size = (warpsPerBlock * k) Candidate elements.
    extern __shared__ char shared_mem[];
    Candidate* candidateBuffers = reinterpret_cast<Candidate*>(shared_mem);
    int* candidateCounts = reinterpret_cast<int*>(candidateBuffers + warpsPerBlock * k);
    float2* sharedData = reinterpret_cast<float2*>(candidateCounts + warpsPerBlock);
    Candidate* scratchBuffer = reinterpret_cast<Candidate*>(sharedData + BATCH_SIZE);

    // Pointers for this warp's candidate buffer and scratch merging region.
    Candidate* myBuffer = candidateBuffers + warpIdInBlock * k;
    Candidate* myScratch = scratchBuffer + warpIdInBlock * k;

    // Initialize the candidate count for this warp.
    if (lane == 0) {
        candidateCounts[warpIdInBlock] = 0;
    }
    // Ensure candidateCounts[warpIdInBlock] is visible to all threads in block.
    __syncwarp();

    // The intermediate result (k nearest neighbors for this query) is maintained privately
    // in registers. To distribute storage, each thread holds a portion: localCount = k/32.
    const int localCount = k / 32;
    // Each thread allocates room for its part (maximum localCount <= 32 since k is at most 1024).
    Candidate localNN[32];
    for (int i = 0; i < localCount; i++) {
        localNN[i].index = -1;
        localNN[i].dist = FLT_MAX;
    }

    //---------------------------------------------------------------------------
    // Function: mergeBuffer
    // Merges the candidate buffer (myBuffer) with the intermediate result (localNN).
    // The merge is performed in several steps:
    //   1. Each thread writes its localNN entries into the scratch buffer (myScratch)
    //      in a contiguous, warp-ordered fashion.
    //   2. The scratch buffer is sorted with Bitonic Sort.
    //   3. The candidate buffer (myBuffer) is also sorted with Bitonic Sort.
    //   4. Each element of the merged result is computed as the minimum of
    //      myBuffer[i] and myScratch[k-i-1] (thus forming a bitonic sequence).
    //   5. The merged array is sorted with Bitonic Sort to produce the updated
    //      intermediate result.
    //   6. The sorted merged result is copied back to localNN.
    // After merging, the candidate buffer count is reset (by the caller).
    //---------------------------------------------------------------------------
    /// @FIXED
    /// auto mergeBuffer = [&]() __device__ {
    auto mergeBuffer = [&]() {
        // Step 1: Write the intermediate result from registers into myScratch.
        // Each thread writes its local part in order.
        for (int i = 0; i < localCount; i++) {
            myScratch[lane * localCount + i] = localNN[i];
        }
        __syncwarp();

        // Step 2: Sort the scratch buffer containing the intermediate result.
        bitonicSort(myScratch, k);
        __syncwarp();

        // Step 3: Sort the candidate buffer (myBuffer) in-place.
        bitonicSort(myBuffer, k);
        __syncwarp();

        // Step 4: Merge the two sorted arrays.
        // For each index i in [0, k), compute:
        //   merged[i] = min(myBuffer[i], myScratch[k - i - 1])
        for (int i = lane; i < k; i += 32) {
            Candidate candA = myBuffer[i];
            Candidate candB = myScratch[k - i - 1];
            Candidate merged;
            merged = (candA.dist < candB.dist) ? candA : candB;
            myScratch[i] = merged;
        }
        __syncwarp();

        // Step 5: Sort the merged result in myScratch.
        bitonicSort(myScratch, k);
        __syncwarp();

        // Step 6: Copy the updated sorted intermediate result back to registers.
        for (int i = 0; i < localCount; i++) {
            localNN[i] = myScratch[lane * localCount + i];
        }
        __syncwarp();
    };

    //--------------------------------------------------------------------------
    // Main loop: Process data points in batches.
    // For each batch, the entire block cooperatively loads data into shared memory.
    // Then each warp iterates over the batch (each lane processing a subset) and computes
    // squared Euclidean distances. If a candidate distance is smaller than the current
    // max in the intermediate result, it is added to the candidate buffer using atomicAdd.
    //---------------------------------------------------------------------------
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        int batchSize = BATCH_SIZE;
        if (batch_start + batchSize > data_count)
            batchSize = data_count - batch_start;

        // Cooperative load of a batch of data points into shared memory.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            sharedData[i] = data[batch_start + i];
        }
        __syncthreads();

        // For each data point in the batch, each warp computes the distance from q.
        for (int i = lane; i < batchSize; i += 32) {
            float2 d = sharedData[i];
            float dx = q.x - d.x;
            float dy = q.y - d.y;
            float dist = dx * dx + dy * dy;

            // Compute the current maximum distance from the intermediate result.
            // Since the intermediate result is maintained sorted (ascending),
            // the k-th nearest neighbor's distance is the maximum.
            float threadMax = localNN[localCount - 1].dist;
            // Perform warp-level reduction over the 32 lanes.
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(0xffffffff, threadMax, offset);
                if (other > threadMax) threadMax = other;
            }
            float currentMax = __shfl_sync(0xffffffff, threadMax, 0);

            // If the new candidate is closer than the current worst candidate, add it.
            if (dist < currentMax) {
                int pos = atomicAdd(&candidateCounts[warpIdInBlock], 1);
                if (pos < k) {
                    myBuffer[pos].index = batch_start + i;
                    myBuffer[pos].dist = dist;
                }
            }
        }
        __syncwarp(); // synchronize warp lanes for candidate buffer state

        // If the candidate buffer is full, merge it with the intermediate result.
        if (candidateCounts[warpIdInBlock] >= k) {
            mergeBuffer();
            if (lane == 0) {
                candidateCounts[warpIdInBlock] = 0; // reset candidate buffer count
            }
            __syncwarp();
        }
        __syncthreads(); // ensure all threads in block are synchronized before next batch load
    }

    // After processing all batches, merge any remaining candidates in the buffer.
    if (candidateCounts[warpIdInBlock] > 0) {
        mergeBuffer();
        if (lane == 0) {
            candidateCounts[warpIdInBlock] = 0;
        }
        __syncwarp();
    }

    //--------------------------------------------------------------------------
    // Write the final sorted intermediate result (k nearest neighbors) to global memory.
    // Each query's result is stored consecutively in the 'result' array.
    // The ordering is based on the global index in the merged array.
    //--------------------------------------------------------------------------
    for (int i = 0; i < localCount; i++) {
        int globalPos = lane * localCount + i; // global offset within the k results for this query
        result[queryIdx * k + globalPos] = std::pair<int, float>(localNN[i].index, localNN[i].dist);
    }
}

//------------------------------------------------------------------------------
// Host function: run_knn
// This function prepares the kernel launch parameters and launches the k-NN kernel.
// It assumes that 'query', 'data', and 'result' are allocated on the device (via cudaMalloc).
//------------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose the number of warps per block; each warp processes one query.
    // For a modern GPU, 4 warps per block (128 threads per block) is a reasonable choice.
    const int warpsPerBlock = 4;
    const int threadsPerBlock = warpsPerBlock * 32;
    // Determine the number of blocks needed so that each warp gets one query.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate the required shared memory size (in bytes):
    //   candidateBuffers: warpsPerBlock * k * sizeof(Candidate)
    //   candidateCounts:  warpsPerBlock * sizeof(int)
    //   sharedData:       BATCH_SIZE * sizeof(float2)
    //   scratchBuffer:    warpsPerBlock * k * sizeof(Candidate)
    size_t sharedMemSize = warpsPerBlock * k * sizeof(Candidate) +
                           warpsPerBlock * sizeof(int) +
                           BATCH_SIZE * sizeof(float2) +
                           warpsPerBlock * k * sizeof(Candidate);

    // Launch the kernel with the computed configuration.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // Note: For production code, error checking and stream synchronization should be performed.
}
