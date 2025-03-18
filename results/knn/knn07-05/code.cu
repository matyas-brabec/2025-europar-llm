// Optimized implementation of k-NN for 2D points using CUDA.
// This implementation assigns one query per warp (32 threads).
// Each warp maintains its intermediate nearest-neighbor result (sorted)
// in shared memory and also uses a shared candidate buffer to accumulate new
// candidates from batches of data points loaded in shared memory.
// When the candidate buffer becomes full (size k), it is merged with the intermediate result
// using a serial Bitonic Sort based merge procedure.
// The code assumes k is a power of two (between 32 and 1024) and data_count >= k.

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

#define BATCH_SIZE 256  // Number of data points processed in each batch

// Structure to represent a candidate neighbor.
// 'idx' is the index of the data point and 'dist' is the squared Euclidean distance.
struct Candidate {
    int idx;
    float dist;
};

// Serial Bitonic Sort on an array of Candidates in shared memory.
// This function is executed by one thread of the warp.
// The sort is in ascending order (smaller distances come first).
__device__ void bitonicSortSerial(Candidate *arr, int n) {
    // For each subsequence length (size), perform compare-swap steps.
    for (int size = 2; size <= n; size *= 2) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Serial loop over all indices in the array.
            for (int i = 0; i < n; i++) {
                int l = i ^ stride;
                if (l > i) {
                    // Determine the sorting direction based on the bit in i.
                    bool ascending = ((i & size) == 0);
                    // Compare based on ascending or descending order.
                    if ( (ascending && (arr[i].dist > arr[l].dist)) ||
                         (!ascending && (arr[i].dist < arr[l].dist)) ) {
                        // Swap the two Candidates.
                        Candidate tmp = arr[i];
                        arr[i] = arr[l];
                        arr[l] = tmp;
                    }
                }
            }
        }
    }
}

// Merge the candidate buffer with the current intermediate knn result.
// This function is called by all threads in the warp (using warp-level synchronization)
// and uses the candidate buffer (candBuff) and the intermediate knn result (knnRes),
// each of length k. The candidate count (candCount) indicates how many candidates are stored
// in the candidate buffer. If candCount < k, the remaining entries are filled with a dummy candidate.
__device__ void mergeBuffers(int warpOffset, Candidate *knnRes, Candidate *candBuff, volatile int *candCount, int k) {
    int lane = threadIdx.x & 31;  // warp lane id

    // 1. Fill the unused candidate buffer entries with dummy candidates.
    int count = *candCount;
    for (int i = lane; i < k; i += 32) {
        if (i >= count) {
            candBuff[i].idx  = -1;
            candBuff[i].dist = FLT_MAX;
        }
    }
    __syncwarp();

    // 2. Sort the candidate buffer (in-place) using Bitonic Sort.
    if (lane == 0) {
        bitonicSortSerial(candBuff, k);
    }
    __syncwarp();

    // 3. Merge the candidate buffer (reversed order) with the intermediate result.
    // For each index i, compute:
    //   merged[i] = min( knnRes[i], candBuff[k-i-1] )
    // where "min" means selecting the candidate with the smaller distance.
    for (int i = lane; i < k; i += 32) {
        Candidate a = knnRes[i];
        Candidate b = candBuff[k - 1 - i];
        Candidate m;
        m.idx  = (a.dist < b.dist) ? a.idx  : b.idx;
        m.dist = (a.dist < b.dist) ? a.dist : b.dist;
        candBuff[i] = m;  // reuse candidate buffer to hold the merged result
    }
    __syncwarp();

    // 4. Sort the merged result stored in candBuff.
    if (lane == 0) {
        bitonicSortSerial(candBuff, k);
    }
    __syncwarp();

    // 5. Copy the merged sorted result back into the intermediate result array.
    for (int i = lane; i < k; i += 32) {
        knnRes[i] = candBuff[i];
    }
    __syncwarp();

    // 6. Reset the candidate buffer count.
    if (lane == 0) {
        *candCount = 0;
    }
    __syncwarp();
}

// Main kernel that processes k-NN for each query (one query per warp).
// gridDim.x * (blockDim.x/32) should be at least the number of queries.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Compute number of warps per block.
    int warpsPerBlock = blockDim.x >> 5;  // equivalent to blockDim.x / 32
    int warpIdInBlock = threadIdx.x >> 5;  // warp index within this block
    int lane = threadIdx.x & 31;           // lane id within the warp

    // Compute global warp id; each warp handles one query.
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (globalWarpId >= query_count) return; // out-of-range queries

    // Partition the shared memory.
    // Layout in shared memory:
    //  [ batch data: BATCH_SIZE * sizeof(float2) ]
    //  [ candidate buffer: (warpsPerBlock * k) * sizeof(Candidate) ]
    //  [ intermediate knn result: (warpsPerBlock * k) * sizeof(Candidate) ]
    //  [ candidate count array: warpsPerBlock * sizeof(int) ]
    extern __shared__ char smem[];
    char *smem_ptr = smem;
    float2 *sharedData = (float2*)smem_ptr;
    smem_ptr += BATCH_SIZE * sizeof(float2);
    Candidate *candBuffBlock = (Candidate*)smem_ptr;
    smem_ptr += warpsPerBlock * k * sizeof(Candidate);
    Candidate *knnResBlock = (Candidate*)smem_ptr;
    smem_ptr += warpsPerBlock * k * sizeof(Candidate);
    int *candCountBlock = (int*)smem_ptr;
    // Pointers for this warp:
    int warpOffset = warpIdInBlock * k;
    Candidate *candBuff = &candBuffBlock[warpOffset];
    Candidate *knnRes = &knnResBlock[warpOffset];
    volatile int *candCount = (volatile int*)&candCountBlock[warpIdInBlock];

    // Initialize the intermediate result (knnRes) and candidate buffer (candBuff) to dummy values.
    for (int i = lane; i < k; i += 32) {
        knnRes[i].idx  = -1;
        knnRes[i].dist = FLT_MAX;
        candBuff[i].idx  = -1;
        candBuff[i].dist = FLT_MAX;
    }
    if (lane == 0) {
        *candCount = 0;
    }
    __syncwarp();

    // Load the query point for this warp.
    float2 q = query[globalWarpId];

    // Register variable to cache the current maximum (i.e., worst) distance.
    // Initially, the worst candidate in knnRes is FLT_MAX.
    float currentMax = FLT_MAX;
    currentMax = __shfl_sync(0xffffffff, currentMax, 0);

    // Process the data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        int batch_size = ((data_count - batch_start) < BATCH_SIZE) ? (data_count - batch_start) : BATCH_SIZE;
        // Load a batch of data points into shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            sharedData[i] = data[batch_start + i];
        }
        __syncthreads();

        // Each warp processes the batch using its 32 lanes.
        for (int i = lane; i < batch_size; i += 32) {
            float2 d = sharedData[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            // Only consider the candidate if its distance is smaller than the current worst.
            if (dist < currentMax) {
                int candidateIndex = batch_start + i; // global index of the data point
                // Atomically reserve a slot in the candidate buffer.
                int pos = atomicAdd((int*)candCount, 1);
                if (pos < k) {
                    candBuff[pos].idx  = candidateIndex;
                    candBuff[pos].dist = dist;
                }
                // If we filled the candidate buffer, merge it with the intermediate knn result.
                if (pos == (k - 1)) {
                    mergeBuffers(warpOffset, knnRes, candBuff, candCount, k);
                    // Update currentMax from knnRes[k-1] (the worst neighbor after merge).
                    if (lane == 0) {
                        currentMax = knnRes[k - 1].dist;
                    }
                    currentMax = __shfl_sync(0xffffffff, currentMax, 0);
                }
            }
        }
        __syncthreads();  // ensure all threads have finished using sharedData before next batch
    }

    // After processing all batches, if there are any pending candidates, merge them.
    if (*candCount > 0) {
        mergeBuffers(warpOffset, knnRes, candBuff, candCount, k);
        if (lane == 0) {
            currentMax = knnRes[k - 1].dist;
        }
        currentMax = __shfl_sync(0xffffffff, currentMax, 0);
    }

    // Write the final sorted k nearest neighbors for this query to global memory.
    // Each query's k results are stored consecutively.
    for (int i = lane; i < k; i += 32) {
        int outIndex = globalWarpId * k + i;
        result[outIndex] = std::make_pair(knnRes[i].idx, knnRes[i].dist);
    }
}

// Host function that launches the k-NN kernel.
// It chooses an appropriate number of threads per block and configures shared memory.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose 128 threads per block (i.e., 4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock >> 5;  // divide by 32
    // Each warp processes one query, so determine the grid dimensions.
    int totalWarps = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    int blocks = totalWarps;

    // Calculate the required shared memory per block:
    //   - Batch buffer: BATCH_SIZE float2 points.
    //   - Candidate buffer: warpsPerBlock * k * sizeof(Candidate).
    //   - Intermediate knn result buffer: warpsPerBlock * k * sizeof(Candidate).
    //   - Candidate count array: warpsPerBlock * sizeof(int).
    size_t sharedMemSize = 0;
    sharedMemSize += BATCH_SIZE * sizeof(float2);
    sharedMemSize += warpsPerBlock * k * sizeof(Candidate);
    sharedMemSize += warpsPerBlock * k * sizeof(Candidate);
    sharedMemSize += warpsPerBlock * sizeof(int);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // (Error checking can be added as needed.)
}