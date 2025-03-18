#include <cuda_runtime.h>
#include <float.h>
#include <utility>   // for std::pair

// -----------------------------------------------------------------------------
// This file implements an optimized CUDA kernel for k-nearest neighbors (k-NN)
// for 2D points using warp-level parallelism. Each warp (32 threads) handles
// one query point. Each query maintains two candidate lists:
//  (1) a private intermediate result of k best neighbors (sorted in ascending order)
//  (2) a shared candidate buffer for newly found candidates.
// When the candidate buffer fills up, it is merged with the intermediate result
// via Bitonic Sort and a special merge procedure.
// Data points are processed in batches that are loaded into shared memory.
// -----------------------------------------------------------------------------

// Structure to represent a candidate neighbor.
// It holds the index of the data point and its squared Euclidean distance.
struct Candidate {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// Device helper: Returns the "minimum" candidate (the one with smaller distance;
// if equal, the one with the lower index).
// -----------------------------------------------------------------------------
__device__ inline Candidate candidate_min(const Candidate &a, const Candidate &b) {
    Candidate res;
    // Compare squared distances; if equal, compare indices.
    if (a.dist < b.dist || (a.dist == b.dist && a.idx <= b.idx))
        res = a;
    else
        res = b;
    return res;
}

// -----------------------------------------------------------------------------
// Device helper: In-place Bitonic Sort over an array 'arr' of Candidate objects.
// 'n' must be a power-of-two and the sort is performed by all 32 threads in a warp.
// The sort arranges candidates in ascending order (lowest distance first).
// This implementation follows the standard Bitonic Sort network pseudocode.
// -----------------------------------------------------------------------------
__device__ void bitonic_sort(Candidate *arr, int n) {
    // We assume blockDim.x == 32 (one warp per block).
    unsigned int tid = threadIdx.x;
    // Outer loop: size of the subsequences being merged.
    for (int size = 2; size <= n; size *= 2) {
        // Inner loop: stride (gap) between elements being compared.
        for (int stride = size / 2; stride > 0; stride /= 2) {
            __syncthreads();  // Synchronize all warp threads.
            // Each thread processes several indices in strided manner.
            for (int i = tid; i < n; i += 32) {
                int ixj = i ^ stride;
                if (ixj > i && ixj < n) {
                    // Determine sorting direction: if (i & size)==0 then ascending.
                    bool ascending = ((i & size) == 0);
                    Candidate a = arr[i];
                    Candidate b = arr[ixj];
                    if (ascending) {
                        // For ascending order, swap if a > b.
                        if (a.dist > b.dist || (a.dist == b.dist && a.idx > b.idx)) {
                            arr[i]   = b;
                            arr[ixj] = a;
                        }
                    } else {
                        // For descending order, swap if a < b.
                        if (a.dist < b.dist || (a.dist == b.dist && a.idx < b.idx)) {
                            arr[i]   = b;
                            arr[ixj] = a;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// Device helper: Merges the sorted intermediate result 'inter' and the sorted
// candidate buffer 'candBuffer' into an updated intermediate result.
// The intermediate result and candidate buffer are arrays of length 'k'.
// The merge is done in two steps:
//   1. For each index i, compute mergeBuffer[i] = min(inter[i], candBuffer[k-i-1]).
//      This constructs a bitonic sequence.
//   2. Bitonic sort the mergeBuffer to obtain the updated sorted intermediate result.
// -----------------------------------------------------------------------------
__device__ void merge_candidates(Candidate *inter, Candidate *candBuffer, Candidate *mergeBuffer, int k) {
    unsigned int tid = threadIdx.x;
    // Step 1: Create merged candidate entries in mergeBuffer.
    for (int i = tid; i < k; i += 32) {
        int j = k - i - 1;
        Candidate a = inter[i];
        Candidate b = candBuffer[j];
        mergeBuffer[i] = candidate_min(a, b);
    }
    __syncthreads();
    // Step 2: Use bitonic sort to sort the mergeBuffer.
    bitonic_sort(mergeBuffer, k);
    // Copy the sorted mergeBuffer back to the intermediate result.
    for (int i = tid; i < k; i += 32) {
        inter[i] = mergeBuffer[i];
    }
    __syncthreads();
}

// -----------------------------------------------------------------------------
// __global__ kernel: Each block (32 threads = one warp) processes one query point.
// The kernel processes the entire set of data points in batches stored in shared memory.
// Each query maintains its private intermediate result (sorted list of k nearest candidates)
// and a shared candidate buffer. When the candidate buffer fills, it is merged with the
// intermediate result.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // We launch with one warp (32 threads) per block.
    int qid = blockIdx.x;  // Each block processes one query.
    if (qid >= query_count) return;  // Out-of-bound check.

    // Each warp's threads.
    int laneId = threadIdx.x;  // Since blockDim.x==32.

    // Load the query point.
    float2 qpt = query[qid];

    // -------------------------------------------------------------------------
    // Shared memory layout (per block, only one warp per block used):
    // [0, k * sizeof(Candidate))             -> intermediate result buffer (k Candidates)
    // [k * sizeof(Candidate), 2*k * sizeof(Candidate)) -> candidate buffer (k Candidates)
    // [2*k * sizeof(Candidate), 3*k * sizeof(Candidate)) -> merge buffer (k Candidates)
    // [3*k * sizeof(Candidate), 3*k * sizeof(Candidate) + sizeof(int)) -> candidate count (1 int)
    // [3*k * sizeof(Candidate) + sizeof(int), ... ) -> data batch buffer (BATCH_SIZE float2 elements)
    // -------------------------------------------------------------------------
    extern __shared__ char smem[];
    // Pointers to per-query shared buffers.
    Candidate *intermediate = (Candidate*) smem;            // Intermediate result [0, k)
    Candidate *candBuffer   = (Candidate*) (smem + 1 * k * sizeof(Candidate)); // Candidate buffer [k, 2*k)
    Candidate *mergeBuffer  = (Candidate*) (smem + 2 * k * sizeof(Candidate)); // Merge scratch [2*k, 3*k)
    int *candCountPtr       = (int*) (smem + 3 * k * sizeof(Candidate)); // Candidate count (single int)
    // Data batch buffer begins after candidate count.
    const int BATCH_SIZE = 256; // Hyper-parameter: batch size for loading data points.
    float2 *dataBatch = (float2*) (smem + 3 * k * sizeof(Candidate) + sizeof(int));

    // Initialize candidate count to 0 (only one warp per block so all threads do this identically).
    if (laneId == 0) {
        *candCountPtr = 0;
    }
    __syncthreads();

    // Initialize the intermediate result array with "infinite" distances.
    // We use -1 for index and FLT_MAX for distance.
    for (int i = laneId; i < k; i += 32) {
        intermediate[i].idx = -1;
        intermediate[i].dist = FLT_MAX;
    }
    __syncthreads();

    // Read the current maximum (k-th neighbor) from the intermediate result.
    // Initially this is FLT_MAX.
    float currentMax;
    if (laneId == 0) {
        currentMax = intermediate[k - 1].dist;
    }
    currentMax = __shfl_sync(0xffffffff, currentMax, 0);

    // Calculate the number of data batches.
    int numBatches = (data_count + BATCH_SIZE - 1) / BATCH_SIZE;

    // Iterate over batches of data points.
    for (int batch = 0; batch < numBatches; batch++)
    {
        int base = batch * BATCH_SIZE;
        int batchCount = ((base + BATCH_SIZE) <= data_count) ? BATCH_SIZE : (data_count - base);

        // Load the current batch of data points into shared memory.
        // Each thread loads several points in a strided manner.
        for (int i = laneId; i < batchCount; i += 32) {
            dataBatch[i] = data[base + i];
        }
        __syncthreads();

        // For each data point in the batch, compute the squared Euclidean distance.
        // If the distance is lower than currentMax, add it to the candidate buffer.
        for (int i = laneId; i < batchCount; i += 32) {
            float2 dpt = dataBatch[i];
            float dx = dpt.x - qpt.x;
            float dy = dpt.y - qpt.y;
            float dist = dx*dx + dy*dy;
            // Only consider if the distance is less than current maximum.
            if (dist < currentMax) {
                // Atomically reserve a slot in the candidate buffer.
                int pos = atomicAdd(candCountPtr, 1);
                if (pos < k) { // Only store if within buffer bounds.
                    candBuffer[pos].idx = base + i;  // Global data index.
                    candBuffer[pos].dist = dist;
                }
            }
        }
        __syncthreads();

        // If the candidate buffer is full (or overfull) then merge it with the intermediate result.
        if (*candCountPtr >= k) {
            // Sort the candidate buffer using Bitonic Sort.
            bitonic_sort(candBuffer, k);
            // Merge the sorted candidate buffer with the intermediate result.
            merge_candidates(intermediate, candBuffer, mergeBuffer, k);
            // Reset the candidate buffer count.
            if (laneId == 0) {
                *candCountPtr = 0;
            }
            __syncthreads();
            // Update currentMax from the updated intermediate result.
            if (laneId == 0) {
                currentMax = intermediate[k - 1].dist;
            }
            currentMax = __shfl_sync(0xffffffff, currentMax, 0);
        }
        __syncthreads();
    } // end batch loop

    // After processing all batches, if there are remaining candidates in the buffer, merge them.
    if (*candCountPtr > 0) {
        int count = *candCountPtr;
        // For indices beyond the filled ones, pad with "infinite" candidates.
        for (int i = laneId; i < k; i += 32) {
            if (i >= count) {
                candBuffer[i].idx = -1;
                candBuffer[i].dist = FLT_MAX;
            }
        }
        __syncthreads();
        bitonic_sort(candBuffer, k);
        merge_candidates(intermediate, candBuffer, mergeBuffer, k);
        if (laneId == 0) {
            *candCountPtr = 0;
            currentMax = intermediate[k - 1].dist;
        }
        currentMax = __shfl_sync(0xffffffff, currentMax, 0);
    }
    __syncthreads();

    // Finally, write the k sorted nearest neighbors for this query to global memory.
    // Global result layout: for query qid, result[qid * k + j] holds the j-th neighbor.
    for (int i = laneId; i < k; i += 32) {
        result[qid * k + i] = std::make_pair(intermediate[i].idx, intermediate[i].dist);
    }
}


// -----------------------------------------------------------------------------
// Host function: run_knn
// This function dispatches the knn_kernel to compute the k-nearest neighbors
// for each query. It assumes that memory for 'query', 'data', and 'result' has
// been allocated on the device (via cudaMalloc). 'k' is a power of two in [32,1024].
// -----------------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k)
{
    // Launch configuration:
    // Use one warp (32 threads) per query. Thus, blockDim = 32 and gridDim = query_count.
    int threadsPerBlock = 32;
    int numBlocks = query_count;

    // Shared memory layout per block:
    // - Intermediate result buffer: k * sizeof(Candidate)
    // - Candidate buffer: k * sizeof(Candidate)
    // - Merge buffer: k * sizeof(Candidate)
    // - Candidate count: sizeof(int)
    // - Data batch buffer: BATCH_SIZE * sizeof(float2)
    const int BATCH_SIZE = 256;
    size_t sharedMemSize = 3 * k * sizeof(Candidate) + sizeof(int) + BATCH_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}
