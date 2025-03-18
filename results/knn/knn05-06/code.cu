// This file implements an optimized CUDA k-nearest neighbors (k-NN) algorithm
// for 2D points in Euclidean space. Each query is processed by one warp
// (32 threads), and candidates are buffered in shared memory before being
// merged with the per-warp intermediate result. The final k nearest neighbors
// (sorted in ascending order by squared distance) for each query are stored in
// the output array.
//
// The main public interface is the run_knn() function.
// Target hardware: modern NVIDIA data‐center GPUs (A100/H100).
// The code uses warp-level primitives (e.g., __shfl_down_sync) and shared memory
// to achieve high performance, while processing the input data in batches.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

// A candidate data point (a potential neighbor)
// 'index' is the index in the data array, and 'dist' is the squared Euclidean distance.
struct Candidate {
    int index;
    float dist;
};

// A simple pair structure (layout-compatible with std::pair<int,float>) used for the result.
struct Pair {
    int first;
    float second;
};

// Helper device function: returns the minimum of two integers.
__device__ inline int min_int(int a, int b) {
    return (a < b) ? a : b;
}

// Warp-level reduction for maximum value using shuffle.
__device__ inline float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Swap two Candidate elements.
__device__ inline void swapCandidate(Candidate &a, Candidate &b) {
    Candidate tmp = a;
    a = b;
    b = tmp;
}

// A device function that partitions the array of Candidate elements (of length n)
// using a quickselect‐like approach so that the first k elements are the k smallest
// (by distance). Then it sorts the first k elements in ascending order via insertion sort.
__device__ void quickselect_sort(Candidate *arr, int n, int k) {
    int l = 0, r = n - 1;
    while (l < r) {
        int pivotIndex = l + (r - l) / 2;
        float pivotValue = arr[pivotIndex].dist;
        swapCandidate(arr[pivotIndex], arr[r]);
        int storeIndex = l;
        for (int i = l; i < r; i++) {
            if (arr[i].dist < pivotValue) {
                swapCandidate(arr[i], arr[storeIndex]);
                storeIndex++;
            }
        }
        swapCandidate(arr[storeIndex], arr[r]);
        if (storeIndex == k)
            break;
        else if (storeIndex < k)
            l = storeIndex + 1;
        else
            r = storeIndex - 1;
    }
    // Insertion sort on the first k elements.
    for (int i = 1; i < k; i++) {
        Candidate key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].dist > key.dist) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// flush_merge() merges the current intermediate per-warp candidate list
// (stored in registers, distributed over 32 threads) and the candidate buffer
// (stored in shared memory) when the buffer is full OR when final flushing is needed.
// It writes the combined (and sorted) k best candidates back into the candidate buffer,
// then reloads them into the per-thread intermediate registers and updates the
// current threshold (largest distance among the k candidates).
// Parameters:
//   intermediate  - per-thread candidate array in registers (size L = k/32).
//   L             - number of candidates per thread (k must be divisible by 32).
//   k             - total number of candidates per warp.
//   warp_id       - warp's index within the block.
//   lane          - thread lane id within the warp (0..31).
//   warpMerge     - pointer to the merge buffer in shared memory.
//   candBuffer    - pointer to the candidate buffer in shared memory.
//   mergeBase     - base offset (index) in warpMerge for this warp (size = 2*k).
//   candBufferBase- base offset (index) in candBuffer for this warp (size = k).
//   warpCandCount - pointer to an array in shared memory that holds the candidate count per warp.
//   currentThreshold - reference to the per-warp threshold value (largest distance among current candidates).
__device__ void flush_merge(Candidate *intermediate, int L, int k, int warp_id, int lane,
/// @FIXED
///                               volatile Candidate *warpMerge, volatile Candidate *candBuffer,
                              Candidate *warpMerge, Candidate *candBuffer,
                              int mergeBase, int candBufferBase, int *warpCandCount, float &currentThreshold) {
    // Step 1: Each thread writes its intermediate candidates to the merge buffer.
    #pragma unroll
    for (int i = 0; i < L; i++) {
        warpMerge[mergeBase + lane * L + i] = intermediate[i];
    }
    __syncwarp();
    // Step 2: Lane 0 pads the candidate buffer (if needed) and copies it into the second half of warpMerge.
    if (lane == 0) {
        int candCount = warpCandCount[warp_id];
        for (int i = candCount; i < k; i++) {
            candBuffer[candBufferBase + i].index = -1;
            candBuffer[candBufferBase + i].dist = FLT_MAX;
        }
        for (int i = 0; i < k; i++) {
            warpMerge[mergeBase + k + i] = candBuffer[candBufferBase + i];
        }
    }
    __syncwarp();
    // Step 3: Lane 0 calls quickselect_sort to select the best k candidates over 2*k entries,
    // then writes the sorted k candidates back into the candidate buffer.
    if (lane == 0) {
        quickselect_sort((Candidate *)(warpMerge + mergeBase), 2 * k, k);
        for (int i = 0; i < k; i++) {
            candBuffer[candBufferBase + i] = warpMerge[mergeBase + i];
        }
        // Reset the candidate count for this warp.
        warpCandCount[warp_id] = 0;
    }
    __syncwarp();
    // Step 4: Each thread loads its portion of the new intermediate candidate list from the candidate buffer.
    #pragma unroll
    for (int i = 0; i < L; i++) {
        intermediate[i] = candBuffer[candBufferBase + lane * L + i];
    }
    // Step 5: Each thread computes the maximum distance among its L intermediate candidates,
    // and then a warp-wide reduction obtains the new threshold.
    float local_max = -1.0f;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        if (intermediate[i].dist > local_max)
            local_max = intermediate[i].dist;
    }
    local_max = warpReduceMax(local_max);
    currentThreshold = local_max;
    __syncwarp();
}

// The k-NN kernel.
// Each warp processes one query. Queries are stored in global memory (query array).
// The data points are processed in batches (tiles), which are cached into shared memory.
// Each warp maintains an intermediate candidate list (of k best neighbors so far)
// distributed among its 32 threads (each holds k/32 candidates).
// A candidate buffer in shared memory (per warp) is used to buffer promising candidates
// before merging with the intermediate result.
// After processing all data batches, a final merge flush is performed (if needed),
// and the sorted k nearest neighbors for the query are written to the output.
__global__ void knn_kernel(const float2 * __restrict__ query, int query_count,
                           const float2 * __restrict__ data, int data_count,
                           Pair * __restrict__ result, int k) {
    // Tile size (number of data points loaded per batch).
    const int TILE_SIZE = 1024;
    // Compute number of warps per block.
    int warpsPerBlock = blockDim.x / 32;
    // Compute the global warp index; each warp processes one query.
    int warp_global = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);
    int lane = threadIdx.x % 32;
    if (warp_global >= query_count) return;

    // Each warp's intermediate candidate list contains 'k' elements;
    // distributed evenly across 32 threads, so each thread holds L = k/32 candidates.
    int L = k / 32;

    // Load the query point for this warp.
    float2 q = query[warp_global];

    // Initialize the per-thread intermediate candidate list in registers.
    Candidate intermediate[64]; // Maximum L is <= 32 (for k up to 1024)
    #pragma unroll
    for (int i = 0; i < L; i++) {
        intermediate[i].index = -1;
        intermediate[i].dist = FLT_MAX;
    }
    float currentThreshold = FLT_MAX;

    // Shared memory layout (dynamically allocated):
    // - First region: Tile of data points (float2) of size TILE_SIZE.
    // - Next: Candidate buffer for warps: warpsPerBlock * k Candidate elements.
    // - Next: Merge buffer for warps: warpsPerBlock * 2 * k Candidate elements.
    // - Next: Warp candidate counts: warpsPerBlock integers.
    extern __shared__ char sharedMem[];
    float2 *tilePoints = (float2 *)sharedMem;
    Candidate *candBuffer = (Candidate *)(sharedMem + TILE_SIZE * sizeof(float2));
    Candidate *warpMerge = (Candidate *)(sharedMem + TILE_SIZE * sizeof(float2)
                                          + warpsPerBlock * k * sizeof(Candidate));
    int *warpCandCount = (int *)(sharedMem + TILE_SIZE * sizeof(float2)
                                  + warpsPerBlock * k * sizeof(Candidate)
                                  + warpsPerBlock * 2 * k * sizeof(Candidate));

    // Compute the base offsets for this warp in the candidate buffer and merge buffer.
    int warp_id = threadIdx.x / 32;         // warp index within the block.
    int candBufferBase = warp_id * k;         // starting index for this warp in candBuffer.
    int mergeBase = warp_id * (2 * k);          // starting index for this warp in warpMerge.

    // Initialize the candidate count for this warp to 0 (done by the first lane).
    if (lane == 0) {
        warpCandCount[warp_id] = 0;
    }
    __syncwarp();

    // Process the data points in batches (tiles).
    for (int batchStart = 0; batchStart < data_count; batchStart += TILE_SIZE) {
        int batchSize = min_int(TILE_SIZE, data_count - batchStart);
        // Load a batch of data points into shared memory (tilePoints).
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            tilePoints[i] = data[batchStart + i];
        }
        __syncthreads();

        // Each warp processes data from the tile.
        // Each thread processes a subset of indices (stride = 32 in the warp).
        for (int i = lane; i < batchSize; i += 32) {
            float2 pt = tilePoints[i];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float dist = dx * dx + dy * dy;

            // If the candidate is promising (closer than the current worst neighbor),
            // insert it into the candidate buffer.
            if (dist < currentThreshold) {
                Candidate cand;
                cand.index = batchStart + i;  // Global data index.
                cand.dist = dist;
                int pos = atomicAdd(&warpCandCount[warp_id], 1);
                if (pos < k) {
                    candBuffer[candBufferBase + pos] = cand;
                }
                // If the candidate buffer is now full, flush and merge.
                if (pos == k - 1) {
                    flush_merge(intermediate, L, k, warp_id, lane, warpMerge,
                                candBuffer, mergeBase, candBufferBase, warpCandCount, currentThreshold);
                }
            }
        }
        __syncthreads();
    } // end for each batch

    // After all batches, if the candidate buffer is not empty, perform a final flush merge.
    int remaining = warpCandCount[warp_id];
    remaining = __shfl_sync(0xffffffff, remaining, 0);
    if (remaining > 0) {
        flush_merge(intermediate, L, k, warp_id, lane, warpMerge,
                    candBuffer, mergeBase, candBufferBase, warpCandCount, currentThreshold);
    }

    // Write the final intermediate candidate list (k sorted nearest neighbors)
    // from the per-thread registers to the global output.
    // Each warp writes out its k candidates to result[warp_global * k ...].
    for (int i = 0; i < L; i++) {
        int outIndex = warp_global * k + lane * L + i;
        result[outIndex].first = intermediate[i].index;
        result[outIndex].second = intermediate[i].dist;
    }
}

// Host function that launches the k-NN kernel.
// query: pointer to an array of query points (float2)
// query_count: number of queries
// data: pointer to an array of data points (float2)
// data_count: number of data points (>= k)
// result: pointer to an output array (std::pair<int, float>) where each query has k results
// k: number of nearest neighbors to compute (power-of-2 between 32 and 1024)
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k) {
    // Choose a block size: 128 threads per block (4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query; thus, the total number of warps needed equals query_count.
    int numWarps = query_count;
    int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Compute the shared memory size per block.
    // Layout:
    // - TILE_SIZE float2 elements for the data tile.
    // - warpsPerBlock * k Candidate elements for the candidate buffer.
    // - warpsPerBlock * 2 * k Candidate elements for the merge buffer.
    // - warpsPerBlock integers for candidate counts.
    const int TILE_SIZE = 1024;
    size_t shmem = 0;
    shmem += TILE_SIZE * sizeof(float2);
    shmem += warpsPerBlock * k * sizeof(Candidate);
    shmem += warpsPerBlock * 2 * k * sizeof(Candidate);
    shmem += warpsPerBlock * sizeof(int);

    // Launch the kernel. The candidate result array is reinterpreted as a pointer to Pair,
    // which is layout-compatible with std::pair<int, float>.
    knn_kernel<<<numBlocks, threadsPerBlock, shmem>>>(query, query_count, data, data_count,
                                                      reinterpret_cast<Pair*>(result), k);
    cudaDeviceSynchronize();
}
