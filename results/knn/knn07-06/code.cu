#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// In this implementation we assume that each warp (32 threads)
// processes one query. Each warp maintains its own "intermediate result"
// (the best k candidates so far) in registers (distributed among its threads)
// and a candidate buffer in shared memory (of size k) for new candidates.
// When the candidate buffer becomes full, we merge its sorted contents with
// the intermediate result using a Bitonic sort–based merge procedure.

// Constants for shared‐memory batch processing.
#define DATA_CHUNK_SIZE 1024  // number of data points loaded per batch from global memory.
#define WARP_SIZE 32

// Structure representing a neighbor (data index and squared L2 distance).
struct Neighbor {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// Device function: bitonic_sort
// This function sorts an array of 'n' Neighbor elements in ascending order
// (based on dist) using the Bitonic Sort algorithm. It assumes that 'n'
// is a power of two and that all threads in a warp (32 threads) collaborate.
// The array 'data' is stored in shared memory and declared volatile to
// prevent undesired optimizations. Each thread loops over indices stepping by WARP_SIZE.
// -----------------------------------------------------------------------------
/// @FIXED
/// __device__ inline void bitonic_sort(volatile Neighbor *data, int n)
__device__ inline void bitonic_sort(Neighbor *data, int n)
{
    // The algorithm loops over sizes doubling each time.
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes indices: starting at its lane id with stride=WARP_SIZE.
            for (int i = threadIdx.x % WARP_SIZE; i < n; i += WARP_SIZE) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    // Determine the sorting direction.
                    bool ascending = ((i & size) == 0);
                    // Swap if out of order.
                    if ( (ascending && (data[i].dist > data[ixj].dist)) ||
                         (!ascending && (data[i].dist < data[ixj].dist)) ) {
                        Neighbor tmp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = tmp;
                    }
                }
            }
            __syncwarp(); // synchronize the warp.
        }
    }
}

// -----------------------------------------------------------------------------
// Device function: warp_merge_candidates
// This function merges the warp's candidate buffer (already sorted) with its
// intermediate result (stored in registers and then copied to shared memory)
// to update the intermediate result.
// Parameters:
//   k           - number of neighbors (size of intermediate result)
//   kPerThread  - number of elements stored per thread in intermediate result registers (k / 32)
//   lane        - lane index within the warp (0..31)
//   warpId      - warp index within the block
//   warpCand    - pointer to the candidate buffers in shared memory (for all warps)
//   warpMerge   - pointer to temporary merge workspace in shared memory (for all warps)
//   warpCount   - pointer to candidate count array in shared memory (for all warps)
//   localIR     - the warp's intermediate result stored in registers (array of kPerThread elements)
//   localMax    - pointer to the warp's local max_distance (in registers) to be updated
// The merge proceeds as follows:
// 1) The warp's candidate buffer is sorted using bitonic_sort.
// 2) Each thread writes its portion of the intermediate result from registers
//    into the merge workspace.
// 3) Each thread (for indices i in its share) computes a merged candidate:
//      merged[i] = min( intermediate_result[i], candidate_buffer[k - i - 1] )
// 4) The merged array (of size k) is then sorted using bitonic_sort.
// 5) The sorted merged result is distributed back to registers (localIR),
//    and the new max_distance is obtained from the last element.
// 6) The candidate buffer count is reset to 0.
// -----------------------------------------------------------------------------
__device__ inline void warp_merge_candidates(
    int k, int kPerThread, int lane, int warpId,
    Neighbor *warpCand,        // candidate buffers for all warps, size: (numWarps * k)
    Neighbor *warpMerge,       // merge workspace for all warps, size: (numWarps * k)
    int *warpCount,            // candidate counts for all warps, size: (numWarps)
    Neighbor localIR[],        // local intermediate result in registers (size: kPerThread)
    float *localMax            // pointer to local max_distance, to update
)
{
    // Calculate pointers for the current warp in shared memory arrays.
    int warpOffset = warpId * k;
    Neighbor *cand = &warpCand[warpOffset];
    Neighbor *mergeArr = &warpMerge[warpOffset];

    // 1) Sort the candidate buffer.
    bitonic_sort(cand, k);
    __syncwarp();

    // 2) Copy intermediate result (from registers) into mergeArr.
    // The intermediate result is distributed among 32 threads: each thread stores kPerThread elements.
    for (int r = 0; r < kPerThread; r++) {
        int pos = lane + r * WARP_SIZE;  // global index in the warp's intermediate result.
        mergeArr[pos] = localIR[r];
    }
    __syncwarp();

    // 3) Merge: for each index i in [0, k), compute merged candidate.
    // Each thread processes indices starting at its lane index.
    for (int i = lane; i < k; i += WARP_SIZE) {
        // Compare the intermediate candidate at position i and the candidate buffer's
        // element at the mirrored position (k - i - 1). We choose the one with lower distance.
        if (mergeArr[i].dist < cand[k - i - 1].dist)
            cand[i] = mergeArr[i];
        // Otherwise, cand[i] already holds the candidate from the candidate buffer.
        // (We write into cand[i] in place.)
    }
    __syncwarp();

    // 4) Sort the merged candidate buffer.
    bitonic_sort(cand, k);
    __syncwarp();

    // 5) Copy sorted merged candidates back into registers (localIR).
    for (int r = 0; r < kPerThread; r++) {
        int pos = lane + r * WARP_SIZE;
        localIR[r] = cand[pos];
    }
    __syncwarp();

    // 6) Update the local max distance from the global sorted result.
    // The maximum (k-th nearest) element is at index k-1. Determine which lane holds it.
    int maxLane = (k - 1) & (WARP_SIZE - 1); // equivalent to (k-1) % 32.
    float newMax;
    if (lane == maxLane) {
        newMax = localIR[(kPerThread - 1)].dist; // The last element of this thread's array.
    }
    // Broadcast newMax from maxLane to all lanes in the warp.
    newMax = __shfl_sync(0xffffffff, newMax, maxLane);
    *localMax = newMax;

    // 7) Reset the candidate buffer count for this warp.
    if (lane == 0)
        warpCount[warpId] = 0;
    __syncwarp();
}

// -----------------------------------------------------------------------------
// Global kernel: knn_kernel
//
// This kernel implements the k-NN search for 2D queries.
// Each warp processes one query. It loads the query point,
// and then iterates over the data points in batches. Each batch of
// data points is loaded into shared memory and each warp computes the
// squared Euclidean distance of its query to these points.
// If a computed distance is less than the current maximum distance
// of the warp's intermediate result, the candidate is added to a
// warp-specific candidate buffer (in shared memory) via atomicAdd.
// When a candidate buffer is full, it is merged with the intermediate result
// using a Bitonic sort–based merge procedure (described above).
//
// After all batches are processed, a final merge is performed if any
// candidates remain in the candidate buffer. Finally, the final k
// nearest neighbors (sorted in ascending order) are written to the output.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(
    const float2 *query, int query_count,
    const float2 *data, int data_count,
    std::pair<int, float> *result,
    int k)
{
    // Determine warp and lane indices.
    int lane   = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x / WARP_SIZE;  // warp id within the block
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    // Global warp index corresponds one-to-one with a query.
    int globalWarpId = blockIdx.x * warpsPerBlock + warpId;
    if (globalWarpId >= query_count)
        return;

    // Each warp processes one query.
    float2 q = query[globalWarpId];

    // Each warp maintains an intermediate result of size k (sorted in ascending order).
    // We distribute this array among the 32 threads: each thread holds (k / 32) elements.
    int kPerThread = k / WARP_SIZE;
    Neighbor localIR[1024 / WARP_SIZE];  // Maximum k is 1024, so maximum kPerThread is 32.
    // Initialize intermediate result with "empty" candidates (max distance).
    for (int i = 0; i < kPerThread; i++) {
        localIR[i].idx  = -1;
        localIR[i].dist = FLT_MAX;
    }
    // Local maximum distance (the current kth nearest neighbor distance).
    float localMax = FLT_MAX;

    // Shared memory layout:
    // [0, DATA_CHUNK_SIZE * sizeof(float2))            -> data chunk buffer (float2 array)
    // [data_chunk_offset, data_chunk_offset + warpsPerBlock * k * sizeof(Neighbor))  -> candidate buffers for each warp
    // [merge_offset, merge_offset + warpsPerBlock * k * sizeof(Neighbor))             -> merge workspace for each warp
    // [count_offset, count_offset + warpsPerBlock * sizeof(int))                      -> candidate counts for each warp
    extern __shared__ unsigned char shared[];
    // Pointer to data chunk buffer.
    float2 *s_data = (float2*)shared;
    size_t offset = DATA_CHUNK_SIZE * sizeof(float2);
    // Pointer to candidate buffers (for all warps).
    Neighbor *warpCand = (Neighbor*)(shared + offset);
    offset += warpsPerBlock * k * sizeof(Neighbor);
    // Pointer to merge workspace (for all warps).
    Neighbor *warpMerge = (Neighbor*)(shared + offset);
    offset += warpsPerBlock * k * sizeof(Neighbor);
    // Pointer to candidate count for each warp.
    int *warpCount = (int*)(shared + offset);

    // Initialize candidate count for this warp to 0.
    if (lane == 0)
        warpCount[warpId] = 0;
    __syncwarp();

    // Process data points in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += DATA_CHUNK_SIZE) {
        int currBatchSize = DATA_CHUNK_SIZE;
        if (batchStart + currBatchSize > data_count)
            currBatchSize = data_count - batchStart;

        // Load current batch from global memory into shared memory.
        // Use all threads in the block.
        for (int i = threadIdx.x; i < currBatchSize; i += blockDim.x) {
            s_data[i] = data[batchStart + i];
        }
        __syncthreads();

        // Each warp processes the loaded batch.
        for (int i = lane; i < currBatchSize; i += WARP_SIZE) {
            // Load data point from shared memory.
            float2 dp = s_data[i];
            float dx = dp.x - q.x;
            float dy = dp.y - q.y;
            float dist = dx * dx + dy * dy;

            // If this distance is promising, add it to the warp's candidate buffer.
            if (dist < localMax) {
                int globalDataIdx = batchStart + i;
                // Atomically increment candidate count for this warp.
                int pos = atomicAdd(&warpCount[warpId], 1);
                if (pos < k) {
                    int index = warpId * k + pos;
                    warpCand[index].idx  = globalDataIdx;
                    warpCand[index].dist = dist;
                }
                // If the candidate buffer just became full, trigger a merge.
                if (pos == k - 1) {
                    __syncwarp();
                    warp_merge_candidates(k, kPerThread, lane, warpId,
                                          warpCand, warpMerge, warpCount,
                                          localIR, &localMax);
                }
            }
        }
        __syncwarp();
        __syncthreads(); // ensure all threads have finished processing the batch.
    }

    // After processing all batches, if there are any remaining candidates in the candidate buffer,
    // we need to merge them with the intermediate result.
    int currCount = warpCount[warpId];
    if (currCount > 0 && currCount < k) {
        // Pad remaining slots with "empty" candidates.
        for (int i = lane; i < k; i += WARP_SIZE) {
            if (i >= currCount) {
                int index = warpId * k + i;
                warpCand[index].idx  = -1;
                warpCand[index].dist = FLT_MAX;
            }
        }
        __syncwarp();
        if (lane == 0)
            warpCount[warpId] = k;
        __syncwarp();
        warp_merge_candidates(k, kPerThread, lane, warpId,
                              warpCand, warpMerge, warpCount,
                              localIR, &localMax);
    }

    // Write the final k nearest neighbors for this query to global memory.
    // The results are stored in row-major order:
    // For query[globalWarpId], result[globalWarpId * k + j] holds the j-th neighbor.
    for (int r = 0; r < kPerThread; r++) {
        int pos = lane + r * WARP_SIZE; // global index in the merged intermediate array
        int resIndex = globalWarpId * k + pos;
        result[resIndex] = std::make_pair(localIR[r].idx, localIR[r].dist);
    }
}

// -----------------------------------------------------------------------------
// Host function: run_knn
// This function sets up the kernel launch parameters and computes the
// required shared memory size before launching the knn_kernel.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose a block size that is a multiple of the warp size.
    int threadsPerBlock = 256;  // 256 threads => 8 warps per block.
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    // Each warp processes one query. Compute grid size accordingly.
    int totalWarps = (query_count + 0 - 1) / 1;  // one warp per query.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute the shared memory size required per block.
    size_t sharedMemSize = 0;
    // Data chunk buffer: DATA_CHUNK_SIZE float2 elements.
    sharedMemSize += DATA_CHUNK_SIZE * sizeof(float2);
    // Candidate buffers: one per warp in the block, each of size k.
    sharedMemSize += warpsPerBlock * k * sizeof(Neighbor);
    // Merge workspace: one per warp, size k.
    sharedMemSize += warpsPerBlock * k * sizeof(Neighbor);
    // Candidate counts: one int per warp.
    sharedMemSize += warpsPerBlock * sizeof(int);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count,
                                                           data, data_count,
                                                           result, k);
    // It's assumed that the caller checks for errors and synchronizes as needed.
}
