#include <cuda_runtime.h>
#include <cfloat>

//------------------------------------------------------------------------------
// This struct encodes a (data_index, squared_distance) pair
// that represents one candidate neighbor.
struct ResultPair {
    int index;
    float dist;
};

//------------------------------------------------------------------------------
// Device helper function: Compute the worst (largest) squared distance among
// a warp's current best list.  In our design, each warp’s best list is stored
// in registers partitioned among warp lanes. Each lane holds "local_count"
// candidates.  This function computes the maximum distance over the candidates
// held by the calling lane and then reduces across the warp.
__device__ float compute_warp_threshold(const ResultPair* my_best, int local_count) {
    float local_max = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < local_count; i++) {
        if (my_best[i].dist > local_max)
            local_max = my_best[i].dist;
    }
    // Warp-level reduction using shuffle to get the maximum distance.
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    return local_max;
}

//------------------------------------------------------------------------------
// Device helper function: Merge the warp's current best list (stored in registers)
// with the candidate buffer (stored in shared memory) that has recently accumulated
// candidates from the data points. The union of these two arrays (of size k + candidate_count)
// is then sorted in ascending order (by distance) using a parallel selection approach,
// and the k best (smallest) candidates are distributed back into the best_list registers.
//
// Parameters:
//   best_local    - each warp lane's local portion of the best list (size = local_count = k/32)
//   k             - total number of best candidates per query
//   local_count   - number of candidates held per warp lane (k/32)
//   candBuffer    - pointer to the candidate buffer (shared memory) of size k for this warp
//   candCount     - number of candidates in candBuffer (may be less than or equal to k)
//   mergeBuffer   - pointer to a merge scratch buffer (shared memory) of size at least (k + candCount)
//   lane          - lane index within the warp (0-31)
__device__ void merge_warp(ResultPair* best_local, int k, int local_count,
                             ResultPair* candBuffer, int candCount,
                             ResultPair* mergeBuffer, int lane) {
    // Step 1: Scatter the best list from registers into mergeBuffer.
    // The best list is distributed: each lane writes its local best candidates into
    // mergeBuffer at positions: lane + 32*i, for i from 0 to local_count-1.
    for (int i = 0; i < local_count; i++) {
        int pos = lane + 32 * i;
        mergeBuffer[pos] = best_local[i];
    }
    // Now, mergeBuffer[0 ... k-1] holds the current best list (k elements).

    // Step 2: Append the candidate buffer (new candidates) into mergeBuffer.
    // Copy candidate buffer elements into mergeBuffer starting at index k.
    int unionSize = k + candCount;  // total number of union elements
    for (int i = lane; i < candCount; i += 32) {
        mergeBuffer[k + i] = candBuffer[i];
    }
    __syncwarp();

    // Step 3: Select the k smallest elements (in sorted order) from mergeBuffer.
    // We use a parallel selection loop. In each iteration, all threads cooperate
    // to find the global minimum among those not yet selected (by marking selected entries with FLT_MAX).
    // The sorted result will be stored into candBuffer (reusing the candidate buffer as temporary output).
    for (int j = 0; j < k; j++) {
        float localMin = FLT_MAX;
        int localMinIdx = -1;
        // Each thread scans the elements assigned to it (stride = 32) over the union array.
        for (int i = lane; i < unionSize; i += 32) {
            float d = mergeBuffer[i].dist;
            if (d < localMin) {
                localMin = d;
                localMinIdx = i;
            }
        }
        // Reduce within the warp to find the global minimum candidate.
        for (int offset = 16; offset > 0; offset /= 2) {
            float otherMin = __shfl_down_sync(0xFFFFFFFF, localMin, offset);
            int otherIdx = __shfl_down_sync(0xFFFFFFFF, localMinIdx, offset);
            if (otherMin < localMin) {
                localMin = otherMin;
                localMinIdx = otherIdx;
            }
        }
        // Lane 0 writes the found minimum element into the sorted output (stored in candBuffer).
        if (lane == 0) {
            candBuffer[j] = mergeBuffer[localMinIdx];
        }
        // Mark the selected element in mergeBuffer as "used" by setting its distance to FLT_MAX.
        if (lane == 0) {
            mergeBuffer[localMinIdx].dist = FLT_MAX;
        }
        __syncwarp();
    }
    // Now, candBuffer[0 ... k-1] holds the sorted best list (in ascending order).
    // Step 4: Gather the sorted results back into the best_local registers.
    int numPerLane = k / 32;
    for (int i = 0; i < numPerLane; i++) {
        int pos = lane + 32 * i;
        best_local[i] = candBuffer[pos];
    }
    __syncwarp();
}

//------------------------------------------------------------------------------
// The main k-NN kernel.  Each warp processes one query point.
// The kernel processes the data points iteratively in batches. Each batch is loaded
// into shared memory, and each warp computes distances from its own query point to the
// data points in the batch. Candidates that are closer (i.e. with smaller squared distances)
// than the current worst (k-th best) are added to a per-warp candidate buffer in shared memory.
// Whenever this buffer fills (or at the end of processing) a merge is performed between the
// candidate buffer and the warp's current best list (held in registers).
/// @FIXED
/// extern "C" __global__ void knn_kernel(
__global__ void knn_kernel(
    const float2* query, int query_count,
    const float2* data, int data_count,
    ResultPair* out,  // Output: array of (index,distance) pairs for each query.
    int k)
{
    // Each warp (32 threads) processes one query.
    int warp_id_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int warpsPerBlock = blockDim.x / 32;
    int globalWarpId = blockIdx.x * warpsPerBlock + warp_id_in_block;
    if (globalWarpId >= query_count) return; // Out-of-bound check.

    // Each warp loads its query point into registers.
    float2 q = query[globalWarpId];

    // Define tile size for batched processing of data points.
    const int TILE_SIZE = 1024;  // Hyperparameter: can be tuned for the target GPU.

    // Shared memory layout:
    // 1. A tile of data points (float2[TILE_SIZE]) loaded from global memory.
    // 2. Per-warp buffers: For each warp, we allocate:
    //    - A candidate buffer of size k (ResultPair), used for accumulating candidate neighbors.
    //    - A merge (scratch) buffer of size 2*k (ResultPair), used during the candidate merge.
    //    - An integer candidate count.
    //
    // The overall shared memory size is computed by the host.
    extern __shared__ char smem[];

    // Pointer to the tile of data points.
    float2* tile = (float2*) smem;

    // Compute offset (in bytes) for per-warp buffers.
    // Each warp gets: candidate buffer (k * sizeof(ResultPair)) +
    //                merge buffer (2*k * sizeof(ResultPair)) +
    //                candidate count (sizeof(int)).
    int perWarpBytes = 3 * k * sizeof(ResultPair) + sizeof(int);
    // Compute pointer to this warp's buffer block.
    char* warpBase = smem + TILE_SIZE * sizeof(float2) + warp_id_in_block * perWarpBytes;
    // Candidate buffer pointer.
    ResultPair* candidateBuffer = (ResultPair*) warpBase;
    // Merge buffer pointer (used as scratch space during merge).
    ResultPair* mergeBuffer = (ResultPair*) (warpBase + k * sizeof(ResultPair));
    // Candidate count pointer.
    int* candCountPtr = (int*) (warpBase + k * sizeof(ResultPair) + 2 * k * sizeof(ResultPair));

    // Initialize candidate buffer count to 0 (done by lane 0).
    if (lane == 0)
        *candCountPtr = 0;
    __syncwarp();

    // Each warp maintains a private copy (in registers) of its best k neighbors.
    // The best list is partitioned evenly among warp lanes.
    // Each lane holds local_count = k/32 candidates.
    int local_count = k / 32;
    ResultPair best_local[32];  // Maximum local_count is k/32 (k max is 1024 => 1024/32 = 32).
    #pragma unroll
    for (int i = 0; i < local_count; i++) {
        best_local[i].index = -1;
        best_local[i].dist = FLT_MAX;
    }
    // Initially, the current threshold is FLT_MAX.
    float currentThreshold = FLT_MAX;

    // Process the data in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += TILE_SIZE) {
        // Determine the number of points in this batch.
        int batchSize = TILE_SIZE;
        if (batchStart + TILE_SIZE > data_count)
            batchSize = data_count - batchStart;

        // Load the current tile of data points from global memory into shared memory.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            tile[i] = data[batchStart + i];
        }
        __syncthreads();

        // Each warp processes the tile.
        // Each warp thread iterates over the data points in the tile in a strided manner.
        for (int i = lane; i < batchSize; i += 32) {
            float2 pt = tile[i];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float d = dx * dx + dy * dy;

            // Only consider candidates that are closer than the current worst neighbor.
            if (d < currentThreshold) {
                // Atomically obtain a slot in the candidate buffer (shared among the warp).
                int pos = atomicAdd(candCountPtr, 1);
                if (pos < k) {
                    candidateBuffer[pos].index = batchStart + i; // Global index into data.
                    candidateBuffer[pos].dist = d;
                }
                // If the candidate buffer has become full, merge it with the best list.
                __syncwarp();
                if (lane == 0 && *candCountPtr >= k) {
                    // Merge: union the best list (k elements) and candidate buffer (k new candidates)
                    merge_warp(best_local, k, local_count, candidateBuffer, k, mergeBuffer, lane);
                    // Reset candidate buffer count.
                    *candCountPtr = 0;
                    // Update the current threshold from the newly merged best list.
                    currentThreshold = compute_warp_threshold(best_local, local_count);
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // After processing all batches, perform a final merge if there are remaining candidates.
    __syncwarp();
    if (*candCountPtr > 0) {
        merge_warp(best_local, k, local_count, candidateBuffer, *candCountPtr, mergeBuffer, lane);
        *candCountPtr = 0;
        currentThreshold = compute_warp_threshold(best_local, local_count);
    } else {
        // Even if no new candidates were added, perform a merge to ensure the best list is sorted.
        merge_warp(best_local, k, local_count, candidateBuffer, 0, mergeBuffer, lane);
    }
    __syncwarp();

    // Write the final sorted best list to global memory.
    // The final, sorted k nearest neighbors for query "globalWarpId" are stored in candidateBuffer.
    // They are written in row-major order: result[globalWarpId * k + j] for j = 0 ... k-1.
    for (int i = lane; i < k; i += 32) {
        out[globalWarpId * k + i] = candidateBuffer[i];
    }
}

//------------------------------------------------------------------------------
// Host entry-point function.  This function wraps the kernel launch and computes
// appropriate grid/block configuration and shared memory allocation.  It expects that
// the query, data and result arrays have been allocated on the device (e.g., via cudaMalloc).
//
// The function's C++ signature is:
//    void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k);
//
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Choose a fixed number of threads per block (must be a multiple of 32).
    int threadsPerBlock = 128;  // 128 threads -> 4 warps per block. This is a tunable hyper-parameter.
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query. Therefore, the number of warps per grid is >= query_count.
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Define TILE_SIZE for batched loading.
    const int TILE_SIZE = 1024;

    // Compute the size of shared memory required:
    // Shared memory layout:
    // 1. TILE_SIZE float2 values for the data tile.
    // 2. Per-warp buffers: For each warp, we require:
    //       - Candidate buffer: k * sizeof(ResultPair)
    //       - Merge buffer: 2*k * sizeof(ResultPair)
    //       - Candidate count: sizeof(int)
    // Total per warp = (3*k * sizeof(ResultPair) + sizeof(int))
    size_t sharedMemSize = TILE_SIZE * sizeof(float2) +
        warpsPerBlock * (3 * k * sizeof(ResultPair) + sizeof(int));

    // Launch the kernel. Cast the result pointer to ResultPair* since its layout matches std::pair<int,float>.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        query, query_count, data, data_count, (ResultPair*)result, k);
    // It is assumed that error checking and synchronization are handled by the caller.
}
