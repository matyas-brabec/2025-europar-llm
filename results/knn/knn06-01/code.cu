#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// The Candidate structure represents a neighbor candidate: its index and squared distance.
struct Candidate {
    int idx;
    float dist;
};

// Insertion sort for a small array of Candidates in shared memory.
// This function sorts 'n' elements in 'data' in ascending order of .dist.
__device__ void insertion_sort(Candidate* data, int n) {
    for (int i = 1; i < n; i++) {
        Candidate temp = data[i];
        int j = i - 1;
        // Shift larger elements to the right.
        while (j >= 0 && data[j].dist > temp.dist) {
            data[j+1] = data[j];
            j--;
        }
        data[j+1] = temp;
    }
}

// Kernel implementing k-NN for 2D points.
// Each warp (32 threads) processes one query. It maintains a private (register/local) sorted
// list of k best neighbors (intermediate result) and a candidate buffer (in shared memory)
// to store potential candidates until the buffer is full. When full, the candidate buffer is merged
// with the private result.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Define a constant batch size (number of data points loaded per batch).
    constexpr int BATCH_SIZE = 1024;
    // Each warp processes one query.
    int warpId_global = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (warpId_global >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[warpId_global];

    // Compute warp-local ID within the block.
    int warpId_in_block = threadIdx.x / 32;

    // Each warp will maintain a private sorted list (the intermediate result) of k candidates.
    // We distribute the k elements evenly across the 32 lanes.
    const int chunk = k / 32;  // k is assumed to be a power of two between 32 and 1024.
    // Private candidate list stored in registers (per lane holds 'chunk' entries).
    /// @FIXED
    /// Candidate priv[chunk];
    Candidate priv[/*MAX_K*/1024 / 32];  // Each thread holds k/32 candidates.
#pragma unroll
    for (int i = 0; i < chunk; i++) {
        // Initialize with "infinite" distance and an invalid index.
        priv[i].idx = -1;
        priv[i].dist = FLT_MAX;
    }
    // max_distance is the distance of the worst (k-th) neighbor in the sorted list.
    float max_distance = FLT_MAX;

    // We now allocate per-block shared memory and partition it among warps.
    // The dynamic shared memory layout is as follows:
    //   [0, offset1): Batch data buffer for data points (BATCH_SIZE float2)
    //   [offset1, offset2): Candidate buffers for each warp, each of size k Candidates.
    //   [offset2, offset3): Merge buffers for each warp, each of size 2*k Candidates.
    //   [offset3, end): Candidate count for each warp (one int per warp).
    extern __shared__ char smem[];
    int offset = 0;
    // Batch buffer for data points.
    float2* batch = reinterpret_cast<float2*>(smem + offset);
    offset += BATCH_SIZE * sizeof(float2);
    // Number of warps per block.
    int warpsPerBlock = blockDim.x / 32;
    // Candidate buffer: each warp has a buffer of k Candidates.
    Candidate* candBufferBase = reinterpret_cast<Candidate*>(smem + offset);
    Candidate* warpCandBuffer = candBufferBase + warpId_in_block * k;
    offset += warpsPerBlock * k * sizeof(Candidate);
    // Merge buffer: each warp has a scratch buffer of 2*k Candidates.
    Candidate* mergeBufferBase = reinterpret_cast<Candidate*>(smem + offset);
    Candidate* warpMergeBuffer = mergeBufferBase + warpId_in_block * (2 * k);
    offset += warpsPerBlock * (2 * k) * sizeof(Candidate);
    // Candidate count buffer: one int per warp.
    int* countBufferBase = reinterpret_cast<int*>(smem + offset);
    int* warpCandCount = countBufferBase + warpId_in_block;
    // Initialize candidate count (only one lane per warp does it).
    if (lane == 0)
        *warpCandCount = 0;
    __syncwarp();

    // Process the data in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += BATCH_SIZE) {
        // Load a batch of data points from global memory into shared memory.
        // Use all threads in the block (not just the warp) to load.
        for (int i = threadIdx.x; i < BATCH_SIZE && (batchStart + i) < data_count; i += blockDim.x) {
            batch[i] = data[batchStart + i];
        }
        __syncthreads();  // Ensure the batch is loaded.

        // Each warp processes the batch: each lane processes a subset of the batch.
        for (int i = lane; i < BATCH_SIZE && (batchStart + i) < data_count; i += 32) {
            float2 pt = batch[i];
            float dx = pt.x - q.x;
            float dy = pt.y - q.y;
            float d = dx * dx + dy * dy;
            // If the computed squared distance is smaller than our current threshold,
            // then it is a candidate.
            if (d < max_distance) {
                // Atomically get a position in the warp's candidate buffer.
                int pos = atomicAdd(warpCandCount, 1);
                if (pos < k) {
                    warpCandBuffer[pos].idx = batchStart + i;  // Global index of this data point.
                    warpCandBuffer[pos].dist = d;
                }
                // If pos >= k, we ignore the candidate.
            }
        }
        __syncwarp();  // Synchronize threads within the warp.

        // If the candidate buffer is full (or overfull), merge it with the private result.
        int curCount = *warpCandCount;
        if (curCount >= k) {
            // We will merge exactly k candidates from the candidate buffer with our current private result.
            // Step 1: Copy the private result to the merge buffer.
            // The private result is distributed among the 32 lanes.
            for (int i = 0; i < chunk; i++) {
                int pos = lane + i * 32;
                if (pos < k)
                    warpMergeBuffer[pos] = priv[i];
            }
            // Step 2: Copy the candidate buffer into the second half of the merge buffer.
            // If the candidate buffer holds less than k valid entries (should not happen here,
            // because we only merge when count>=k), then fill the remainder with FLT_MAX.
            for (int i = lane; i < k; i += 32) {
                Candidate cand = warpCandBuffer[i];
                if (i >= curCount) {
                    cand.idx = -1;
                    cand.dist = FLT_MAX;
                }
                warpMergeBuffer[k + i] = cand;
            }
            __syncwarp();

            // Step 3: Sort the merge buffer (of 2*k candidates) using a simple insertion sort.
            // For simplicity, we let lane 0 perform the sort serially.
            if (lane == 0) {
                int mergeCount = 2 * k;
                insertion_sort(warpMergeBuffer, mergeCount);
            }
            __syncwarp();

            // Step 4: The first k elements in the merge buffer become the new private result.
            for (int i = 0; i < chunk; i++) {
                int pos = lane + i * 32;
                if (pos < k)
                    priv[i] = warpMergeBuffer[pos];
            }
            __syncwarp();

            // Step 5: Update max_distance from the k-th best candidate.
            // The k-th candidate is at index (k-1). Determine which lane holds that.
            int kthLane = (k - 1) & 31;
            int kthIndex = (k - 1) / 32;
            if (lane == kthLane)
                max_distance = priv[kthIndex].dist;
            max_distance = __shfl_sync(0xFFFFFFFF, max_distance, kthLane);
            // Reset candidate buffer count for the next round.
            if (lane == 0)
                *warpCandCount = 0;
            __syncwarp();
        }
        __syncthreads(); // Ensure all block threads complete before loading next batch.
    } // end for each batch

    // After all batches, if any candidates remain in the candidate buffer, merge them.
    int finalCount = *warpCandCount;
    if (finalCount > 0) {
        // Merge private result with remaining candidates.
        // Copy private result into merge buffer.
        for (int i = 0; i < chunk; i++) {
            int pos = lane + i * 32;
            if (pos < k)
                warpMergeBuffer[pos] = priv[i];
        }
        // Copy the candidate buffer in positions [k, 2*k). Fill unused positions (if any) with FLT_MAX.
        for (int i = lane; i < k; i += 32) {
            Candidate cand = warpCandBuffer[i];
            if (i >= finalCount) {
                cand.idx = -1;
                cand.dist = FLT_MAX;
            }
            warpMergeBuffer[k + i] = cand;
        }
        __syncwarp();
        if (lane == 0) {
            int mergeCount = 2 * k;
            insertion_sort(warpMergeBuffer, mergeCount);
        }
        __syncwarp();
        // Update private result with the best k candidates.
        for (int i = 0; i < chunk; i++) {
            int pos = lane + i * 32;
            if (pos < k)
                priv[i] = warpMergeBuffer[pos];
        }
        __syncwarp();
        // Update max_distance.
        int kthLane = (k - 1) & 31;
        int kthIndex = (k - 1) / 32;
        if (lane == kthLane)
            max_distance = priv[kthIndex].dist;
        max_distance = __shfl_sync(0xFFFFFFFF, max_distance, kthLane);
    }

    // Finally, write the private sorted candidate list (the k nearest neighbors) to global memory.
    // Each query's results are stored contiguously in row-major order.
    int outBase = warpId_global * k;
    for (int i = 0; i < chunk; i++) {
        int pos = lane + i * 32;
        if (pos < k) {
            // Use the pair (index, distance) representation.
            result[outBase + pos] = std::pair<int, float>(priv[i].idx, priv[i].dist);
        }
    }
}

// Host interface function. This function configures and launches the kernel.
// It assumes that 'query', 'data', and 'result' have been allocated in device memory (via cudaMalloc)
// and that the input parameters are valid.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose the number of warps per block.
    // We choose 4 warps per block (i.e. 128 threads per block) as a reasonable heuristic.
    const int warpsPerBlock = 4;
    const int threadsPerBlock = warpsPerBlock * 32;
    // The total number of warps needed is one per query.
    int totalWarps = query_count;
    int blocks = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Compute shared memory size required.
    // Layout:
    //   batch buffer: BATCH_SIZE * sizeof(float2)
    //   candidate buffers: (warpsPerBlock * k * sizeof(Candidate))
    //   merge buffers: (warpsPerBlock * (2*k) * sizeof(Candidate))
    //   candidate count: (warpsPerBlock * sizeof(int))
    constexpr int BATCH_SIZE = 1024;
    size_t sharedMemSize = BATCH_SIZE * sizeof(float2)
                           + warpsPerBlock * k * sizeof(Candidate)
                           + warpsPerBlock * (2 * k) * sizeof(Candidate)
                           + warpsPerBlock * sizeof(int);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        query, query_count, data, data_count, result, k);

    // It is assumed that error checking and synchronization are handled by the caller.
    // For example, one might call cudaDeviceSynchronize() after this.
}
