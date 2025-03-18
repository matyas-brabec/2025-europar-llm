#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// -----------------------------------------------------------------------------
// This file implements a CUDA kernel for k‐nearest neighbors (k-NN) search in 2D.
// For each query point (float2), the kernel finds its k nearest neighbors among
// a large set of data points (float2) using the squared Euclidean distance. 
//
// The design assigns one warp (32 threads) per query. Each warp processes the 
// entire data set iteratively in batches. In each batch the block loads data 
// points into shared memory. Then each warp’s threads compute distances for 
// points in the batch, using a simple local buffering scheme to collect a few 
// candidate distances that are “good” (i.e. less than the current threshold).
//
// The per–query intermediate candidate list (of size k) is maintained in shared 
// memory (one contiguous segment per warp) and is updated (by lane 0 of the warp)
// with candidate updates from all lanes once per batch. (For simplicity and to 
// still leverage warp‐communication we let each thread in the warp compute a 
// local “buffer” of candidate results and then lane 0 collects them using warp 
// shuffles.)  This candidate list is kept sorted (best candidates at low indices)
// so that the “threshold” (largest candidate in top k, at index k–1) is always 
// available for pruning.
// 
// The final candidate list for each query is then written to global memory in 
// sorted order (lowest distance first).  Input arrays and result arrays are 
// assumed to have been allocated by cudaMalloc.
// 
// Hyper-parameters used in this implementation:
//    BATCH_SIZE = 256     -- number of data points processed in one batch.
//    LOCAL_BUFFER_SIZE = 4  -- each thread collects up to 4 candidate results per batch.
//    Block size = 128 threads (4 warps per block).
//
// NOTES:
//  - Each warp (32 threads) is assigned one query point.
//  - Shared memory is used to cache the batch of data points as well as to store 
//    each warp's candidate list (of k results) and its current threshold.
//  - Warp-level synchronization uses __syncwarp() and __shfl_sync().
// -----------------------------------------------------------------------------

// Kernel: one warp per query.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result,
                           int k)
{
    // Define batch size and local buffer size.
    const int BATCH_SIZE = 256;
    const int LOCAL_BUFFER_SIZE = 4; // Each thread collects up to 4 candidates per batch.

    // Determine warp ID and lane ID.
    int warpIdInBlock = threadIdx.x / 32;     // Which warp in the block.
    int lane = threadIdx.x % 32;                // Lane index within the warp.
    // Global warp id, each warp is assigned one query.
    int warpsPerBlock = blockDim.x / 32;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (globalWarpId >= query_count)
        return; // No query assigned.

    // Load the query point for this warp.
    float2 q = query[globalWarpId];

    // -------------------------------------------------------------------------
    // Shared memory layout (per block):
    //    1. Batch cache for data points: sdata, array of BATCH_SIZE float2
    //       (stored as 2*BATCH_SIZE floats).
    //    2. Warp candidate lists: for each warp in the block, we reserve k floats 
    //       for distances and k ints for indices.
    //    3. Warp thresholds: for each warp, one float holding the worst (largest)
    //       distance in its current candidate list.
    //
    // All arrays are packed consecutively.
    // -------------------------------------------------------------------------
    extern __shared__ char smem[];
    // (1) sdata: BATCH_SIZE float2 => 2*BATCH_SIZE floats.
    float *sdata = (float*)smem; 
    // Offset (in bytes) for next part:
    int offset = BATCH_SIZE * 2 * sizeof(float);
    // (2) Warp candidate distances: numWarps * k floats.
    float *warp_knn_dists = (float*)(smem + offset);
    offset += warpsPerBlock * k * sizeof(float);
    // (2) Warp candidate indices: numWarps * k ints.
    int *warp_knn_idxs = (int*)(smem + offset);
    offset += warpsPerBlock * k * sizeof(int);
    // (3) Warp thresholds: one float per warp.
    float *warp_thresholds = (float*)(smem + offset);
    // For this warp, pointers to its candidate list in shared memory:
    float *candDists = &warp_knn_dists[warpIdInBlock * k];
    int   *candIdxs  = &warp_knn_idxs[warpIdInBlock * k];

    // Initialize the candidate list and threshold.
    if(lane == 0) {
        // Initialize candidate list with "infinite" distances.
        for (int i = 0; i < k; i++) {
            candDists[i] = FLT_MAX;
            candIdxs[i] = -1;
        }
        // The current threshold is the worst candidate: index k-1.
        warp_thresholds[warpIdInBlock] = FLT_MAX;
    }
    // Make sure all threads see the initialized candidate list.
    __syncwarp();

    // -------------------------------------------------------------------------
    // Process all data points in batches.
    // Each batch is loaded from global memory into sdata.
    // -------------------------------------------------------------------------
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {

        // Load current batch of data points from global memory to shared memory.
        int idx = batch_start + threadIdx.x;
        if (idx < data_count) {
            float2 pt = data[idx];
            int pos = (idx - batch_start) * 2;
            sdata[pos]     = pt.x;
            sdata[pos + 1] = pt.y;
        }
        // (Optionally, threads with idx >= data_count could write dummy values, but not required.)
        __syncthreads(); // Ensure batch has been loaded.

        // Determine number of valid points in this batch.
        int current_batch = (data_count - batch_start > BATCH_SIZE) ? BATCH_SIZE : (data_count - batch_start);

        // Load current warp threshold, used for pruning.
        // Only lane 0 holds the latest updated threshold; broadcast to all lanes.
        float current_threshold = warp_thresholds[warpIdInBlock];
        current_threshold = __shfl_sync(0xffffffff, current_threshold, 0);

        // Each thread in the warp processes a subset of the batch.
        // We use a small local buffer (of size LOCAL_BUFFER_SIZE) per thread to store 
        // candidate distances and indices from this batch.
        float localBuf[LOCAL_BUFFER_SIZE];
        int   localIdxBuf[LOCAL_BUFFER_SIZE];
        int localCount = 0;
        // Initialize local buffer with "infinite" candidates.
        for (int i = 0; i < LOCAL_BUFFER_SIZE; i++) {
            localBuf[i] = FLT_MAX;
            localIdxBuf[i] = -1;
        }

        // Loop over data points in the batch (stride = warp size).
        for (int i = lane; i < current_batch; i += 32) {
            // Read point from shared memory.
            float px = sdata[2*i];
            float py = sdata[2*i + 1];
            // Compute squared Euclidean distance.
            float dx = q.x - px;
            float dy = q.y - py;
            float dist = dx*dx + dy*dy;
            // Prune if not better than current threshold.
            if (dist >= current_threshold)
                continue;
            // Insert candidate (dist, global index) into local buffer (sorted insertion).
            // Linear scan to find insertion position.
            int pos = localCount; // default: at end.
            for (int j = 0; j < localCount; j++) {
                if (dist < localBuf[j]) { 
                    pos = j;
                    break;
                }
            }
            // If buffer not full, increase localCount.
            if (localCount < LOCAL_BUFFER_SIZE) {
                for (int j = localCount; j > pos; j--) {
                    localBuf[j] = localBuf[j-1];
                    localIdxBuf[j] = localIdxBuf[j-1];
                }
                localBuf[pos] = dist;
                localIdxBuf[pos] = batch_start + i; // global index of the candidate.
                localCount++;
            }
            else { // Buffer full: check if new candidate beats the worst candidate.
                if (dist < localBuf[LOCAL_BUFFER_SIZE - 1]) {
                    // Insert candidate in sorted order.
                    for (int j = LOCAL_BUFFER_SIZE - 1; j > pos; j--) {
                        localBuf[j] = localBuf[j-1];
                        localIdxBuf[j] = localIdxBuf[j-1];
                    }
                    localBuf[pos] = dist;
                    localIdxBuf[pos] = batch_start + i;
                }
            }
        } // end loop over batch

        // Now, each thread has up to LOCAL_BUFFER_SIZE candidate pairs from this batch.
        // We now update the global (warp) candidate list with these candidates.
        // Let lane 0 of the warp collect all candidate entries from every lane using
        // warp shuffle.
        if (lane == 0) {
            // For each lane in the warp (there are 32 lanes), fetch its LOCAL_BUFFER_SIZE candidates.
            for (int src = 0; src < 32; src++) {
                // For each candidate in the local buffer of lane "src".
                for (int j = 0; j < LOCAL_BUFFER_SIZE; j++) {
                    // Use __shfl_sync to retrieve candidate distance and index from lane src.
                    float cand = __shfl_sync(0xffffffff, localBuf[j], src);
                    int candIdx = __shfl_sync(0xffffffff, localIdxBuf[j], src);
                    // Only update if candidate is valid and improves the global candidate list.
                    if (cand < FLT_MAX && cand < warp_thresholds[warpIdInBlock]) {
                        // Update procedure:
                        // If the candidate is better than the worst in the candidate list
                        // (stored in candDists[k-1]), then insert it in sorted order.
                        if (cand < candDists[k - 1]) {
                            int pos = k - 1;
                            // Linear search backward until the correct insertion location is found.
                            while (pos > 0 && cand < candDists[pos - 1]) {
                                // Shift candidate list element right.
                                candDists[pos] = candDists[pos - 1];
                                candIdxs[pos] = candIdxs[pos - 1];
                                pos--;
                            }
                            candDists[pos] = cand;
                            candIdxs[pos] = candIdx;
                            // Update the stored threshold (worst candidate).
                            warp_thresholds[warpIdInBlock] = candDists[k - 1];
                        }
                    }
                }
            }
        }
        // Synchronize the warp so that candidate list updates are visible before next batch.
        __syncwarp();
        __syncthreads(); // Also synchronize block since shared memory (sdata) will be reused.
    } // end for each batch

    // After processing all batches, lane 0 of the warp writes the final sorted candidate list
    // (k nearest neighbors) to global memory for its query.
    if (lane == 0) {
        int queryIdx = globalWarpId;
        for (int i = 0; i < k; i++) {
            // Write candidate index and squared distance.
            // Using std::pair's members: first is index, second is distance.
            result[queryIdx * k + i] = std::pair<int, float>(candIdxs[i], candDists[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface function.
// This function configures and launches the knn_kernel.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block configuration: 128 threads per block => 4 warps per block,
    // each warp processes one query.
    const int blockThreads = 128;
    const int warpsPerBlock = blockThreads / 32;
    // Each block processes one query per warp.
    int gridBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute shared memory size per block.
    // Shared memory layout:
    //    (1) sdata: BATCH_SIZE * 2 float => BATCH_SIZE*2*sizeof(float)
    //    (2) warp candidate distances: (warpsPerBlock * k) floats.
    //    (3) warp candidate indices: (warpsPerBlock * k) ints.
    //    (4) warp thresholds: (warpsPerBlock * 1) float.
    const int BATCH_SIZE = 256;
    size_t shmemSize = 0;
    shmemSize += BATCH_SIZE * 2 * sizeof(float);         // sdata
    shmemSize += warpsPerBlock * k * sizeof(float);        // warp candidate distances
    shmemSize += warpsPerBlock * k * sizeof(int);          // warp candidate indices
    shmemSize += warpsPerBlock * sizeof(float);            // warp thresholds

    // Launch the kernel.
    knn_kernel<<< gridBlocks, blockThreads, shmemSize >>>
        (query, query_count, data, data_count, result, k);
    
    // Optionally, one may synchronize and/or check for errors.
    cudaDeviceSynchronize();
}