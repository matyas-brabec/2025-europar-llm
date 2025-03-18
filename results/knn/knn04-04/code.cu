#include <cuda_runtime.h>
#include <utility>
#include <limits>
#include <cassert>

// Define an extremely large float value as infinity.
#define INF std::numeric_limits<float>::infinity()

// The batch size for loading data points into shared memory.
// This value was chosen as a trade‐off between shared memory usage and re‐use.
#define BATCH_SIZE 1024

// The CUDA kernel that computes k-nearest neighbors for 2D float2 query points.
// Each warp (32 threads) processes one query point. The intermediate k candidate
// neighbors (index and squared distance) are stored privately across the warp:
// each thread holds k/32 candidates in a small sorted array (in descending order).
// After processing all data points in batches (loaded into shared memory),
// one thread (lane 0) in the warp merges the per‐thread candidate lists and writes
// the final sorted result (in ascending order of distance) into the global output.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Determine warp and lane identifiers.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31; // lane id within warp
    int warpId = globalThreadId >> 5; // warp id

    // Each warp processes one query.
    int queryIndex = warpId;
    if (queryIndex >= query_count) return;

    // Load the query point from global memory.
    float2 q = query[queryIndex];

    // Each warp will maintain 'k' candidates distributed among its 32 lanes.
    // Let local_k = k/32 be the number of candidate entries per thread.
    const int local_k = k >> 5;  // k divided by 32; k is guaranteed to be a power of two between 32 and 1024.

    // Each thread keeps its own candidate list in registers.
    // The candidate lists are maintained sorted in descending order by distance.
    // That is, the 0-th element of each list holds the worst (largest) distance among its entries.
    float local_d[32];    // maximum local size is 1024/32 = 32.
    int   local_idx[32];  // corresponding data point indices.
    #pragma unroll
    for (int i = 0; i < local_k; i++) {
        local_d[i] = INF;
        local_idx[i] = -1;
    }
    // Initially, each thread's worst candidate is at index 0 (value INF).

    // Declare shared memory for a batch of data points.
    // The shared memory buffer has been allocated by the host (BATCH_SIZE * sizeof(float2)).
    extern __shared__ char shared_buffer[];
    float2 *sdata = reinterpret_cast<float2*>(shared_buffer);

    // Process the data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Determine the number of points in this batch.
        int batch_size = BATCH_SIZE;
        if (batch_start + batch_size > data_count)
            batch_size = data_count - batch_start;

        // Load the batch from global memory into shared memory cooperatively.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            sdata[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the entire batch is loaded.

        // Each warp processes the batch.
        // We assign data points from shared memory to warp lanes in a round-robin fashion.
        for (int j = lane; j < batch_size; j += 32) {
            // Load the candidate data point.
            float2 pt = sdata[j];
            // Compute squared Euclidean distance.
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float dist = dx * dx + dy * dy;

            // For efficient pruning, compute the global threshold for this warp.
            // Each lane's candidate list worst value is local_d[0] (largest in its sorted candidate list).
            float myWorst = local_d[0];
            unsigned fullMask = 0xFFFFFFFF;
            // Warp-level reduction to compute the maximum (worst) value among the 32 lanes.
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(fullMask, myWorst, offset);
                myWorst = fmaxf(myWorst, other);
            }
            float globalWorst = __shfl_sync(fullMask, myWorst, 0);

            // If the computed distance is not promising compared to the global worst candidate,
            // skip further processing.
            if (dist >= globalWorst)
                continue;

            // In this lane, check if this candidate improves the local candidate list.
            // Since the list is sorted in descending order (worst is at index 0),
            // if dist is smaller than local worst, then insert it.
            if (dist < local_d[0]) {
                // Replace the worst candidate with the new candidate.
                local_d[0] = dist;
                local_idx[0] = batch_start + j;

                // Bubble the new candidate upward in the sorted order.
                // That is, repeatedly swap with the next element if it is smaller.
                int pos = 0;
                while (pos < local_k - 1 && local_d[pos] < local_d[pos + 1]) {
                    float tmp_d = local_d[pos];
                    local_d[pos] = local_d[pos + 1];
                    local_d[pos + 1] = tmp_d;

                    int tmp_idx = local_idx[pos];
                    local_idx[pos] = local_idx[pos + 1];
                    local_idx[pos + 1] = tmp_idx;

                    pos++;
                }
            }
        }
        // Synchronize warp (and block) before next batch.
        __syncwarp();
        __syncthreads();
    } // end batch loop

    // After processing all batches, each warp has 32 sorted candidate lists (one per lane)
    // each of length local_k, for a total of k candidates.
    // These per-thread lists are currently sorted in descending order (largest first),
    // so the best candidate (smallest distance) is at the end of each list.
    // We need to merge these 32 sorted lists into one sorted list in ascending order.

    // Let lane 0 in the warp perform the merge.
    if (lane == 0) {
        // Temporary arrays for holding merged data.
        // Maximum k is 1024.
        float merged_d[1024];
        int merged_idx[1024];

        // For each lane in the warp, retrieve its local candidate list using warp shuffles.
        // We reverse each list so that it becomes sorted in ascending order.
        // Each thread's local candidate list is stored in registers; lane 0 grabs every candidate from all lanes.
        for (int r = 0; r < 32; r++) {
            for (int j = 0; j < local_k; j++) {
                // Retrieve the candidate from lane r, slot j.
                float cand = __shfl_sync(0xFFFFFFFF, local_d[j], r);
                int cand_idx = __shfl_sync(0xFFFFFFFF, local_idx[j], r);
                // Since the list in lane r is in descending order, reversing it gives ascending order.
                // The reversed index is: (local_k - 1 - j).
                int pos = r * local_k + (local_k - 1 - j);
                merged_d[pos] = cand;
                merged_idx[pos] = cand_idx;
            }
        }

        // Each of the 32 subarrays of size local_k (total k elements) is now sorted in ascending order.
        // Perform a 32-way merge using a simple pointer-based algorithm.
        int pointers[32];
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            pointers[r] = 0;  // pointer into subarray r
        }

        // Temporary array to hold the final sorted k candidates.
        std::pair<int, float> final_candidates[1024];

        // Merge k elements.
        for (int p = 0; p < k; p++) {
            float best_val = INF;
            int best_lane = -1;
            // For each subarray, check the current candidate pointed by pointers[r].
            for (int r = 0; r < 32; r++) {
                if (pointers[r] < local_k) {
                    int pos = r * local_k + pointers[r];
                    float val = merged_d[pos];
                    if (val < best_val) {
                        best_val = val;
                        best_lane = r;
                    }
                }
            }
            // Record the best candidate.
            int pos = best_lane * local_k + pointers[best_lane];
            final_candidates[p] = std::pair<int, float>(merged_idx[pos], merged_d[pos]);
            pointers[best_lane]++; // move pointer in the winning subarray
        }

        // Write the final merged result for this query to global memory.
        // Each query occupies k consecutive entries in the result array.
        int out_base = queryIndex * k;
        for (int p = 0; p < k; p++) {
            result[out_base + p] = final_candidates[p];
        }
    }
}

// Host interface function for running k-NN.
// Each query point in the 'query' array is processed by one warp in the kernel.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block size as 256 threads (8 warps per block).
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    // Total number of warps required is equal to query_count (one warp per query).
    int totalWarps = query_count;
    int numBlocks = (totalWarps + warps_per_block - 1) / warps_per_block;
    
    // Launch the kernel with dynamic shared memory size = BATCH_SIZE * sizeof(float2).
    knn_kernel<<<numBlocks, threads_per_block, BATCH_SIZE * sizeof(float2)>>>(
        query, query_count, data, data_count, result, k);

    // Optionally (in production code) add error checking and device synchronization:
    // cudaError_t err = cudaGetLastError();
    // assert(err == cudaSuccess);
    // cudaDeviceSynchronize();
}