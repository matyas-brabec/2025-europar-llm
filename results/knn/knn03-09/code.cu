#include <cuda_runtime.h>
#include <cfloat>      // for FLT_MAX
#include <utility>     // for std::pair

// Define algorithm hyper‐parameters.
// WARP_SIZE: number of threads per warp (set to 32).
// THREADS_PER_BLOCK: number of threads per CUDA thread block.
// BATCH_SIZE: number of data points processed in one shared‐memory batch.
// MAX_LOCAL_CAND: maximum number of candidates per warp thread (k_max/32, since k_max=1024).
#define WARP_SIZE         32
#define THREADS_PER_BLOCK 128
#define BATCH_SIZE        512
#define MAX_LOCAL_CAND    32

// In our design, each warp processes one query.
// The k nearest neighbors for one query are stored in a private candidate list of size k.
// This candidate list is distributed among the 32 threads of the warp.
// Each thread is responsible for k_local = k / 32 entries and uses local arrays (in registers/local memory)
// to hold its part of the candidate list. Initially all distances are set to FLT_MAX (i.e. no candidate).
// As the kernel iterates through the data points (loaded in shared memory in batches),
// each thread computes the squared Euclidean distance from its assigned query point to the data point
// and then (if the candidate distance is below the current worst candidate in the warp)
// performs a warp‐synchronous update of the candidate list.
// At the end, lane 0 of each warp gathers the candidate list from all 32 lanes,
// sorts the k candidates in ascending order by distance, and writes the result to global memory.

__global__ 
void knn_kernel(const float2 *query, int query_count,
                const float2 *data, int data_count,
                std::pair<int, float> *result,
                int k)
{
    // Each warp processes one query.
    // Compute global warp (query) index and lane id.
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = global_thread_id / WARP_SIZE;   // warp index (each warp handles one query)
    int laneId = threadIdx.x % WARP_SIZE;          // lane index within warp

    if (warpId >= query_count)
        return;  // excess threads do nothing

    // Load the query point for this warp.
    float2 q = query[warpId];

    // Determine number of candidates each thread holds.
    // k is guaranteed to be a power-of-two between 32 and 1024; so k / WARP_SIZE is an integer in [1,32].
    int nlocal = k / WARP_SIZE;

    // Per-thread candidate list: each thread holds nlocal candidate distances and indices.
    // Initially, all candidates are set to FLT_MAX (i.e., no real candidate) and index -1.
    float candDist[MAX_LOCAL_CAND];
    int   candIdx [MAX_LOCAL_CAND];
    for (int j = 0; j < nlocal; j++) {
        candDist[j] = FLT_MAX;
        candIdx[j]  = -1;
    }

    // Declare shared memory to cache a batch of data points.
    // Each thread block allocates a shared memory array of BATCH_SIZE float2 values.
    __shared__ float2 shared_data[BATCH_SIZE];

    // Process the data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Determine number of points in this batch.
        int batch_size = BATCH_SIZE;
        if (batch_start + batch_size > data_count)
            batch_size = data_count - batch_start;

        // Load the current batch from global memory into shared memory.
        // Threads in the block collaboratively load the batch.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            shared_data[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the batch is loaded before proceeding

        // Each warp processes the loaded batch.
        // Distribute the work among the 32 threads in the warp:
        // Each thread processes shared_data indices starting from its laneId with stride WARP_SIZE.
        for (int i = laneId; i < batch_size; i += WARP_SIZE) {
            float2 pt = shared_data[i];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float d = dx * dx + dy * dy;
            int global_index = batch_start + i;

            // --- Begin candidate update ---
            // We want to update the warp's k-candidate list if this candidate is better than
            // the current worst candidate. The candidate list is distributed in parallel across lanes.
            // First, each lane computes the maximum (i.e. worst) candidate in its local list.
            float local_max = candDist[0];
            int   local_max_idx = 0;
            for (int j = 1; j < nlocal; j++) {
                float val = candDist[j];
                if (val > local_max) {
                    local_max = val;
                    local_max_idx = j;
                }
            }
            // Next, perform a warp-level reduction to find the global worst candidate (largest distance)
            // among all lanes. We'll use warp shuffle operations.
            float warp_max = local_max;
            int warp_max_lane = laneId;    // record which lane holds the worst candidate
            int warp_max_sub = local_max_idx; // record index within that lane's candidate list

            // Reduce within the warp.
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                // Each lane reads the maximum value from the lane 'offset' positions below.
                float other = __shfl_down_sync(0xffffffff, warp_max, offset);
                int   other_lane = __shfl_down_sync(0xffffffff, warp_max_lane, offset);
                int   other_sub = __shfl_down_sync(0xffffffff, warp_max_sub, offset);
                if (other > warp_max) {
                    warp_max = other;
                    warp_max_lane = other_lane;
                    warp_max_sub = other_sub;
                }
            }
            // Now, warp_max is the worst candidate distance in the warp's candidate list.
            // If the new candidate d is better than warp_max, update the candidate list.
            if (d < warp_max) {
                // Only the lane that owns the worst candidate performs the update.
                if (laneId == warp_max_lane) {
                    candDist[warp_max_sub] = d;
                    candIdx[warp_max_sub]  = global_index;
                }
            }
            __syncwarp();  // Ensure all lanes see the update before next candidate
            // --- End candidate update ---
        }
        __syncthreads();  // Ensure all warps are done with this batch before loading the next
    } // end for each batch

    // At this point, the warp's candidate list (distributed among its 32 lanes) holds k candidates.
    // They are unsorted. We now gather all k candidates into temporary arrays and sort them.
    int total_candidates = nlocal * WARP_SIZE;  // This equals k.
    // Let lane 0 of the warp perform the gathering and sorting.
    if (laneId == 0) {
        // Allocate temporary arrays in local memory. Maximum k is 1024.
        float finalDist[1024];
        int   finalIdx[1024];

        // For each lane in the warp, for each candidate in that lane,
        // use warp shuffle to retrieve the candidate from that lane.
        for (int r = 0; r < WARP_SIZE; r++) {
            for (int j = 0; j < nlocal; j++) {
                float d_val = __shfl_sync(0xffffffff, candDist[j], r);
                int   i_val = __shfl_sync(0xffffffff, candIdx[j], r);
                finalDist[r * nlocal + j] = d_val;
                finalIdx[r * nlocal + j]  = i_val;
            }
        }

        // Simple insertion sort on the collected candidate list in ascending order.
        for (int i = 1; i < total_candidates; i++) {
            float key = finalDist[i];
            int   key_idx = finalIdx[i];
            int j = i - 1;
            while (j >= 0 && finalDist[j] > key) {
                finalDist[j+1] = finalDist[j];
                finalIdx[j+1] = finalIdx[j];
                j--;
            }
            finalDist[j+1] = key;
            finalIdx[j+1] = key_idx;
        }

        // Write the sorted k nearest neighbors to global memory.
        // For query 'warpId', result[warpId * k + i] holds the i-th nearest neighbor.
        int base = warpId * total_candidates;
        for (int i = 0; i < total_candidates; i++) {
            result[base + i] = std::pair<int, float>(finalIdx[i], finalDist[i]);
        }
    }
}


// Host interface function.
// This function launches the CUDA kernel to compute the k nearest neighbors for 2D points in Euclidean space.
// It assumes that the input arrays 'query', 'data' and 'result' have been allocated on the device
// (e.g. via cudaMalloc) and that data_count >= k.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Each warp (32 threads) processes one query.
    // With THREADS_PER_BLOCK threads per block, we have (THREADS_PER_BLOCK / 32) queries per block.
    int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    int block_count = (query_count + warps_per_block - 1) / warps_per_block;

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(block_count);

    // Launch the kernel.
    // Shared memory size is BATCH_SIZE * sizeof(float2) bytes.
    knn_kernel<<<gridDim, blockDim>>>(query, query_count, data, data_count, result, k);
    
    // Synchronize and check for any runtime errors.
    cudaDeviceSynchronize();
}