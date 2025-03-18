#include <cuda_runtime.h>
#include <cfloat>       // For FLT_MAX
#include <utility>      // For std::pair

// Define the batch size for loading data points into shared memory.
// This value is chosen as a trade‐off between occupancy and shared memory usage.
#define BATCH_SIZE 1024

//------------------------------------------------------------------------------
// K-NN CUDA kernel:
//   Each warp (32 threads) processes one query point.
//   For each query, the warp maintains a private candidate list of k neighbors
//   distributed over its 32 threads. Each thread holds k/32 candidates in registers.
//   The kernel processes the large data array in batches. Each batch of data points
//   is cooperatively loaded into shared memory by the block. Then each warp processes
//   the batch using warp-level parallelism and updates its candidate list via insertion.
//   Finally, one thread (lane 0) in each warp gathers the per-thread candidate lists,
//   merges and sorts them, and writes the k nearest neighbors to global memory.
//------------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Compute global thread and warp IDs.
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / warpSize;      // Each warp has 32 threads.
    int lane = global_tid & (warpSize - 1);     // lane = global_tid % 32

    // Guard: if warp_id exceeds the number of queries, exit.
    if (warp_id >= query_count)
        return;

    // Each warp processes one query.
    float2 q = query[warp_id];

    // Determine the number of candidates each thread must maintain.
    // Since k is a power-of-two between 32 and 1024 and each warp has 32 threads,
    // each thread holds local_k = k / 32 candidate entries.
    int local_k = k >> 5;  // Equivalent to k / 32.

    // Allocate per-thread candidate arrays in registers.
    // The maximum possible local_k is 1024/32 == 32.
    float local_d[32];
    int   local_idx[32];

    // Initialize each thread's candidate list with "infinite" distances.
    for (int i = 0; i < local_k; i++) {
        local_d[i] = FLT_MAX;
        local_idx[i] = -1;
    }

    // Allocate shared memory for loading a batch of data points.
    __shared__ float2 s_data[BATCH_SIZE];

    // Process all data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Determine number of points in this batch.
        int batch_size = BATCH_SIZE;
        if (batch_start + batch_size > data_count)
            batch_size = data_count - batch_start;

        // Cooperative loading: Each thread in the block loads one or more points.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            s_data[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the entire batch is loaded.

        // Each warp now processes the batch.
        // Each thread in the warp loops over its assigned data points in the batch.
        for (int j = lane; j < batch_size; j += warpSize) {
            // Load point from shared memory.
            float2 d = s_data[j];

            // Compute squared Euclidean distance to the query.
            float dx = q.x - d.x;
            float dy = q.y - d.y;
            float dist = dx * dx + dy * dy;
            int data_index = batch_start + j;

            // Update the thread's candidate list if this distance is smaller than its worst candidate.
            if (dist < local_d[local_k - 1]) {
                // Insertion sort into the sorted candidate list (ascending order).
                int pos = local_k - 1;
                while (pos > 0 && dist < local_d[pos - 1]) {
                    local_d[pos] = local_d[pos - 1];
                    local_idx[pos] = local_idx[pos - 1];
                    pos--;
                }
                local_d[pos] = dist;
                local_idx[pos] = data_index;
            }
        }
        // Synchronize threads in the warp to ensure all candidate updates are done.
        __syncwarp();
        __syncthreads();  // Ensure batch processing complete before next batch load.
    } // End of batching loop.

    // At this point, each thread in the warp holds local_k candidate entries.
    // Total candidates for the warp equals k = local_k * 32.
    // Let lane 0 in the warp gather all candidates, merge and sort them.
    if (lane == 0) {
        // Allocate final candidate arrays in registers.
        // We use a fixed maximum size of 1024 for k.
        float final_d[1024];
        int   final_idx[1024];

        // Gather the per-thread candidate lists from all 32 lanes via warp shuffles.
        for (int src_lane = 0; src_lane < warpSize; src_lane++) {
            for (int i = 0; i < local_k; i++) {
                // __shfl_sync broadcasts the candidate from src_lane.
                float cand_d = __shfl_sync(0xFFFFFFFF, local_d[i], src_lane);
                int   cand_idx = __shfl_sync(0xFFFFFFFF, local_idx[i], src_lane);
                final_d[src_lane * local_k + i] = cand_d;
                final_idx[src_lane * local_k + i] = cand_idx;
            }
        }

        // Perform an insertion sort on the final candidate list to order by ascending distance.
        for (int i = 1; i < k; i++) {
            float key_d = final_d[i];
            int key_idx = final_idx[i];
            int j = i - 1;
            while (j >= 0 && final_d[j] > key_d) {
                final_d[j + 1] = final_d[j];
                final_idx[j + 1] = final_idx[j];
                j--;
            }
            final_d[j + 1] = key_d;
            final_idx[j + 1] = key_idx;
        }

        // Write the sorted k nearest neighbors to the global result array.
        // For query number warp_id, write its k neighbors consecutively.
        for (int i = 0; i < k; i++) {
            result[warp_id * k + i] = std::pair<int, float>(final_idx[i], final_d[i]);
        }
    }
}

//------------------------------------------------------------------------------
// Host interface function: run_knn
// This function configures and launches the CUDA kernel for the k-NN computation.
// It assumes that all pointers (query, data, result) refer to device memory
// allocated via cudaMalloc, and that data_count >= k.
//------------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose the number of threads per block.
    // Here we select 256 threads per block.
    constexpr int THREADS_PER_BLOCK = 256;

    // Each warp (32 threads) processes one query.
    // Total warps required equals the number of queries.
    int total_warps = query_count;
    /// @FIXED
    /// int total_threads = total_warps * warpSize;  // warpSize is 32.
    int total_threads = total_warps * 32;  // warpSize is 32.

    // Compute grid dimensions.
    int numBlocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel.
    // The kernel uses statically allocated shared memory of size BATCH_SIZE * sizeof(float2).
    knn_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(query, query_count,
                                                 data, data_count,
                                                 result, k);

    // Optionally, error checking and synchronization can be added:
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { /* Handle error */ }
    // cudaDeviceSynchronize();
}
