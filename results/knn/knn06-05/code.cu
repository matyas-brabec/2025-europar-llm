#include <cuda_runtime.h>
#include <utility>

// Optimized CUDA kernel for k-Nearest Neighbors (k-NN) in 2D Euclidean space.
// Each query is processed by one warp (32 threads).
// The algorithm processes the global data in batches cached in shared memory.
// Each warp maintains two per-query buffers in shared memory:
//   • A candidate buffer that temporarily collects candidate neighbors.
//   • A best-result buffer that stores the current k best (nearest) neighbors in sorted order
//     (i.e. s_best_dist[warp * k + (k-1)] is the current max squared distance).
// When the candidate buffer is full, its contents are merged into the best-result array;
// a similar merge is done at the end for any remaining candidates.
// Atomic operations, warp shuffles and thread-block synchronizations are used for coordination.
 
// __global__ kernel. 'query' and 'data' are in global memory; 'result' is written to global memory.
// 'k' is a power-of-two between 32 and 1024.
__global__ void knn_kernel(const float2 *query,
                           int query_count,
                           const float2 *data,
                           int data_count,
                           std::pair<int, float> *result,
                           int k)
{
    // Each warp (32 threads) processes one query.
    // Calculate warp-local indices:
    int lane = threadIdx.x & 31; // thread index within warp [0,31]
    int warp_in_block = threadIdx.x >> 5; // warp index within the block
    int warps_per_block = blockDim.x >> 5;
    // Global warp index = block index * warps_per_block + warp_in_block.
    int global_warp = blockIdx.x * warps_per_block + warp_in_block;
    if (global_warp >= query_count)
        return;
 
    // Load the query point for this warp.
    float2 q = query[global_warp];
 
    // Dynamic shared memory layout:
    // [0, BATCH_SIZE*sizeof(float2))            : data batch storage (float2 array)
    // [BATCH_SIZE*sizeof(float2), ... )           : candidate buffer and best-result arrays per warp.
    //
    // Layout (all arrays are contiguous; all indices are per block):
    // 1. s_data_batch: float2 s_data_batch[BATCH_SIZE]
    // 2. s_cand_count: int s_cand_count[warps_per_block]
    // 3. s_cand_idx  : int s_cand_idx[warps_per_block * k]
    // 4. s_cand_dist : float s_cand_dist[warps_per_block * k]
    // 5. s_best_idx  : int s_best_idx[warps_per_block * k]
    // 6. s_best_dist : float s_best_dist[warps_per_block * k]
    //
    // Choose a batch size for loading 'data' (tunable parameter).
    const int BATCH_SIZE = 256;
    extern __shared__ char shared_mem[];
    size_t offset = 0;
 
    // 1. Data batch buffer.
    float2 *s_data_batch = reinterpret_cast<float2*>(shared_mem + offset);
    offset += BATCH_SIZE * sizeof(float2);
 
    // 2. Candidate buffer count for each warp.
    int *s_cand_count = reinterpret_cast<int*>(shared_mem + offset);
    offset += warps_per_block * sizeof(int);
 
    // 3. Candidate buffer indices.
    int *s_cand_idx = reinterpret_cast<int*>(shared_mem + offset);
    offset += warps_per_block * k * sizeof(int);
 
    // 4. Candidate buffer distances.
    float *s_cand_dist = reinterpret_cast<float*>(shared_mem + offset);
    offset += warps_per_block * k * sizeof(float);
 
    // 5. Best-result buffer indices (the current best k neighbors).
    int *s_best_idx = reinterpret_cast<int*>(shared_mem + offset);
    offset += warps_per_block * k * sizeof(int);
 
    // 6. Best-result buffer distances.
    float *s_best_dist = reinterpret_cast<float*>(shared_mem + offset);
    offset += warps_per_block * k * sizeof(float);
 
    // Initialize candidate count for this warp.
    if (lane == 0)
        s_cand_count[warp_in_block] = 0;
 
    // Initialize the best-result buffer for this query.
    // The best buffer is sorted in ascending order: best_result[0] is the smallest distance, [k-1] is the worst.
    for (int i = lane; i < k; i += 32)
    {
        s_best_dist[warp_in_block * k + i] = 1e20f; // large initial value
        s_best_idx[warp_in_block * k + i] = -1;
    }
 
    // Synchronize the block to ensure shared memory initialization is complete.
    __syncthreads();
 
    // Process the global data in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE)
    {
        // Compute the current batch size (may be less than BATCH_SIZE at the end).
        int current_batch_size = (data_count - batch_start < BATCH_SIZE) ? (data_count - batch_start) : BATCH_SIZE;
 
        // Load the current batch of data points into shared memory.
        for (int i = threadIdx.x; i < current_batch_size; i += blockDim.x)
        {
            s_data_batch[i] = data[batch_start + i];
        }
        // Wait for all threads in the block to finish loading the batch.
        __syncthreads();
 
        // Each warp processes the batch from shared memory.
        // Each warp's threads loop over the batch with stride 32.
        for (int i = lane; i < current_batch_size; i += 32)
        {
            // Load data point from shared memory.
            float2 d = s_data_batch[i];
            // Compute squared Euclidean distance between query and data point.
            float dx = q.x - d.x;
            float dy = q.y - d.y;
            float dist = dx * dx + dy * dy;
 
            // Read the current maximum (worst) distance in the best-result buffer.
            // (The array s_best_dist is sorted in ascending order.)
            float current_max = s_best_dist[warp_in_block * k + (k - 1)];
 
            // If the computed distance is better than the current worst, add it as a candidate.
            if (dist < current_max)
            {
                // Use atomicAdd on the candidate count for this warp to get an insertion position.
                int pos = atomicAdd(&s_cand_count[warp_in_block], 1);
                if (pos < k)
                {
                    s_cand_idx[warp_in_block * k + pos] = batch_start + i; // global data index
                    s_cand_dist[warp_in_block * k + pos] = dist;
                }
                // If this candidate fills the candidate buffer, trigger a merge.
                if (pos == k - 1)
                {
                    // Merge the candidate buffer with the best-result buffer.
                    // For simplicity, we assign the merge work to lane 0 of the warp.
                    if (lane == 0)
                    {
                        int count = s_cand_count[warp_in_block];
                        // Iterate over each candidate in the candidate buffer.
                        for (int j = 0; j < count; j++)
                        {
                            float cand_dist = s_cand_dist[warp_in_block * k + j];
                            int cand_idx = s_cand_idx[warp_in_block * k + j];
                            // If candidate is better than the worst element in the best-result buffer...
                            if (cand_dist < s_best_dist[warp_in_block * k + (k - 1)])
                            {
                                // Find insertion position in the sorted best-result buffer.
                                int pos_ins = 0;
                                while (pos_ins < k && cand_dist >= s_best_dist[warp_in_block * k + pos_ins])
                                    pos_ins++;
                                // If candidate improves on any entry, insert it.
                                if (pos_ins < k)
                                {
                                    // Shift the worse elements down to make room for the candidate.
                                    for (int t = k - 1; t > pos_ins; t--)
                                    {
                                        s_best_dist[warp_in_block * k + t] = s_best_dist[warp_in_block * k + t - 1];
                                        s_best_idx[warp_in_block * k + t] = s_best_idx[warp_in_block * k + t - 1];
                                    }
                                    // Insert the candidate.
                                    s_best_dist[warp_in_block * k + pos_ins] = cand_dist;
                                    s_best_idx[warp_in_block * k + pos_ins] = cand_idx;
                                }
                            }
                        }
                        // Reset the candidate buffer count for this warp.
                        s_cand_count[warp_in_block] = 0;
                    }
                    // Synchronize warp threads to ensure the merge is visible.
                    __syncwarp();
                }
            }
        }
 
        // Synchronize the entire block before loading the next batch.
        __syncthreads();
    } // End of batches
 
    // After processing all batches, merge any remaining candidates in the candidate buffer.
    if (s_cand_count[warp_in_block] > 0)
    {
        if (lane == 0)
        {
            int count = s_cand_count[warp_in_block];
            for (int j = 0; j < count; j++)
            {
                float cand_dist = s_cand_dist[warp_in_block * k + j];
                int cand_idx = s_cand_idx[warp_in_block * k + j];
                if (cand_dist < s_best_dist[warp_in_block * k + (k - 1)])
                {
                    int pos_ins = 0;
                    while (pos_ins < k && cand_dist >= s_best_dist[warp_in_block * k + pos_ins])
                        pos_ins++;
                    if (pos_ins < k)
                    {
                        for (int t = k - 1; t > pos_ins; t--)
                        {
                            s_best_dist[warp_in_block * k + t] = s_best_dist[warp_in_block * k + t - 1];
                            s_best_idx[warp_in_block * k + t] = s_best_idx[warp_in_block * k + t - 1];
                        }
                        s_best_dist[warp_in_block * k + pos_ins] = cand_dist;
                        s_best_idx[warp_in_block * k + pos_ins] = cand_idx;
                    }
                }
            }
            s_cand_count[warp_in_block] = 0;
        }
        __syncwarp();
    }
    __syncthreads(); // Ensure all shared memory updates are complete.
 
    // Write out the final k nearest neighbors for this query to global memory.
    // Use warp cooperation: each thread in the warp writes its assigned portion.
    for (int i = lane; i < k; i += 32)
    {
        int idx = s_best_idx[warp_in_block * k + i];
        float d   = s_best_dist[warp_in_block * k + i];
        result[global_warp * k + i] = std::pair<int, float>(idx, d);
    }
}
 
// Host interface function.
// query: array of query points (float2)
// query_count: number of queries (each query is one float2)
// data: array of data points (float2)
// data_count: number of data points (>= k)
// result: output array (row-major) of std::pair<int, float> results (size = query_count * k)
// k: number of nearest neighbors per query (power-of-two between 32 and 1024)
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Choose block size = 128 threads (i.e. 4 warps per block).
    int blockDim_x = 128;
    int warps_per_block = blockDim_x >> 5;  // 128/32 = 4 warps per block.
    // Each warp processes one query; so grid dimension (in warps) is:
    int gridDim_x = (query_count + warps_per_block - 1) / warps_per_block;
 
    // Calculate the shared memory requirement:
    // Shared memory layout:
    //   s_data_batch: BATCH_SIZE * sizeof(float2)
    //   s_cand_count: warps_per_block * sizeof(int)
    //   s_cand_idx  : warps_per_block * k * (sizeof(int))
    //   s_cand_dist : warps_per_block * k * (sizeof(float))
    //   s_best_idx  : warps_per_block * k * (sizeof(int))
    //   s_best_dist : warps_per_block * k * (sizeof(float))
    const int BATCH_SIZE = 256;
    size_t shared_mem_size = BATCH_SIZE * sizeof(float2)
                             + warps_per_block * sizeof(int)
                             + warps_per_block * k * (sizeof(int) + sizeof(float))  // candidate buffer
                             + warps_per_block * k * (sizeof(int) + sizeof(float)); // best-result buffer
 
    // Launch the kernel.
    knn_kernel<<<gridDim_x, blockDim_x, shared_mem_size>>>(query, query_count, data, data_count, result, k);
}
