#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

//------------------------------------------------------------------------------
// This struct mirrors std::pair<int, float> but is defined explicitly so that
// it can be used in device code.
struct Pair {
    int first;
    float second;
};

//------------------------------------------------------------------------------
// The CUDA kernel that computes the k-nearest neighbors (using squared Euclidean distance)
// for 2D query points from a large set of data points. Each warp (32 threads)
// processes one query. Each thread in the warp will process a disjoint subset of data points
// (loaded in batches into shared memory) and maintain a private, sorted (ascending by distance)
// list of L = k/32 candidate neighbors. After processing all data points,
// the warp cooperatively merges the 32 candidate sub-lists into one sorted list of k neighbors.
// The final sorted list (in increasing order of squared distance) for each query is
// stored in the result array in row-major order.
/// @FIXED
/// extern "C"

__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           Pair *result, int k) {
    // Each warp processes one query.
    int warp_id = threadIdx.x / 32;       // warp index within the block.
    int lane = threadIdx.x & 31;            // lane index within the warp.
    int warpsPerBlock = blockDim.x / 32;

    // Map each warp to a query index.
    int queryIdx = blockIdx.x * warpsPerBlock + warp_id;
    if (queryIdx >= query_count)
        return;

    // Load the query point; only lane 0 loads from global memory,
    // then broadcast it to all lanes in the warp.
    float2 q;
    if (lane == 0)
        q = query[queryIdx];
    /// @FIXED (-1:+2)
    /// q = __shfl_sync(0xFFFFFFFF, q, 0);
    q.x = __shfl_sync(0xFFFFFFFF, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFF, q.y, 0);

    // Each warp will maintain k candidates sorted in ascending order.
    // Each thread in the warp is responsible for k_sub = k/32 candidates.
    const int k_sub = k / 32;  // k is guaranteed to be a power-of-two between 32 and 1024.
    /// @FIXED
    /// float best_dist[k_sub];
    float best_dist[/*MAX_K*/1024 / 32];  // Each thread holds k/32 candidates.
    /// @FIXED
    /// int best_idx[k_sub];
    int best_idx[/*MAX_K*/1024 / 32];  // Each thread holds k/32 candidates.
    // Initialize candidates with FLT_MAX and a dummy index.
#pragma unroll
    for (int i = 0; i < k_sub; i++) {
        best_dist[i] = FLT_MAX;
        best_idx[i] = -1;
    }

    // We'll process data points in batches; choose a batch size that
    // efficiently utilizes shared memory. (BATCH_SIZE=1024 is chosen here.)
    const int BATCH_SIZE = 1024;

    // Shared memory is used to cache a batch of data points.
    // The first part of shared memory is used for data points.
    extern __shared__ char shared_memory[];
    float2 *s_data = reinterpret_cast<float2*>(shared_memory);
    // A candidate merging scratch area (per block) will be placed after s_data.
    // (Its offset is computed in the host launcher; here we only use s_data in the batch loop.)

    // Process the complete data array in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        int batch_size = (data_count - batch_start < BATCH_SIZE) ? (data_count - batch_start) : BATCH_SIZE;

        // Cooperative load: each thread in the block loads data points into shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            s_data[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the batch is loaded.

        // Each warp processes this batch: each lane processes a subset of data points.
        for (int i = lane; i < batch_size; i += 32) {
            // Compute squared Euclidean distance between query q and data point.
            float2 p = s_data[i];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dist = dx * dx + dy * dy;
            int global_idx = batch_start + i;  // Global index of this data point.

            // Update the lanes' local candidate list if the new candidate is better than the worst one.
            if (dist < best_dist[k_sub - 1]) {
                // Since the list is maintained in sorted order (ascending),
                // the worst candidate is at index (k_sub - 1).
                int pos = k_sub - 1;
                // Shift candidates to the right until finding the insertion position.
                while (pos > 0 && best_dist[pos - 1] > dist) {
                    best_dist[pos] = best_dist[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                    pos--;
                }
                best_dist[pos] = dist;
                best_idx[pos] = global_idx;
            }
        }
        __syncthreads();  // Ensure all threads are done before loading the next batch.
    } // End of batch processing.

    //------------------------------------------------------------------------------
    // Now, each lane in this warp has k_sub candidates (sorted in ascending order).
    // We need to merge these 32 sorted sub-lists (total k candidates) into one sorted list.
    // We perform a warp-level tournament: in each of the k iterations we select the smallest
    // candidate among the current heads of the 32 lists.
    // Each lane holds a pointer (local_idx) into its candidate list.
    int local_ptr = 0;

    // For each position in the final merged output.
    for (int out_idx = 0; out_idx < k; out_idx++) {
        // Each lane sets its current candidate value (or FLT_MAX if its list is exhausted).
        float my_val = (local_ptr < k_sub) ? best_dist[local_ptr] : FLT_MAX;
        int my_cand_idx = (local_ptr < k_sub) ? best_idx[local_ptr] : -1;

        // Perform a warp-wide reduction to determine the minimum candidate among all lanes.
        unsigned mask = 0xFFFFFFFF;
        float min_val = my_val;
        int min_lane = lane;
        // Use iterative shuffles to reduce within the warp.
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(mask, min_val, offset);
            int other_lane = __shfl_down_sync(mask, min_lane, offset);
            if (other_val < min_val) {
                min_val = other_val;
                min_lane = other_lane;
            }
        }
        // The candidate value min_val is the smallest among the current candidates.
        // Retrieve the corresponding candidate index from the winning lane.
        int winning_idx = __shfl_sync(mask, my_cand_idx, min_lane);

        // Write the winning candidate to the final result.
        // The final result for query queryIdx is stored in result[queryIdx * k + out_idx].
        if (lane == 0) {
            result[queryIdx * k + out_idx].first = winning_idx;
            result[queryIdx * k + out_idx].second = min_val;
        }

        // The lane that contributed the minimum candidate increments its local pointer.
        if (lane == min_lane && local_ptr < k_sub)
            local_ptr++;

        __syncwarp();  // Synchronize warp lanes before next iteration.
    }
}

//------------------------------------------------------------------------------
// Host function interface for k-NN.
// This function launches the CUDA kernel with suitable grid/block dimensions
// and dynamic shared memory allocation. It processes 'query_count' 2D query points
// against 'data_count' 2D data points; for each query the k nearest neighbors
// (their indices and squared distances) are stored in 'result' in row-major order.
/// @FIXED
/// extern "C"

void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose 128 threads per block (i.e. 4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate dynamic shared memory size:
    // - BATCH_SIZE elements for data caching: BATCH_SIZE * sizeof(float2)
    // - One merge scratch area per warp: warpsPerBlock * k * sizeof(Pair)
    const int BATCH_SIZE = 1024;
    size_t sharedMemSize = BATCH_SIZE * sizeof(float2) + warpsPerBlock * k * sizeof(Pair);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count,
                                                           reinterpret_cast<Pair*>(result), k);
    // Optionally, CUDA error checking can be added here.
}
