#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>       // for std::pair
#include <float.h>       // for FLT_MAX

// -----------------------------------------------------------------------------
// This CUDA kernel implements the k-nearest neighbors (k-NN) algorithm for 2D
// points in Euclidean space. It is optimized for modern NVIDIA GPUs (e.g., A100,
// H100) and uses warp‐level parallelism: one warp (32 threads) processes one query.
// Each warp maintains its k best (i.e. smallest squared Euclidean distance) neighbors,
// stored in registers partitioned evenly among its 32 threads.
//
// The input data points are processed in batches which are cooperatively loaded
// into shared memory by the thread block. Then each warp’s threads compute the
// distances from the query point to the points in shared memory in parallel.
//
// Each thread holds a private candidate list (of size LOCAL_SIZE = k/32) sorted
// in ascending order (best candidates at lower indices) so that the worst candidate
// in that slot is at index LOCAL_SIZE-1. After processing all data batches, the
// 32 candidate lists are merged using warp shuffles so that the final k candidates
// (in complete sorted order) are written to global memory.
//
// The squared Euclidean distance is computed without taking square roots.
// -----------------------------------------------------------------------------

// Number of data points loaded per batch from global memory into shared memory.
// This hyper-parameter can be tuned based on the GPU's shared memory capacity.
#define BATCH_SIZE 1024

// The kNN kernel. One warp (32 threads) processes one query point.
__global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                           const float2* __restrict__ data, int data_count,
                           std::pair<int, float>* __restrict__ result, int k)
{
    // Identify the warp within the block.
    int warpIdInBlock = threadIdx.x / 32;
    int lane = threadIdx.x & 31;  // lane id in [0,31]
    int warpsPerBlock = blockDim.x / 32;
    // Global query index equals blockIdx * (warps per block) + warpIdInBlock.
    int query_id = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (query_id >= query_count)
        return;

    // Load the query point for this warp.
    float2 q;
    if (lane == 0) {
        q = query[query_id];
    }
    // Broadcast query coordinates to all threads within the warp.
    q.x = __shfl_sync(0xffffffff, q.x, 0);
    q.y = __shfl_sync(0xffffffff, q.y, 0);

    // Each warp maintains a private candidate list of its k best neighbors.
    // We partition the k candidates evenly among the 32 threads.
    const int LOCAL_SIZE = k / 32;  // (k is a power-of-two between 32 and 1024)
    // Each thread holds LOCAL_SIZE candidate records: a distance and an index.
    /// @FIXED
    /// float local_d[LOCAL_SIZE];
    float local_d[/*MAX_K*/1024 / 32];  // Each thread holds k/32 candidates.
    /// @FIXED
    /// int   local_idx[LOCAL_SIZE];
    int   local_idx[/*MAX_K*/1024 / 32];  // Each thread holds k/32 candidates.
    // Initialize candidate lists with "infinite" distance and invalid index.
    #pragma unroll
    for (int i = 0; i < LOCAL_SIZE; i++) {
        local_d[i] = FLT_MAX;
        local_idx[i] = -1;
    }

    // Use shared memory to cache a batch of data points.
    // The shared memory size is BATCH_SIZE * sizeof(float2).
    extern __shared__ float2 shared_data[];

    // Process the global data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Determine the number of points in this batch.
        int batch_size = (batch_start + BATCH_SIZE <= data_count) ? BATCH_SIZE : (data_count - batch_start);

        // Cooperative load of batch data into shared memory.
        // Each thread in the block loads multiple points (if needed) using a strided loop.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            shared_data[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the batch is completely loaded.

        // Each warp processes the batch from shared memory.
        // Each thread in the warp processes a subset of points (using a stride of warp size).
        for (int i = lane; i < batch_size; i += 32) {
            float2 d_pt = shared_data[i];
            float dx = q.x - d_pt.x;
            float dy = q.y - d_pt.y;
            float dist = dx * dx + dy * dy;  // Squared Euclidean distance

            // Update the local candidate list if this candidate is better than
            // the worst candidate currently stored (which is at index LOCAL_SIZE-1,
            // since the list is maintained in ascending order).
            if (dist < local_d[LOCAL_SIZE - 1]) {
                // Find the insertion position using linear search.
                int pos = LOCAL_SIZE - 1;
                // Shift larger candidates to the right.
                while (pos > 0 && dist < local_d[pos - 1]) {
                    local_d[pos] = local_d[pos - 1];
                    local_idx[pos] = local_idx[pos - 1];
                    pos--;
                }
                local_d[pos] = dist;
                local_idx[pos] = batch_start + i;  // Global index of the data point.
            }
        }
        __syncthreads();  // Synchronize before loading the next batch.
    } // end for each batch

    // -------------------------------------------------------------------------
    // Merge step: The 32 threads in the warp now hold a total of k candidate
    // records (each with LOCAL_SIZE = k/32 candidates) in sorted (ascending) order.
    // We perform a 32-way merge to produce a single sorted list of k candidates.
    // Each thread maintains a pointer (initially 0) to the head of its local list.
    // In each iteration, the warp collectively selects the smallest candidate
    // among the current heads. Then, the thread that contributed the candidate
    // increments its pointer. This procedure runs for k iterations.
    // -------------------------------------------------------------------------
    int ptr = 0;  // Local pointer for each thread's candidate list.
    for (int r = 0; r < k; r++) {
        // Each thread computes its current candidate value and index.
        float myVal = (ptr < LOCAL_SIZE) ? local_d[ptr] : FLT_MAX;
        int   myIdx = (ptr < LOCAL_SIZE) ? local_idx[ptr] : -1;
        __syncwarp();  // Make sure all lanes have their current candidate.

        // Let lane 0 in the warp perform a serial 32-way reduction.
        float bestVal;
        int bestIdx;
        int bestLane;
        if (lane == 0) {
            bestVal = FLT_MAX;
            bestIdx = -1;
            bestLane = -1;
            // Loop over all 32 lanes to select the candidate with the smallest distance.
            for (int i = 0; i < 32; i++) {
                float cand = __shfl_sync(0xffffffff, myVal, i);
                int   candIdx = __shfl_sync(0xffffffff, myIdx, i);
                if (cand < bestVal) {
                    bestVal = cand;
                    bestIdx = candIdx;
                    bestLane = i;
                }
            }
        }
        // Broadcast the winning lane and candidate from lane 0 to all lanes.
        bestLane = __shfl_sync(0xffffffff, bestLane, 0);
        bestVal  = __shfl_sync(0xffffffff, bestVal, 0);
        bestIdx  = __shfl_sync(0xffffffff, bestIdx, 0);

        // The thread whose current candidate was chosen increments its pointer.
        if (lane == bestLane && ptr < LOCAL_SIZE)
            ptr++;

        __syncwarp();  // Ensure the pointer update is visible to all lanes.

        // Only lane 0 writes the merged candidate into the global result array.
        if (lane == 0) {
            result[query_id * k + r] = std::pair<int, float>(bestIdx, bestVal);
        }
    }
} // end kernel

// -----------------------------------------------------------------------------
// Host interface: run_knn
//   query       : array of query points (float2), one per query.
//   query_count : number of query points.
//   data        : array of data points (float2).
//   data_count  : number of data points (>= k).
//   result      : output array of k nearest neighbors per query, stored as a
//                 row-major array of std::pair<int, float> (index, squared distance).
//   k           : number of nearest neighbors to find (power-of-two between 32 and 1024).
// -----------------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k)
{
    // Choose 256 threads per block.
    const int threads_per_block = 256;
    // Each warp (32 threads) processes one query; hence, each block processes:
    int warps_per_block = threads_per_block / 32;
    int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Allocate shared memory per block: BATCH_SIZE data points (each float2).
    size_t sharedMemSize = BATCH_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<num_blocks, threads_per_block, sharedMemSize>>>(query, query_count,
                                                                 data, data_count,
                                                                 result, k);
    // (Optional) Check for launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // In production code one may call cudaDeviceReset() and report the error.
    }
}
