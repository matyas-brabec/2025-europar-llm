// This complete CUDA C++ source implements an optimized k‐nearest neighbors (k‐NN)
// kernel for 2D points using squared Euclidean distance. Each query is processed
// by a single warp (32 threads). Each thread in the warp maintains a private candidate
// list of size (k/32) in registers. The global data points are processed in tiles
// that are loaded into shared memory. After processing all tiles, the warp‐private
// candidate arrays (totaling k candidates per query) are merged via warp shuffle
// and reduction to output the k nearest neighbors sorted in increasing order of distance.
//
// The interface is:
//   void run_knn(const float2 *query, int query_count, const float2 *data,
//                int data_count, std::pair<int, float> *result, int k);
// where for query i, result[i*k + j] (0 <= j < k) is the j-th nearest neighbor
// (with data point index and squared distance).
//
// This code assumes modern NVIDIA GPUs (A100/H100) and the latest CUDA toolkit.

#include <cuda_runtime.h>
#include <float.h>    // For FLT_MAX
#include <utility>    // For std::pair

// Define tile size for loading data into shared memory.
// We choose 1024 points per tile; each point is a float2 (8 bytes),
// so shared memory per block will be 1024 * 8 = 8192 bytes.
#define TILE_SIZE 1024

// The CUDA kernel that computes k-NN for 2D points.
// Each warp processes one query. The query id is computed from the warp id in the block.
// Each thread in the warp maintains its own private candidate list of size (k/32).
// After processing all data tiles, the warp cooperatively merges the candidate lists.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp (of 32 threads) processes one query.
    // Compute warp lane id and warp id.
    int lane = threadIdx.x & 31;              // threadIdx.x % 32
    int warp_id = threadIdx.x >> 5;             // threadIdx.x / 32

    // Each block contains (blockDim.x/32) warps.
    int warps_per_block = blockDim.x >> 5;
    int query_id = blockIdx.x * warps_per_block + warp_id;
    if (query_id >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[query_id];

    // Each thread will maintain a private candidate list of size (k/32).
    // Since k is a power-of-two between 32 and 1024, k/32 is between 1 and 32.
    int cand_per_thread = k >> 5; // equivalent to k/32

    // Declare candidate distance and index arrays in registers.
    // Maximum possible candidate list size is 32.
    float cand_dists[32];
    int   cand_idxs[32];
    // Initialize candidates to "worst" (FLT_MAX) and invalid index.
    for (int i = 0; i < cand_per_thread; i++) {
        cand_dists[i] = FLT_MAX;
        cand_idxs[i] = -1;
    }

    // Declare pointer to shared memory for the current tile.
    // "extern __shared__" uses dynamically allocated shared memory.
    extern __shared__ float2 shared_data[];

    // Loop over the data points in tiles cached in shared memory.
    for (int tile_offset = 0; tile_offset < data_count; tile_offset += TILE_SIZE) {
        // Determine number of points in the current tile.
        int tile_size = (data_count - tile_offset < TILE_SIZE) ? (data_count - tile_offset) : TILE_SIZE;

        // Cooperative load of data points into shared memory.
        // Each thread in the block loads elements at a stride of blockDim.x.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_data[i] = data[tile_offset + i];
        }
        __syncthreads();

        // Each warp processes the current tile.
        // The 32 threads in the warp collaborate: each thread processes a subset of tile points.
        for (int i = lane; i < tile_size; i += 32) {
            float2 dpt = shared_data[i];
            float dx = q.x - dpt.x;
            float dy = q.y - dpt.y;
            float dist = dx * dx + dy * dy;

            // Each thread updates its own candidate list.
            // First, find the worst candidate in the local array (i.e. the maximum distance).
            float worst = cand_dists[0];
            int worst_idx = 0;
            for (int j = 1; j < cand_per_thread; j++) {
                if (cand_dists[j] > worst) {
                    worst = cand_dists[j];
                    worst_idx = j;
                }
            }
            // If the new candidate is better than the worst in this thread's list,
            // replace the worst candidate.
            if (dist < worst) {
                cand_dists[worst_idx] = dist;
                cand_idxs[worst_idx] = tile_offset + i;  // Global data index.
            }
        }
        __syncthreads();
    } // End of tile loop.

    // At this point, each thread in the warp has a private candidate list.
    // The total number of candidate points for the query is k (cand_per_thread * 32).
    // We now merge these candidate lists in sorted order (by ascending distance)
    // using warp-level shuffles. Each iteration extracts one candidate (the best remaining).
    unsigned int full_mask = 0xffffffff;
    // Loop k times to extract the k nearest neighbors in order.
    for (int s = 0; s < k; s++) {
        // Each thread scans its own candidate list for the best (lowest distance) candidate.
        float local_min = FLT_MAX;
        int local_candidate_idx = -1;
        int local_data_idx = -1;
        for (int j = 0; j < cand_per_thread; j++) {
            float d = cand_dists[j];
            if (d < local_min) {
                local_min = d;
                local_candidate_idx = j;
                local_data_idx = cand_idxs[j];
            }
        }

        // Perform a warp-level reduction to find the global best candidate.
        // Each thread contributes its local minimum candidate.
        float best_val = local_min;
        int best_data_idx = local_data_idx;
        int best_thread = lane;         // Record origin: the lane id of current thread.
        int best_local_idx = local_candidate_idx;  // Index within the candidate array.
        // Reduction using __shfl_down_sync.
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(full_mask, best_val, offset);
            int other_data_idx = __shfl_down_sync(full_mask, best_data_idx, offset);
            int other_thread = __shfl_down_sync(full_mask, best_thread, offset);
            int other_local_idx = __shfl_down_sync(full_mask, best_local_idx, offset);
            if (other_val < best_val) {
                best_val = other_val;
                best_data_idx = other_data_idx;
                best_thread = other_thread;
                best_local_idx = other_local_idx;
            }
        }
        // Broadcast the global best candidate to all lanes.
        best_val = __shfl_sync(full_mask, best_val, 0);
        best_data_idx = __shfl_sync(full_mask, best_data_idx, 0);
        best_thread = __shfl_sync(full_mask, best_thread, 0);
        best_local_idx = __shfl_sync(full_mask, best_local_idx, 0);

        // Lane 0 writes the global best candidate for this query.
        // The j-th output (j == s) is the s-th nearest neighbor.
        if (lane == 0) {
            result[query_id * k + s] = std::pair<int, float>(best_data_idx, best_val);
        }
        // The thread that owns the winning candidate removes it by marking it as FLT_MAX.
        if (lane == best_thread && best_local_idx >= 0) {
            cand_dists[best_local_idx] = FLT_MAX;
            cand_idxs[best_local_idx] = -1;
        }
        __syncwarp(full_mask);
    }
}

// Host function that sets up and launches the knn_kernel.
// Assumptions:
//   - 'query', 'data', and 'result' point to GPU memory allocated via cudaMalloc.
//   - data_count >= k and k is a power of two between 32 and 1024.
//   - A large number of data points and queries make GPU parallelism beneficial.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose the number of threads per block.
    // We use 128 threads per block (4 warps per block).
    int threads_per_block = 128;
    int warps_per_block = threads_per_block >> 5;  // 128/32 = 4

    // Compute the number of blocks needed so that each warp gets one query.
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Calculate the shared memory size required per block.
    int shared_mem_size = TILE_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        query, query_count, data, data_count, result, k
    );
    // Optionally, check for errors and synchronize.
    cudaDeviceSynchronize();
}