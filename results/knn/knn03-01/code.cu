#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ---------------------------------------------------------------------------------------
// This file implements an optimized CUDA kernel for 2D k-nearest neighbors (k-NN)
// using a single warp (32 threads) per query. The code assumes that k is a power-of-two
// in the range [32, 1024], and that each query finds its k nearest neighbors based
// on squared Euclidean distance.
// 
// Each warp loads its query point and maintains a private candidate list of k neighbors,
// which is distributed among its 32 lanes. Each lane holds k/32 candidates in a local
// array (in registers) and processes a part of the data points loaded in shared memory.
// 
// The overall algorithm proceeds in two phases:
//  1. Distance computation & candidate update:
//     The input data is processed in batches. For each batch, the entire block cooperatively
//     loads a chunk of data points into shared memory. Then, each warp iterates over the
//     batch (with each lane processing a strided subset) and computes squared distances to its
//     query point. Each lane updates its local candidate list (kept unsorted) by replacing
//     the worst candidate if a better one is found.
//  2. Candidate merging & sorting:
//     After all batches, each lane sorts its local candidate list in ascending order (by distance).
//     Then, the 32 sorted lists (one per lane) are merged into one sorted list of k candidates
//     using a warp-collaborative merging loop using warp shuffles. The final sorted k results
//     for each query are written to the global results array in row-major order.
// ---------------------------------------------------------------------------------------

// Define constants for the batch size and warp size.
#define BATCH_SIZE 1024      // Number of data points processed per batch (fits in shared memory)
#define WARP_SIZE 32         // Size of a warp in CUDA
#define MAX_LOCAL_BUCKET_SIZE 32  // Maximum size for per-lane candidate list (k/32 <= 1024/32)

// The CUDA kernel that processes queries in parallel (one warp per query).
// Each warp finds the k nearest neighbors (smallest squared distances) from the data points.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Compute the warp-level index.
    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (globalWarpId >= query_count) return; // Out-of-bound warp; no work.

    // Load the query point assigned to this warp.
    // All lanes in the warp use the same query point.
    float2 q = query[globalWarpId];

    // Each warp maintains a private candidate list distributed across its 32 lanes.
    // The candidate list will have size k, and each lane holds k/32 candidates.
    int local_bucket_size = k / WARP_SIZE; // Guaranteed to be an integer.
    // Declare per-lane arrays to hold candidate distances and corresponding indices.
    // We use a fixed-size array of MAX_LOCAL_BUCKET_SIZE elements (max k/32).
    float candidateD[MAX_LOCAL_BUCKET_SIZE];
    int candidateI[MAX_LOCAL_BUCKET_SIZE];
    // Initialize the candidate list with maximum possible distances.
    for (int i = 0; i < local_bucket_size; i++) {
        candidateD[i] = FLT_MAX;
        candidateI[i] = -1;
    }

    // Declare shared memory for a batch of data points.
    extern __shared__ float2 shared_data[];

    // Process the global data points iteratively in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Determine the number of data points in the current batch.
        int batch_size = BATCH_SIZE;
        if (batch_start + batch_size > data_count) {
            batch_size = data_count - batch_start;
        }
        // Cooperative loading: Each thread in the block loads some data points into shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            shared_data[i] = data[batch_start + i];
        }
        __syncthreads(); // Ensure the batch is fully loaded before processing.

        // Each warp processes the shared memory batch:
        // Each lane processes the data points in a strided fashion (stride = warp size).
        for (int j = lane; j < batch_size; j += WARP_SIZE) {
            float2 d = shared_data[j];
            float dx = q.x - d.x;
            float dy = q.y - d.y;
            float dist = dx * dx + dy * dy;
            int data_index = batch_start + j;

            // For the new candidate, find the worst candidate (max distance)
            // in the lane’s local candidate list.
            float max_val = candidateD[0];
            int max_idx = 0;
            for (int l = 1; l < local_bucket_size; l++) {
                if (candidateD[l] > max_val) {
                    max_val = candidateD[l];
                    max_idx = l;
                }
            }
            // If the new distance is smaller than the worst candidate, update the candidate list.
            if (dist < max_val) {
                candidateD[max_idx] = dist;
                candidateI[max_idx] = data_index;
            }
        }
        __syncthreads(); // Ensure all warps finish processing the batch.
    } // End of batching over data points

    // Stage 2: Each lane sorts its own local candidate list (of length local_bucket_size)
    // in ascending order using a simple insertion sort.
    for (int i = 1; i < local_bucket_size; i++) {
        float key = candidateD[i];
        int keyI = candidateI[i];
        int j = i - 1;
        while (j >= 0 && candidateD[j] > key) {
            candidateD[j + 1] = candidateD[j];
            candidateI[j + 1] = candidateI[j];
            j--;
        }
        candidateD[j + 1] = key;
        candidateI[j + 1] = keyI;
    }

    // Stage 3: Merge the 32 sorted candidate lists (one per lane) into one sorted list of k entries.
    // Each lane maintains a pointer (initially 0) into its sorted candidate list.
    int ptr = 0; // Local pointer for this lane.
    // The merged output will be generated in k iterations.
    for (int merge_idx = 0; merge_idx < k; merge_idx++) {
        // Each lane selects its current candidate value if available; otherwise, return FLT_MAX.
        float my_val = (ptr < local_bucket_size) ? candidateD[ptr] : FLT_MAX;
        int my_idx = (ptr < local_bucket_size) ? candidateI[ptr] : -1;

        // Lane 0 collects candidate values from all lanes using warp shuffles.
        float best_val;
        int best_idx;
        int best_lane;
        if (lane == 0) {
            best_val = my_val;
            best_idx = my_idx;
            best_lane = 0;
            // Iterate through lanes 1 to 31 to find the overall minimum candidate.
            for (int w = 1; w < WARP_SIZE; w++) {
                float w_val = __shfl_sync(0xffffffff, my_val, w);
                int w_idx = __shfl_sync(0xffffffff, my_idx, w);
                if (w_val < best_val) {
                    best_val = w_val;
                    best_idx = w_idx;
                    best_lane = w;
                }
            }
        }
        // Broadcast the winning lane, best candidate value, and candidate index from lane 0 to all lanes.
        int winning_lane = __shfl_sync(0xffffffff, (lane == 0 ? best_lane : 0), 0);
        best_val = __shfl_sync(0xffffffff, (lane == 0 ? best_val : 0.0f), 0);
        best_idx = __shfl_sync(0xffffffff, (lane == 0 ? best_idx : 0), 0);

        // Only lane 0 writes the selected candidate to the global result array.
        if (lane == 0) {
            result[globalWarpId * k + merge_idx] = std::pair<int, float>(best_idx, best_val);
        }
        // The lane that contributed the best candidate increments its pointer.
        if (lane == winning_lane) {
            ptr++;
        }
        __syncwarp(0xffffffff);  // Ensure all lanes update their pointer before next merge iteration.
    }
}

// Host-side function that configures and launches the k-NN kernel.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Define the block size as 256 threads (i.e. 8 warps per block).
    int block_size = 256;
    int warps_per_block = block_size / WARP_SIZE;
    // Each warp processes one query. Calculate the grid dimension accordingly.
    int grid_size = (query_count + warps_per_block - 1) / warps_per_block;
    // Shared memory per block: enough to hold BATCH_SIZE data points (each of type float2).
    size_t shared_mem_size = BATCH_SIZE * sizeof(float2);
    // Launch the kernel.
    knn_kernel<<<grid_size, block_size, shared_mem_size>>>(query, query_count,
                                                           data, data_count,
                                                           result, k);
    // Synchronize to ensure kernel completion.
    cudaDeviceSynchronize();
}