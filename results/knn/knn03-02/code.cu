#include <cuda_runtime.h>
#include <utility>

// Define a very large value to initialize candidate distances.
#define INF 1e30f
// Define the batch size for loading data points into shared memory.
// This value may be tuned depending on the GPU architecture.
#define BATCH_SIZE 1024

// The CUDA kernel that computes k-nearest neighbors for 2D points.
// Each warp (32 threads) processes one query point.
// Each thread in the warp holds a private candidate list of size L = k/32.
// The kernel iterates over the data points in batches loaded into shared memory.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int,float> *result, int k) {
    // Each warp processes one query.
    // Compute the global warp id. (Each warp has 32 threads.)
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (global_warp_id >= query_count) return;

    // Compute the lane id within the warp (0 to 31).
    int lane_id = threadIdx.x & 31;

    // Load the query point for this warp.
    // Only lane 0 reads from global memory; then broadcast using warp shuffle.
    float2 q;
    if (lane_id == 0) {
        q = query[global_warp_id];
    }
    // Broadcast the query point to every lane in the warp.
    q.x = __shfl_sync(0xFFFFFFFF, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFF, q.y, 0);

    // Each thread in the warp will maintain a private candidate list.
    // Let L = k / 32 (since k is a power of two between 32 and 1024, L is an integer between 1 and 32).
    int L = k / 32;
    // Private arrays stored in registers for candidate distances and corresponding indices.
    // They are initialized to INF (so that any real distance will be an improvement).
    float localCandDist[32];
    int   localCandIdx[32];
    for (int i = 0; i < L; i++) {
        localCandDist[i] = INF;
        localCandIdx[i] = -1;
    }

    // Declare shared memory for a batch (tile) of data points.
    __shared__ float2 s_data[BATCH_SIZE];

    // Process the data points in batches.
    for (int base = 0; base < data_count; base += BATCH_SIZE) {
        // Determine the number of points to load for this batch.
        int tile_size = BATCH_SIZE;
        if (base + tile_size > data_count)
            tile_size = data_count - base;

        // Load the batch of data points into shared memory.
        // All threads in the block cooperatively load s_data.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            s_data[i] = data[base + i];
        }
        __syncthreads(); // Ensure the tile is fully loaded.

        // Each warp processes the loaded batch.
        // Distribute the work within the warp: each thread processes indices starting at its lane id.
        for (int i = lane_id; i < tile_size; i += 32) {
            float2 dp = s_data[i];  // Loaded data point.
            float dx = q.x - dp.x;
            float dy = q.y - dp.y;
            float dist = dx * dx + dy * dy;  // Squared Euclidean distance.
            int   idx = base + i;            // Global index of the data point.

            // Find the worst candidate (largest distance) in this thread's candidate list.
            float max_val = localCandDist[0];
            int max_pos = 0;
            for (int j = 1; j < L; j++) {
                if (localCandDist[j] > max_val) {
                    max_val = localCandDist[j];
                    max_pos = j;
                }
            }
            // If the current point is closer than the worst candidate,
            // update the candidate list by replacing the worst candidate.
            if (dist < max_val) {
                localCandDist[max_pos] = dist;
                localCandIdx[max_pos]  = idx;
            }
        }
        // Synchronize the warp (though __syncwarp is sufficient for warp-level communication).
        __syncwarp();
        // Synchronize the block before the next batch loads into shared memory.
        __syncthreads();
    }

    // Now, each thread in the warp holds its private list of L candidate pairs.
    // In the next step, we sort each thread's local candidate list in ascending order (by distance).
    // Since L is small (<= 32), a simple insertion sort is efficient.
    for (int i = 1; i < L; i++) {
        float key_dist = localCandDist[i];
        int key_idx = localCandIdx[i];
        int j = i - 1;
        while (j >= 0 && localCandDist[j] > key_dist) {
            localCandDist[j + 1] = localCandDist[j];
            localCandIdx[j + 1]  = localCandIdx[j];
            j--;
        }
        localCandDist[j + 1] = key_dist;
        localCandIdx[j + 1]  = key_idx;
    }

    // Now perform a warp-level merge of candidate lists.
    // Each warp has 32 sorted sublists (one per lane), each of length L.
    // We will merge these 32 sublists into one sorted list of length k.
    // For this, we use warp shuffle instructions so that one thread (lane 0) gathers all the candidates.
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    // Fixed-size arrays to hold the merged candidate data.
    // Since k is at most 1024, we allocate arrays of size 1024.
    float mergedDist[1024];
    int   mergedIdx[1024];
    if (lane_id == 0) {
        // Gather sorted candidate lists from all lanes.
        // For each lane r and each candidate position j, use __shfl_sync to retrieve the value from lane r.
        for (int r = 0; r < 32; r++) {
            for (int j = 0; j < L; j++) {
                float cand = __shfl_sync(FULL_MASK, localCandDist[j], r);
                int   idxCand = __shfl_sync(FULL_MASK, localCandIdx[j], r);
                mergedDist[r * L + j] = cand;
                mergedIdx[r * L + j]  = idxCand;
            }
        }

        // Now perform a k-way merge of the 32 sorted sublists.
        // We use an array "indices" of length 32 to track the current position in each sublist.
        int indices[32];
        for (int r = 0; r < 32; r++) {
            indices[r] = 0;  // Start at the beginning of each sublist.
        }
        float sortedDist[1024];
        int sortedIdx[1024];
        // For each position t in the final merged list (from 0 to k-1):
        for (int t = 0; t < k; t++) {
            float min_val = INF;
            int min_lane = -1;
            // Find the sublist which currently has the smallest candidate.
            for (int r = 0; r < 32; r++) {
                if (indices[r] < L) {
                    float d = mergedDist[r * L + indices[r]];
                    if (d < min_val) {
                        min_val = d;
                        min_lane = r;
                    }
                }
            }
            // Add the selected candidate to the final sorted list.
            sortedDist[t] = mergedDist[min_lane * L + indices[min_lane]];
            sortedIdx[t]  = mergedIdx[min_lane * L + indices[min_lane]];
            indices[min_lane]++;  // Advance the pointer in that sublist.
        }

        // Write the sorted k-nearest neighbors (index and squared distance) to the global result array.
        int query_idx = global_warp_id;
        for (int t = 0; t < k; t++) {
            result[query_idx * k + t] = std::pair<int,float>(sortedIdx[t], sortedDist[t]);
        }
    }
    // End of kernel.
}

// Host interface for launching the k-NN kernel.
// The function configures CUDA kernel launch parameters and synchronizes on completion.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int,float> *result, int k) {
    // Choose 256 threads per block (8 warps per block).
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    // Launch the kernel with the required amount of dynamic shared memory.
    // Shared memory size is BATCH_SIZE elements of float2.
    knn_kernel<<<blocks, threadsPerBlock, BATCH_SIZE * sizeof(float2)>>>(
        query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}