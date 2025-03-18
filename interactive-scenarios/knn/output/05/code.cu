#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <utility>

// Define the tile size for caching data points in shared memory.
#define TILE_SIZE 256

// Compute the squared Euclidean distance between two 2D points.
__device__ float squared_distance(float2 a, float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Device function: Each thread maintains a private sorted list of candidates of size 'local_k'.
// Inserts a new candidate (with its index and squared distance) into the local list if it is better
// than the current worst candidate. The list is maintained in ascending order.
template <int local_k>
__device__ void insert_local_candidate(int local_indices[local_k], float local_dists[local_k], int index, float dist) {
    if (dist < local_dists[local_k - 1]) {
        int pos = local_k - 1;
        while (pos > 0 && local_dists[pos - 1] > dist) {
            local_dists[pos] = local_dists[pos - 1];
            local_indices[pos] = local_indices[pos - 1];
            pos--;
        }
        local_dists[pos] = dist;
        local_indices[pos] = index;
    }
}

// Templated CUDA kernel for k-NN search where K is the total number of nearest neighbors.
// Each query is processed by one warp (32 threads). The threads in a warp cooperatively load
// batches of data into shared memory and update their private candidate lists. Then, the 32 private
// lists (each of size K/32) are merged into one sorted final list using dynamic shared memory.
template <int K>
__global__ void knn_kernel_templated(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result) {
    const int warpSize = 32;
    // Each thread holds a private candidate list of size local_k.
    const int local_k = K / warpSize;

    // Determine warp and lane indices.
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;  // One warp per query.
    int lane = threadIdx.x % warpSize;           // Lane index within the warp.

    // If the warp_id exceeds the number of queries, exit.
    if (warp_id >= query_count) return;

    // Load the query point for this warp.
    float2 q = query[warp_id];

    // Each thread in the warp maintains its private candidate list in registers.
    int local_indices[local_k];
    float local_dists[local_k];
    #pragma unroll
    for (int i = 0; i < local_k; i++) {
        local_dists[i] = std::numeric_limits<float>::infinity();
        local_indices[i] = -1;
    }

    // Process the data points in batches using a statically allocated shared memory tile.
    __shared__ float2 s_tile[TILE_SIZE];

    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (batch_start + TILE_SIZE > data_count)
            tile_size = data_count - batch_start;

        // Load the current batch of data points into shared memory cooperatively.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            s_tile[i] = data[batch_start + i];
        }
        __syncthreads();

        // Each warp processes the tile: each thread handles a subset based on its lane index.
        for (int i = lane; i < tile_size; i += warpSize) {
            float2 d = s_tile[i];
            float dist = squared_distance(q, d);
            // Update the thread's private candidate list.
            insert_local_candidate<local_k>(local_indices, local_dists, batch_start + i, dist);
        }
        __syncthreads();
    }

    // Merge the 32 private candidate lists (total K candidates) into one final sorted list.
    // Allocate dynamic shared memory for merging. The memory is partitioned per warp.
    // The dynamic shared memory is organized as two arrays: one for indices and one for distances.
    int warps_per_block = blockDim.x / warpSize;
    int warp_local_id = threadIdx.x / warpSize; // Warp index within the block.
    extern __shared__ char merge_mem[];
    int *s_indices = (int*) merge_mem;
    float *s_dists = (float*) (merge_mem + warps_per_block * K * sizeof(int));
    // Each warp gets K contiguous slots.
    int *warp_indices = s_indices + warp_local_id * K;
    float *warp_dists = s_dists + warp_local_id * K;

    // Each thread writes its private candidate list into its designated portion of the warp's shared memory.
    #pragma unroll
    for (int i = 0; i < local_k; i++) {
        warp_indices[lane * local_k + i] = local_indices[i];
        warp_dists[lane * local_k + i] = local_dists[i];
    }
    __syncwarp();

    // Let lane 0 of the warp merge the candidates.
    if (lane == 0) {
        int final_indices[K];
        float final_dists[K];
        // Copy the merged candidates from shared memory.
        for (int i = 0; i < K; i++) {
            final_indices[i] = warp_indices[i];
            final_dists[i] = warp_dists[i];
        }
        // Sort the K candidates using insertion sort (ascending order by distance).
        for (int i = 1; i < K; i++) {
            int idx = final_indices[i];
            float d = final_dists[i];
            int j = i;
            while (j > 0 && final_dists[j - 1] > d) {
                final_dists[j] = final_dists[j - 1];
                final_indices[j] = final_indices[j - 1];
                j--;
            }
            final_dists[j] = d;
            final_indices[j] = idx;
        }
        // Write the final sorted k nearest neighbors to global memory.
        for (int i = 0; i < K; i++) {
            result[warp_id * K + i] = std::make_pair(final_indices[i], final_dists[i]);
        }
    }
}

// Host function to dispatch the templated k-NN kernel based on the runtime k value.
// 'k' is assumed to be a power of two between 32 and 1024 (inclusive).
// The dynamic shared memory size is computed as:
//   (warps per block) * K * (sizeof(int) + sizeof(float))
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    int threads_per_block = 256;           // Each block has 256 threads (8 warps).
    int warps_per_block = threads_per_block / 32;
    // Each query is processed by one warp: total threads = query_count * 32.
    int total_threads = query_count * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size = warps_per_block * k * (sizeof(int) + sizeof(float));

    switch (k) {
        case 32:
            knn_kernel_templated<32><<<blocks, threads_per_block, shared_mem_size>>>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_kernel_templated<64><<<blocks, threads_per_block, shared_mem_size>>>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_kernel_templated<128><<<blocks, threads_per_block, shared_mem_size>>>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_kernel_templated<256><<<blocks, threads_per_block, shared_mem_size>>>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_kernel_templated<512><<<blocks, threads_per_block, shared_mem_size>>>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_kernel_templated<1024><<<blocks, threads_per_block, shared_mem_size>>>(query, query_count, data, data_count, result);
            break;
        default:
            // Unsupported k value.
            return;
    }
    cudaDeviceSynchronize();
}
