#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <utility>

#define TILE_SIZE 256  // Tile size for caching data points in shared memory.

// -----------------------------------------------------------------------------
// Device function: Compute squared Euclidean distance between two 2D points.
// -----------------------------------------------------------------------------
__device__ float squared_distance(float2 a, float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// -----------------------------------------------------------------------------
// Device function: Merge the candidate buffer with the intermediate result.
// This function is executed by lane 0 of the warp. It first sorts the candidate
// buffer (which may be unsorted) using insertion sort and then merges it with
// the intermediate candidate list (both sorted in ascending order), updating the
// intermediate result with the best k candidates.
// -----------------------------------------------------------------------------
template <int K>
__device__ void merge_buffer_with_intermediate(int *inter_indices, float *inter_dists,
                                               int *buffer_indices, float *buffer_dists,
                                               int buffer_count) {
    // Allocate temporary local arrays to hold sorted candidates from the buffer.
    // We assume buffer_count <= K.
    int sorted_buffer_indices[K];
    float sorted_buffer_dists[K];

    // Copy the candidate buffer into local arrays.
    for (int i = 0; i < buffer_count; i++) {
        sorted_buffer_indices[i] = buffer_indices[i];
        sorted_buffer_dists[i] = buffer_dists[i];
    }
    // Insertion sort on the candidate buffer.
    for (int i = 1; i < buffer_count; i++) {
        int key_idx = sorted_buffer_indices[i];
        float key_dist = sorted_buffer_dists[i];
        int j = i - 1;
        while (j >= 0 && sorted_buffer_dists[j] > key_dist) {
            sorted_buffer_dists[j + 1] = sorted_buffer_dists[j];
            sorted_buffer_indices[j + 1] = sorted_buffer_indices[j];
            j--;
        }
        sorted_buffer_dists[j + 1] = key_dist;
        sorted_buffer_indices[j + 1] = key_idx;
    }

    // Merge the sorted intermediate candidate list (size K) with the sorted candidate
    // buffer (size buffer_count). The merged result has size (K + buffer_count),
    // from which we keep only the best K candidates.
    const int total = K + buffer_count;
    int merged_indices[2 * K];   // Maximum possible size if buffer_count equals K.
    float merged_dists[2 * K];
    int i = 0, j = 0, m = 0;
    while (m < total && (i < K || j < buffer_count)) {
        float d1 = (i < K) ? inter_dists[i] : std::numeric_limits<float>::infinity();
        float d2 = (j < buffer_count) ? sorted_buffer_dists[j] : std::numeric_limits<float>::infinity();
        if (d1 <= d2) {
            merged_dists[m] = d1;
            merged_indices[m] = inter_indices[i];
            i++;
        } else {
            merged_dists[m] = d2;
            merged_indices[m] = sorted_buffer_indices[j];
            j++;
        }
        m++;
    }
    // Update the intermediate candidate list with the best K candidates.
    for (int i = 0; i < K; i++) {
        inter_dists[i] = merged_dists[i];
        inter_indices[i] = merged_indices[i];
    }
}

// -----------------------------------------------------------------------------
// Templated CUDA kernel for k-nearest neighbors (k-NN) using one warp (32 threads)
// per query. For each query, a shared memory buffer of size k is allocated to hold:
//   - The intermediate candidate list (sorted ascending by distance)
//   - A candidate buffer used to enqueue new candidates
//   - A candidate count variable tracking the number of enqueued candidates.
// While processing data points loaded into a shared memory tile, candidates with a
// distance less than the current worst candidate (k-th nearest neighbor) are enqueued
// into the candidate buffer. Whenever the buffer fills (i.e. reaches k elements),
// it is merged with the intermediate candidate list, and the buffer is reset.
// -----------------------------------------------------------------------------
template <int K>
__global__ void knn_kernel_templated(const float2 *query, int query_count,
                                     const float2 *data, int data_count,
                                     std::pair<int, float> *result) {
    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize; // Each warp processes one query.
    int lane = threadIdx.x % warpSize;
    if (warp_id >= query_count)
        return;

    // Load the query point (shared by all threads in the warp).
    float2 q = query[warp_id];

    // -------------------------------------------------------------------------
    // Allocate a per-warp shared memory region for candidate data.
    // Shared memory layout for each warp:
    //   - Intermediate candidate list: K ints (indices) and K floats (dists)
    //   - Candidate buffer: K ints (indices) and K floats (dists)
    //   - Candidate count: 1 int.
    // Total size per warp = 2*K*(sizeof(int)+sizeof(float)) + sizeof(int).
    // -------------------------------------------------------------------------
    extern __shared__ char shared_mem[];
    const size_t per_warp_size = 2 * K * (sizeof(int) + sizeof(float)) + sizeof(int);
    char *warp_mem = shared_mem + ((threadIdx.x / warpSize) * per_warp_size);
    int *inter_indices = (int *)warp_mem;                 // Intermediate candidate indices (size K).
    float *inter_dists = (float *)(inter_indices + K);      // Intermediate candidate distances (size K).
    int *buffer_indices = (int *)(inter_dists + K);         // Candidate buffer indices (size K).
    float *buffer_dists = (float *)(buffer_indices + K);    // Candidate buffer distances (size K).
    int *candidate_count = (int *)(buffer_dists + K);       // Candidate count (single int).

    // Initialize the intermediate candidate list with worst-case values and candidate count to 0.
    for (int i = lane; i < K; i += warpSize) {
        inter_indices[i] = -1;
        inter_dists[i] = std::numeric_limits<float>::infinity();
    }
    if (lane == 0)
        *candidate_count = 0;
    __syncwarp();

    // Shared memory tile for data points.
    __shared__ float2 tile[TILE_SIZE];

    // Process input data in batches (tiles).
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (batch_start + TILE_SIZE > data_count)
            tile_size = data_count - batch_start;
        // Load current tile into shared memory cooperatively.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[batch_start + i];
        }
        __syncthreads();

        // Process each data point in the tile assigned to this warp.
        for (int i = lane; i < tile_size; i += warpSize) {
            float d = squared_distance(q, tile[i]);
            // Read current worst candidate from the intermediate candidate list.
            float threshold = inter_dists[K - 1];
            if (d >= threshold)
                continue;
            // Enqueue the candidate into the candidate buffer.
            int pos = atomicAdd(candidate_count, 1);
            if (pos < K) {
                buffer_indices[pos] = batch_start + i;
                buffer_dists[pos] = d;
            }
        }
        __syncwarp();

        // If the candidate buffer is full, merge it with the intermediate candidate list.
        if (*candidate_count >= K) {
            if (lane == 0) {
                int count = *candidate_count;
                // Clamp the count to K in case atomicAdd overcounted.
                if (count > K)
                    count = K;
                merge_buffer_with_intermediate<K>(inter_indices, inter_dists,
                                                  buffer_indices, buffer_dists, count);
                *candidate_count = 0;
            }
            __syncwarp();
        }
        __syncthreads();  // Ensure the tile is processed before loading the next batch.
    }

    // After processing all tiles, if any candidates remain in the buffer, merge them.
    if (*candidate_count > 0) {
        if (lane == 0) {
            int count = *candidate_count;
            if (count > K)
                count = K;
            merge_buffer_with_intermediate<K>(inter_indices, inter_dists,
                                              buffer_indices, buffer_dists, count);
            *candidate_count = 0;
        }
        __syncwarp();
    }

    // Write the final intermediate candidate list (k nearest neighbors) to global memory.
    for (int i = lane; i < K; i += warpSize) {
        result[warp_id * K + i] = std::make_pair(inter_indices[i], inter_dists[i]);
    }
}

// -----------------------------------------------------------------------------
// Host function: Dispatch the templated k-NN kernel based on the runtime value of k.
// k is assumed to be a power of two between 32 and 1024 (inclusive).
// The shared memory size per block is computed from the number of warps per block and
// the per-warp memory block size.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    int threads_per_block = 256;  // 256 threads per block (8 warps per block).
    int total_warps = query_count; // One warp per query.
    int total_threads = total_warps * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    int warps_per_block = threads_per_block / 32;
    // Compute per-warp shared memory size in bytes.
    size_t per_warp_size = 2 * k * (sizeof(int) + sizeof(float)) + sizeof(int);
    // Total shared memory per block.
    size_t shared_mem_size = warps_per_block * per_warp_size;

    switch(k) {
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
            return;
    }
    cudaDeviceSynchronize();
}
