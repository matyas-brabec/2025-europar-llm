#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <utility>

#define TILE_SIZE 256    // Tile size for caching data points in shared memory.
#define LOCAL_L 8        // Number of local candidates each thread collects per tile.
#define WARP_SIZE 32     // Number of threads per warp.

// -----------------------------------------------------------------------------
// Device function: Compute squared Euclidean distance between two 2D points.
// -----------------------------------------------------------------------------
__device__ float squared_distance(float2 a, float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// -----------------------------------------------------------------------------
// Device function: Merge an unsorted candidate buffer with the intermediate
// candidate list. The candidate buffer (of n elements) is first sorted using
// insertion sort and then merged with the sorted intermediate list (of size K),
// updating the intermediate list with the best K candidates.
// -----------------------------------------------------------------------------
template <int K>
__device__ void merge_buffer_with_intermediate(int *inter_indices, float *inter_dists,
                                               int *buffer_indices, float *buffer_dists,
                                               int n) {
    // Temporary arrays to hold sorted buffer candidates.
    int sorted_buf_idx[WARP_SIZE * LOCAL_L];
    float sorted_buf_dists[WARP_SIZE * LOCAL_L];
    for (int i = 0; i < n; i++) {
        sorted_buf_idx[i] = buffer_indices[i];
        sorted_buf_dists[i] = buffer_dists[i];
    }
    // Insertion sort the candidate buffer.
    for (int i = 1; i < n; i++) {
        int key_idx = sorted_buf_idx[i];
        float key_dist = sorted_buf_dists[i];
        int j = i - 1;
        while (j >= 0 && sorted_buf_dists[j] > key_dist) {
            sorted_buf_dists[j + 1] = sorted_buf_dists[j];
            sorted_buf_idx[j + 1] = sorted_buf_idx[j];
            j--;
        }
        sorted_buf_dists[j + 1] = key_dist;
        sorted_buf_idx[j + 1] = key_idx;
    }
    // Merge the sorted intermediate list (size K) with the sorted buffer (size n).
    const int total = K + n;
    int merged_idx[2 * K];    // Sufficient when n <= K.
    float merged_dists[2 * K];
    int i = 0, j = 0, m = 0;
    while (m < total && (i < K || j < n)) {
        float d1 = (i < K) ? inter_dists[i] : std::numeric_limits<float>::infinity();
        float d2 = (j < n) ? sorted_buf_dists[j] : std::numeric_limits<float>::infinity();
        if (d1 <= d2) {
            merged_dists[m] = d1;
            merged_idx[m] = inter_indices[i];
            i++;
        } else {
            merged_dists[m] = d2;
            merged_idx[m] = sorted_buf_idx[j];
            j++;
        }
        m++;
    }
    // Update the intermediate candidate list with the best K candidates.
    for (int i = 0; i < K; i++) {
        inter_dists[i] = merged_dists[i];
        inter_indices[i] = merged_idx[i];
    }
}

// -----------------------------------------------------------------------------
// Templated CUDA kernel for k-nearest neighbors (k-NN) using one warp (32 threads)
// per query. Each warp uses a shared memory region that holds:
//   - An intermediate candidate list of K indices and K distances (sorted ascending).
//   - A candidate buffer of (WARP_SIZE * LOCAL_L) indices and distances.
// While processing a shared memory tile of data points, each thread collects up to
// LOCAL_L local candidates (in registers) that are closer than the current worst
// candidate in the intermediate list. After processing a tile, all threads write
// their local candidates into the candidate buffer, and then the warp cooperatively
// merges the buffer with the intermediate candidate list.
// -----------------------------------------------------------------------------
template <int K>
__global__ void knn_kernel_templated(const float2 *query, int query_count,
                                     const float2 *data, int data_count,
                                     std::pair<int, float> *result) {
    // Each warp processes one query.
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (warp_id >= query_count)
        return;

    // Load query point (shared by all threads in the warp).
    float2 q = query[warp_id];

    // -------------------------------------------------------------------------
    // Allocate per-warp shared memory region.
    // Layout per warp:
    //   - Intermediate candidate list: K ints + K floats.
    //   - Candidate buffer: (WARP_SIZE * LOCAL_L) ints + (WARP_SIZE * LOCAL_L) floats.
    // Total per-warp size = K*(sizeof(int)+sizeof(float)) +
    //                      (WARP_SIZE * LOCAL_L)*(sizeof(int)+sizeof(float)).
    // -------------------------------------------------------------------------
    extern __shared__ char shared_mem[];
    int warps_per_block = blockDim.x / WARP_SIZE;
    int warp_local_id = threadIdx.x / WARP_SIZE;
    size_t per_warp_size = K * (sizeof(int) + sizeof(float)) +
                           (WARP_SIZE * LOCAL_L) * (sizeof(int) + sizeof(float)) +
                           sizeof(int); // Adding space for candidate_count.
    char *warp_mem = shared_mem + warp_local_id * per_warp_size;
    int *inter_indices = (int *)warp_mem;                // Intermediate candidate indices (size K).
    float *inter_dists = (float *)(inter_indices + K);     // Intermediate candidate distances (size K).
    int *buffer_indices = (int *)(warp_mem + K * (sizeof(int) + sizeof(float)));  // Candidate buffer indices.
    float *buffer_dists = (float *)(buffer_indices + WARP_SIZE * LOCAL_L);          // Candidate buffer distances.
    int *candidate_count = (int *)(buffer_dists + WARP_SIZE * LOCAL_L);             // Candidate count.

    // Initialize intermediate candidate list.
    for (int i = lane; i < K; i += WARP_SIZE) {
        inter_indices[i] = -1;
        inter_dists[i] = std::numeric_limits<float>::infinity();
    }
    if (lane == 0)
        *candidate_count = 0;
    __syncwarp();

    // Each thread maintains a local candidate list in registers.
    int local_indices[LOCAL_L];
    float local_dists[LOCAL_L];
    for (int i = 0; i < LOCAL_L; i++) {
        local_indices[i] = -1;
        local_dists[i] = std::numeric_limits<float>::infinity();
    }

    // Shared memory tile for data points.
    __shared__ float2 tile[TILE_SIZE];

    // Process input data in tiles.
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (batch_start + TILE_SIZE > data_count)
            tile_size = data_count - batch_start;
        // Load tile into shared memory.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[batch_start + i];
        }
        __syncthreads();

        // Each thread processes its portion of the tile.
        for (int i = lane; i < tile_size; i += WARP_SIZE) {
            float d = squared_distance(q, tile[i]);
            float threshold = inter_dists[K - 1];
            if (d >= threshold)
                continue;
            // Insertion sort into local candidate list.
            if (d < local_dists[LOCAL_L - 1]) {
                int pos = LOCAL_L - 1;
                while (pos > 0 && local_dists[pos - 1] > d) {
                    local_dists[pos] = local_dists[pos - 1];
                    local_indices[pos] = local_indices[pos - 1];
                    pos--;
                }
                local_dists[pos] = d;
                local_indices[pos] = batch_start + i;
            }
        }
        __syncwarp();

        // Write each thread's local candidate list into the candidate buffer.
        for (int i = 0; i < LOCAL_L; i++) {
            buffer_indices[lane * LOCAL_L + i] = local_indices[i];
            buffer_dists[lane * LOCAL_L + i] = local_dists[i];
        }
        __syncwarp();

        // Enqueue logic: if any thread enqueued a candidate that overflows the buffer,
        // perform a merge. All lanes participate based on a ballot.
        int pos = atomicAdd((unsigned int*)&buffer_indices[WARP_SIZE * LOCAL_L], 0); // Dummy read.
        // Instead, use a shared variable for candidate count.
        if (*candidate_count >= WARP_SIZE * LOCAL_L) {
            // Perform a merge.
            if (lane == 0) {
                int count = *candidate_count;
                if (count > WARP_SIZE * LOCAL_L)
                    count = WARP_SIZE * LOCAL_L;
                merge_buffer_with_intermediate<K>(inter_indices, inter_dists,
                                                  buffer_indices, buffer_dists, count);
                *candidate_count = 0;
            }
            __syncwarp();
        }
    }

    // After processing all tiles, if any candidates remain in the buffer, merge them.
    int final_count = *candidate_count;
    if (final_count > 0) {
        __syncwarp();
        if (lane == 0) {
            if (final_count > WARP_SIZE * LOCAL_L)
                final_count = WARP_SIZE * LOCAL_L;
            merge_buffer_with_intermediate<K>(inter_indices, inter_dists,
                                              buffer_indices, buffer_dists, final_count);
            *candidate_count = 0;
        }
        __syncwarp();
    }

    // Write final intermediate candidate list (k nearest neighbors) to global memory.
    for (int i = lane; i < K; i += WARP_SIZE) {
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
    int total_threads = total_warps * WARP_SIZE;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    int warps_per_block = threads_per_block / WARP_SIZE;
    size_t per_warp_size = k * (sizeof(int) + sizeof(float)) +
                           (WARP_SIZE * LOCAL_L) * (sizeof(int) + sizeof(float)) + sizeof(int);
    size_t shared_mem_size = warps_per_block * per_warp_size;

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
            return;
    }
    cudaDeviceSynchronize();
}
