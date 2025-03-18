#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <utility>

#define TILE_SIZE 256    // Tile size for caching data points in shared memory.
#define LOCAL_L 8        // Each thread collects up to LOCAL_L candidates per tile.
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
// candidate list. Both arrays are assumed to have 'n' elements in the candidate
// buffer (n = WARP_SIZE * LOCAL_L) and the intermediate candidate list has K
// elements sorted in ascending order (best candidates first).
//
// The candidate buffer is first sorted using insertion sort, then merged
// with the intermediate candidate list, and the best K candidates are stored
// back into the intermediate candidate list.
// -----------------------------------------------------------------------------
template <int K>
__device__ void merge_buffer_with_intermediate(int *inter_indices, float *inter_dists,
                                               int *buffer_indices, float *buffer_dists,
                                               int n) {
    // Local arrays for sorting the candidate buffer.
    int sorted_buf_idx[WARP_SIZE * LOCAL_L];
    float sorted_buf_dists[WARP_SIZE * LOCAL_L];

    // Copy candidate buffer into local arrays.
    for (int i = 0; i < n; i++) {
        sorted_buf_idx[i] = buffer_indices[i];
        sorted_buf_dists[i] = buffer_dists[i];
    }
    // Insertion sort on the candidate buffer (n elements).
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

    // Merge the sorted intermediate candidate list (size K) with the sorted
    // candidate buffer (size n) to form a merged array of size (K + n).
    const int total = K + n;
    int merged_idx[2 * K];    // Maximum total when n <= K.
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
// per query. For each query, a shared memory region is allocated (per warp) to hold:
//   (a) The intermediate candidate list: K indices and K distances (sorted ascending).
//   (b) A candidate buffer: WARP_SIZE * LOCAL_L indices and distances.
// Each thread processes a subset of data points from a shared memory tile and collects
// up to LOCAL_L local candidates (in registers) that are closer than the current worst
// candidate in the intermediate result. After processing a tile, all threads write their
// local candidate lists into the candidate buffer, and lane 0 merges the buffer with the
// intermediate candidate list. Then the local lists are cleared for the next tile.
// -----------------------------------------------------------------------------
template <int K>
__global__ void knn_kernel_templated(const float2 *query, int query_count,
                                     const float2 *data, int data_count,
                                     std::pair<int, float> *result) {
    const int warpSize = WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / warpSize;  // Each warp processes one query.
    int lane = threadIdx.x % warpSize;
    if (warp_id >= query_count)
        return;

    // Load the query point (all lanes in the warp use the same query).
    float2 q = query[warp_id];

    // -------------------------------------------------------------------------
    // Shared memory per warp layout:
    //   - Intermediate candidate list: K ints and K floats.
    //   - Candidate buffer: (warpSize * LOCAL_L) ints and (warpSize * LOCAL_L) floats.
    // Total per-warp size = K*(sizeof(int)+sizeof(float)) + (warpSize*LOCAL_L)*(sizeof(int)+sizeof(float)).
    // -------------------------------------------------------------------------
    extern __shared__ char shared_mem[];
    int warps_per_block = blockDim.x / warpSize;
    size_t inter_offset = warps_per_block * 0;  // start of intermediate list.
    // Base pointer for this warp's shared memory.
    char *warp_mem = shared_mem + ((threadIdx.x / warpSize) * (K * (sizeof(int) + sizeof(float)) +
                         warpSize * LOCAL_L * (sizeof(int) + sizeof(float)));
    int *inter_indices = (int *)warp_mem;                 // Intermediate candidate indices (size K).
    float *inter_dists = (float *)(inter_indices + K);      // Intermediate candidate distances (size K).
    int *cand_buffer_indices = (int *)(warp_mem + K * (sizeof(int) + sizeof(float)));  // Candidate buffer.
    float *cand_buffer_dists = (float *)(cand_buffer_indices + warpSize * LOCAL_L);      // Candidate buffer.

    // Initialize the intermediate candidate list with worst-case values.
    for (int i = lane; i < K; i += warpSize) {
        inter_indices[i] = -1;
        inter_dists[i] = std::numeric_limits<float>::infinity();
    }
    __syncwarp();

    // Each thread maintains its own local candidate list in registers.
    int local_indices[LOCAL_L];
    float local_dists[LOCAL_L];
    for (int i = 0; i < LOCAL_L; i++) {
        local_indices[i] = -1;
        local_dists[i] = std::numeric_limits<float>::infinity();
    }

    // Shared memory tile for loading data points.
    __shared__ float2 tile[TILE_SIZE];

    // Process data points in batches (tiles).
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (batch_start + TILE_SIZE > data_count)
            tile_size = data_count - batch_start;

        // Load current tile into shared memory cooperatively.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[batch_start + i];
        }
        __syncthreads();

        // Each thread processes its portion of the tile.
        for (int i = lane; i < tile_size; i += warpSize) {
            float d = squared_distance(q, tile[i]);
            // Read current worst candidate from intermediate candidate list.
            float threshold = inter_dists[K - 1];
            if (d >= threshold)
                continue;
            // Insertion sort into local candidate list (of size LOCAL_L).
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

        // Each thread writes its LOCAL_L candidates into the candidate buffer.
        for (int i = 0; i < LOCAL_L; i++) {
            cand_buffer_indices[lane * LOCAL_L + i] = local_indices[i];
            cand_buffer_dists[lane * LOCAL_L + i] = local_dists[i];
        }
        __syncwarp();

        // Lane 0 merges the candidate buffer with the intermediate candidate list.
        if (lane == 0) {
            // The candidate buffer has T = warpSize * LOCAL_L elements.
            const int T = warpSize * LOCAL_L;
            merge_buffer_with_intermediate<K>(inter_indices, inter_dists,
                                              cand_buffer_indices, cand_buffer_dists, T);
        }
        __syncwarp();

        // Reset local candidate lists for next tile.
        for (int i = 0; i < LOCAL_L; i++) {
            local_indices[i] = -1;
            local_dists[i] = std::numeric_limits<float>::infinity();
        }
        __syncwarp();
    }

    // After processing all tiles, lane 0 writes the final intermediate candidate list
    // (the k nearest neighbors for this query) to global memory.
    if (lane == 0) {
        for (int i = 0; i < K; i++) {
            result[warp_id * K + i] = std::make_pair(inter_indices[i], inter_dists[i]);
        }
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
    // Compute per-warp shared memory size:
    //   Intermediate candidate list: k * (sizeof(int) + sizeof(float))
    //   Candidate buffer: (WARP_SIZE * LOCAL_L) * (sizeof(int) + sizeof(float))
    size_t per_warp_size = k * (sizeof(int) + sizeof(float)) +
                           (WARP_SIZE * LOCAL_L) * (sizeof(int) + sizeof(float));
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
