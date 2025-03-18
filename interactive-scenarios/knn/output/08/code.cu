#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <utility>

#define TILE_SIZE 256         // Tile size for loading data into shared memory.
#define LOCAL_L 8             // Each thread collects LOCAL_L candidates from each tile.

// -----------------------------------------------------------------------------
// Device function: Compute the squared Euclidean distance between two 2D points.
// -----------------------------------------------------------------------------
__device__ float squared_distance(float2 a, float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// -----------------------------------------------------------------------------
// Device function: In-warp parallel bitonic sort for an array of T elements,
// stored in shared memory. T must be a power of two (here T = warpSize * LOCAL_L).
// The sort orders the candidate distances in ascending order and applies the same
// swaps to the corresponding indices.
// -----------------------------------------------------------------------------
__device__ void bitonic_sort_in_warp(int *arr_indices, float *arr_dists, int T) {
    int lane = threadIdx.x % 32;
    // k is the size of the subsequence to be merged.
    for (int k = 2; k <= T; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each thread processes elements spaced by warpSize.
            for (int i = lane; i < T; i += 32) {
                int ixj = i ^ j;
                if (ixj > i) {
                    // Determine the sorting order: ascending if the bit k is 0.
                    bool ascending = ((i & k) == 0);
                    float d_i = arr_dists[i];
                    float d_ixj = arr_dists[ixj];
                    bool swap = (ascending && d_i > d_ixj) || (!ascending && d_i < d_ixj);
                    if (swap) {
                        // Swap distances.
                        float temp = arr_dists[i];
                        arr_dists[i] = arr_dists[ixj];
                        arr_dists[ixj] = temp;
                        // Swap indices.
                        int temp_idx = arr_indices[i];
                        arr_indices[i] = arr_indices[ixj];
                        arr_indices[ixj] = temp_idx;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// -----------------------------------------------------------------------------
// Device function: For a given output index r (0 <= r < K), merge two sorted
// candidate arrays— the global candidate list (of length K) and the tile candidate
// list (of length T)— and return the candidate at merged position r (i.e. the r-th
// best candidate). This is done by binary search partitioning.
// -----------------------------------------------------------------------------
__device__ void merge_candidate_at_index(int r,
    const int *global_indices, const float *global_dists, int K,
    const int *tile_indices, const float *tile_dists, int T,
    int &out_idx, float &out_dist)
{
    int low = max(0, r - T);
    int high = min(r, K);
    while (low < high) {
        int mid = (low + high) >> 1;
        int j = r - mid;
        if (mid < K && j > 0 && tile_dists[j - 1] > global_dists[mid])
            low = mid + 1;
        else
            high = mid;
    }
    int i = low;
    int j = r - i;
    if (i < K && (j >= T || global_dists[i] <= tile_dists[j])) {
        out_idx = global_indices[i];
        out_dist = global_dists[i];
    } else {
        out_idx = tile_indices[j];
        out_dist = tile_dists[j];
    }
}

// -----------------------------------------------------------------------------
// Templated CUDA kernel implementing k-nearest neighbors (k-NN) for 2D points.
// Each query is processed by one warp (32 threads). The kernel loads the data
// points in tiles, each thread computes a local candidate list from its portion,
// and then the warp cooperatively updates its global candidate list by merging
// in the new candidates concurrently using warp-level parallelism.
// K is the total number of nearest neighbors per query.
// -----------------------------------------------------------------------------
template <int K>
__global__ void knn_kernel_templated(const float2 *query, int query_count,
                                     const float2 *data, int data_count,
                                     std::pair<int, float> *result) {
    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;  // Each warp processes one query.
    int lane = threadIdx.x % warpSize;
    if (warp_id >= query_count)
        return;

    // Load the query point (all lanes in the warp share the same query).
    float2 q = query[warp_id];

    // -------------------------------------------------------------------------
    // Shared memory is partitioned per warp. Each warp has:
    //   (a) A global candidate list (of length K) to hold the best candidates so far,
    //       stored as two arrays: one for indices and one for distances.
    //   (b) A merge buffer for tile candidates (of length T = warpSize * LOCAL_L),
    //       stored as two arrays.
    // The layout in shared memory per block is as follows:
    //   [Global Candidate Lists for all warps]
    //   [Merge Buffers for all warps]
    // -------------------------------------------------------------------------
    extern __shared__ char shared_mem[];
    int warps_per_block = blockDim.x / warpSize;
    int warp_id_in_block = threadIdx.x / warpSize;

    // Global candidate list for this warp.
    int *warp_global_indices = (int*)shared_mem + warp_id_in_block * K;
    float *warp_global_dists = (float*)(shared_mem + warps_per_block * K * sizeof(int))
                                + warp_id_in_block * K;

    // Merge buffer for tile candidates for this warp.
    const int T = warpSize * LOCAL_L;
    int *warp_tile_indices = (int*)(shared_mem + warps_per_block * (K * sizeof(int) + K * sizeof(float)))
                               + warp_id_in_block * T;
    float *warp_tile_dists = (float*)(shared_mem + warps_per_block * (K * sizeof(int) + K * sizeof(float))
                                      + warps_per_block * T * sizeof(int))
                               + warp_id_in_block * T;

    // Initialize the global candidate list with worst-case values.
    for (int i = lane; i < K; i += warpSize) {
        warp_global_indices[i] = -1;
        warp_global_dists[i] = std::numeric_limits<float>::infinity();
    }
    __syncwarp();

    // Shared memory tile for data points (shared by the entire block).
    __shared__ float2 tile[TILE_SIZE];

    // Process the data points in batches loaded into shared memory.
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (batch_start + TILE_SIZE > data_count)
            tile_size = data_count - batch_start;

        // Load the current tile from global memory into shared memory.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[batch_start + i];
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // Each thread in the warp processes a subset of the tile (stride = warpSize)
        // and builds a private local candidate list of size LOCAL_L.
        // ---------------------------------------------------------------------
        int local_candidates[LOCAL_L];
        float local_dists[LOCAL_L];
        for (int i = 0; i < LOCAL_L; i++) {
            local_candidates[i] = -1;
            local_dists[i] = std::numeric_limits<float>::infinity();
        }
        for (int i = lane; i < tile_size; i += warpSize) {
            float d = squared_distance(q, tile[i]);
            // Insertion sort into the local candidate list.
            if (d < local_dists[LOCAL_L - 1]) {
                int pos = LOCAL_L - 1;
                while (pos > 0 && local_dists[pos - 1] > d) {
                    local_dists[pos] = local_dists[pos - 1];
                    local_candidates[pos] = local_candidates[pos - 1];
                    pos--;
                }
                local_dists[pos] = d;
                local_candidates[pos] = batch_start + i;
            }
        }
        __syncwarp();

        // Each thread writes its LOCAL_L candidates into its portion of the merge buffer.
        for (int i = 0; i < LOCAL_L; i++) {
            warp_tile_indices[lane * LOCAL_L + i] = local_candidates[i];
            warp_tile_dists[lane * LOCAL_L + i] = local_dists[i];
        }
        __syncwarp();

        // ---------------------------------------------------------------------
        // All 32 threads in the warp cooperatively sort the merge buffer of T
        // candidates using in-warp parallel bitonic sort.
        // ---------------------------------------------------------------------
        bitonic_sort_in_warp(warp_tile_indices, warp_tile_dists, T);
        __syncwarp();

        // ---------------------------------------------------------------------
        // Update the global candidate list by merging it with the sorted tile
        // candidate list. Instead of having a single thread do the merge, each lane
        // computes a segment of the merged result concurrently.
        // The merged array has length (K + T) but we only keep the best K candidates.
        // Each thread processes num_per_thread = K / warpSize output positions.
        // ---------------------------------------------------------------------
        int num_per_thread = K / warpSize;
        int start = lane * num_per_thread;
        int end = start + num_per_thread;
        // Temporary storage for the merged segment computed by this thread.
        int local_merged_indices[32];   // Maximum possible segment size.
        float local_merged_dists[32];
        for (int r = start; r < end; r++) {
            int merged_idx;
            float merged_dist;
            merge_candidate_at_index(r, warp_global_indices, warp_global_dists, K,
                                     warp_tile_indices, warp_tile_dists, T,
                                     merged_idx, merged_dist);
            local_merged_indices[r - start] = merged_idx;
            local_merged_dists[r - start] = merged_dist;
        }
        __syncwarp();
        // Write the merged segment back into the global candidate list.
        for (int r = start; r < end; r++) {
            warp_global_indices[r] = local_merged_indices[r - start];
            warp_global_dists[r] = local_merged_dists[r - start];
        }
        __syncwarp();
        __syncthreads();  // Ensure the tile buffer is free for the next batch.
    }

    // After all batches, lane 0 of the warp writes the final global candidate list
    // (the k nearest neighbors for this query) to global memory.
    if (lane == 0) {
        for (int i = 0; i < K; i++) {
            result[warp_id * K + i] = std::make_pair(warp_global_indices[i], warp_global_dists[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// Host function: Dispatch the k-NN kernel.
// Assumes k is a power of two between 32 and 1024 (inclusive).
// The shared memory size per block is computed based on the number of warps per block,
// K, and the merge buffer size.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    int threads_per_block = 256;  // 256 threads per block => 8 warps per block.
    int total_warps = query_count; // One warp per query.
    int total_threads = total_warps * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    int warps_per_block = threads_per_block / 32;
    // Compute shared memory per block:
    // For each warp, we need:
    //   - Global candidate list: k ints + k floats.
    //   - Merge buffer for tile candidates: (32 * LOCAL_L) ints + (32 * LOCAL_L) floats.
    size_t shm_per_warp = k * sizeof(int) + k * sizeof(float) +
                          (32 * LOCAL_L) * (sizeof(int) + sizeof(float));
    size_t shared_mem_size = warps_per_block * shm_per_warp;

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
            // Unsupported k value.
            return;
    }
    cudaDeviceSynchronize();
}
