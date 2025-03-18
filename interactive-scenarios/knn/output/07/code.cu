#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <utility>

// -----------------------------------------------------------------------------
// This kernel implements k-nearest neighbors (k-NN) for 2D points using a
// single warp (32 threads) per query. The warp processes the data in tiles
// loaded into shared memory, and each warp maintains a global candidate list
// (of size K) in shared memory. Within each tile, each thread computes a local
// candidate list (of size LOCAL_L) for its assigned portion. The 32 local lists
// are merged tile-by-tile into the warp's global candidate list using dynamic
// shared memory. Appropriate warp and block synchronizations are used.
// -----------------------------------------------------------------------------

// Tile size for caching data points in shared memory.
#define TILE_SIZE 256
// Each thread in a warp maintains LOCAL_L candidates from each tile.
#define LOCAL_L 8

// Compute squared Euclidean distance between two 2D points.
__device__ float squared_distance(float2 a, float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Templated kernel where K is the total number of nearest neighbors per query.
// One warp (32 threads) processes one query.
template <int K>
__global__ void knn_kernel_templated(const float2 *query, int query_count,
                                     const float2 *data, int data_count,
                                     std::pair<int, float> *result) {
    const int warpSize = 32;
    // Each warp processes one query.
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;  // query index
    int lane = threadIdx.x % warpSize;           // lane index within the warp
    if (warp_id >= query_count)
        return;

    // Load the query point (all threads in the warp use the same query).
    float2 q = query[warp_id];

    // -------------------------------------------------------------------------
    // Allocate per-warp global candidate list in dynamic shared memory.
    // The layout in dynamic shared memory per block is as follows for each warp:
    //   [ Global Candidate List ]: K ints (indices) followed by K floats (dists)
    //   [ Merge Buffer ]: (32 * LOCAL_L) ints followed by (32 * LOCAL_L) floats.
    // Each block has (blockDim.x/32) warps.
    // -------------------------------------------------------------------------
    int warps_per_block = blockDim.x / warpSize;
    int warp_id_in_block = threadIdx.x / warpSize;
    extern __shared__ char shared_mem[];

    // Pointers for global candidate list.
    int *global_indices = (int*)shared_mem;
    float *global_dists = (float*)(shared_mem + warps_per_block * K * sizeof(int));

    // Pointers for merge buffer (for candidate merging from the current tile).
    int *merge_indices = (int*)(shared_mem + warps_per_block * (K * sizeof(int) + K * sizeof(float)));
    float *merge_dists = (float*)(shared_mem + warps_per_block * (K * sizeof(int) + K * sizeof(float)) +
                                  warps_per_block * 32 * LOCAL_L * sizeof(int));

    // Get pointers for this warp.
    int *warp_global_indices = global_indices + warp_id_in_block * K;
    float *warp_global_dists = global_dists + warp_id_in_block * K;
    int *warp_merge_indices = merge_indices + warp_id_in_block * (32 * LOCAL_L);
    float *warp_merge_dists = merge_dists + warp_id_in_block * (32 * LOCAL_L);

    // Initialize the global candidate list for this warp with worst-case values.
    for (int i = lane; i < K; i += warpSize) {
        warp_global_indices[i] = -1;
        warp_global_dists[i] = std::numeric_limits<float>::infinity();
    }
    __syncwarp();

    // Shared memory tile for data points (shared by all threads in the block).
    __shared__ float2 tile[TILE_SIZE];

    // Process data points in batches loaded into shared memory.
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (batch_start + TILE_SIZE > data_count)
            tile_size = data_count - batch_start;

        // Load the current tile into shared memory cooperatively by all threads in the block.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[batch_start + i];
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // Each thread in the warp computes distances for its assigned subset of
        // the tile and maintains a private local candidate list of size LOCAL_L.
        // ---------------------------------------------------------------------
        int local_indices[LOCAL_L];
        float local_dists[LOCAL_L];
        // Initialize local candidate list.
        for (int i = 0; i < LOCAL_L; i++) {
            local_indices[i] = -1;
            local_dists[i] = std::numeric_limits<float>::infinity();
        }
        // Process tile elements with stride = warpSize.
        for (int i = lane; i < tile_size; i += warpSize) {
            float d = squared_distance(q, tile[i]);
            // Insertion sort into the local list.
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

        // Write each thread's local candidate list into the warp's merge buffer.
        for (int i = 0; i < LOCAL_L; i++) {
            warp_merge_indices[lane * LOCAL_L + i] = local_indices[i];
            warp_merge_dists[lane * LOCAL_L + i] = local_dists[i];
        }
        __syncwarp();

        // ---------------------------------------------------------------------
        // Lane 0 of the warp merges the 32*LOCAL_L candidates from this tile.
        // The merged tile candidate list (of T candidates, where T = 32*LOCAL_L)
        // is sorted in ascending order (best candidates first).
        // ---------------------------------------------------------------------
        if (lane == 0) {
            const int T = warpSize * LOCAL_L;
            int tile_candidates_indices[T];
            float tile_candidates_dists[T];
            for (int i = 0; i < T; i++) {
                tile_candidates_indices[i] = warp_merge_indices[i];
                tile_candidates_dists[i] = warp_merge_dists[i];
            }
            // Insertion sort for T candidates.
            for (int i = 1; i < T; i++) {
                int idx = tile_candidates_indices[i];
                float d = tile_candidates_dists[i];
                int j = i;
                while (j > 0 && tile_candidates_dists[j - 1] > d) {
                    tile_candidates_dists[j] = tile_candidates_dists[j - 1];
                    tile_candidates_indices[j] = tile_candidates_indices[j - 1];
                    j--;
                }
                tile_candidates_dists[j] = d;
                tile_candidates_indices[j] = idx;
            }
            // -----------------------------------------------------------------
            // Merge the sorted tile candidates (T elements) with the current
            // global candidate list (K elements) to update the global list.
            // Both lists are sorted in ascending order.
            // -----------------------------------------------------------------
            int merged_indices[K];
            float merged_dists[K];
            int i = 0, j = 0, m = 0;
            while (m < K && (i < K || j < T)) {
                float d1 = (i < K) ? warp_global_dists[i] : std::numeric_limits<float>::infinity();
                float d2 = (j < T) ? tile_candidates_dists[j] : std::numeric_limits<float>::infinity();
                if (d1 <= d2) {
                    merged_dists[m] = d1;
                    merged_indices[m] = warp_global_indices[i];
                    i++;
                } else {
                    merged_dists[m] = d2;
                    merged_indices[m] = tile_candidates_indices[j];
                    j++;
                }
                m++;
            }
            // Update the global candidate list with the merged results.
            for (int i = 0; i < K; i++) {
                warp_global_dists[i] = merged_dists[i];
                warp_global_indices[i] = merged_indices[i];
            }
        }
        __syncwarp();
        __syncthreads();  // Ensure the shared memory tile is free for the next batch.
    }

    // After processing all tiles, lane 0 writes the final candidate list to global memory.
    if (lane == 0) {
        for (int i = 0; i < K; i++) {
            result[warp_id * K + i] = std::make_pair(warp_global_indices[i], warp_global_dists[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// Host function to dispatch the k-NN kernel.
// Each query is processed by one warp (32 threads). The runtime parameter 'k'
// is assumed to be a power of two between 32 and 1024 (inclusive). The shared
// memory size is computed based on the number of warps per block, k, and LOCAL_L.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    int threads_per_block = 256; // 256 threads per block => 8 warps per block.
    // One warp per query.
    int total_warps = query_count;
    int total_threads = total_warps * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    int warps_per_block = threads_per_block / 32;
    // Compute shared memory size per block:
    // For each warp, we need:
    //   - k ints for global candidate indices
    //   - k floats for global candidate distances
    //   - (32 * LOCAL_L) ints for merge buffer
    //   - (32 * LOCAL_L) floats for merge buffer
    size_t shm_per_warp = k * sizeof(int) + k * sizeof(float) +
                          32 * LOCAL_L * (sizeof(int) + sizeof(float));
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
