#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>      // for FLT_MAX
#include <utility>     // for std::pair

// -----------------------------------------------------------------------------
// In this implementation, each query is processed by one warp (32 threads).
// Each warp maintains its own private candidate list of k nearest neighbors,
// distributed evenly across its 32 lanes. In particular, each thread holds
// k/32 candidates in its private array. For every new data point (loaded in
// tiles into shared memory) the warp’s lanes cooperatively compute the squared
// Euclidean distance to the query and update their candidate lists.
// After processing all data tiles, each warp locally sorts its  (k/32)-element
// lists (each lane sorts its private L candidates). Then, using a per-warp
// shared buffer (allocated in dynamic shared memory), the 32 lanes write their
// candidates in an interleaved fashion so that the combined buffer holds k
// unsorted candidates. Next, lane 0 of the warp collects these k candidates,
// performs an in-place bitonic sort (the input size k is always a power of two)
// to order them by ascending distance, and writes the sorted candidates back to
// the shared buffer. Finally, the warp’s threads cooperatively write the sorted
// k-nearest candidates to global memory.
// 
// The kernel iterates over the data points using a tile (batch) size defined by
// TILE_SIZE. The tile is loaded from global memory to shared memory once per
// iteration to amortize memory latency.
// ----------------------------------------------------------------------------- 

// Define the tile size for caching data points in shared memory.
#define TILE_SIZE 1024

// CUDA kernel implementing k-NN for 2D points.
// Each warp processes one query.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Determine which warp (within the block and globally) we are in.
    int warp_id = threadIdx.x / 32;      // warp index within block
    int lane     = threadIdx.x & 31;       // lane index within warp
    int warpsPerBlock = blockDim.x / 32;
    int global_warp_id = blockIdx.x * warpsPerBlock + warp_id;
    if (global_warp_id >= query_count)
        return;  // if there is no query assigned to this warp, exit.

    // Load the query point for this warp.
    float2 q = query[global_warp_id];

    // Each warp will maintain k nearest neighbor candidates.
    // They are distributed evenly among the 32 threads in the warp.
    // Let L = k/32: each thread stores L candidates.
    int L = k / 32;

    // Each thread maintains a private candidate list (local arrays) of size L.
    // Each candidate is a pair: (data point index, squared distance).
    // Initialize distances to FLT_MAX and indices to -1.
    float local_dists[32];  // maximal L is 1024/32 = 32.
    int   local_inds[32];
#pragma unroll
    for (int i = 0; i < L; i++) {
        local_dists[i] = FLT_MAX;
        local_inds[i]  = -1;
    }

    // -------------------------------------------------------------------------
    // Shared memory allocation:
    // The dynamic shared memory is partitioned as follows:
    // 1. A tile for caching data points: TILE_SIZE elements of type float2.
    // 2. A per-warp candidate merge buffer: each warp uses k elements of
    //    type std::pair<int, float>. The number of warps per block is (blockDim.x/32).
    // The total shared memory size (in bytes) should be set accordingly at kernel launch.
    // -------------------------------------------------------------------------
    extern __shared__ char shared_mem[];
    // Shared memory region for data tile.
    float2 *tile = reinterpret_cast<float2*>(shared_mem);
    // Compute pointer to candidate merge buffer (placed after the tile).
    std::pair<int, float> *warp_candidates =
         reinterpret_cast<std::pair<int, float>*>(shared_mem + TILE_SIZE * sizeof(float2));

    // Calculate the number of tiles needed to cover all data points.
    int num_tiles = (data_count + TILE_SIZE - 1) / TILE_SIZE;

    // Process data points in tiles.
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Load a tile of data points from global memory into shared memory.
        // Let only the first TILE_SIZE threads in the block (if blockDim.x > TILE_SIZE)
        // do the loading.
        int load_idx = tile_idx * TILE_SIZE + threadIdx.x;
        if (threadIdx.x < TILE_SIZE) {
            if (load_idx < data_count)
                tile[threadIdx.x] = data[load_idx];
            else {
                // For out-of-bound indices, fill with dummy values.
                tile[threadIdx.x].x = 0.f;
                tile[threadIdx.x].y = 0.f;
            }
        }
        __syncthreads(); // ensure the tile is loaded before use

        // Determine the number of valid data points in this tile.
        int tile_points = TILE_SIZE;
        if (tile_idx == num_tiles - 1)
            tile_points = data_count - tile_idx * TILE_SIZE;

        // Each warp processes the current tile. The lanes cooperate in a strided loop.
        for (int i = lane; i < tile_points; i += 32) {
            float2 d_pt = tile[i];
            float dx = q.x - d_pt.x;
            float dy = q.y - d_pt.y;
            float dist = dx * dx + dy * dy;  // squared Euclidean distance
            int data_index = tile_idx * TILE_SIZE + i;

            // Update the thread’s private candidate list.
            // Find the worst (largest) distance in the local candidate array.
            float worst = local_dists[0];
            int worst_pos = 0;
#pragma unroll
            for (int j = 1; j < L; j++) {
                if (local_dists[j] > worst) {
                    worst = local_dists[j];
                    worst_pos = j;
                }
            }
            // If the new candidate is better, update the candidate at the worst position.
            if (dist < worst) {
                local_dists[worst_pos] = dist;
                local_inds[worst_pos]  = data_index;
            }
        }
        // Synchronize within the warp (not strictly needed here, but good practice).
        __syncwarp();
        __syncthreads(); // synchronize all threads before reusing shared tile memory
    } // end for each tile

    // Each thread now has an unsorted list of L candidate distances/indices.
    // Sort the local candidate list in ascending order using insertion sort.
    for (int i = 1; i < L; i++) {
        float key_val = local_dists[i];
        int   key_ind = local_inds[i];
        int j = i - 1;
        while (j >= 0 && local_dists[j] > key_val) {
            local_dists[j + 1] = local_dists[j];
            local_inds[j + 1]  = local_inds[j];
            j--;
        }
        local_dists[j + 1] = key_val;
        local_inds[j + 1]  = key_ind;
    }

    // Write each thread's sorted local list to the warp candidate merge buffer.
    // The candidates are interleaved: each thread writes its i-th candidate to location (i*32 + lane).
    std::pair<int, float> *myWarpBuffer = warp_candidates + warp_id * k;
#pragma unroll
    for (int i = 0; i < L; i++) {
        myWarpBuffer[i * 32 + lane] = std::pair<int, float>(local_inds[i], local_dists[i]);
    }
    __syncwarp();

    // Now the per-warp buffer "myWarpBuffer" holds k candidate pairs (unsorted).
    // We need to sort these k candidates (by ascending distance) to obtain the final k-NN list.
    // For simplicity, we let lane 0 of the warp perform a serial bitonic sort on the k elements.
    if (lane == 0) {
        // Allocate a temporary array (on local memory) to hold the k candidates.
        // Maximum k is 1024.
        std::pair<int, float> sorted[1024];
#pragma unroll
        for (int i = 0; i < k; i++) {
            sorted[i] = myWarpBuffer[i];
        }
        // Bitonic sort: k is guaranteed to be a power-of-two.
        for (int size = 2; size <= k; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = 0; i < k; i++) {
                    int j = i ^ stride;
                    if (j > i) {
                        // Determine sorting direction.
                        bool ascending = ((i & size) == 0);
                        if (ascending) {
                            if (sorted[i].second > sorted[j].second) {
                                std::pair<int, float> tmp = sorted[i];
                                sorted[i] = sorted[j];
                                sorted[j] = tmp;
                            }
                        } else {
                            if (sorted[i].second < sorted[j].second) {
                                std::pair<int, float> tmp = sorted[i];
                                sorted[i] = sorted[j];
                                sorted[j] = tmp;
                            }
                        }
                    }
                }
            }
        }
        // Write back the sorted array to the warp candidate buffer.
        for (int i = 0; i < k; i++) {
            myWarpBuffer[i] = sorted[i];
        }
    }
    __syncwarp(); // ensure all lanes see the sorted candidates

    // Finally, write the k sorted results for this query to global memory.
    // Each warp writes to result[global_warp_id * k ... global_warp_id * k + k - 1].
    int base_index = global_warp_id * k;
    for (int i = lane; i < k; i += 32) {
        result[base_index + i] = myWarpBuffer[i];
    }
}

// -----------------------------------------------------------------------------
// Host interface function that launches the CUDA kernel.
// The caller is responsible for allocating 'query', 'data', and 'result' on device.
// k is a power of 2 between 32 and 1024 inclusive.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose threads-per-block. Each warp (32 threads) processes one query.
    // We choose 128 threads per block (i.e. 4 warps per block).
    const int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Total number of warps (queries) to process.
    int totalWarps = query_count;
    int numBlocks = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate shared memory size (in bytes):
    // 1. For data tile: TILE_SIZE * sizeof(float2)
    // 2. For candidate merge buffers: (warpsPerBlock * k) * sizeof(std::pair<int, float>)
    size_t shmemSize = TILE_SIZE * sizeof(float2) +
                       warpsPerBlock * k * sizeof(std::pair<int, float>);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, shmemSize>>>(query, query_count,
                                                          data, data_count,
                                                          result, k);
    // Optionally, one might add error checking of the kernel launch here.
}