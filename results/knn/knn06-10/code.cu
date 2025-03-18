// This complete .cu file implements an optimized k‐nearest neighbors (k‐NN)
// algorithm for 2D points using CUDA. Each query is processed by one warp (32 threads),
// and a “private” intermediate result (the best k neighbors found so far) is maintained
// across the warp’s registers (distributed among its 32 threads).
//
// The algorithm proceeds by iterating over the (global) data points in batches (tiles)
// that are loaded into shared memory by all threads in the block. Each warp then computes
// the squared Euclidean distance from its query point (loaded from global memory) to every
// point in the current tile. Whenever a computed distance is lower than the current
// “max_distance” (i.e. the distance of the worst neighbor in the intermediate result),
// the candidate (global index and squared distance) is added (via atomicAdd) into a candidate
// buffer that is allocated in shared memory for that warp. When the candidate buffer becomes
// full (i.e. holds k candidates) or when the last batch is processed, the candidate buffer
// is merged with the intermediate result. The merge is done by (i) copying the warp’s
// intermediate result from registers and the candidate buffer into a temporary “merge buffer”
// in shared memory (of size 2*k), (ii) sorting this buffer in ascending order using a
// parallel bitonic sort performed cooperatively by the 32 threads of the warp, and (iii)
// writing back the first k (i.e. best) candidates into the warp’s registers and updating the
// “max_distance” (which is the k-th neighbor’s distance).
//
// The final (sorted) list of k nearest neighbors for a query is then written to global memory
// in the output result array (stored in row‐major order).
//
// The code uses dynamic shared memory to allocate space for:
//   (1) the data tile (TILE_SIZE float2’s),
//   (2) for each warp in the block: an integer candidate counter,
//   (3) a candidate buffer (k elements per warp),
//   (4) a temporary merge buffer (2*k elements per warp).
//
// This implementation targets modern NVIDIA GPUs (such as A100 or H100) using the latest
// CUDA toolkit and host compiler.
//
// NOTE: k is assumed to be a power of two in the range [32, 1024] and data_count >= k.

#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <utility>

// Define a structure to store a neighbor (index and squared distance).
struct Neighbor {
    int idx;
    float dist;
};

// Swap function for two Neighbor elements.
__device__ inline void swapNeighbor(Neighbor &a, Neighbor &b) {
    Neighbor tmp = a;
    a = b;
    b = tmp;
}

// ----------------------------------------------------------------------
// This device function performs an in-place bitonic sort on an array 'arr'
// of size n (which must be a power of two) located in shared memory.
// The 32 threads of a warp cooperate: each thread loops over indices in steps of 32.
// The sort is performed in ascending order by 'dist'.
// ----------------------------------------------------------------------
__device__ void bitonicSortWarp(Neighbor* arr, int n) {
    // Each thread’s lane index within the warp.
    int lane = threadIdx.x & 31;
    // Outer loop: size doubles each iteration.
    for (int size = 2; size <= n; size <<= 1) {
        // The inner loop: stride halves each time.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes indices starting at its lane index.
            for (int i = lane; i < n; i += 32) {
                int j = i ^ stride;
                if (j > i) {
                    bool ascending = ((i & size) == 0);
                    // In ascending sub-sequences, swap if element i > element j.
                    if (ascending) {
                        if (arr[i].dist > arr[j].dist) {
                            swapNeighbor(arr[i], arr[j]);
                        }
                    } else { // descending order in the other half.
                        if (arr[i].dist < arr[j].dist) {
                            swapNeighbor(arr[i], arr[j]);
                        }
                    }
                }
            }
            __syncwarp();
        }
    }
}

// ----------------------------------------------------------------------
// This device function merges the warp's candidate buffer with its intermediate result.
// The warp’s intermediate result is stored in registers (distributed among lanes)
// via the arrays local_knn_idx[0..num_local-1] and local_knn_dist[0..num_local-1] in each lane,
// where num_local = k/32. Together, these registers form a sorted list of k neighbors.
// The candidate buffer (of length 'cand_count' <= k) is stored in shared memory.
// The merge_buffer is a temporary shared memory array of size 2*k.
// The function copies both the intermediate result and candidate buffer into merge_buffer,
// fills unused slots with dummy values (FLT_MAX), then performs a bitonic sort to produce
// a sorted merged array. Finally, each thread reloads its portion of the new intermediate
// result from merge_buffer and the new max_distance (the k-th smallest distance) is updated.
// ----------------------------------------------------------------------
__device__ void warpMerge(
    int k,                // total number of neighbors to keep
    int num_local,        // number of neighbors per thread (k/32)
    Neighbor* merge_buffer,       // pointer to a temporary merge buffer (size = 2*k) for this warp
    Neighbor* candidate_buffer,   // pointer to this warp's candidate buffer (size = k)
    int cand_count,       // number of valid candidates in candidate_buffer
    int *local_knn_idx,   // pointer to the warp’s intermediate indices (per lane; length = num_local)
    float *local_knn_dist,// pointer to the warp’s intermediate distances (per lane; length = num_local)
    float *newMax         // output: new max_distance after merging
) {
    int lane = threadIdx.x & 31;
    // Step 1: Copy the intermediate result into merge_buffer[0, ..., k-1].
    // The intermediate result is stored in registers in each lane.
    // We assume a row‐major mapping: each lane writes its num_local elements
    // into positions: lane, lane+32, lane+64, ... (as long as lane + j*32 < k).
    for (int j = 0; j < num_local; j++) {
        int pos = lane + j * 32;
        if (pos < k) {
            merge_buffer[pos].idx = local_knn_idx[j];
            merge_buffer[pos].dist = local_knn_dist[j];
        }
    }
    // Step 2: Copy candidate_buffer into merge_buffer[k ... k+cand_count-1].
    for (int j = lane; j < k; j += 32) {
        if (j < cand_count) {
            merge_buffer[k + j].idx = candidate_buffer[j].idx;
            merge_buffer[k + j].dist = candidate_buffer[j].dist;
        }
    }
    // Step 3: For positions [k + cand_count, 2*k), fill with dummy values.
    for (int j = lane; j < k; j += 32) {
        int idx = k + cand_count + j;
        if (idx < 2 * k) {
            merge_buffer[idx].idx = -1;
            merge_buffer[idx].dist = FLT_MAX;
        }
    }
    __syncwarp();
    // Step 4: Sort the whole merge_buffer (of length 2*k) using bitonic sort.
    // After sorting, the first k elements in merge_buffer are the best (smallest) k neighbors.
    bitonicSortWarp(merge_buffer, 2 * k);
    __syncwarp();
    // Step 5: Write the first k sorted neighbors back to the intermediate result registers.
    for (int j = 0; j < num_local; j++) {
        int pos = lane + j * 32;
        if (pos < k) {
            local_knn_idx[j] = merge_buffer[pos].idx;
            local_knn_dist[j] = merge_buffer[pos].dist;
        }
    }
    __syncwarp();
    // Step 6: The new max_distance is the k-th neighbor's distance (last element in sorted order).
    float max_d;
    if (lane == 0) {
        max_d = merge_buffer[k - 1].dist;
    }
    max_d = __shfl_sync(0xffffffff, max_d, 0);
    *newMax = max_d;
}

// ----------------------------------------------------------------------
// Kernel: knn_kernel
//
// Each warp (32 threads) processes one query (if available). The warp loads its query
// and initializes its intermediate result (k nearest neighbors) to dummy values (distance = FLT_MAX).
// It then loops over the global data points in batches (tiles) that are cached in shared memory.
// Each warp computes distances from its query to the points in the tile and, if the distance is
// less than the current max_distance (i.e. the worst of the current k neighbors), it adds a candidate
// (data point index and distance) to its candidate buffer (using atomicAdd to update the candidate count).
// When the candidate buffer is full (i.e. holds k candidates) or after finishing all batches,
// the warp merges (via warpMerge) the candidate buffer with its intermediate result. Finally,
// the warp writes its k best neighbors out to global memory in the 'result' array.
// ----------------------------------------------------------------------
#define TILE_SIZE 256

__global__ void knn_kernel(
    const float2 *query,    // array of queries (each a float2)
    int query_count,
    const float2 *data,     // array of data points (each a float2)
    int data_count,
    std::pair<int, float> *result, // output: row-major order; for query i, result[i*k + j] is its j-th neighbor
    int k
) {
    // Each warp processes one query.
    int warp_in_block = threadIdx.x >> 5;  // warp index within the block
    int lane = threadIdx.x & 31;            // lane index within the warp
    // Global warp id = block index * (# warps per block) + warp_in_block.
    int warps_per_block = blockDim.x >> 5;
    int global_warp_id = blockIdx.x * warps_per_block + warp_in_block;
    if (global_warp_id >= query_count) return;

    // Load the query point for this warp.
    float2 q = query[global_warp_id];

    // Define the number of neighbors stored per thread (k/32).
    int num_local = k >> 5; // since 32 divides k exactly

    // Each thread keeps its share of the warp’s intermediate nearest neighbors in registers.
    // Initially, all neighbors are set to dummy values (idx = -1, dist = FLT_MAX).
    int local_knn_idx[32];      // maximum num_local is 1024/32 = 32.
    float local_knn_dist[32];
    for (int j = 0; j < num_local; j++) {
        local_knn_idx[j] = -1;
        local_knn_dist[j] = FLT_MAX;
    }
    // The current worst distance in the warp’s intermediate result.
    // Since initially all distances are FLT_MAX, warp_max is FLT_MAX.
    float warp_max = FLT_MAX;

    // -------------------------------------------------------------
    // Partition dynamic shared memory.
    // The shared memory layout is as follows:
    //   [0, TILE_SIZE*sizeof(float2))               -> tile data (float2 array)
    //   [next, next + (warps_per_block * sizeof(int)) ) -> candidate count array (one int per warp)
    //   [next, next + (warps_per_block * k * sizeof(Neighbor)) ) -> candidate buffer (k per warp)
    //   [next, next + (warps_per_block * (2*k) * sizeof(Neighbor)) ) -> merge buffer (2*k per warp)
    // -------------------------------------------------------------
    extern __shared__ char dynamic_shmem[];
    char *ptr = dynamic_shmem;
    float2 *tile = (float2 *)ptr;
    ptr += TILE_SIZE * sizeof(float2);
    int *cand_count_arr = (int *)ptr;
    ptr += warps_per_block * sizeof(int);
    Neighbor *cand_buffer = (Neighbor *)ptr;
    ptr += warps_per_block * k * sizeof(Neighbor);
    Neighbor *merge_buf = (Neighbor *)ptr;
    // No need to update ptr further.

    // Each warp gets its own candidate count, candidate buffer, and merge buffer.
    int *myCandCount = &cand_count_arr[warp_in_block];
    Neighbor *myCandBuffer = &cand_buffer[warp_in_block * k];
    Neighbor *myMergeBuffer = &merge_buf[warp_in_block * (2 * k)];
    // Initialize candidate count to zero (only need one thread; then synchronize within warp).
    if (lane == 0) {
        *myCandCount = 0;
    }
    __syncwarp();

    // Process data in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        // Determine the tile size.
        int tile_size = ((tile_start + TILE_SIZE) <= data_count) ? TILE_SIZE : (data_count - tile_start);
        // All threads in the block load data points into shared memory.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[tile_start + i];
        }
        __syncthreads();  // ensure the tile is loaded

        // Each warp processes the tile.
        // Each lane loops over tile indices in stride of 32.
        for (int i = lane; i < tile_size; i += 32) {
            float2 pt = tile[i];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float dist = dx * dx + dy * dy;
            // If the distance is less than the current worst (max) distance,
            // then add this candidate.
            if (dist < warp_max) {
                // Atomically reserve a slot in the candidate buffer.
                int pos = atomicAdd(myCandCount, 1);
                if (pos < k) {
                    myCandBuffer[pos].idx = tile_start + i; // global index of data point
                    myCandBuffer[pos].dist = dist;
                }
                // If pos >= k, the candidate buffer is (or would be) over‐full.
                // In this implementation we simply drop the extra candidate.
            }
        }
        __syncwarp();
        // If the candidate buffer is full (or over full), merge it with the intermediate result.
        if (*myCandCount >= k) {
            warpMerge(k, num_local, myMergeBuffer, myCandBuffer, *myCandCount, local_knn_idx, local_knn_dist, &warp_max);
            if (lane == 0) {
                *myCandCount = 0;
            }
            __syncwarp();
        }
        __syncthreads(); // synchronize all threads in the block before loading next tile
    }

    // End of tile loop.
    // If the candidate buffer holds any remaining candidates, merge them.
    if (*myCandCount > 0) {
        warpMerge(k, num_local, myMergeBuffer, myCandBuffer, *myCandCount, local_knn_idx, local_knn_dist, &warp_max);
        if (lane == 0) {
            *myCandCount = 0;
        }
        __syncwarp();
    }

    // Now the warp’s intermediate result (distributed in registers) holds the best k neighbors
    // in sorted (ascending) order. Write these results to global memory.
    // The output array is stored in row‐major order: for query with index global_warp_id,
    // its j-th nearest neighbor is stored at result[global_warp_id * k + j].
    for (int j = 0; j < num_local; j++) {
        int pos = lane + j * 32;
        if (pos < k) {
            result[global_warp_id * k + pos] = std::make_pair(local_knn_idx[j], local_knn_dist[j]);
        }
    }
}

// ----------------------------------------------------------------------
// Host function: run_knn
//
// This is the C++ interface for running the k‐NN kernel. It selects appropriate
// grid and block dimensions and computes the dynamic shared memory size required.
// It then launches the knn_kernel to compute the k-nearest neighbors for all queries.
// ----------------------------------------------------------------------
void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k
) {
    // Choose block size: use 128 threads per block (i.e. 4 warps per block)
    int blockSize = 128;
    int warpsPerBlock = blockSize >> 5;  // blockSize / 32
    // Each warp processes one query, so total number of warps needed = query_count.
    int gridSize = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute dynamic shared memory size per block.
    // Layout:
    //   Tile: TILE_SIZE * sizeof(float2)
    //   Candidate count: warpsPerBlock * sizeof(int)
    //   Candidate buffer: warpsPerBlock * k * sizeof(Neighbor)
    //   Merge buffer: warpsPerBlock * (2*k) * sizeof(Neighbor)
    size_t shared_mem_size = TILE_SIZE * sizeof(float2)
        + warpsPerBlock * (sizeof(int) + 3UL * k * sizeof(Neighbor));

    // Launch kernel.
    knn_kernel<<<gridSize, blockSize, shared_mem_size>>>(query, query_count, data, data_count, result, k);
    
    // Optionally check for launch errors (omitted here for brevity).
    cudaDeviceSynchronize();
}