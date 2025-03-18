#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// -----------------------------------------------------------------------------
// This implementation of k-nearest neighbors (k-NN) for 2D points uses a
// single warp (32 threads) per query. Each warp maintains a private (register-
// resident) sorted intermediate result of its current best k candidates and a
// candidate buffer in shared memory for new candidates. The new candidates are
// accumulated in the candidate buffer, and when full, merged with the
// intermediate result using Bitonic Sort and a special “min‐merge” procedure.
// Data points are processed in batches that are loaded cooperatively by the
// entire thread block into shared memory.
// 
// Shared memory layout (per block):
//   - Candidate buffers: warps_per_block * k Candidates.
//   - Candidate counts: warps_per_block * sizeof(int).
//   - Data batch: batch_size * sizeof(float2) (for the current tile of data).
//   - Merge temporary buffers: warps_per_block * k Candidates.
// 
// The kernel uses warp‐synchronous programming (with __syncwarp) for intra‐warp
// collaboration and atomicAdd for candidate buffer updates. The candidate merge
// routine follows the prescribed three‐step procedure using Bitonic Sort.
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// Structure to hold a candidate (data index and distance).
// -----------------------------------------------------------------------------
struct Candidate {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// Device function: Bitonic Sort over an array of Candidate elements in shared
// memory. This function is executed cooperatively by the 32 threads of a warp.
// n must be a power of two (and in our algorithm, k is a power of 2 between 32
// and 1024).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void bitonic_sort(Candidate* arr, int n)
{
    // use lane id within the warp for parallel work.
    unsigned lane = threadIdx.x & 0x1F; // equivalent to threadIdx.x % 32
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread iterates over indices assigned to it in steps of warp size.
            for (int i = lane; i < n; i += 32) {
                int j = i ^ stride;
                if (j > i && j < n) {
                    // Determine sorting order (ascending) for the current subsequence.
                    bool ascending = ((i & size) == 0);
                    if (ascending) {
                        if (arr[i].dist > arr[j].dist) {
                            Candidate tmp = arr[i];
                            arr[i] = arr[j];
                            arr[j] = tmp;
                        }
                    } else {
                        if (arr[i].dist < arr[j].dist) {
                            Candidate tmp = arr[i];
                            arr[i] = arr[j];
                            arr[j] = tmp;
                        }
                    }
                }
            }
            __syncwarp(); // synchronize warp lanes
        }
    }
}

// -----------------------------------------------------------------------------
// The main kernel that computes the k-nearest neighbors for 2D queries.
// Each warp processes one query.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k,
                           int batch_size)
{
    // Each warp (32 threads) processes one query.
    const int warpSize = 32;
    int warp_id = threadIdx.x / warpSize;       // local warp id in the block
    int lane = threadIdx.x & 0x1F;                // lane id [0,31]
    int warps_per_block = blockDim.x / warpSize;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    if (global_warp_id >= query_count)
        return;
    
    // Load query point for this warp.
    float2 q = query[global_warp_id];
    
    // Each warp keeps a private, sorted intermediate result of size k.
    // Distribute the k elements among the 32 threads equally.
    int num_per_thread = k / warpSize;  // Guaranteed to be an integer (k is power of 2)
    Candidate knn_local[32];            // Maximum needed is 32 elements per thread.
    #pragma unroll
    for (int i = 0; i < num_per_thread; i++) {
        knn_local[i].idx = -1;
        knn_local[i].dist = FLT_MAX;
    }
    // Compute current max distance from the intermediate result.
    int owner = (k - 1) / num_per_thread;
    int localIdx = (k - 1) % num_per_thread;
    float my_max = (lane == owner) ? knn_local[localIdx].dist : 0.0f;
    unsigned mask = 0xffffffff;
    float max_distance = __shfl_sync(mask, my_max, owner);
    
    // -------------------------------------------------------------------------
    // Shared memory layout:
    //   [0, warps_per_block * k * sizeof(Candidate))      -> Candidate buffer for each warp.
    //   [warps_per_block * k * sizeof(Candidate),
    //    warps_per_block * k * sizeof(Candidate) + warps_per_block * sizeof(int)) -> Candidate count per warp.
    //   Next comes the data tile, batch_size * sizeof(float2).
    //   Next comes temporary merge buffer: warps_per_block * k * sizeof(Candidate).
    // -------------------------------------------------------------------------
    extern __shared__ char shared_mem[];
    // Candidate buffers.
    Candidate *cand_buffer = reinterpret_cast<Candidate*>(shared_mem);
    // Candidate counts (one int per warp).
    int *cand_count = reinterpret_cast<int*>(shared_mem + warps_per_block * k * sizeof(Candidate));
    // Data tile for input data points.
    int data_tile_offset = warps_per_block * k * sizeof(Candidate) + warps_per_block * sizeof(int);
    float2 *data_tile = reinterpret_cast<float2*>(shared_mem + data_tile_offset);
    // Temporary merge buffer.
    int merge_buffer_offset = data_tile_offset + batch_size * sizeof(float2);
    Candidate *merge_buffer = reinterpret_cast<Candidate*>(shared_mem + merge_buffer_offset);
    
    // Initialize candidate count for this warp.
    if (lane == 0)
        cand_count[warp_id] = 0;
    __syncwarp();

    // Process input data in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += batch_size) {
        // Determine number of data points in this batch.
        int batch_len = ((batch_start + batch_size) <= data_count) ? batch_size : (data_count - batch_start);
        
        // Load batch of data points into shared memory cooperatively by the block.
        for (int i = threadIdx.x; i < batch_len; i += blockDim.x) {
            data_tile[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the batch is fully loaded.
        
        // Each warp processes the data in the shared tile.
        for (int i = lane; i < batch_len; i += warpSize) {
            float2 pt = data_tile[i];
            float dx = pt.x - q.x;
            float dy = pt.y - q.y;
            float dist = dx*dx + dy*dy; // Squared Euclidean distance.
            // Candidate is accepted only if its distance is less than the current kth best.
            if (dist < max_distance) {
                // Atomically reserve a slot in the candidate buffer.
                int pos = atomicAdd(&cand_count[warp_id], 1);
                if (pos < k) {
                    cand_buffer[warp_id * k + pos].idx = batch_start + i;
                    cand_buffer[warp_id * k + pos].dist = dist;
                }
            }
        }
        __syncwarp();
        __syncthreads(); // Synchronize block before next use of shared memory.
        
        // If candidate buffer is full, merge it with the intermediate result.
        if (cand_count[warp_id] >= k) {
            // Each warp's candidate buffer starts at:
            Candidate *buffer = cand_buffer + warp_id * k;
            // Step 1: Sort the candidate buffer using Bitonic Sort.
            bitonic_sort(buffer, k);
            __syncwarp();
            // Step 2: Merge candidate buffer with intermediate result.
            // For each index i in [0,k), compute merged[i] = min(buffer[i], intermediate[k-i-1]).
            Candidate *merged = merge_buffer + warp_id * k;
            for (int i = lane; i < k; i += warpSize) {
                int j = k - i - 1;
                int owner_thread = j / num_per_thread;
                int localPos = j % num_per_thread;
                Candidate inter;
                // Retrieve the (k - i - 1)-th candidate from the intermediate result using warp shuffle.
                inter.idx  = __shfl_sync(mask, knn_local[localPos].idx, owner_thread);
                inter.dist = __shfl_sync(mask, knn_local[localPos].dist, owner_thread);
                Candidate cand = buffer[i];
                Candidate merged_val = (cand.dist < inter.dist) ? cand : inter;
                merged[i] = merged_val;
            }
            __syncwarp();
            // Step 3: Sort the merged array to get an updated intermediate result.
            bitonic_sort(merged, k);
            __syncwarp();
            // Update private intermediate result from the merged array.
            for (int i = lane; i < k; i += warpSize) {
                int owner_thread = i / num_per_thread;
                int localPos = i % num_per_thread;
                Candidate newCand = merged[i];
                if (lane == owner_thread && localPos < num_per_thread)
                    knn_local[localPos] = newCand;
            }
            __syncwarp();
            // Update kth best distance (max_distance) from the updated intermediate.
            if (lane == (k - 1) / num_per_thread)
                my_max = knn_local[(k - 1) % num_per_thread].dist;
            max_distance = __shfl_sync(mask, my_max, (k - 1) / num_per_thread);
            // Reset candidate count for this warp.
            if (lane == 0)
                cand_count[warp_id] = 0;
            __syncwarp();
        }
    }  // End of batch loop.
    
    // After processing all batches, merge any remaining candidates in the candidate buffer.
    int final_count = cand_count[warp_id];
    if (final_count > 0) {
        Candidate *buffer = cand_buffer + warp_id * k;
        // Fill unused buffer slots with dummy candidates (FLT_MAX).
        for (int i = lane; i < k; i += warpSize) {
            if (i >= final_count) {
                buffer[i].idx = -1;
                buffer[i].dist = FLT_MAX;
            }
        }
        __syncwarp();
        // Sort the candidate buffer.
        bitonic_sort(buffer, k);
        __syncwarp();
        Candidate *merged = merge_buffer + warp_id * k;
        for (int i = lane; i < k; i += warpSize) {
            int j = k - i - 1;
            int owner_thread = j / num_per_thread;
            int localPos = j % num_per_thread;
            Candidate inter;
            inter.idx  = __shfl_sync(mask, knn_local[localPos].idx, owner_thread);
            inter.dist = __shfl_sync(mask, knn_local[localPos].dist, owner_thread);
            Candidate cand = buffer[i];
            Candidate merged_val = (cand.dist < inter.dist) ? cand : inter;
            merged[i] = merged_val;
        }
        __syncwarp();
        bitonic_sort(merged, k);
        __syncwarp();
        for (int i = lane; i < k; i += warpSize) {
            int owner_thread = i / num_per_thread;
            int localPos = i % num_per_thread;
            Candidate newCand = merged[i];
            if (lane == owner_thread && localPos < num_per_thread)
                knn_local[localPos] = newCand;
        }
        __syncwarp();
        if (lane == (k - 1) / num_per_thread)
            my_max = knn_local[(k - 1) % num_per_thread].dist;
        max_distance = __shfl_sync(mask, my_max, (k - 1) / num_per_thread);
        if (lane == 0)
            cand_count[warp_id] = 0;
        __syncwarp();
    }
    
    // Write out the final sorted k-nearest neighbors to global memory.
    // The final intermediate result is distributed among the 32 warp threads.
    for (int i = lane; i < k; i += warpSize) {
        int owner_thread = i / num_per_thread;
        int localPos = i % num_per_thread;
        Candidate cand;
        cand.idx  = __shfl_sync(mask, knn_local[localPos].idx, owner_thread);
        cand.dist = __shfl_sync(mask, knn_local[localPos].dist, owner_thread);
        result[global_warp_id * k + i] = std::pair<int, float>(cand.idx, cand.dist);
    }
}

// -----------------------------------------------------------------------------
// Host interface function: run_knn.
// This function launches the CUDA kernel to compute the k-nearest neighbors.
// Input arrays 'query', 'data', and 'result' are expected to be allocated with
// cudaMalloc. 'query' and 'data' consist of float2 values stored consecutively,
// and 'result' is an array of std::pair<int, float> stored consecutively.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose a batch size for processing data points in shared memory.
    const int batch_size = 1024;
    // Define block size in threads. Each warp (32 threads) processes one query.
    const int block_threads = 256; // e.g., 256 threads per block (8 warps per block)
    int warps_per_block = block_threads / 32;
    
    // Compute grid dimensions: one warp per query.
    int total_warps = query_count;
    int grid_blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    
    // Compute shared memory size required per block:
    //   candidate buffer: warps_per_block * k * sizeof(Candidate)
    //   candidate count:  warps_per_block * sizeof(int)
    //   data tile:        batch_size * sizeof(float2)
    //   merge temporary:  warps_per_block * k * sizeof(Candidate)
    size_t sharedMemSize = warps_per_block * k * sizeof(Candidate) +
                           warps_per_block * sizeof(int) +
                           batch_size * sizeof(float2) +
                           warps_per_block * k * sizeof(Candidate);
    
    // Launch the kernel.
    knn_kernel<<<grid_blocks, block_threads, sharedMemSize>>>(query, query_count,
                                                              data, data_count,
                                                              result, k, batch_size);
    cudaDeviceSynchronize();
}