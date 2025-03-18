#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// This kernel implements a k-nearest neighbors (k-NN) search for 2D points using squared Euclidean distance.
// Each query is processed by one warp (32 threads). The warp cooperatively maintains a private reservoir
// of k candidate neighbors (divided evenly among its 32 threads). It iterates over the data in batches
// (tiles) loaded into shared memory and updates its reservoir using warp‐level reductions and local insertion.
// Finally, the warp “merges” its 32 sorted sublists (one per thread) into one sorted list using a
// cooperative bitonic sort in shared memory, and writes out the final sorted k neighbors.
// Assumptions:
//   - k is a power-of-two between 32 and 1024.
//   - Each of the arrays query, data, and result has been allocated on the device (via cudaMalloc).
//   - BlockDim.x is chosen as a multiple of 32; in the host launcher below we choose 128 threads per block.
//   - For final merge we allocate a shared memory buffer sized for 4 warps * (maximum k == 1024) elements.
//      (This works when blockDim.x==128 i.e. 4 warps per block.)
//   - The squared Euclidean distance is used (no square root).
//
// Note: We use __syncthreads() to synchronize threads in a block and __syncwarp() to synchronize threads within a warp.
// Warp-level shuffles (using __shfl_down_sync) are used to compute the current global worst candidate quickly.

__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result,
                           int k) {
    // Each warp processes one query.
    int warpId = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (warpId >= query_count)
        return;
    
    // Load the query point for this warp.
    float2 q = query[warpId];
    
    // Each warp will maintain a private reservoir of k candidates.
    // Each thread holds k_per_lane candidates (k/32 each).
    int k_per_lane = k / 32;
    
    // Declare local arrays to hold candidates for this thread.
    // Maximum k_per_lane is 1024/32 = 32.
    float localDists[32];
    int   localIdx[32];
#pragma unroll
    for (int i = 0; i < k_per_lane; i++) {
        localDists[i] = FLT_MAX;
        localIdx[i] = -1;
    }
    
    // Define tile (batch) size for loading data into shared memory.
    // Each tile holds TILE_SIZE points.
    const int TILE_SIZE = 1024;
    // Shared memory for caching a batch of data points.
    __shared__ float2 shData[TILE_SIZE];
    
    // Process the data array iteratively in tiles.
    for (int tile_offset = 0; tile_offset < data_count; tile_offset += TILE_SIZE) {
        int tile_size = (tile_offset + TILE_SIZE <= data_count) ? TILE_SIZE : (data_count - tile_offset);
        // Load the current data tile into shared memory.
        for (int i = threadIdx.x; i < TILE_SIZE && (tile_offset + i) < data_count; i += blockDim.x) {
            shData[i] = data[tile_offset + i];
        }
        __syncthreads();
        
        // Each warp processes this tile.
        // Each lane processes points with indices (lane, lane+32, lane+64, ...) within the tile.
        for (int i = lane; i < tile_size; i += 32) {
            float2 d = shData[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            int dataIndex = tile_offset + i;
            
            // Compute the current global worst among the warp's reservoir.
            // First, each thread finds the maximum (worst) value in its local reservoir.
            float local_max = localDists[0];
#pragma unroll
            for (int j = 1; j < k_per_lane; j++) {
                if (localDists[j] > local_max)
                    local_max = localDists[j];
            }
            // Use warp-level reduction (with __shfl_down_sync) to find the maximum of all lanes.
            unsigned mask = 0xFFFFFFFF;
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(mask, local_max, offset);
                if (other > local_max)
                    local_max = other;
            }
            float global_thresh = local_max;
            
            // If the current candidate is better than the worst candidate in the warp...
            if (dist < global_thresh) {
                // Each thread checks its own reservoir and replaces its worst candidate if needed.
                float my_max = localDists[0];
                int my_max_idx = 0;
#pragma unroll
                for (int j = 1; j < k_per_lane; j++) {
                    if (localDists[j] > my_max) {
                        my_max = localDists[j];
                        my_max_idx = j;
                    }
                }
                if (dist < my_max) {
                    localDists[my_max_idx] = dist;
                    localIdx[my_max_idx] = dataIndex;
                }
            }
        }
        __syncwarp(); // ensure warp is coordinated
        __syncthreads(); // ensure all threads are ready before next tile load
    }
    
    // At this point each thread holds k_per_lane candidates,
    // so the warp’s entire reservoir has k candidates (unsorted).
    // Now, each thread sorts its own local reservoir in ascending order.
    for (int i = 1; i < k_per_lane; i++) {
        float key_dist = localDists[i];
        int key_idx = localIdx[i];
        int j = i - 1;
        while (j >= 0 && localDists[j] > key_dist) {
            localDists[j + 1] = localDists[j];
            localIdx[j + 1] = localIdx[j];
            j--;
        }
        localDists[j + 1] = key_dist;
        localIdx[j + 1] = key_idx;
    }
    
    // --- Final merge: merge the 32 sorted sublists (one per thread) into one sorted list of k elements.
    // We use a cooperative bitonic sort in shared memory.
    // For each block we assume blockDim.x==128 (4 warps per block). Therefore we allocate a buffer
    // of size (warps per block)*k. Each warp writes its k candidates into a contiguous segment.
    // (Since k <= 1024 and 4 warps, the maximum shared memory required is 4*1024*sizeof(pair<int,float>) bytes.)
    __shared__ std::pair<int, float> finalBuffer[4 * 1024]; // works when blockDim.x==128 and k<=1024
    int warp_in_block = threadIdx.x / 32;  // which warp (0..3) in this block
    int base_sort = warp_in_block * k;      // starting index for this warp's region in finalBuffer

    // Each thread writes its sorted local candidates into finalBuffer in a transposed order.
    // (Each thread of the warp writes its k_per_lane candidates into positions spaced by warp width.)
    for (int i = 0; i < k_per_lane; i++) {
        int pos = lane + i * 32; // covers range 0 ... k-1 (since k=k_per_lane*32)
        finalBuffer[base_sort + pos] = std::make_pair(localIdx[i], localDists[i]);
    }
    __syncthreads();
    
    // Now, perform a parallel bitonic sort on the array finalBuffer[base_sort ... base_sort+k-1].
    // k is a power-of-two.
    for (int size = 2; size <= k; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            __syncthreads();  // synchronize all threads in block
            // Each thread in the warp loops over indices in its region in a strided loop.
            for (int i = lane; i < k; i += 32) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    // Determine the sorting direction: ascending if the bit corresponding to 'size' is 0.
                    bool ascending = ((i & size) == 0);
                    std::pair<int, float> a = finalBuffer[base_sort + i];
                    std::pair<int, float> b = finalBuffer[base_sort + ixj];
                    if ((ascending && a.second > b.second) || (!ascending && a.second < b.second)) {
                        finalBuffer[base_sort + i] = b;
                        finalBuffer[base_sort + ixj] = a;
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // Finally, each warp writes its sorted k nearest neighbors to the output global memory.
    for (int i = lane; i < k; i += 32) {
        result[warpId * k + i] = finalBuffer[base_sort + i];
    }
}

// Host function to launch the kernel.
// For each query[i], result[i*k + j] will contain a pair (data-index, squared-distance)
// corresponding to its j-th nearest neighbor (ordered in ascending distance).
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block size as a multiple of 32.
    // Here we choose 128 threads per block so that each block processes 128/32 = 4 queries.
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Compute grid size (number of blocks) so that there is one warp per query.
    int gridWarps = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    int blocks = gridWarps;
    
    // Launch the k-NN kernel.
    knn_kernel<<<blocks, threadsPerBlock>>>(query, query_count, data, data_count, result, k);
    
    // Optionally, one might call cudaDeviceSynchronize() and check for errors.
}