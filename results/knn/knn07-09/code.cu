#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// The Candidate structure holds an index into the "data" array and the squared Euclidean distance.
struct Candidate {
    int index;
    float dist;
};

//---------------------------------------------------------------------
// Device function: bitonicSortShared
// Performs an in-place bitonic sort on an array of Candidate elements 
// stored in shared memory. The array length "n" is assumed to be a power of two.
// The sorting is done cooperatively by the 32 threads in a warp using __syncwarp().
//---------------------------------------------------------------------
__device__ void bitonicSortShared(Candidate *arr, int n) {
    int lane = threadIdx.x % 32;  // lane index in the warp
    // Outer loop: subarray sizes double each iteration.
    for (int size = 2; size <= n; size <<= 1) {
        // Inner loop: compare and swap elements with decreasing stride.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes indices starting at its lane id, stepping by warp size.
            for (int i = lane; i < n; i += 32) {
                int ixj = i ^ stride;  // partner index using bitwise XOR
                if (ixj > i && ixj < n) {
                    // Determine sorting direction: ascending if bit "size" of i is 0, descending otherwise.
                    bool ascending = ((i & size) == 0);
                    Candidate a = arr[i];
                    Candidate b = arr[ixj];
                    if ((ascending && a.dist > b.dist) || (!ascending && a.dist < b.dist)) {
                        // Swap arr[i] and arr[ixj]
                        arr[i] = b;
                        arr[ixj] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

//---------------------------------------------------------------------
// Device function: performMerge
// Merges the candidate buffer (stored in shared memory) with the warp's private
// intermediate result (stored in registers). The candidate buffer and the private
// intermediate result (of total length k, distributed among 32 threads, each holding k/32)
// are merged in two steps as described by:
//   1. Sort the candidate buffer using bitonicSortShared.
//   2. Copy the private intermediate result into a temporary shared memory buffer.
//   3. For each index i in [0,k), replace candidate_buffers[i] with the minimum between 
//      candidate_buffers[i] and merge_tmp[k - i - 1].
//   4. Sort the merged result using bitonicSortShared.
//   5. Copy the merged result back into the private intermediate result.
// The new maximum distance is then taken from the last element of the merged (sorted)
// candidate buffer and broadcast to all threads in the warp.
//---------------------------------------------------------------------
__device__ float performMerge(int warp_id, Candidate *inter, Candidate *candidate_buffers, Candidate *merge_tmp, int k) {
    int lane = threadIdx.x % 32;
    int m = k / 32;  // each thread holds m intermediate candidates
    int warp_offset = warp_id * k;

    // Step 1: Sort the candidate buffer (in shared memory) for this warp.
    bitonicSortShared(&candidate_buffers[warp_offset], k);

    // Step 2: Copy the private intermediate result (stored in registers) into merge_tmp (shared memory).
    for (int j = 0; j < m; j++) {
        merge_tmp[warp_offset + lane + j * 32] = inter[j];
    }
    __syncwarp();

    // Step 3: Merge the candidate buffer and the copied intermediate result.
    // For each index i in [0, k), take the element with smaller distance between candidate_buffers[i]
    // and merge_tmp[k - i - 1].
    for (int i = lane; i < k; i += 32) {
        int partner = k - i - 1;
        Candidate cand_buf = candidate_buffers[warp_offset + i];
        Candidate cand_inter = merge_tmp[warp_offset + partner];
        Candidate merged = (cand_buf.dist < cand_inter.dist) ? cand_buf : cand_inter;
        candidate_buffers[warp_offset + i] = merged;
    }
    __syncwarp();

    // Step 4: Sort the merged candidate buffer to produce a properly ordered intermediate result.
    bitonicSortShared(&candidate_buffers[warp_offset], k);
    __syncwarp();

    // Step 5: Copy the sorted merged result back into the private intermediate result.
    for (int j = 0; j < m; j++) {
        inter[j] = candidate_buffers[warp_offset + lane + j * 32];
    }
    __syncwarp();

    // Step 6: Update max_distance from the last element of the candidate buffer (largest in sorted order).
    float new_max = candidate_buffers[warp_offset + k - 1].dist;
    // Broadcast the new max distance from lane 0 to all other lanes in the warp.
    new_max = __shfl_sync(0xffffffff, new_max, 0);
    return new_max;
}

//---------------------------------------------------------------------
// Device kernel: knn_kernel
// Each warp (32 threads) processes one query point to compute its k nearest 
// neighbors from the provided data set (using squared Euclidean distances).
// The data points are processed in batches that are cached in shared memory.
// Each warp maintains a private (register) copy of the current best k candidates 
// ("intermediate result"), along with a shared candidate buffer for candidate 
// accumulation. When the candidate buffer fills, it is merged with the private 
// intermediate result using a bitonic sort based procedure.
// The final sorted k nearest neighbors for a query are written to the output array.
//---------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count, const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Each warp (32 threads) processes one query.
    int warp_lane = threadIdx.x % 32;
    int warp_in_block = threadIdx.x / 32;  // warp index within this block
    int global_warp_id = blockIdx.x * (blockDim.x / 32) + warp_in_block;
    if (global_warp_id >= query_count) return;  // No query assigned to this warp.

    // Load the query point.
    float2 q = query[global_warp_id];

    // Number of candidates per thread in the private intermediate result.
    int m = k / 32;  // Since k is a power of 2 between 32 and 1024, m is an integer.
    // Private intermediate result stored in registers (distributed across warp lanes).
    Candidate inter[32];  // Maximum m is 32 when k==1024.
    for (int j = 0; j < m; j++) {
        inter[j].index = -1;
        inter[j].dist = FLT_MAX;
    }
    // Private variable that tracks the current maximum (worst) distance among the k best candidates.
    float max_distance = FLT_MAX;

    // Shared memory layout (allocated as a contiguous region):
    //   [0, candidateBufferSize): candidate buffers for each warp (each of size k).
    //   [candidateBufferSize, candidateBufferSize + candidateCountSize): candidate_counts for each warp.
    //   [tileOffset, tileOffset + TILE_SIZE * sizeof(float2) ): tile for data points.
    //   [mergeTmpOffset, ... ): merge_tmp buffers for each warp (each of size k).
    extern __shared__ char shared_mem[];
    int numWarps = blockDim.x / 32;
    int candidateBufferSize = numWarps * k * sizeof(Candidate);
    int candidateCountSize = numWarps * sizeof(int);
    int tileOffset = candidateBufferSize + candidateCountSize;
    const int TILE_SIZE = 256;  // Chosen batch size for data points.
    int tileSizeBytes = TILE_SIZE * sizeof(float2);
    int mergeTmpOffset = tileOffset + tileSizeBytes;

    Candidate *candidate_buffers = (Candidate*)shared_mem; 
    int *candidate_counts = (int*)(shared_mem + candidateBufferSize);
    float2 *tile = (float2*)(shared_mem + tileOffset);
    Candidate *merge_tmp = (Candidate*)(shared_mem + mergeTmpOffset);

    // Initialize the candidate count for this warp.
    if (warp_lane == 0) {
        candidate_counts[warp_in_block] = 0;
    }
    __syncthreads();

    // Process the data points iteratively in batches loaded into shared memory.
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        int current_tile_size = (data_count - batch_start) < TILE_SIZE ? (data_count - batch_start) : TILE_SIZE;
        // Cooperative loading of the current tile from global memory into shared memory.
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            if (i < current_tile_size) {
                tile[i] = data[batch_start + i];
            }
        }
        __syncthreads();

        // Each warp processes the points in the current tile.
        for (int i = warp_lane; i < current_tile_size; i += 32) {
            float2 pt = tile[i];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float dist = dx * dx + dy * dy;
            // If the point is closer than the worst candidate so far...
            if (dist < max_distance) {
                // Global index of the candidate.
                int global_data_idx = batch_start + i;
                // Atomically add a candidate into the per-warp candidate buffer.
                int pos = atomicAdd(&candidate_counts[warp_in_block], 1);
                if (pos < k) {
                    int index_in_buffer = warp_in_block * k + pos;
                    candidate_buffers[index_in_buffer].index = global_data_idx;
                    candidate_buffers[index_in_buffer].dist = dist;
                }
            }
        }
        __syncthreads();

        // If the candidate buffer is full (i.e., has at least k entries), merge it with the intermediate result.
        if (candidate_counts[warp_in_block] >= k) {
            max_distance = performMerge(warp_in_block, inter, candidate_buffers, merge_tmp, k);
            if (warp_lane == 0) {
                candidate_counts[warp_in_block] = 0;  // Reset candidate buffer count.
            }
            __syncwarp();
        }
        __syncthreads();
    }

    // After processing all batches, merge any remaining candidates in the candidate buffer.
    if (candidate_counts[warp_in_block] > 0) {
        max_distance = performMerge(warp_in_block, inter, candidate_buffers, merge_tmp, k);
        if (warp_lane == 0) {
            candidate_counts[warp_in_block] = 0;
        }
        __syncwarp();
    }

    // Write the sorted k nearest neighbors from the private intermediate result to global memory.
    // The private result is distributed such that each thread holds m elements; the overall ordering is:
    //   global index = warp_lane + j * 32, for j in 0 .. m-1.
    for (int j = 0; j < m; j++) {
        int out_idx = global_warp_id * k + (warp_lane + j * 32);
        result[out_idx] = std::pair<int, float>(inter[j].index, inter[j].dist);
    }
}

//---------------------------------------------------------------------
// Host function: run_knn
// This function launches the CUDA kernel to compute the k-nearest neighbors.
// It sets up an appropriate grid configuration (128 threads per block, i.e. 4 warps per block)
// so that each warp processes one query. It also computes the required shared memory size.
//---------------------------------------------------------------------
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose block size: 128 threads per block (i.e. 4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute shared memory size.
    // Layout:
    //   candidate_buffers: warpsPerBlock * k * sizeof(Candidate)
    //   candidate_counts: warpsPerBlock * sizeof(int)
    //   tile: TILE_SIZE * sizeof(float2) with TILE_SIZE = 256
    //   merge_tmp: warpsPerBlock * k * sizeof(Candidate)
    size_t sharedMemSize = warpsPerBlock * k * sizeof(Candidate)     // candidate_buffers
                          + warpsPerBlock * sizeof(int)               // candidate_counts
                          + 256 * sizeof(float2)                        // tile for data points
                          + warpsPerBlock * k * sizeof(Candidate);      // merge_tmp

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // Optionally, check for errors here.
}