// This complete CUDA source code implements an optimized k‐nearest neighbors (k‐NN) algorithm for 2D points.
// It processes each query by one warp (32 threads). Each warp maintains its “intermediate result” of its k best candidates
// in registers (distributed evenly among the 32 threads) and a per‐query candidate buffer stored in shared memory.
// The input data points are processed in batches (tiles) loaded cooperatively into shared memory.
// When the candidate buffer becomes full, a warp‐synchronous merge routine is invoked that uses
// a simple rank‐selection algorithm to combine the candidate buffer with the intermediate result.
// Finally, a final odd–even transposition sort is performed over the k best candidates so that the results
// are in increasing order (i.e. j-th neighbor in result[i*k+j]) before writing the answer to global memory.
//
// The interface function is:
//   void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k);
//
// This source code is self‐contained and does not allocate any extra global (device) memory other than shared memory.

#include <cuda_runtime.h>
#include <float.h>

// Define a simple Pair structure that matches std::pair<int,float> layout.
struct Pair {
    int first;    // index
    float second; // squared Euclidean distance
};

// Tile size: number of data points loaded per batch into shared memory.
#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

//------------------------------------------------------------------------------
// __device__ inline merge_warp
// This function merges the current intermediate result and candidate buffer (which is full)
// for a given query (warp) using a rank‐selection algorithm. The union of the two arrays
// (each of size k) is stored in a warp‐private region of the merge scratch area.
// Then, each warp thread computes the rank (number of union elements with strictly smaller distance)
// for a subset of union elements. If a candidate’s rank is less than k, it is written to the new
// intermediate result in the candidate buffer area (overwriting previous contents). The new intermediate
// result is then copied back to the registers and the candidate buffer count is reset.
//------------------------------------------------------------------------------
__device__ inline void merge_warp(
    int k,              // total number of candidates per query
    int warp_id,        // warp index within block (0 <= warp_id < (blockDim.x/32))
    int lane,           // thread lane id (0<=lane<32)
    int local_k,        // number of elements stored per thread in the intermediate result (k/32)
    Pair *local_intermediate, // per-thread array of intermediate results (in registers)
    Pair *s_candidate,  // candidate buffer for all warps, layout: [warp][0..k-1]
    Pair *s_merge,      // merge scratch area for all warps, layout: [warp][0..(2*k-1)]
    int *s_candidate_count  // candidate count per warp, one int per warp
)
{
    // Each warp uses a contiguous region of size 2*k in s_merge.
    // Compute base pointer for this warp.
    Pair *merge_ptr = s_merge + warp_id * (2 * k);

    // Step 1: Each thread writes its local intermediate results from registers into the first k positions.
    // The intermediate result is distributed among 32 threads; each thread holds local_k elements.
    for (int j = 0; j < local_k; j++) {
        int index = lane + j * 32;  // index in [0, k)
        merge_ptr[index] = local_intermediate[j];
    }
    // Step 2: Each thread loads candidate buffer for this warp into merge_ptr at offset k.
    for (int j = 0; j < local_k; j++) {
        int index = lane + j * 32;  // each thread loads local_k elements; total k elements.
        merge_ptr[k + index] = s_candidate[warp_id * k + index];
    }
    __syncwarp();

    // Step 3: For each union element (total 2*k), compute its rank.
    // Each thread processes a subset with stride 32.
    for (int i = lane; i < 2 * k; i += 32) {
        Pair cand = merge_ptr[i];
        int r = 0;
        // Linear scan over the union array.
        for (int j = 0; j < 2 * k; j++) {
            // Count elements with strictly smaller distance.
            // (Assume distances are distinct; ties are not resolved specially.)
            if (merge_ptr[j].second < cand.second) {
                r++;
            }
        }
        // If candidate's rank is less than k, it belongs in the new intermediate result.
        if (r < k) {
            s_candidate[warp_id * k + r] = cand;
        }
    }
    __syncwarp();

    // Step 4: Update local_intermediate registers from the new intermediate result stored in s_candidate.
    for (int j = 0; j < local_k; j++) {
        int index = lane + j * 32;
        local_intermediate[j] = s_candidate[warp_id * k + index];
    }
    // Step 5: Reset the candidate buffer count for this warp.
    if (lane == 0) {
        s_candidate_count[warp_id] = 0;
    }
    __syncwarp();
}

//------------------------------------------------------------------------------
// __device__ inline final_sort_warp
// Once all data batches have been processed and the final intermediate result is ready in registers,
// we perform a final sort (in increasing order of distance) for the query. This routine copies the warp's
// k candidates (distributed among 32 threads) to a contiguous region in shared memory and then applies
// an odd–even transposition sort. Finally, the sorted candidates are copied back to registers.
//------------------------------------------------------------------------------
__device__ inline void final_sort_warp(
    int k,                      // total number of candidates per query
    int warp_id,                // warp index within block
    int lane,                   // thread lane id
    int local_k,                // k/32 (number of elements per thread)
    Pair *local_intermediate,   // per-thread intermediate result (in registers)
    Pair *s_candidate           // candidate buffer; used here as temporary storage (layout: [warp][0..k-1])
)
{
    // Base pointer for this warp's candidate buffer.
    Pair *cand_ptr = s_candidate + warp_id * k;

    // Step 1: Copy local_intermediate (distributed in registers) to the contiguous candidate buffer.
    for (int j = 0; j < local_k; j++) {
        int index = lane + j * 32;
        cand_ptr[index] = local_intermediate[j];
    }
    __syncwarp();

    // Step 2: Perform odd–even transposition sort over the k elements.
    // For k iterations, each thread compares and possibly swaps candidates for indices assigned to it.
    for (int iter = 0; iter < k; iter++) {
        // Each thread processes all indices i (with stride 32) where the pairing applies.
        for (int i = lane; i < k - 1; i += 32) {
            // In even phase (iter even), compare pairs (0,1), (2,3), ...;
            // in odd phase (iter odd), compare pairs (1,2), (3,4), ...
            if ((i & 1) == (iter & 1)) {
                int idx = i;
                int idx_next = i + 1;
                Pair a = cand_ptr[idx];
                Pair b = cand_ptr[idx_next];
                if (a.second > b.second) {
                    // Swap
                    cand_ptr[idx] = b;
                    cand_ptr[idx_next] = a;
                }
            }
        }
        __syncwarp();
    }

    // Step 3: Copy the sorted result back to local_intermediate registers.
    for (int j = 0; j < local_k; j++) {
        int index = lane + j * 32;
        local_intermediate[j] = cand_ptr[index];
    }
    __syncwarp();
}

//------------------------------------------------------------------------------
// __global__ knn_kernel
// This kernel is launched with a blockDim that is a multiple of 32. Each warp processes one query.
// The input queries and data points are stored in global memory. Data points are processed in batches;
// each batch (tile) is loaded cooperatively into shared memory. Each warp computes distances for its query,
// and if a candidate point is closer than the worst candidate so far, it is appended to the candidate buffer.
// When the candidate buffer fills (i.e., k candidates), the merge routine is called.
// After all batches are processed, a final merge is performed if candidates remain in the buffer,
// followed by a final sort and writing the k nearest neighbors (index and squared distance) to the result array.
//------------------------------------------------------------------------------
__global__ void knn_kernel(
    const float2 *query,    // array of query points
    int query_count,        // number of queries
    const float2 *data,     // array of data points
    int data_count,         // number of data points
    Pair *result,           // output array [query_count * k] (row-major)
    int k                   // number of neighbors per query (power-of-two, 32 <= k <= 1024)
)
{
    // Each warp (32 threads) processes one query.
    int tid = threadIdx.x;
    int lane = tid & 31;           // thread's lane id within its warp
    int warp_id = tid >> 5;        // warp id within the block
    int numWarps = blockDim.x >> 5; // number of warps per block

    // Global query index computed per warp.
    int query_idx = blockIdx.x * numWarps + warp_id;
    if (query_idx >= query_count)
        return;

    // Load the query point into registers.
    float qx, qy;
    if (lane == 0) {
        qx = query[query_idx].x;
        qy = query[query_idx].y;
    }
    // Broadcast query coordinates to all lanes in the warp.
    qx = __shfl_sync(0xffffffff, qx, 0);
    qy = __shfl_sync(0xffffffff, qy, 0);

    // Each warp will maintain its k nearest candidates in registers.
    // They are distributed evenly among the 32 threads.
    int local_k = k >> 5;  // k/32 elements per thread
    Pair local_intermediate[32];  // maximum local_k is 1024/32 = 32
    // Initialize each slot to "infinity" (FLT_MAX) with an invalid index.
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (i < local_k) {
            local_intermediate[i].first = -1;
            local_intermediate[i].second = FLT_MAX;
        }
    }

    // Set up pointers into dynamic shared memory.
    // The shared memory layout is as follows:
    // [ s_data         ]: TILE_SIZE float2 elements for the data tile.
    // [ s_candidate    ]: (numWarps * k) Pair elements for candidate buffers.
    // [ s_merge        ]: (numWarps * 2 * k) Pair elements for merge scratch.
    // [ s_candidate_count ]: (numWarps) int elements for candidate counts.
    extern __shared__ char shared_mem[];
    float2 *s_data = (float2*) shared_mem;
    Pair *s_candidate = (Pair*)(s_data + TILE_SIZE);
    Pair *s_merge = (Pair*)(s_candidate + numWarps * k);
    int *s_candidate_count = (int*)(s_merge + numWarps * 2 * k);

    // Initialize candidate count for this warp.
    if (lane == 0) {
        s_candidate_count[warp_id] = 0;
    }
    __syncwarp();

    // Process the data points in batches (tiles).
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        // Determine the number of data points in this batch.
        int tile_count = (batch_start + TILE_SIZE <= data_count) ? TILE_SIZE : (data_count - batch_start);
        // Cooperative loading: each thread in the block loads elements of the current tile.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_data[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure entire tile is loaded.

        // Each warp processes the tile: iterate over data points in the tile in a warp-stride loop.
        for (int i = lane; i < tile_count; i += 32) {
            float2 point = s_data[i];
            float dx = qx - point.x;
            float dy = qy - point.y;
            float dist = dx * dx + dy * dy;

            // Compute the current worst (maximum) distance in the intermediate result.
            float local_max = -1.0f;
            for (int j = 0; j < local_k; j++) {
                if (local_intermediate[j].second > local_max)
                    local_max = local_intermediate[j].second;
            }
            // Reduce across the warp to obtain the maximum distance.
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(0xffffffff, local_max, offset);
                if (other > local_max) local_max = other;
            }
            float current_threshold = local_max;

            // Only consider this data point if its distance is smaller than the worst candidate so far.
            if (dist < current_threshold) {
                // Append candidate to the candidate buffer.
                // Use atomicAdd to update the candidate count for this warp.
                int pos = atomicAdd(&s_candidate_count[warp_id], 1);
                Pair cand;
                cand.first = batch_start + i;  // Global index of data point.
                cand.second = dist;
                s_candidate[warp_id * k + pos] = cand;
                // When the candidate buffer becomes full (only one thread will observe pos == k-1), merge.
                if (pos == k - 1) {
                    merge_warp(k, warp_id, lane, local_k, local_intermediate, s_candidate, s_merge, s_candidate_count);
                }
            }
        }
        __syncthreads();  // Ensure all warps finished processing this tile.
    }

    // After processing all data batches, merge any remaining candidates in the candidate buffer.
    int cand_count = s_candidate_count[warp_id];
    if (cand_count > 0) {
        // Pad the candidate buffer with dummy candidates (FLT_MAX) so that its size becomes k.
        if (lane == 0) {
            for (int i = cand_count; i < k; i++) {
                s_candidate[warp_id * k + i].first = -1;
                s_candidate[warp_id * k + i].second = FLT_MAX;
            }
            s_candidate_count[warp_id] = k;
        }
        __syncwarp();
        merge_warp(k, warp_id, lane, local_k, local_intermediate, s_candidate, s_merge, s_candidate_count);
    }

    // Final sort: sort the k intermediate candidates in increasing order (by squared distance).
    final_sort_warp(k, warp_id, lane, local_k, local_intermediate, s_candidate);

    // Write the final sorted k-nearest neighbors to global memory.
    // The output for query[query_idx] is stored in result[query_idx * k + j] for j in 0..k-1.
    int base = query_idx * k;
    for (int i = lane; i < k; i += 32) {
        // The sorted result was written into s_candidate for this warp during final_sort_warp.
        Pair res = s_candidate[warp_id * k + i];
        // Write as std::pair<int,float> equivalent: (first, second).
        result[base + i].first = res.first;
        result[base + i].second = res.second;
    }
}

//------------------------------------------------------------------------------
// Host interface: run_knn
// This function sets up and launches the knn_kernel with appropriate parameters.
// It computes the grid and block dimensions, calculates the required dynamic shared memory size,
// and then launches the kernel.
//------------------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_knn(
void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k
)
{
    // Choose number of threads per block.
    // We'll use 256 threads per block; note that each block contains blockDim.x/32 (i.e., 8) warps,
    // so each block processes 8 queries.
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock >> 5; // = 8
    int queriesPerBlock = warpsPerBlock;
    int gridSize = (query_count + queriesPerBlock - 1) / queriesPerBlock;

    // Compute the required dynamic shared memory size per block.
    // Shared memory layout (in bytes):
    //   s_data: TILE_SIZE float2 elements --> TILE_SIZE * sizeof(float2)
    //   s_candidate: (warpsPerBlock * k) Pair elements --> warpsPerBlock * k * sizeof(Pair)
    //   s_merge: (warpsPerBlock * 2 * k) Pair elements --> warpsPerBlock * 2 * k * sizeof(Pair)
    //   s_candidate_count: (warpsPerBlock) int elements --> warpsPerBlock * sizeof(int)
    size_t sharedMemSize = TILE_SIZE * sizeof(float2)
                         + warpsPerBlock * k * sizeof(Pair)
                         + warpsPerBlock * 2 * k * sizeof(Pair)
                         + warpsPerBlock * sizeof(int);

    // Launch the kernel.
    knn_kernel<<<gridSize, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count,
                                                             reinterpret_cast<Pair *>(result), k);
    // Optionally, one might add cudaDeviceSynchronize() and error checking here.
}
