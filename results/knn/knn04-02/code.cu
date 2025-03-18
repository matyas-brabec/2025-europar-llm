#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// -----------------------------------------------------------------------------
// This CUDA code implements a highly‐optimized k‑Nearest Neighbors (k‑NN)
// kernel for 2D points in squared Euclidean space. Each query is processed by
// a single warp (32 threads). The kernel loads the dataset in tiles (batches)
// into shared memory, then each warp computes distances from its query point
// to the cached data points. Each warp maintains its own private candidate list
// of k best (lowest distance) neighbors in distributed registers (each of the
// 32 threads holds k/32 candidates). The candidate list is updated cooperatively
// by the warp using warp-level shuffles and synchronization. Finally, the k
// candidates are sorted (in ascending order by distance) and written to global
// memory. Note: k is a power‐of‐2 between 32 and 1024.
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// Define a constant TILE_SIZE for iterative processing of data points.
// Each tile of data points is loaded into shared memory.
#define TILE_SIZE 512

// -----------------------------------------------------------------------------
// Define a simple Pair struct that is layout-compatible with std::pair<int,float>.
struct Pair {
    int first;    // data index in the dataset
    float second; // squared distance to the query point
};

// -----------------------------------------------------------------------------
// The knn_kernel processes multiple queries in parallel; each warp processes
// one query. The candidate list is maintained in registers across the warp.
// At the end, each warp sorts its candidate list and writes the result to global
// memory in row‐major order.
//
// Parameters:
//    query       - pointer to an array of float2 query points.
//    query_count - number of queries.
//    data        - pointer to an array of float2 data points.
//    data_count  - number of data points in the dataset.
//    result      - pointer to an array of std::pair<int, float> for output.
//                  For query i, result[i*k + j] is the j‐th nearest neighbor.
//    k           - number of nearest neighbors to return (power‐of‐2 between 32 and 1024).
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2* query, int query_count, const float2* data, int data_count,
                           std::pair<int, float>* result, int k)
{
    // Each warp (32 threads) processes one query
    int lane = threadIdx.x & 31;              // thread’s lane index in the warp
    int warpIdInBlock = threadIdx.x >> 5;       // warp index within the block
    int warpsPerBlock = blockDim.x >> 5;        // number of warps per block
    int globalWarpId = warpIdInBlock + blockIdx.x * warpsPerBlock;
    if (globalWarpId >= query_count) return;    // out‐of‐range queries

    // Load the query point for this warp
    float2 q;
    if (lane == 0) {
        q = query[globalWarpId];
    }
    // Broadcast query coordinates from lane 0 to the whole warp
    q.x = __shfl_sync(0xffffffff, q.x, 0);
    q.y = __shfl_sync(0xffffffff, q.y, 0);

    // Each warp will maintain a candidate list of k points.
    // We distribute the k candidates evenly across 32 threads.
    // Let L = k/32. (Since k is a power‐of‐2 and at least 32, division is exact.)
    int L = k >> 5;  // = k/32
    // Each thread in the warp holds L candidates in registers.
    // Initialize each candidate’s distance to FLT_MAX and index to -1.
    // The list is maintained unsorted; we update by replacing the current worst.
    float cand[32];  // maximum L is 32 when k==1024.
    int   ind[32];
#pragma unroll
    for (int i = 0; i < L; i++) {
        cand[i] = FLT_MAX;
        ind[i]  = -1;
    }

    // -------------------------------------------------------------------------
    // Shared memory for caching a tile of data points.
    // The whole block cooperatively loads TILE_SIZE data points.
    __shared__ float2 sdata[TILE_SIZE];

    // Process the dataset in batches ("tiles")
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        // Determine tile size (handle last batch if it is smaller)
        int tile_size = (batch_start + TILE_SIZE <= data_count) ? TILE_SIZE : (data_count - batch_start);
        // Each thread loads a subset of the tile into shared memory
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            sdata[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure all data in the tile is loaded

        // Each warp processes the TILE_SIZE points in shared memory.
        // The 32 threads in the warp stride through the tile.
        for (int j = lane; j < tile_size; j += 32) {
            float2 p = sdata[j];
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float d = dx*dx + dy*dy;
            int data_idx = batch_start + j;  // global index of the data point

            // -----------------------------------------------------------------
            // Candidate update routine:
            // We want to update our candidate list (stored unsorted in registers)
            // if d (the distance to the current data point) is better than the
            // current worst candidate in the candidate list.
            // First, compute (in each thread) the maximum (worst) candidate in its L entries.
            float local_max = cand[0];
            int local_max_index = 0;
#pragma unroll
            for (int i = 1; i < L; i++) {
                if (cand[i] > local_max) {
                    local_max = cand[i];
                    local_max_index = i;
                }
            }
            // Next, use warp-level reduction to determine the global worst candidate
            // (i.e. the maximum distance) across the warp's candidate list.
            float warp_max = local_max;
            int warp_max_lane = lane;
            int warp_max_pos = local_max_index;
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(0xffffffff, warp_max, offset);
                int other_lane = __shfl_down_sync(0xffffffff, warp_max_lane, offset);
                int other_pos = __shfl_down_sync(0xffffffff, warp_max_pos, offset);
                if (other > warp_max) {
                    warp_max = other;
                    warp_max_lane = other_lane;
                    warp_max_pos = other_pos;
                }
            }

            // If the new distance 'd' is not better than the worst candidate in our list, skip.
            if (d >= warp_max)
                continue;

            // Otherwise, we want to update the candidate list.
            // To allow for concurrent candidate updates across threads in the warp,
            // we use a warp-synchronous election: each thread that has a candidate
            // to insert holds it in a local variable; then the warp picks (via ballot)
            // the best candidate (i.e. the one with the smallest d) among those.
            float candidate_val = d;       // candidate to insert
            int candidate_idx   = data_idx;  // corresponding data index

            // Loop until all threads’ candidate update requests from this data point are processed.
            unsigned int active = __ballot_sync(0xffffffff, candidate_val < FLT_MAX);
            while (active) {
                int vote = __ffs(active) - 1;  // select lowest lane among those with active candidate
                float winner_candidate = __shfl_sync(0xffffffff, candidate_val, vote);
                int winner_idx = __shfl_sync(0xffffffff, candidate_idx, vote);

                // Recompute the current worst candidate in the candidate list.
                float local_max2 = cand[0];
                int local_max_index2 = 0;
#pragma unroll
                for (int i = 1; i < L; i++) {
                    if (cand[i] > local_max2) {
                        local_max2 = cand[i];
                        local_max_index2 = i;
                    }
                }
                float current_warp_max = local_max2;
                int current_warp_max_lane = lane;
                int current_warp_max_pos = local_max_index2;
                for (int offset = 16; offset > 0; offset /= 2) {
                    float other = __shfl_down_sync(0xffffffff, current_warp_max, offset);
                    int other_lane = __shfl_down_sync(0xffffffff, current_warp_max_lane, offset);
                    int other_pos = __shfl_down_sync(0xffffffff, current_warp_max_pos, offset);
                    if (other > current_warp_max) {
                        current_warp_max = other;
                        current_warp_max_lane = other_lane;
                        current_warp_max_pos = other_pos;
                    }
                }
                // If the winning candidate from our election is better than the current worst,
                // then update that slot in the candidate list.
                if (winner_candidate < current_warp_max) {
                    if (lane == current_warp_max_lane) {
                        cand[current_warp_max_pos] = winner_candidate;
                        ind[current_warp_max_pos] = winner_idx;
                    }
                }
                __syncwarp();
                // Mark the winning candidate as processed.
                if (lane == vote) {
                    candidate_val = FLT_MAX;
                }
                __syncwarp();
                active = __ballot_sync(0xffffffff, candidate_val < FLT_MAX);
            }
            // End of candidate update for data point j
        }
        __syncthreads();  // Ensure all warps have finished with this tile before it gets overwritten
    }
    // End processing all data tiles. At this point each warp's candidate list
    // (distributed over registers among its 32 threads) holds the k best (unsorted) candidates.

    // -------------------------------------------------------------------------
    // To output results in sorted order, copy the candidate list into dynamic shared
    // memory, then sort it (using a simple serial sort performed by lane 0).
    // Each warp is allocated a contiguous segment of k Pair elements in dynamic shared mem.
    // The total dynamic shared mem size (in bytes) must be set to (warpsPerBlock * k * sizeof(Pair)).
    extern __shared__ char shared_mem[];  // dynamic shared memory provided by host
    Pair* warp_candidates = (Pair*)shared_mem;
    // Each warp uses a segment of k Pair elements.
    Pair* my_candidates = warp_candidates + (warpIdInBlock * k);

    // Each thread writes its L candidates from registers into shared memory.
    // We store them in order so that the final list is stored contiguously.
    for (int i = 0; i < L; i++) {
        int pos = lane + 32 * i; // position in the k-sized candidate list
        if (pos < k) {
            my_candidates[pos].first  = ind[i];
            my_candidates[pos].second = cand[i];
        }
    }
    __syncwarp();  // Ensure all candidate values are written before sorting

    // Let one thread per warp (lane 0) sort the candidate list in shared memory.
    // We use simple selection sort (k is small, maximum 1024 elements).
    if (lane == 0) {
        for (int i = 0; i < k - 1; i++) {
            int min_idx = i;
            for (int j = i + 1; j < k; j++) {
                if (my_candidates[j].second < my_candidates[min_idx].second) {
                    min_idx = j;
                }
            }
            // Swap the found minimum with element i.
            Pair temp = my_candidates[i];
            my_candidates[i] = my_candidates[min_idx];
            my_candidates[min_idx] = temp;
        }
    }
    __syncwarp();  // Ensure sorting is complete

    // Write the sorted candidate list to the global result array.
    // For query globalWarpId, the k results are stored consecutively.
    for (int i = lane; i < k; i += 32) {
        // The output type is std::pair<int, float> and our Pair is layout-compatible.
        result[globalWarpId * k + i] = std::pair<int, float>(my_candidates[i].first, my_candidates[i].second);
    }
}

// -----------------------------------------------------------------------------
// Host interface: run_knn
//
// Launches the k-NN kernel with optimized parameters. Each warp processes one query,
// and the dataset is processed in batches (tiles) loaded into shared memory.
// The dynamic shared memory allocation is used for staging the per-warp candidate
// lists (each of size k).
//
// Parameters are as described in the interface:
//    query       - pointer to device memory for query points (float2).
//    query_count - number of queries (each query is a float2).
//    data        - pointer to device memory for data points (float2).
//    data_count  - number of data points.
//    result      - pointer to device memory for the results (std::pair<int,float>).
//                  For query i, result[i*k + j] contains the j-th nearest neighbor's
//                  index and squared distance.
//    k           - number of nearest neighbors to compute (power-of-2 between 32 and 1024).
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose hyper-parameters:
    // Use 128 threads per block (i.e. 4 warps per block) for example.
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query.
    int numWarps = query_count;
    int blocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Allocate dynamic shared memory for per-warp candidate staging.
    // Each block requires (warpsPerBlock * k * sizeof(Pair)) bytes.
    int sharedMemSize = warpsPerBlock * k * sizeof(Pair);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);

    // Synchronize (error checking omitted for brevity)
    cudaDeviceSynchronize();
}