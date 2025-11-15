#include <cuda_runtime.h>
#include <float.h>
#include <utility>

/*
 * Optimized brute-force k-NN for 2D points on modern NVIDIA GPUs (e.g., A100/H100).
 *
 * Algorithm overview (per query point):
 *   - Maintain the current k nearest neighbors in shared memory.
 *   - Process the entire dataset in tiles of size k (CHUNK = k).
 *   - For each tile:
 *       * Compute distances for up to k new data points and place them in the
 *         second half of a shared array of size 2*k.
 *       * Run an in-place bitonic sort on the 2*k candidate neighbors
 *         (k previous best + k new ones).
 *       * Keep only the first k entries (smallest distances) as the new top-k.
 *   - After all tiles are processed, write the final top-k list for the query
 *     into the output array.
 *
 * Notes:
 *   - k is guaranteed to be a power of two between 32 and 1024 (inclusive),
 *     and data_count >= k.
 *   - Distance measure is squared Euclidean distance in 2D.
 *   - No extra device memory is allocated; only shared memory and registers are used.
 *   - The kernel is launched with one block per query; the number of threads per
 *     block is min(256, k), which balances parallelism and shared-memory usage.
 */

constexpr int MAX_K = 1024;  // Maximum allowed k as per problem statement

// In-place bitonic sort of (distance, index) pairs in ascending order by distance.
// 'count' must be a power of two and <= 2 * MAX_K.
//
// This implementation is block-wide: all threads in the block cooperate.
// Each thread processes multiple elements with stride blockDim.x.
__device__ inline void bitonic_sort_pairs(float *dist, int *idx, int count)
{
    // Standard bitonic sort network
    for (int k = 2; k <= count; k <<= 1) {
        // Merge bitonic sequences of length 'k'
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();  // ensure previous stage's writes are visible

            for (int i = threadIdx.x; i < count; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i && ixj < count) {
                    // Determine whether this subsequence is sorted ascending or descending
                    bool ascending = ((i & k) == 0);

                    float di = dist[i];
                    float dj = dist[ixj];
                    int   idi = idx[i];
                    int   idj = idx[ixj];

                    // Swap if out of order for the current direction
                    bool do_swap = ascending ? (di > dj) : (di < dj);
                    if (do_swap) {
                        dist[i]  = dj;
                        dist[ixj] = di;
                        idx[i]   = idj;
                        idx[ixj] = idi;
                    }
                }
            }
        }
    }
    __syncthreads();  // final synchronization before returning
}

// Kernel: computes k nearest neighbors for each query point.
__global__ void knn_kernel_2d(
    const float2 *__restrict__ query,
    int query_count,
    const float2 *__restrict__ data,
    int data_count,
    std::pair<int, float> *__restrict__ result,
    int k)
{
    // Shared memory arrays for candidate neighbors:
    // We allocate space for up to 2 * MAX_K pairs (distance, index).
    __shared__ float s_dist[2 * MAX_K];
    __shared__ int   s_idx [2 * MAX_K];

    const int q_idx = blockIdx.x;
    if (q_idx >= query_count || k <= 0 || k > MAX_K) {
        return;
    }

    const int pair_count = 2 * k;  // number of elements actually used in shared arrays

    // Load the query point for this block
    const float2 q = query[q_idx];

    // Initialize shared arrays: set all distances in [0, 2*k) to a large value (infinity)
    // and indices to -1.
    const float INF = FLT_MAX;
    for (int i = threadIdx.x; i < pair_count; i += blockDim.x) {
        s_dist[i] = INF;
        s_idx[i]  = -1;
    }
    __syncthreads();

    // Process the data points in tiles of size 'chunk' = k.
    const int chunk = k;

    for (int base = 0; base < data_count; base += chunk) {
        int tile_count = data_count - base;
        if (tile_count > chunk) {
            tile_count = chunk;
        }

        // Compute distances for this tile and store them into the second half
        // of shared arrays: indices [k, k + tile_count).
        // Remaining entries up to [k, 2*k) are set to INF.
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
            int slot = k + i;  // position in shared arrays for new candidates
            if (i < tile_count) {
                int data_idx = base + i;
                float2 p = data[data_idx];

                float dx = p.x - q.x;
                float dy = p.y - q.y;
                float dist = dx * dx + dy * dy;

                s_dist[slot] = dist;
                s_idx[slot]  = data_idx;
            } else if (slot < pair_count) {
                // Fill unused slots (for incomplete tiles) with INF
                s_dist[slot] = INF;
                s_idx[slot]  = -1;
            }
        }
        __syncthreads();

        // Merge the previous top-k (stored in s_dist[0..k-1]) with the new
        // tile's candidates (stored in s_dist[k..k+tile_count-1]) into a new
        // top-k set. We do this by sorting the first 2*k elements using a
        // bitonic sort; the k smallest distances end up in positions [0..k-1].
        bitonic_sort_pairs(s_dist, s_idx, pair_count);

        // After the sort, the first k entries represent the updated global top-k.
        // The second half [k..2k) will be overwritten by the next tile.
    }

    // Write the final top-k neighbors for this query out to global memory.
    const int out_base = q_idx * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        int out_idx = out_base + i;
        result[out_idx].first  = s_idx[i];
        result[out_idx].second = s_dist[i];
    }
    // No need for further synchronization; each block writes to a disjoint
    // region of the result array.
}

// Host interface: launches the kernel.
// All pointers (query, data, result) are assumed to reside in device memory.
void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0 || k > MAX_K) {
        return;
    }

    // Choose the number of threads per block.
    // k is a power of two between 32 and 1024, inclusive.
    // Use up to 256 threads per block; for small k, use k threads so that
    // each candidate in a tile can be handled in parallel.
    int threads_per_block = (k < 256) ? k : 256;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(query_count, 1, 1);  // one block per query

    knn_kernel_2d<<<grid_dim, block_dim>>>(query, query_count, data, data_count, result, k);

    // Optional: synchronize for correctness in simple usage patterns.
    // In performance-critical code, consider exposing async behavior to the caller.
    cudaDeviceSynchronize();
}