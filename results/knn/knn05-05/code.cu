#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <utility>

// Device-side representation of (index, distance) pair.
// This is assumed to be layout-compatible with std::pair<int, float>.
struct PairIFDevice {
    int   first;
    float second;
};

static_assert(sizeof(PairIFDevice) == sizeof(std::pair<int, float>),
              "PairIFDevice must match std::pair<int,float> layout");

/**
 * Warp-level bitonic sort of (distance, index) pairs stored in shared memory.
 *
 * - dist[idx] and ind[idx] are arrays of length n in shared memory.
 * - n must be a power of two (here, n = 2 * k, with k also a power of two).
 * - The whole block participates, but in this code the block is a single warp
 *   (32 threads), so __syncthreads() is equivalent to __syncwarp().
 *
 * After the function returns, (dist, ind) is sorted in ascending order of dist.
 */
__device__ __forceinline__
void bitonic_sort_pair_asc(float *dist, int *ind, int n)
{
    const int tid = threadIdx.x;

    // Standard bitonic sort network for power-of-two n
    for (int k = 2; k <= n; k <<= 1) {
        // j is the distance to the comparison partner
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Iterate over all indices this thread is responsible for
            for (int i = tid; i < n; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    float di = dist[i];
                    float dj = dist[ixj];
                    int   ii = ind[i];
                    int   ij = ind[ixj];

                    // 'up' indicates ascending order for this subsequence
                    bool up = ((i & k) == 0);

                    // Swap when (di > dj) for ascending, or (di < dj) for descending.
                    if ((di > dj) == up) {
                        dist[i]  = dj;
                        dist[ixj] = di;
                        ind[i]   = ij;
                        ind[ixj] = ii;
                    }
                }
            }
            __syncthreads(); // Synchronize warp/block after each compare/exchange stage
        }
    }
}

/**
 * Kernel computing k-NN for 2D points.
 *
 * - One query point is processed by exactly one warp (here, a whole block).
 * - For each query, we keep:
 *   - A private intermediate result (top-k neighbors) in shared memory.
 *   - A per-query candidate buffer of size k in shared memory.
 * - The input data is processed in batches of size k; each batch is loaded
 *   into shared memory once and reused by the warp.
 * - After each batch, we merge intermediate result and candidates by sorting
 *   2*k elements with a bitonic network, then keeping the best k.
 */
__global__
void knn_kernel(const float2 * __restrict__ query,
                int                       query_count,
                const float2 * __restrict__ data,
                int                       data_count,
                PairIFDevice      * __restrict__ result,
                int                       k)
{
    // One warp (block) per query
    const int query_idx = blockIdx.x;
    if (query_idx >= query_count) {
        return;
    }

    // Shared memory layout:
    // [0, k)       : float2 cached data points (current batch)
    // [k, k+2k)    : float distances (top-k + candidates)
    // [k+2k, ... ) : int indices   (top-k + candidates)
    extern __shared__ unsigned char smem[];
    float2 *s_points  = reinterpret_cast<float2*>(smem);
    float  *s_dists   = reinterpret_cast<float*>(s_points + k);
    int    *s_indices = reinterpret_cast<int*>(s_dists + 2 * k);

    // Aliases for intermediate result and candidate buffer
    float *topk_dist = s_dists;         // [0 .. k-1]   : intermediate top-k distances
    float *cand_dist = s_dists + k;     // [k .. 2k-1]  : candidate distances for current batch
    int   *topk_idx  = s_indices;       // [0 .. k-1]   : intermediate top-k indices
    int   *cand_idx  = s_indices + k;   // [k .. 2k-1]  : candidate indices for current batch

    // Initialize intermediate result and candidate buffer with "infinite" values.
    // This effectively means no neighbors have been found yet.
    for (int i = threadIdx.x; i < 2 * k; i += blockDim.x) {
        s_dists[i]   = FLT_MAX;
        s_indices[i] = -1;
    }
    __syncthreads();

    // Each thread loads its own copy of the query point (L1/L2 cache will
    // broadcast, so this is efficient and simple).
    const float2 q = query[query_idx];

    // Threshold: current distance of the k-th nearest neighbor in intermediate result.
    // Start with +infinity (no pruning at the beginning).
    float threshold = FLT_MAX;

    // Batch size: number of data points cached per iteration.
    // We choose it equal to k so that the candidate buffer (of size k) is
    // sufficient to hold all potential candidates from a single batch.
    const int tile_size = k;

    // Process all data points in batches.
    for (int base = 0; base < data_count; base += tile_size) {
        int tile_count = data_count - base;
        if (tile_count > tile_size) tile_count = tile_size;

        // Load the current batch of data points to shared memory.
        // The whole block (single warp) cooperates.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_points[i] = data[base + i];
        }
        __syncthreads();

        // Reset the candidate buffer for this batch to "invalid".
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            cand_dist[i] = FLT_MAX;
            cand_idx[i]  = -1;
        }
        __syncthreads();

        // Each thread processes a subset of the batch and fills the candidate buffer.
        // Only candidates closer than the current k-th nearest neighbor in the
        // intermediate result are stored; the rest are effectively skipped.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            const float2 p = s_points[i];
            const float  dx = p.x - q.x;
            const float  dy = p.y - q.y;
            const float  dist = dx * dx + dy * dy;

            // Skip candidates that are not closer than the current threshold.
            if (dist < threshold) {
                cand_dist[i] = dist;
                cand_idx[i]  = base + i;  // Global index in the data array
            }
        }
        __syncthreads();

        // Merge the candidate buffer with the intermediate result.
        // We represent both as a single array of 2*k elements in shared memory
        // and perform a bitonic sort on (distance, index) pairs.
        //
        // After sorting in ascending order of distance:
        //   - topk_dist[0 .. k-1] and topk_idx[0 .. k-1] contain the k nearest
        //     neighbors seen so far (intermediate result), sorted by distance.
        //   - The remaining entries contain worse neighbors or invalid placeholders.
        bitonic_sort_pair_asc(s_dists, s_indices, 2 * k);

        // Update the threshold (distance of the current k-th nearest neighbor)
        // for pruning in subsequent batches.
        threshold = topk_dist[k - 1];

        __syncthreads();
    }

    // All data points have been processed; the final top-k neighbors for this
    // query are in topk_dist/topk_idx. Write them back to global memory.
    //
    // Layout of 'result':
    //   For query i, result[i * k + j] is its j-th nearest neighbor:
    //     - result[i * k + j].first  = index of data point
    //     - result[i * k + j].second = squared Euclidean distance
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int out_index = query_idx * k + i;
        result[out_index].first  = topk_idx[i];
        result[out_index].second = topk_dist[i];
    }
}

/**
 * Host wrapper launching the k-NN kernel.
 *
 * query       : device pointer to query points (float2)
 * query_count : number of query points
 * data        : device pointer to data points (float2)
 * data_count  : number of data points
 * result      : device pointer to result array of std::pair<int, float>
 * k           : number of nearest neighbors to find
 *
 * Assumptions:
 *   - data_count >= k
 *   - k is a power of two between 32 and 1024, inclusive
 *   - All pointers refer to device memory allocated via cudaMalloc.
 */
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // One warp (32 threads) per query, one warp per block.
    constexpr int WARP_SIZE         = 32;
    constexpr int WARPS_PER_BLOCK   = 1;
    constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

    // Shared memory requirements per block:
    //
    // - k * sizeof(float2):  cached data batch
    // - 2k * sizeof(float) :  distances (top-k + candidates)
    // - 2k * sizeof(int)   :  indices   (top-k + candidates)
    const size_t shared_bytes =
        static_cast<size_t>(k) * sizeof(float2) +
        static_cast<size_t>(2 * k) * sizeof(float) +
        static_cast<size_t>(2 * k) * sizeof(int);

    using PairIFHost = std::pair<int, float>;
    static_assert(sizeof(PairIFDevice) == sizeof(PairIFHost),
                  "PairIFDevice must be compatible with std::pair<int,float>");

    // One block per query.
    const int blocks = query_count;

    knn_kernel<<<blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<PairIFDevice*>(result),
        k
    );

    // The caller may wish to perform cudaDeviceSynchronize() or error checking.
}