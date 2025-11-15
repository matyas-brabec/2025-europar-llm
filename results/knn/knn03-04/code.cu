#include <cuda_runtime.h>
#include <utility>

// -----------------------------------------------------------------------------
// Optimized k-NN for 2D points on modern NVIDIA GPUs (A100/H100).
//
// Design overview (see function and kernel comments below for details):
//   * One warp (32 threads) processes one query point.
//   * Each block processes multiple queries (WARPS_PER_BLOCK).
//   * Data points are streamed in tiles into shared memory.
//   * All threads in the block cooperatively load each tile.
//   * Within each warp, threads compute distances from the shared tile to
//     their query, then use warp shuffles to feed candidates to lane 0.
//   * Lane 0 maintains a max-heap of size k in shared memory (per warp).
//   * At the end, lane 0 heap-sorts the heap to ascending order and writes
//     out k nearest neighbor indices and distances.
// -----------------------------------------------------------------------------

// Tunable parameters (chosen for a good balance on A100/H100-like GPUs).
static constexpr int KNN_WARP_SIZE        = 32;   // fixed by hardware
static constexpr int KNN_WARPS_PER_BLOCK  = 4;    // warps (queries) per block
static constexpr int KNN_THREADS_PER_BLOCK =
    KNN_WARP_SIZE * KNN_WARPS_PER_BLOCK;

// Maximum k supported (per problem statement: power of two, 32..1024).
static constexpr int KNN_MAX_K            = 1024;

// Number of data points per tile loaded into shared memory.
// Each point is float2 (8 bytes), so 2048 points -> 16 KB per block.
static constexpr int KNN_TILE_SIZE        = 2048;

// -----------------------------------------------------------------------------
// Heap utilities (max-heap on distances).
// We maintain a max-heap of size <= k to keep the k smallest distances seen.
// The root (index 0) holds the current worst (largest) distance.
// Only lane 0 in each warp calls these helpers.
// -----------------------------------------------------------------------------

// Insert (dist, idx) into a max-heap of capacity k.
//   heap_dist / heap_idx: arrays of length >= k.
//   heap_size: current number of elements in the heap (0 <= heap_size <= k).
//   k: maximum allowed heap size (== requested k).
__device__ __forceinline__
void knn_heap_insert(float* heap_dist,
                     int*   heap_idx,
                     int&   heap_size,
                     int    k,
                     float  dist,
                     int    idx)
{
    // If we still have room, insert and sift up.
    if (heap_size < k) {
        int pos = heap_size;
        heap_dist[pos] = dist;
        heap_idx[pos]  = idx;

        // Sift-up to maintain max-heap by distance.
        while (pos > 0) {
            int parent = (pos - 1) >> 1;
            if (heap_dist[parent] >= heap_dist[pos])
                break;

            // Swap with parent.
            float tmpd       = heap_dist[parent];
            heap_dist[parent]= heap_dist[pos];
            heap_dist[pos]   = tmpd;

            int tmpi         = heap_idx[parent];
            heap_idx[parent] = heap_idx[pos];
            heap_idx[pos]    = tmpi;

            pos = parent;
        }
        ++heap_size;
    } else {
        // Heap already full: root holds the largest (worst) distance.
        // Reject if candidate is worse than current worst.
        if (dist >= heap_dist[0])
            return;

        // Replace root with new candidate, then sift down.
        int pos = 0;
        heap_dist[0] = dist;
        heap_idx[0]  = idx;

        while (true) {
            int left  = (pos << 1) + 1;
            if (left >= k)
                break;
            int right = left + 1;

            // Find the child with the larger distance.
            int largest = left;
            if (right < k && heap_dist[right] > heap_dist[left])
                largest = right;

            if (heap_dist[pos] >= heap_dist[largest])
                break;

            // Swap with the larger child.
            float tmpd        = heap_dist[pos];
            heap_dist[pos]    = heap_dist[largest];
            heap_dist[largest]= tmpd;

            int tmpi          = heap_idx[pos];
            heap_idx[pos]     = heap_idx[largest];
            heap_idx[largest] = tmpi;

            pos = largest;
        }
    }
}

// Heapsort for a max-heap (in-place).
// After this, heap_dist[0..heap_size-1] will be sorted in ascending order
// (smallest distance at index 0).
__device__ __forceinline__
void knn_heap_sort(float* heap_dist,
                   int*   heap_idx,
                   int    heap_size)
{
    // Standard in-place heapsort using the existing max-heap.
    for (int end = heap_size - 1; end > 0; --end) {
        // Move current max (root) to position 'end'.
        float tmpd      = heap_dist[0];
        heap_dist[0]    = heap_dist[end];
        heap_dist[end]  = tmpd;

        int tmpi        = heap_idx[0];
        heap_idx[0]     = heap_idx[end];
        heap_idx[end]   = tmpi;

        // Restore max-heap property for [0, end).
        int pos = 0;
        while (true) {
            int left = (pos << 1) + 1;
            if (left >= end)
                break;
            int right  = left + 1;
            int largest = left;
            if (right < end && heap_dist[right] > heap_dist[left])
                largest = right;

            if (heap_dist[pos] >= heap_dist[largest])
                break;

            float tmpd2     = heap_dist[pos];
            heap_dist[pos]  = heap_dist[largest];
            heap_dist[largest] = tmpd2;

            int tmpi2       = heap_idx[pos];
            heap_idx[pos]   = heap_idx[largest];
            heap_idx[largest] = tmpi2;

            pos = largest;
        }
    }
}

// -----------------------------------------------------------------------------
// CUDA kernel: one warp per query, shared-memory tiling of data.
// Each warp keeps its k best neighbors in a max-heap stored in shared memory.
// -----------------------------------------------------------------------------
__global__ void knn_kernel_2d(const float2* __restrict__ query,
                              int                      query_count,
                              const float2* __restrict__ data,
                              int                      data_count,
                              std::pair<int, float>* __restrict__ result,
                              int                      k)
{
    // Shared memory:
    //   * s_points     : tile of data points reused by all warps in block.
    //   * s_heap_dist  : per-warp max-heap distances.
    //   * s_heap_idx   : per-warp max-heap indices.
    __shared__ float2 s_points[KNN_TILE_SIZE];
    __shared__ float  s_heap_dist[KNN_WARPS_PER_BLOCK * KNN_MAX_K];
    __shared__ int    s_heap_idx [KNN_WARPS_PER_BLOCK * KNN_MAX_K];

    const int warp_size       = KNN_WARP_SIZE;
    const int warps_per_block = KNN_WARPS_PER_BLOCK;

    const int lane            = threadIdx.x & (warp_size - 1); // 0..31
    const int warp_in_block   = threadIdx.x >> 5;              // 0..warps_per_block-1
    const int global_warp     = blockIdx.x * warps_per_block + warp_in_block;

    const bool active_warp    = (global_warp < query_count);

    // Per-warp heap pointers into shared memory.
    float* heap_dist = s_heap_dist + warp_in_block * KNN_MAX_K;
    int*   heap_idx  = s_heap_idx  + warp_in_block * KNN_MAX_K;

    // Load query point (one per warp) and broadcast to all lanes via shuffle.
    float qx = 0.0f;
    float qy = 0.0f;
    int   heap_size = 0;  // only lane 0 actually uses this for the heap

    if (active_warp && lane == 0) {
        float2 q = query[global_warp];
        qx = q.x;
        qy = q.y;
    }

    const unsigned full_mask = 0xffffffffu;
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);

    // Iterate over data in tiles, loading each tile into shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += KNN_TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > KNN_TILE_SIZE)
            tile_size = KNN_TILE_SIZE;

        // All threads in the block cooperatively load this tile.
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
            s_points[idx] = data[tile_start + idx];
        }

        // Ensure tile is fully loaded before any warp uses it.
        __syncthreads();

        if (active_warp) {
            // Each warp processes the tile for its query.
            // We process the tile in blocks of 'warp_size' points so that
            // each lane handles at most one point per iteration.
            for (int base = 0; base < tile_size; base += warp_size) {
                int i = base + lane;

                // Compute candidate distance for this lane (or mark invalid).
                int   global_index = -1;   // <0 marks an invalid candidate
                float dist         = 0.0f; // dummy initialization

                if (i < tile_size) {
                    float2 p = s_points[i];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    dist         = dx * dx + dy * dy;
                    global_index = tile_start + i;
                }

                // Warp-wide: feed all 32 lane candidates to lane 0 via shuffles.
                // Invalid lanes (global_index < 0) are simply skipped by lane 0.
#pragma unroll
                for (int src = 0; src < warp_size; ++src) {
                    int   idx_src  = __shfl_sync(full_mask, global_index, src);
                    float dist_src = __shfl_sync(full_mask, dist,         src);
                    if (lane == 0 && idx_src >= 0) {
                        knn_heap_insert(heap_dist, heap_idx,
                                        heap_size, k,
                                        dist_src, idx_src);
                    }
                }
            }
        }

        // Ensure all warps are done reading this tile before loading next tile.
        __syncthreads();
    }

    // After scanning all data points, each active warp has a heap of size k
    // containing the k nearest neighbors (unsorted, in max-heap form).
    if (active_warp && lane == 0) {
        if (heap_size > 1) {
            knn_heap_sort(heap_dist, heap_idx, heap_size);
        }

        // Write out results for this query in ascending order of distance.
        int out_base = global_warp * k;
        for (int j = 0; j < k; ++j) {
            result[out_base + j].first  = heap_idx[j];
            result[out_base + j].second = heap_dist[j];
        }
    }
}

// -----------------------------------------------------------------------------
// Host-side entry point.
//   query       : device pointer to array of float2 (size query_count).
//   query_count : number of queries.
//   data        : device pointer to array of float2 (size data_count).
//   data_count  : number of data points.
//   result      : device pointer to array of std::pair<int,float> of size
//                 query_count * k. For query i, result[i * k + j] holds the
//                 j-th nearest neighbor.
//   k           : number of neighbors (power of two, 32..1024).
//
// Assumptions (per problem statement):
//   * All pointers are allocated with cudaMalloc and point to device memory.
//   * data_count >= k.
//   * k is a power of two, 32 <= k <= 1024.
//   * query_count and data_count are large enough to benefit from GPU.
// -----------------------------------------------------------------------------
void run_knn(const float2*             query,
             int                       query_count,
             const float2*             data,
             int                       data_count,
             std::pair<int, float>*    result,
             int                       k)
{
    // Guard against degenerate cases (no work to do).
    if (query_count <= 0 || data_count <= 0 || k <= 0)
        return;

    // k must not exceed our compile-time maximum (as specified).
    if (k > KNN_MAX_K)
        return; // or handle error as appropriate

    const int warps_per_block = KNN_WARPS_PER_BLOCK;
    const int threads_per_block = KNN_THREADS_PER_BLOCK;

    // One warp per query.
    int total_warps = query_count;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    // Launch the kernel. It is asynchronous; synchronize or check errors
    // outside this function if desired.
    knn_kernel_2d<<<blocks, threads_per_block>>>(
        query, query_count,
        data,  data_count,
        result,
        k
    );
}