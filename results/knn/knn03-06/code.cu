#include <cuda_runtime.h>
#include <utility>

/*
 * Optimized CUDA implementation of k-nearest neighbors (k-NN) for 2D points.
 *
 * Design:
 * - Each query point is processed by a single warp (32 threads). We use one warp per block.
 * - The data points are processed in tiles (batches). Each tile is loaded into shared memory
 *   by all threads in the block (warp).
 * - For each query, we maintain a max-heap of current k best neighbors (indices + squared distances)
 *   in shared memory, private to that warp/query.
 * - In each iteration over a tile:
 *     * Each thread computes the distance to one candidate point and writes it into shared memory.
 *     * Thread 0 sequentially inserts the 32 candidates into the shared heap.
 * - After processing all data points, thread 0 performs heap extraction to produce the final
 *   sorted list of k nearest neighbors for that query and writes them to the result array.
 *
 * Notes:
 * - Distances are squared Euclidean (no sqrt) as requested.
 * - k is assumed to be a power of two in [32, 1024]; the implementation supports all k <= 1024.
 * - No additional device memory allocations are performed; only shared memory is used.
 */

namespace {

static constexpr int WARP_SIZE = 32;
static constexpr int MAX_K     = 1024;   // maximum supported k
static constexpr int TILE_SIZE = 4096;   // number of data points per tile

// Device-side struct layout-compatible with std::pair<int, float>.
struct DevicePair {
    int   first;
    float second;
};

// Sift an element up in a max-heap (by distance).
__device__ __forceinline__ void heap_sift_up(float *heap_dist,
                                             int   *heap_idx,
                                             int    pos)
{
    // Standard binary heap sift-up to maintain max-heap property.
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        float d_pos    = heap_dist[pos];
        float d_parent = heap_dist[parent];

        if (d_parent >= d_pos) {
            break;
        }

        // Swap parent and current
        heap_dist[pos]   = d_parent;
        heap_idx[pos]    = heap_idx[parent];
        heap_dist[parent] = d_pos;
        heap_idx[parent]  = heap_idx[pos];

        pos = parent;
    }
}

// Sift an element down in a max-heap (by distance).
__device__ __forceinline__ void heap_sift_down(float *heap_dist,
                                               int   *heap_idx,
                                               int    heap_size,
                                               int    pos)
{
    // Standard binary heap sift-down to maintain max-heap property.
    while (true) {
        int left  = (pos << 1) + 1;
        int right = left + 1;
        int largest = pos;

        if (left < heap_size && heap_dist[left] > heap_dist[largest]) {
            largest = left;
        }
        if (right < heap_size && heap_dist[right] > heap_dist[largest]) {
            largest = right;
        }
        if (largest == pos) {
            break;
        }

        float d_pos     = heap_dist[pos];
        float d_largest = heap_dist[largest];
        int   i_pos     = heap_idx[pos];
        int   i_largest = heap_idx[largest];

        heap_dist[pos]    = d_largest;
        heap_idx[pos]     = i_largest;
        heap_dist[largest] = d_pos;
        heap_idx[largest]  = i_pos;

        pos = largest;
    }
}

__global__ void knn_kernel(const float2 * __restrict__ query,
                           int                     query_count,
                           const float2 * __restrict__ data,
                           int                     data_count,
                           int                     k,
                           DevicePair * __restrict__ result)
{
    // One warp (block) per query.
    const int qid  = blockIdx.x;
    const int lane = threadIdx.x;  // 0..31, single warp per block

    if (qid >= query_count) {
        return;
    }

    // Shared memory used per block (i.e., per query).
    __shared__ float2 tile[TILE_SIZE];         // cached data points
    __shared__ float  heap_dist[MAX_K];        // distances in max-heap
    __shared__ int    heap_idx[MAX_K];         // indices in max-heap
    __shared__ int    heap_size;               // current heap size (<= k)
    __shared__ float  cand_dist[WARP_SIZE];    // candidate distances (one per thread)
    __shared__ int    cand_idx[WARP_SIZE];     // candidate indices (one per thread)

    // Load query point into registers.
    const float2 q = query[qid];

    // Initialize heap for this query.
    if (lane == 0) {
        heap_size = 0;
    }
    __syncthreads();

    int offset         = 0;
    int data_remaining = data_count;

    // Iterate over data in tiles.
    while (data_remaining > 0) {
        const int tile_len = (data_remaining > TILE_SIZE) ? TILE_SIZE : data_remaining;

        // Load current tile into shared memory cooperatively.
        for (int i = lane; i < tile_len; i += WARP_SIZE) {
            tile[i] = data[offset + i];
        }
        __syncthreads();

        // Process tile in batches of WARP_SIZE points.
        for (int base = 0; base < tile_len; base += WARP_SIZE) {
            const int local_idx = base + lane;
            const int global_idx = offset + local_idx;

            if (local_idx < tile_len) {
                const float2 p = tile[local_idx];
                const float dx = p.x - q.x;
                const float dy = p.y - q.y;
                cand_dist[lane] = dx * dx + dy * dy;  // squared Euclidean distance
                cand_idx[lane]  = global_idx;
            } else {
                // Mark invalid candidate.
                cand_idx[lane]  = -1;
                cand_dist[lane] = 0.0f;
            }
            __syncthreads();

            // Thread 0 merges candidates into the heap.
            if (lane == 0) {
                int hs = heap_size;

                #pragma unroll
                for (int i = 0; i < WARP_SIZE; ++i) {
                    const int   cidx  = cand_idx[i];
                    const float cdist = cand_dist[i];

                    if (cidx < 0) {
                        continue;  // skip invalid
                    }

                    if (hs < k) {
                        // Heap not full: insert new element and sift up.
                        int pos = hs++;
                        heap_dist[pos] = cdist;
                        heap_idx[pos]  = cidx;
                        heap_sift_up(heap_dist, heap_idx, pos);
                    } else {
                        // Heap full: only accept if candidate is better (smaller distance)
                        // than current worst (root of max-heap).
                        if (cdist >= heap_dist[0]) {
                            continue;
                        }
                        heap_dist[0] = cdist;
                        heap_idx[0]  = cidx;
                        heap_sift_down(heap_dist, heap_idx, k, 0);
                    }
                }

                heap_size = hs;
            }
            __syncthreads();
        }

        offset         += tile_len;
        data_remaining -= tile_len;
    }

    // Extract top-k neighbors from heap and store in ascending order of distance.
    if (lane == 0) {
        int hs = heap_size;  // should be >= k since data_count >= k

        // Pop from max-heap: each pop returns the current worst among the k best.
        // Store them from the end so that the final order is ascending.
        for (int pos = k - 1; pos >= 0; --pos) {
            const int   best_idx  = heap_idx[0];
            const float best_dist = heap_dist[0];

            // Move last element to root and reduce heap size.
            --hs;
            if (hs > 0) {
                heap_idx[0]  = heap_idx[hs];
                heap_dist[0] = heap_dist[hs];
                heap_sift_down(heap_dist, heap_idx, hs, 0);
            }

            const int out_offset = qid * k + pos;
            result[out_offset].first  = best_idx;
            result[out_offset].second = best_dist;
        }
    }
}

} // anonymous namespace

// Host-side entry point.
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // This implementation supports k up to MAX_K.
    if (k > MAX_K) {
        // For this task, k is guaranteed to be <= 1024, so we do not handle this case.
        return;
    }

    const dim3 block_dim(WARP_SIZE, 1, 1);     // one warp per block
    const dim3 grid_dim(query_count, 1, 1);    // one block per query

    // Reinterpret std::pair<int, float>* as layout-compatible DevicePair*.
    DevicePair *device_result = reinterpret_cast<DevicePair*>(result);

    knn_kernel<<<grid_dim, block_dim>>>(query,
                                        query_count,
                                        data,
                                        data_count,
                                        k,
                                        device_result);
}