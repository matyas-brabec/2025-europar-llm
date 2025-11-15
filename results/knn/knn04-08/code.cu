#include <cuda_runtime.h>
#include <utility>

// Simple POD type matching the memory layout of std::pair<int, float>
struct PairIF {
    int   first;
    float second;
};

// Tunable kernel parameters
constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 4;                     // 4 warps => 128 threads per block
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int MAX_K             = 1024;                  // maximum supported k
constexpr int TILE_SIZE         = 1024;                  // number of data points cached per block in shared memory

// ---------------------------------------------------------------------------
// Heap utilities: max-heap on distances (largest distance at root).
// Used per warp to maintain the current k nearest neighbors for its query.
// Only lane 0 of each warp mutates the heap; other lanes feed candidates.
// ---------------------------------------------------------------------------

__device__ __forceinline__
void heap_insert_or_update(float dist,
                           int   idx,
                           float *heap_dist,
                           int   *heap_idx,
                           int   k,
                           int   &heap_size)
{
    // Insert into heap if there is still room.
    if (heap_size < k) {
        int pos = heap_size;
        heap_size++;

        // Bubble up to maintain max-heap on distance.
        while (pos > 0) {
            int parent = (pos - 1) >> 1;
            float parent_dist = heap_dist[parent];
            if (parent_dist >= dist) {
                break;
            }
            heap_dist[pos] = parent_dist;
            heap_idx[pos]  = heap_idx[parent];
            pos = parent;
        }
        heap_dist[pos] = dist;
        heap_idx[pos]  = idx;
    } else {
        // Heap is full: check if candidate is better (smaller distance) than current worst.
        float worst_dist = heap_dist[0];
        if (dist >= worst_dist) {
            return; // no improvement
        }

        // Replace root and bubble down.
        int pos = 0;

        while (true) {
            int left = (pos << 1) + 1;
            if (left >= k) {
                break; // no children
            }
            int right = left + 1;

            int largest = left;
            float largest_dist = heap_dist[left];
            if (right < k && heap_dist[right] > largest_dist) {
                largest = right;
                largest_dist = heap_dist[right];
            }

            // If the incoming element is larger than or equal to the largest child,
            // heap property is satisfied.
            if (dist >= largest_dist) {
                break;
            }

            // Move largest child up.
            heap_dist[pos] = largest_dist;
            heap_idx[pos]  = heap_idx[largest];
            pos = largest;
        }

        heap_dist[pos] = dist;
        heap_idx[pos]  = idx;
    }
}

// Process up to 32 candidates (one per lane) for a single warp's heap.
// All threads in the warp participate via warp-wide ballot and shuffles.
// Lane 0 performs the actual heap updates; others supply candidate values.
__device__ __forceinline__
void warp_heap_process_candidates(float dist,
                                  int   idx,
                                  float *heap_dist,
                                  int   *heap_idx,
                                  int   k,
                                  int   *heap_size_ptr,
                                  int   lane_id)
{
    const unsigned full_mask = 0xffffffffu;

    // Determine which lanes have a valid candidate this iteration.
    unsigned active = __ballot_sync(full_mask, idx >= 0);

    // Sequentially feed all candidates to lane 0 in lane-id order.
    while (active) {
        int src_lane = __ffs(active) - 1;  // index of next lane with a candidate

        float c_dist = __shfl_sync(full_mask, dist, src_lane);
        int   c_idx  = __shfl_sync(full_mask, idx,  src_lane);

        if (lane_id == 0) {
            int heap_size = *heap_size_ptr;
            heap_insert_or_update(c_dist, c_idx, heap_dist, heap_idx, k, heap_size);
            *heap_size_ptr = heap_size;
        }

        // Clear this lane's bit and synchronize warp before processing next candidate.
        active &= active - 1;
        __syncwarp(full_mask);
    }
}

// Heapsort on a max-heap (dist, idx) of size n.
// After completion, dist[0..n-1] and idx[0..n-1] are sorted in ascending order of distance.
__device__ __forceinline__
void heap_sort(float *dist, int *idx, int n)
{
    for (int i = n - 1; i > 0; --i) {
        // Swap root with element i.
        float tmp_d = dist[0];
        dist[0] = dist[i];
        dist[i] = tmp_d;

        int tmp_i = idx[0];
        idx[0] = idx[i];
        idx[i] = tmp_i;

        // Restore heap property for the reduced heap [0, i).
        int pos = 0;
        while (true) {
            int left = (pos << 1) + 1;
            if (left >= i) {
                break;
            }
            int right = left + 1;

            int largest = left;
            float largest_dist = dist[left];
            if (right < i && dist[right] > largest_dist) {
                largest = right;
                largest_dist = dist[right];
            }

            if (dist[pos] >= largest_dist) {
                break;
            }

            // Swap parent with largest child.
            float td = dist[pos];
            dist[pos] = dist[largest];
            dist[largest] = td;

            int ti = idx[pos];
            idx[pos] = idx[largest];
            idx[largest] = ti;

            pos = largest;
        }
    }
}

// ---------------------------------------------------------------------------
// KNN kernel: one warp per query.
// - Each block has WARPS_PER_BLOCK warps => WARPS_PER_BLOCK queries per block.
// - Data points are processed in tiles; each tile is cached in shared memory.
// - Warps compute distances from their query to all points in the tile.
// - Distances are fed into a per-warp max-heap of size k in shared memory.
// - Final heap is sorted and written to the result array.
// ---------------------------------------------------------------------------

__global__
void knn_kernel_2d(const float2 * __restrict__ query,
                   int                    query_count,
                   const float2 * __restrict__ data,
                   int                    data_count,
                   PairIF * __restrict__  result,
                   int                    k)
{
    // Shared memory:
    //  - s_data:   cached tile of data points (all warps use).
    //  - s_heap_dist / s_heap_idx: per-warp heaps (MAX_K entries per warp).
    //  - s_heap_size: per-warp heap sizes.
    __shared__ float2 s_data[TILE_SIZE];
    __shared__ float  s_heap_dist[WARPS_PER_BLOCK * MAX_K];
    __shared__ int    s_heap_idx [WARPS_PER_BLOCK * MAX_K];
    __shared__ int    s_heap_size[WARPS_PER_BLOCK];

    const int thread_id     = threadIdx.x;
    const int warp_local_id = thread_id / WARP_SIZE;        // warp index within block
    const int lane_id       = thread_id & (WARP_SIZE - 1);  // lane index within warp
    const int warp_global_id = blockIdx.x * WARPS_PER_BLOCK + warp_local_id;

    const bool warp_active = (warp_global_id < query_count);

    // Pointers to this warp's heap storage in shared memory.
    float *heap_dist = s_heap_dist + warp_local_id * MAX_K;
    int   *heap_idx  = s_heap_idx  + warp_local_id * MAX_K;
    int   *heap_size_ptr = &s_heap_size[warp_local_id];

    // Initialize per-warp heap size.
    if (warp_active && lane_id == 0) {
        *heap_size_ptr = 0;
    }

    // Load query point into registers for the warp, using a broadcast from lane 0.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active) {
        float2 q;
        if (lane_id == 0) {
            q = query[warp_global_id];
        }
        qx = __shfl_sync(0xffffffffu, q.x, 0);
        qy = __shfl_sync(0xffffffffu, q.y, 0);
    }

    // Process data in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_SIZE) tile_size = TILE_SIZE;

        // Load tile into shared memory: all threads in the block participate.
        for (int i = thread_id; i < tile_size; i += blockDim.x) {
            s_data[i] = data[tile_start + i];
        }

        __syncthreads();  // Ensure tile is fully loaded before any warp uses it.

        if (warp_active) {
            // Each warp processes all points in the tile.
            // Within the warp, lanes stride through the tile with step WARP_SIZE.
            for (int offset = 0; offset < tile_size; offset += WARP_SIZE) {
                int idx_in_tile = offset + lane_id;

                float dist = 0.0f;
                int   data_idx = -1;  // mark invalid by default

                if (idx_in_tile < tile_size) {
                    float2 p = s_data[idx_in_tile];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    dist = dx * dx + dy * dy;
                    data_idx = tile_start + idx_in_tile;
                }

                // Update this warp's heap with up to 32 candidates in parallel.
                warp_heap_process_candidates(dist,
                                             data_idx,
                                             heap_dist,
                                             heap_idx,
                                             k,
                                             heap_size_ptr,
                                             lane_id);
            }
        }

        __syncthreads();  // All warps must finish using this tile before it is overwritten.
    }

    // Finalize: convert heap to sorted list and write results for this query.
    if (warp_active && lane_id == 0) {
        int heap_size = *heap_size_ptr;

        // data_count is guaranteed >= k, so heap_size should be at least k.
        if (heap_size > k) {
            heap_size = k;
        }

        // Heapsort to get neighbors ordered by increasing distance.
        heap_sort(heap_dist, heap_idx, heap_size);

        // Write results to global memory.
        PairIF *out = result + warp_global_id * k;
        for (int i = 0; i < k; ++i) {
            out[i].first  = heap_idx[i];
            out[i].second = heap_dist[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Host interface
// ---------------------------------------------------------------------------

void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    // Assumptions:
    //  - query, data, and result are device pointers allocated by cudaMalloc.
    //  - data_count >= k.
    //  - k is a power of two between 32 and 1024 inclusive.

    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    // Launch kernel. Cast result pointer to PairIF* as layouts are compatible (POD with int and float).
    knn_kernel_2d<<<grid_dim, block_dim>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<PairIF*>(result),
        k
    );

    // Optional: synchronize or error-checking can be added by the caller.
}