#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// -----------------------------------------------------------------------------
// CUDA k-NN for 2D points (float2) using one warp per query.
//
// Each query is processed by a single warp (32 threads).
// - The data points are processed in tiles cached in shared memory.
// - The warp maintains a max-heap of size k containing the k smallest
//   distances seen so far (squared Euclidean distances).
// - The heap is distributed across threads in the warp in a lane-strided
//   fashion to keep everything in registers.
// - Warp shuffles are used to read/modify heap entries that reside in
//   different lanes.
// - Initial k candidates are collected unsorted, then a bottom-up heap build
//   is performed. Remaining candidates are inserted using heap updates.
// - After all data is processed, an in-place heapsort generates sorted
//   neighbors (ascending distance), then results are written to global memory.
//
// Assumptions:
// - k is a power of two between 32 and 1024 inclusive (so k is a multiple of 32).
// - data_count >= k.
// - query_count and data_count are large enough to benefit from GPU parallelism.
// - All pointers passed to run_knn are allocated via cudaMalloc.
// -----------------------------------------------------------------------------

// Warp and block configuration.
static constexpr int WARP_SIZE         = 32;
static constexpr int WARPS_PER_BLOCK   = 8;      // 8 warps per block => 256 threads.
static constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

// Shared-memory tile size for data points.
// 4096 * sizeof(float2) = 32 KB per block, which is reasonable on modern GPUs.
static constexpr int TILE_SIZE = 4096;

// Maximum supported k (as per problem statement).
static constexpr int MAX_K = 1024;

// Number of heap elements per lane (lane-strided storage).
// MAX_K is 1024 and WARP_SIZE is 32, so this equals 32.
static constexpr int MAX_HEAP_ELEMS_PER_LANE = (MAX_K + WARP_SIZE - 1) / WARP_SIZE;

// Convenience macro for full warp mask.
static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;


// -----------------------------------------------------------------------------
// Heap helper functions (device-side, warp-synchronous).
//
// Heap representation:
//   - Max-heap of up to 'heap_size' elements, indexes [0, heap_size-1].
//   - Logical index i is mapped to (lane, slot):
//           lane = i % WARP_SIZE
//           slot = i / WARP_SIZE
//   - Each lane owns MAX_HEAP_ELEMS_PER_LANE "slots" in registers:
//           dist_heap[slot], idx_heap[slot]
//
// All helper functions are warp-synchronous: they must be called by all threads
// in a warp with the same arguments. Communication between lanes uses
// __shfl_sync, and updates to heap entries are performed by the owning lane.
// -----------------------------------------------------------------------------

// Bubble-down (heapify-down) from a specific node index 'start_idx' in a
// max-heap. Runs entirely within one warp.
__device__ __forceinline__
void warp_heapify_down(float *dist_heap,
                       int   *idx_heap,
                       int    heap_size,
                       int    start_idx,
                       int    lane_id)
{
    int idx = start_idx;

    while (true) {
        int left  = idx * 2 + 1;
        if (left >= heap_size) {
            // No children; heap property satisfied.
            break;
        }
        int right = left + 1;

        // Map node indices to (lane, slot).
        int lane_idx  = idx   & (WARP_SIZE - 1);
        int slot_idx  = idx   >> 5; // divide by 32
        int lane_left = left  & (WARP_SIZE - 1);
        int slot_left = left  >> 5;
        int lane_right = 0;
        int slot_right = 0;

        // Gather parent and left child distances.
        float dist_idx  = __shfl_sync(FULL_MASK, dist_heap[slot_idx],  lane_idx);
        float dist_left = __shfl_sync(FULL_MASK, dist_heap[slot_left], lane_left);

        // Gather right child distance if it exists.
        float dist_right = -FLT_MAX;
        bool has_right = (right < heap_size);
        if (has_right) {
            lane_right = right & (WARP_SIZE - 1);
            slot_right = right >> 5;
            dist_right = __shfl_sync(FULL_MASK, dist_heap[slot_right], lane_right);
        }

        // Lane 0 decides which node (parent/left/right) has the largest distance.
        int  max_idx   = idx;
        float max_dist = dist_idx;
        if (dist_left > max_dist) {
            max_dist = dist_left;
            max_idx  = left;
        }
        if (has_right && dist_right > max_dist) {
            max_dist = dist_right;
            max_idx  = right;
        }

        // Broadcast max_idx from lane 0 to all lanes.
        max_idx = __shfl_sync(FULL_MASK, max_idx, 0);
        if (max_idx == idx) {
            // Heap property satisfied at this node.
            break;
        }

        // Swap node at idx with node at max_idx (their distances and indices).
        int child_idx   = max_idx;
        int lane_child  = child_idx & (WARP_SIZE - 1);
        int slot_child  = child_idx >> 5;

        float dist_node  = __shfl_sync(FULL_MASK, dist_heap[slot_idx],   lane_idx);
        int   id_node    = __shfl_sync(FULL_MASK, idx_heap[slot_idx],    lane_idx);
        float dist_child = __shfl_sync(FULL_MASK, dist_heap[slot_child], lane_child);
        int   id_child   = __shfl_sync(FULL_MASK, idx_heap[slot_child],  lane_child);

        if (lane_id == lane_idx) {
            dist_heap[slot_idx] = dist_child;
            idx_heap[slot_idx]  = id_child;
        }
        if (lane_id == lane_child) {
            dist_heap[slot_child] = dist_node;
            idx_heap[slot_child]  = id_node;
        }

        idx = child_idx; // Continue bubbling down at child's position.
    }
}


// Build a max-heap in-place from arbitrary values in [0, heap_size-1] using
// bottom-up heap construction.
__device__ __forceinline__
void warp_build_max_heap(float *dist_heap,
                         int   *idx_heap,
                         int    heap_size,
                         int    lane_id)
{
    // Start from the last parent node and heapify downwards.
    for (int i = (heap_size / 2) - 1; i >= 0; --i) {
        warp_heapify_down(dist_heap, idx_heap, heap_size, i, lane_id);
    }
}


// In-place heap sort on the max-heap, producing ascending order by distance.
// After this, the logical indices [0, heap_size-1] contain sorted neighbors
// (smallest distance at index 0).
__device__ __forceinline__
void warp_heap_sort_ascending(float *dist_heap,
                              int   *idx_heap,
                              int    heap_size,
                              int    lane_id)
{
    // Standard heapsort using a max-heap.
    for (int end = heap_size - 1; end > 0; --end) {
        // Swap root (index 0) with element at 'end'.
        int root_idx = 0;
        int lane_root = root_idx & (WARP_SIZE - 1);
        int slot_root = root_idx >> 5;
        int lane_end  = end      & (WARP_SIZE - 1);
        int slot_end  = end      >> 5;

        float dist_root = __shfl_sync(FULL_MASK, dist_heap[slot_root], lane_root);
        int   id_root   = __shfl_sync(FULL_MASK, idx_heap[slot_root],  lane_root);
        float dist_end  = __shfl_sync(FULL_MASK, dist_heap[slot_end],  lane_end);
        int   id_end    = __shfl_sync(FULL_MASK, idx_heap[slot_end],   lane_end);

        if (lane_id == lane_root) {
            dist_heap[slot_root] = dist_end;
            idx_heap[slot_root]  = id_end;
        }
        if (lane_id == lane_end) {
            dist_heap[slot_end] = dist_root;
            idx_heap[slot_end]  = id_root;
        }

        // Restore heap property for the reduced heap [0, end-1].
        warp_heapify_down(dist_heap, idx_heap, end, 0, lane_id);
    }
}


// -----------------------------------------------------------------------------
// k-NN kernel: one warp per query.
// -----------------------------------------------------------------------------
__global__
void knn_kernel(const float2 * __restrict__ query,
                int                    query_count,
                const float2 * __restrict__ data,
                int                    data_count,
                std::pair<int, float> * __restrict__ result,
                int                    k)
{
    extern __shared__ float2 shared_data[]; // Shared-memory tile of data points.

    const int thread_id      = threadIdx.x;
    const int lane_id        = thread_id & (WARP_SIZE - 1);
    const int warp_id_in_block = thread_id / WARP_SIZE;
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    const bool warp_active = (global_warp_id < query_count);

    // Each warp corresponds to one query index.
    const int query_idx = global_warp_id;

    // Per-lane heap storage (distributed heap for this query).
    float dist_heap[MAX_HEAP_ELEMS_PER_LANE];
    int   idx_heap [MAX_HEAP_ELEMS_PER_LANE];

    // Early exit for warps without an assigned query: they still must
    // participate in __syncthreads() but skip all query-dependent work.
    if (!warp_active) {
        // Still need to participate in shared memory loads and synchronizations.
        for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
            int tile_len = data_count - tile_start;
            if (tile_len > TILE_SIZE) tile_len = TILE_SIZE;

            // Cooperative load of data tile into shared memory.
            for (int i = thread_id; i < tile_len; i += blockDim.x) {
                shared_data[i] = data[tile_start + i];
            }
            __syncthreads();
            __syncthreads();
        }
        return;
    }

    // Load query point for this warp.
    float2 q = query[query_idx];

    // Total number of data points processed so far for this query.
    int processed = 0;

    // Once 'processed == k', heap will be constructed from the first k entries.
    bool heap_built = false;

    // Main loop over data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_len = data_count - tile_start;
        if (tile_len > TILE_SIZE) tile_len = TILE_SIZE;

        // Load this tile of data into shared memory using the whole block.
        for (int i = thread_id; i < tile_len; i += blockDim.x) {
            shared_data[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each active warp processes the shared tile against its query.
        // We process tile entries in groups of WARP_SIZE. In each group,
        // each lane computes distance for one point, then candidates are
        // fed into the warp-local heap in a warp-synchronous manner.
        for (int base = 0; base < tile_len; base += WARP_SIZE) {
            int idx_in_tile = base + lane_id;
            bool valid = (idx_in_tile < tile_len);

            float cand_dist = 0.0f;
            int   cand_idx  = -1;

            if (valid) {
                float2 p = shared_data[idx_in_tile];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                cand_dist = dx * dx + dy * dy;           // squared Euclidean distance
                cand_idx  = tile_start + idx_in_tile;    // global index of data point
            }

            // Process the WARP_SIZE candidates in a deterministic order
            // (lane 0 to lane 31). Each candidate is handled by the whole warp
            // when updating the heap.
            #pragma unroll
            for (int src_lane = 0; src_lane < WARP_SIZE; ++src_lane) {
                bool src_valid = __shfl_sync(FULL_MASK, valid, src_lane);
                if (!src_valid) {
                    continue;
                }

                float dist = __shfl_sync(FULL_MASK, cand_dist, src_lane);
                int   idx  = __shfl_sync(FULL_MASK, cand_idx,  src_lane);

                if (processed < k) {
                    // Fill initial buffer of k elements (unsorted).
                    int insert_idx = processed;
                    int owner_lane = insert_idx & (WARP_SIZE - 1);
                    int owner_slot = insert_idx >> 5;

                    if (lane_id == owner_lane) {
                        dist_heap[owner_slot] = dist;
                        idx_heap[owner_slot]  = idx;
                    }

                    ++processed;

                    // Once we have k elements, build the initial max-heap.
                    if (processed == k) {
                        warp_build_max_heap(dist_heap, idx_heap, k, lane_id);
                        heap_built = true;
                    }
                } else {
                    // For remaining elements, use heap insertion:
                    // replace heap root if the new candidate is closer.
                    if (!heap_built) {
                        // Safety: in case processed starts >= k for some reason.
                        warp_build_max_heap(dist_heap, idx_heap, k, lane_id);
                        heap_built = true;
                    }

                    // Broadcast current root distance from lane 0.
                    float root_dist = __shfl_sync(FULL_MASK, dist_heap[0], 0);

                    if (dist < root_dist) {
                        // Replace root with the new, closer candidate.
                        if (lane_id == 0) {
                            dist_heap[0] = dist;
                            idx_heap[0]  = idx;
                        }
                        // Restore heap property.
                        warp_heapify_down(dist_heap, idx_heap, k, 0, lane_id);
                    }

                    ++processed;
                }
            }
        }

        __syncthreads();
    }

    // After all data points have been processed, dist_heap/idx_heap contain
    // the k nearest neighbor candidates in a max-heap (largest distance at root).
    // Convert to ascending order using heap sort.
    warp_heap_sort_ascending(dist_heap, idx_heap, k, lane_id);

    // Write sorted results to global memory.
    // Each lane writes its own lane-strided entries.
    int slots_per_lane = k / WARP_SIZE; // k is a multiple of 32 by assumption.
    for (int slot = 0; slot < slots_per_lane; ++slot) {
        int j = lane_id + slot * WARP_SIZE; // logical index in [0, k-1]
        int out_index = query_idx * k + j;

        result[out_index].first  = idx_heap[slot];
        result[out_index].second = dist_heap[slot];
    }
}


// -----------------------------------------------------------------------------
// Host wrapper: launches the k-NN kernel.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    // Compute grid configuration: one warp per query.
    int total_warps = (query_count + 0); // 1 warp per query
    int blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim(blocks, 1, 1);

    // Shared memory size: TILE_SIZE float2's per block.
    size_t shared_mem_bytes = static_cast<size_t>(TILE_SIZE) * sizeof(float2);

    knn_kernel<<<grid_dim, block_dim, shared_mem_bytes>>>(
        query, query_count, data, data_count, result, k
    );

    // Optionally synchronize here to ensure completion before returning.
    cudaDeviceSynchronize();
}