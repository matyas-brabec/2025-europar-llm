#include <cuda_runtime.h>
#include <utility>
#include <cfloat>
#include <stdint.h>

// Optimized CUDA k-NN (k is power-of-two between 32 and 1024) for 2D points.
// Each query is processed by a single warp (32 threads). A thread block contains
// multiple warps; the block collaboratively loads a batch (tile) of data points
// into shared memory. Each warp then computes distances from its query to all
// cached data points and cooperatively maintains a per-warp top-k using a
// 32-ary max-heap stored in shared memory. Updating the heap uses all 32 threads
// cooperatively (warp-level parallelism) and supports multiple candidate updates
// per tile. At the end, the heap is converted to ascending order via heap-sort
// and written to the output.
//
// Key implementation notes:
// - 32-ary heap yields depth <= 2 for k <= 1024 (since 32^2 = 1024), making
//   each sift-down very cheap while maximally utilizing a warp for the argmax
//   over the 32 children at each heap level.
// - Data points are processed in batches (tiles) loaded into shared memory by
//   the entire block. Each warp then iterates over this tile in groups of 32
//   candidates at a time.
// - For each group of up to 32 candidates, we cooperatively determine winners
//   (those with distance < current heap root). Among those, we iteratively pick
//   the smallest winner (warp argmin) and perform a replace-root operation on
//   the heap cooperatively with the warp. This uses multiple threads in the
//   warp to update the intermediate result with multiple candidates per tile.
// - Each warp maintains a private heap of distances and indices in shared memory,
//   with no additional global memory allocations.
//
// Tunable parameters:
// - WARPS_PER_BLOCK: number of warps per block (4 => 128 threads).
// - BATCH_SIZE: number of data points per tile loaded into shared memory.
//
// These are selected to balance shared memory usage and occupancy for A100/H100.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable hyper-parameters.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 8192  // BATCH_SIZE * sizeof(float2) + WARPS_PER_BLOCK * k * 8 bytes must fit in SMEM
#endif

// Full warp mask for modern architectures (assumes 32-thread warps).
static __device__ __forceinline__ unsigned full_warp_mask() { return 0xFFFFFFFFu; }

// Compute squared Euclidean distance between float2 points.
static __device__ __forceinline__ float sqr_distance(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // Use FMA to reduce instruction count.
    return fmaf(dx, dx, dy * dy);
}

// Warp-wide argmax reduction for (value, index) pairs.
// Returns the maximum value and its absolute array index (not lane index).
// The reduction uses all 32 lanes; for invalid participants, pass value = -INFINITY.
static __device__ __forceinline__ void warp_argmax(float value, int index, float &max_val_out, int &max_idx_out) {
    float v = value;
    int   i = index;
    // Tree-reduction using shfl_down. Full warp participates.
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float v_other = __shfl_down_sync(full_warp_mask(), v, offset);
        int   i_other = __shfl_down_sync(full_warp_mask(), i, offset);
        if (v_other > v) {
            v = v_other;
            i = i_other;
        }
    }
    // Broadcast the result from lane 0 to all lanes.
    max_val_out = __shfl_sync(full_warp_mask(), v, 0);
    max_idx_out = __shfl_sync(full_warp_mask(), i, 0);
}

// Warp-wide argmin reduction across lanes; lanes not in 'active_mask' should pass value=+INFINITY and any lane_id.
// Returns minimal value and the lane id that held it.
static __device__ __forceinline__ void warp_argmin_lane(float value, int lane_id, float &min_val_out, int &min_lane_out) {
    float v = value;
    int   l = lane_id;
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float v_other = __shfl_down_sync(full_warp_mask(), v, offset);
        int   l_other = __shfl_down_sync(full_warp_mask(), l, offset);
        // Prefer strictly smaller; tie-breaker on lower lane id (optional)
        if ((v_other < v) || ((v_other == v) && (l_other < l))) {
            v = v_other;
            l = l_other;
        }
    }
    min_val_out  = __shfl_sync(full_warp_mask(), v, 0);
    min_lane_out = __shfl_sync(full_warp_mask(), l, 0);
}

// Sift-down operation for a 32-ary max-heap stored in shared memory.
// Cooperatively executed by an entire warp. 'temp_dist'/'temp_idx' are the
// values to be placed starting at 'start_idx'. The heap has 'heap_size' elements.
//
// Heap layout (0-based):
//  - Node i has up to 32 children at positions [32*i + 1, ..., 32*i + 32] (within [0, heap_size-1]).
static __device__ __forceinline__ void warp_heap_sift_down(float *heap_dists, int *heap_idxs, int heap_size,
                                                           int start_idx, float temp_dist, int temp_idx,
                                                           int lane) {
    int i = start_idx;
    // Common warp path; no divergence among lanes.
    while (true) {
        int child_start = i * WARP_SIZE + 1;
        if (child_start >= heap_size) {
            // No children; place here.
            if (lane == 0) {
                heap_dists[i] = temp_dist;
                heap_idxs[i]  = temp_idx;
            }
            return;
        }
        int child_count = heap_size - child_start;
        if (child_count > WARP_SIZE) child_count = WARP_SIZE;

        // Each lane inspects one child; lanes beyond child_count set value to -inf.
        int child_pos = child_start + lane;
        float child_val = (lane < child_count) ? heap_dists[child_pos] : -CUDART_INF_F;
        int child_abs_idx = child_pos;

        // Compute argmax among children.
        float max_child_val;
        int   max_child_abs_idx;
        warp_argmax(child_val, child_abs_idx, max_child_val, max_child_abs_idx);

        // If temp is larger or equal than the maximum child, place temp here.
        if (!(temp_dist < max_child_val)) {
            if (lane == 0) {
                heap_dists[i] = temp_dist;
                heap_idxs[i]  = temp_idx;
            }
            return;
        }

        // Move the maximum child up.
        if (lane == 0) {
            heap_dists[i] = max_child_val;
            heap_idxs[i]  = heap_idxs[max_child_abs_idx];
        }
        // Continue sifting down.
        i = max_child_abs_idx;
        // Loop until leaf reached.
    }
}

// Build a 32-ary max-heap in-place for the first 'heap_size' elements of arrays.
// All lanes in the warp cooperate on each sift-down.
static __device__ __forceinline__ void warp_heap_build(float *heap_dists, int *heap_idxs, int heap_size, int lane) {
    if (heap_size <= 1) return;
    // Last internal node index in 32-ary heap: floor((n - 2) / 32).
    int last_internal = (heap_size - 2) / WARP_SIZE;
    for (int i = last_internal; i >= 0; --i) {
        float temp_dist = heap_dists[i];
        int   temp_idx  = heap_idxs[i];
        warp_heap_sift_down(heap_dists, heap_idxs, heap_size, i, temp_dist, temp_idx, lane);
    }
}

// Replace the root (maximum) of the 32-ary max-heap with (new_dist, new_idx) and fix heap property.
// All lanes in the warp cooperate via argmax of children at each level.
static __device__ __forceinline__ void warp_heap_replace_root(float *heap_dists, int *heap_idxs, int heap_size,
                                                              float new_dist, int new_idx, int lane) {
    if (heap_size == 0) return;
    // Place new at root then sift-down.
    warp_heap_sift_down(heap_dists, heap_idxs, heap_size, 0, new_dist, new_idx, lane);
}

// Pop the heap root (maximum) and replace it with the last element, then sift-down.
// The popped pair is returned via references (only lane 0 uses them).
static __device__ __forceinline__ void warp_heap_pop(float *heap_dists, int *heap_idxs, int &heap_size,
                                                     float &out_dist, int &out_idx, int lane) {
    // Read root
    float root_dist = heap_dists[0];
    int   root_idx  = heap_idxs[0];
    if (lane == 0) {
        out_dist = root_dist;
        out_idx  = root_idx;
    }
    // Move last to root and reduce heap size
    int new_size = heap_size - 1;
    if (new_size > 0) {
        float last_dist = heap_dists[new_size];
        int   last_idx  = heap_idxs[new_size];
        // Set heap_size to new_size (uniform across warp)
        heap_size = new_size;
        // Sift-down the last element placed at root
        warp_heap_sift_down(heap_dists, heap_idxs, heap_size, 0, last_dist, last_idx, lane);
    } else {
        // Heap becomes empty
        heap_size = new_size;
    }
}

// Main KNN kernel: each warp processes one query.
__global__ void knn_kernel(const float2 * __restrict__ query, int query_count,
                           const float2 * __restrict__ data,  int data_count,
                           std::pair<int, float> * __restrict__ result, int k) {
    // Shared memory layout:
    // [0 .. BATCH_SIZE-1]: float2 tile of data points
    // Then per-warp heaps:
    //   distances: WARPS_PER_BLOCK * k floats
    //   indices:   WARPS_PER_BLOCK * k ints
    extern __shared__ unsigned char smem[];
    // Data tile at the beginning
    float2 *s_data = reinterpret_cast<float2*>(smem);
    size_t s_data_bytes = static_cast<size_t>(BATCH_SIZE) * sizeof(float2);
    // Per-warp heaps after the data tile
    float *s_heap_dists_base = reinterpret_cast<float*>(smem + s_data_bytes);
    int   *s_heap_idxs_base  = reinterpret_cast<int*>(s_heap_dists_base + WARPS_PER_BLOCK * k);

    const int tid   = threadIdx.x;
    const int lane  = tid & (WARP_SIZE - 1);
    const int warp  = tid >> 5; // warp index within block
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp;

    const bool warp_active = (warp_global < query_count);

    // Warp-private heap pointers in shared memory.
    float *heap_dists = s_heap_dists_base + warp * k;
    int   *heap_idxs  = s_heap_idxs_base  + warp * k;

    // Load query for this warp
    float2 q;
    if (warp_active) {
        // Lane 0 loads query and broadcasts to other lanes
        if (lane == 0) {
            q = query[warp_global];
        }
        q.x = __shfl_sync(full_warp_mask(), q.x, 0);
        q.y = __shfl_sync(full_warp_mask(), q.y, 0);
    } else {
        // Avoid uninitialized use in inactive warps
        q.x = 0.0f; q.y = 0.0f;
    }

    // Initialize heap size and (optionally) content.
    int heap_size = 0;

    // Process the dataset in batches loaded into shared memory
    for (int base = 0; base < data_count; base += BATCH_SIZE) {
        int tile_count = data_count - base;
        if (tile_count > BATCH_SIZE) tile_count = BATCH_SIZE;

        // Block-wide cooperative load into shared memory
        for (int i = tid; i < tile_count; i += blockDim.x) {
            s_data[i] = data[base + i];
        }
        __syncthreads();

        // Each warp processes this tile. Iterate in groups of 32 points (one per lane)
        for (int j = 0; j < tile_count; j += WARP_SIZE) {
            // Local lane's candidate index within tile and global index
            int local_idx = j + lane;
            bool valid = (local_idx < tile_count);

            // Compute candidate distance (or INF if invalid)
            float2 p;
            if (valid) {
                p = s_data[local_idx];
            } else {
                p.x = 0.0f; p.y = 0.0f;
            }
            float cand_dist = valid ? sqr_distance(q, p) : CUDART_INF_F;
            int   cand_gidx = base + local_idx; // global index in data array

            if (!warp_active) {
                continue; // inactive warps just participate in __syncthreads outside
            }

            // Phase 1: Fill the heap until it has k elements.
            if (heap_size < k) {
                // Number of valid candidates in this group
                int group_valid = tile_count - j;
                if (group_valid > WARP_SIZE) group_valid = WARP_SIZE;

                int capacity_left = k - heap_size;
                int nfill = (capacity_left < group_valid) ? capacity_left : group_valid;

                if (lane < nfill) {
                    // Append directly to heap arrays (unsorted build buffer)
                    heap_dists[heap_size + lane] = cand_dist;
                    heap_idxs [heap_size + lane] = cand_gidx;
                }
                // Update heap_size uniformly across warp
                heap_size += nfill;

                // If we just completed filling to k, build the 32-ary max-heap cooperatively
                if (heap_size == k) {
                    // Build heap over first k elements
                    warp_heap_build(heap_dists, heap_idxs, k, lane);

                    // Process leftover candidates in this group if any
                    int leftover = group_valid - nfill;
                    if (leftover > 0) {
                        // Determine current threshold (root of max-heap)
                        float root = heap_dists[0];
                        // Active mask of leftover lanes only
                        bool is_leftover_lane = (lane >= nfill) && (lane < group_valid);
                        float active_val = (is_leftover_lane && (cand_dist < root)) ? cand_dist : CUDART_INF_F;

                        // Iteratively pick minimal among active and replace root
                        // (All threads cooperate via argmin and heap replace)
                        // We do not re-evaluate activity after replacements because the threshold only decreases.
                        unsigned active_mask = __ballot_sync(full_warp_mask(), is_leftover_lane && (cand_dist < root));
                        while (active_mask) {
                            // Compute argmin among currently active lanes
                            float min_val;
                            int   min_lane;
                            warp_argmin_lane(( (active_mask >> lane) & 1u ) ? cand_dist : CUDART_INF_F, lane, min_val, min_lane);
                            // Broadcast winner's data
                            float win_dist = __shfl_sync(full_warp_mask(), cand_dist, min_lane);
                            int   win_idx  = __shfl_sync(full_warp_mask(), cand_gidx,  min_lane);

                            // Replace root with winner and fix heap
                            warp_heap_replace_root(heap_dists, heap_idxs, k, win_dist, win_idx, lane);

                            // Clear this winner from active set
                            if (lane == min_lane) {
                                // Make this lane inactive
                                active_mask &= ~(1u << min_lane);
                            }
                            // Broadcast the updated active_mask to all lanes
                            active_mask = __shfl_sync(full_warp_mask(), active_mask, min_lane);
                        }
                    }
                }
                // Continue to next group
                continue;
            }

            // Phase 2: Heap is full; process current group of candidates against current threshold.
            // Read current heap root (threshold)
            float root = heap_dists[0];

            // Determine which lanes have candidates better than current root
            bool is_winner = valid && (cand_dist < root);
            unsigned win_mask = __ballot_sync(full_warp_mask(), is_winner);

            // Repeatedly extract the smallest among winners and replace heap root
            while (win_mask) {
                float min_val;
                int   min_lane;
                warp_argmin_lane( ((win_mask >> lane) & 1u) ? cand_dist : CUDART_INF_F, lane, min_val, min_lane);

                // Broadcast winner's data to all lanes
                float win_dist = __shfl_sync(full_warp_mask(), cand_dist, min_lane);
                int   win_idx  = __shfl_sync(full_warp_mask(), cand_gidx,  min_lane);

                // Replace root cooperatively and restore heap property
                warp_heap_replace_root(heap_dists, heap_idxs, k, win_dist, win_idx, lane);

                // Clear this winner bit from mask (only the winner lane modifies it; then broadcast)
                if (lane == min_lane) {
                    win_mask &= ~(1u << min_lane);
                }
                win_mask = __shfl_sync(full_warp_mask(), win_mask, min_lane);
            }
        }

        __syncthreads(); // Ensure all warps are done before the next tile load
    }

    // Finalize: transform heap into ascending order and write to result.
    if (warp_active) {
        int out_count = k;
        // Heap-sort: repeatedly pop the maximum to the end, resulting in ascending order.
        while (out_count > 0) {
            float top_dist = 0.0f;
            int   top_idx  = -1;
            // Pop root to (top_dist, top_idx)
            warp_heap_pop(heap_dists, heap_idxs, out_count, top_dist, top_idx, lane);
            if (lane == 0) {
                // Write to result; popping max to the end yields ascending order in [0..k-1]
                int out_pos = out_count; // after decrement, out_count is the index for this max
                // Store pair: index and distance
                result[warp_global * k + out_pos].first  = top_idx;
                result[warp_global * k + out_pos].second = top_dist;
            }
            // Continue until heap empty
        }
    }
}

// Host-side launcher. Assumes device pointers are valid and cudaMalloc'd.
// k is a power-of-two between 32 and 1024 inclusive and data_count >= k.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k) {
    // Grid configuration: one warp per query
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    int warps_total = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(warps_total, 1, 1);

    // Dynamic shared memory size:
    // - Data tile
    // - Per-warp heaps: WARPS_PER_BLOCK * k * (sizeof(float) + sizeof(int))
    size_t smem_size = static_cast<size_t>(BATCH_SIZE) * sizeof(float2)
                     + static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    knn_kernel<<<grid, block, smem_size>>>(query, query_count, data, data_count, result, k);
}