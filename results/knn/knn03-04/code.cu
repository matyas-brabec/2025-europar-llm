#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// This implementation computes k-NN for 2D points using one warp (32 threads) per query.
// It processes the data points in batches loaded into shared memory by the entire block.
// Each warp maintains a distributed top-k (across its 32 threads), with k guaranteed to be
// a power of two between 32 and 1024. Thus, each lane owns S = k / 32 slots. We store
// each lane's S candidates in registers. The warp cooperatively maintains a global "gate"
// equal to the current worst (largest) distance among the distributed top-k to reject
// non-improving candidates quickly. When a replacement happens, the owning lane updates
// its local maximum and the warp recomputes the global gate via a warp reduction.
// After scanning all data, each lane sorts its local S candidates (ascending). The warp
// then performs a 32-way merge to produce the final sorted top-k list for the query and
// writes it to the result array.
//
// Notes:
// - Distances are squared L2 norm (no sqrt).
// - No extra global memory is allocated; only shared memory for data tiles is used.
// - The kernel assumes a modern NVIDIA GPU (A100/H100). It uses warp shuffles for
//   intra-warp communication and __syncthreads to guard shared memory tiles.
// - The block size is chosen as 4 warps (128 threads). The data tile size is 4096 points,
//   which consumes 32 KB of shared memory for float2 points. You can tune these constants
//   for different GPUs if desired.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tune these for the target GPU if needed.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4   // 4 warps = 128 threads per block
#endif

#ifndef TILE_POINTS
#define TILE_POINTS 4096    // Number of float2 points per shared-memory tile (32 KB)
#endif

// Maximum per-lane capacity: k <= 1024 and warp size is 32 => S_max = 32
#define MAX_S (1024 / WARP_SIZE)

static __device__ __forceinline__ unsigned full_warp_mask() {
    // All 32 lanes active
    return 0xFFFFFFFFu;
}

// Warp-wide argmax reduction with broadcast of the winning value and lane.
// Ties are broken in favor of the larger lane index to ensure deterministic behavior.
static __device__ __forceinline__ void warp_argmax_broadcast(float val_in, int lane_id, float &max_val_out, int &max_lane_out, unsigned mask) {
    float v = val_in;
    int   l = lane_id;
    // Tree reduction using shuffle down
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float v_other = __shfl_down_sync(mask, v, offset);
        int   l_other = __shfl_down_sync(mask, l, offset);
        if (v_other > v || (v_other == v && l_other > l)) {
            v = v_other;
            l = l_other;
        }
    }
    // Broadcast final max value and lane from lane 0 of the warp
    max_val_out  = __shfl_sync(mask, v, 0);
    max_lane_out = __shfl_sync(mask, l, 0);
}

// Warp-wide argmin reduction with broadcast of the winning value and lane.
// Ties are broken in favor of the smaller lane index for stable ordering in outputs.
static __device__ __forceinline__ void warp_argmin_broadcast(float val_in, int lane_id, float &min_val_out, int &min_lane_out, unsigned mask) {
    float v = val_in;
    int   l = lane_id;
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float v_other = __shfl_down_sync(mask, v, offset);
        int   l_other = __shfl_down_sync(mask, l, offset);
        if (v_other < v || (v_other == v && l_other < l)) {
            v = v_other;
            l = l_other;
        }
    }
    min_val_out  = __shfl_sync(mask, v, 0);
    min_lane_out = __shfl_sync(mask, l, 0);
}

// Insert a candidate (dist, idx) into the warp-distributed top-k if it improves (dist < gate).
// The global worst (gate) resides at (gate_owner_lane, that lane's local_max_slot).
// After replacement, the owner lane recomputes its local maximum and the warp recomputes the gate.
static __device__ __forceinline__ void warp_topk_try_insert(
    float cand_dist, int cand_idx,
    float &gate, int &gate_owner_lane, int &gate_owner_slot,
    float local_dist[MAX_S], int local_idx[MAX_S], int S,
    float &local_max_val, int &local_max_slot,
    int lane_id, unsigned mask)
{
    if (cand_dist >= gate) return; // Not improving the current top-k
    // Replace the current global worst in the owning lane
    if (lane_id == gate_owner_lane) {
        // Replace at the local max slot
        local_dist[local_max_slot] = cand_dist;
        local_idx[local_max_slot]  = cand_idx;

        // Recompute local maximum (worst in this lane's S candidates)
        // S is at most 32; a linear scan is fine.
        float mx_val = local_dist[0];
        int   mx_pos = 0;
#pragma unroll
        for (int i = 1; i < MAX_S; ++i) {
            if (i >= S) break;
            float v = local_dist[i];
            if (v > mx_val) {
                mx_val = v;
                mx_pos = i;
            }
        }
        local_max_val  = mx_val;
        local_max_slot = mx_pos;
    }
    __syncwarp(mask);

    // Recompute the global worst (gate) among all lanes' local maxima.
    float new_gate;
    int   new_owner_lane;
    warp_argmax_broadcast(local_max_val, lane_id, new_gate, new_owner_lane, mask);
    int new_owner_slot = __shfl_sync(mask, local_max_slot, new_owner_lane);

    gate            = new_gate;
    gate_owner_lane = new_owner_lane;
    gate_owner_slot = new_owner_slot;
    __syncwarp(mask);
}

// Insertion sort of a small array (size S <= 32) in ascending order.
static __device__ __forceinline__ void small_insertion_sort(float dist_arr[MAX_S], int idx_arr[MAX_S], int S) {
#pragma unroll
    for (int i = 1; i < MAX_S; ++i) {
        if (i >= S) break;
        float key_d = dist_arr[i];
        int   key_i = idx_arr[i];
        int j = i - 1;
        while (j >= 0 && dist_arr[j] > key_d) {
            dist_arr[j + 1] = dist_arr[j];
            idx_arr[j + 1]  = idx_arr[j];
            --j;
        }
        dist_arr[j + 1] = key_d;
        idx_arr[j + 1]  = key_i;
    }
}

// Kernel implementing batched, warp-cooperative k-NN for 2D points.
__global__ void knn2d_kernel(const float2 * __restrict__ query,
                             int query_count,
                             const float2 * __restrict__ data,
                             int data_count,
                             std::pair<int, float> * __restrict__ result,
                             int k)
{
    extern __shared__ unsigned char smem_raw[];
    // Shared memory tile for data points
    float2 *s_points = reinterpret_cast<float2*>(smem_raw);

    const int lane_id        = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block  = threadIdx.x >> 5; // threadIdx.x / 32
    const int warps_per_block = WARPS_PER_BLOCK;
    const int warp_global    = blockIdx.x * warps_per_block + warp_in_block;
    const bool warp_active   = (warp_global < query_count);
    const unsigned warp_mask = full_warp_mask();

    // Compute per-lane capacity S = k / 32 (k guaranteed divisible by 32).
    const int S = k >> 5;

    // Load and broadcast query point for this warp.
    float qx = 0.0f, qy = 0.0f;
    if (warp_active && lane_id == 0) {
        float2 q = query[warp_global];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(warp_mask, qx, 0);
    qy = __shfl_sync(warp_mask, qy, 0);

    // Per-lane local storage for distributed top-k
    float local_dist[MAX_S];
    int   local_idx[MAX_S];
#pragma unroll
    for (int i = 0; i < MAX_S; ++i) {
        local_dist[i] = FLT_MAX;
        local_idx[i]  = -1;
    }

    // Each lane tracks its local maximum (worst) among its S candidates.
    float local_max_val  = FLT_MAX;
    int   local_max_slot = 0;

    // Initialize global gate (worst among all lanes).
    float gate_val;
    int   gate_owner_lane;
    warp_argmax_broadcast(local_max_val, lane_id, gate_val, gate_owner_lane, warp_mask);
    int   gate_owner_slot = __shfl_sync(warp_mask, local_max_slot, gate_owner_lane);

    // Process data in shared-memory tiles
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        int tile_count = data_count - base;
        if (tile_count > TILE_POINTS) tile_count = TILE_POINTS;

        // Load tile cooperatively by all threads in the block
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_points[i] = data[base + i];
        }
        __syncthreads();

        // Each warp processes the tile for its query
        if (warp_active) {
            for (int offset = 0; offset < tile_count; offset += WARP_SIZE) {
                int j = offset + lane_id;

                // Compute candidate distance for this lane
                float cand_dist = FLT_MAX;
                int   cand_idx  = -1;
                if (j < tile_count) {
                    float2 p = s_points[j];
                    float dx = qx - p.x;
                    float dy = qy - p.y;
                    cand_dist = dx * dx + dy * dy; // squared L2 distance
                    cand_idx  = base + j;
                }

                // Sequentially attempt insertion for all 32 candidates in this group.
#pragma unroll
                for (int t = 0; t < WARP_SIZE; ++t) {
                    float d2  = __shfl_sync(warp_mask, cand_dist, t);
                    int   idx = __shfl_sync(warp_mask, cand_idx,  t);
                    // Try inserting if better than current gate
                    warp_topk_try_insert(d2, idx,
                                         gate_val, gate_owner_lane, gate_owner_slot,
                                         local_dist, local_idx, S,
                                         local_max_val, local_max_slot,
                                         lane_id, warp_mask);
                }
            }
        }
        __syncthreads();
    }

    // Finalization for active warps: sort per-lane local lists and produce globally sorted k-NN
    if (warp_active) {
        // Sort each lane's local list ascending (by distance)
        small_insertion_sort(local_dist, local_idx, S);
        __syncwarp(warp_mask);

        // Multi-way merge of 32 sorted lists (each of size S) to produce the final k sorted neighbors.
        int head = 0; // this lane's head pointer into its local list
        for (int t = 0; t < k; ++t) {
            float my_val = (head < S) ? local_dist[head] : FLT_MAX;
            // Find minimum distance among lanes
            float min_val;
            int   min_lane;
            warp_argmin_broadcast(my_val, lane_id, min_val, min_lane, warp_mask);
            // Retrieve the index corresponding to the winning lane's current head
            int min_idx = __shfl_sync(warp_mask, (head < S ? local_idx[head] : -1), min_lane);
            // Lane 0 writes the output
            if (lane_id == 0) {
                // Store as j-th nearest neighbor for this query (sorted ascending by distance)
                std::pair<int, float> &dst = result[warp_global * k + t];
                dst.first  = min_idx;
                dst.second = min_val;
            }
            // Winning lane advances its head
            if (lane_id == min_lane) {
                ++head;
            }
            __syncwarp(warp_mask);
        }
    }
}

// Host API entry point
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Choose grid/block dimensions
    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int num_warps = (query_count + warps_per_block - 1) / warps_per_block;
    dim3 grid(num_warps, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    // Shared memory size: only the data tile is stored in shared memory (float2 * TILE_POINTS)
    size_t shared_bytes = sizeof(float2) * TILE_POINTS;

    // Launch kernel
    knn2d_kernel<<<grid, block, shared_bytes>>>(query, query_count, data, data_count, result, k);

    // Optionally, synchronize or check for errors here if desired by the caller.
    // cudaDeviceSynchronize();
}