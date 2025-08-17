#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <utility>

// This implementation assigns one warp (32 threads) to process one query point.
// The data points are processed in tiles loaded into shared memory by the whole block.
// Each warp maintains an intermediate private top-k (indices and distances) in shared memory,
// updated using warp-synchronous operations. After scanning all data points, the per-warp
// top-k list is sorted using a warp-synchronous bitonic sort and written to the output.

// Tunable parameters (chosen for modern data center GPUs like A100/H100)
#ifndef KNN_WARPS_PER_BLOCK
#define KNN_WARPS_PER_BLOCK 8                // 8 warps per block => 256 threads per block
#endif

#ifndef KNN_THREADS_PER_BLOCK
#define KNN_THREADS_PER_BLOCK (KNN_WARPS_PER_BLOCK * 32)
#endif

// Helper: warp size constant
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Device helper: warp-synchronous argmax reduction on values, returning the max value
// in lane 0 and the lane index where the maximum resides (also in lane 0).
// All lanes participate with the same mask; the output is valid in lane 0.
// After calling this, broadcast with __shfl_sync(mask, value, 0) to share among lanes.
__device__ __forceinline__
void warp_argmax_lane0(float val_in, int lane_id, float &max_val_out_lane0, int &max_lane_out_lane0) {
    unsigned mask = 0xffffffffu;
    float best_val = val_in;
    int   best_lane = lane_id;

    // Reduce using pair (value, lane) with preference for larger value
    // Note: Since distances are non-negative, no special NaN handling is needed here.
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(mask, best_val, offset);
        int   other_lane = __shfl_down_sync(mask, best_lane, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_lane = other_lane;
        }
    }
    if (lane_id == 0) {
        max_val_out_lane0 = best_val;
        max_lane_out_lane0 = best_lane;
    }
}

// Device helper: recompute the local (per-lane) maximum within a lane's contiguous slice
// of the warp's top-k shared array.
// Each warp owns 'k' entries in shared memory. Each lane owns a slice of L = k / 32 entries.
// Arguments:
//   topk_dists_base: pointer to the beginning of this warp's distance array in shared memory
//   L:               entries per lane (k / 32)
//   lane_id:         0..31 lane index
// Outputs:
//   local_max_val:   maximum distance value found in this lane's slice
//   local_max_pos:   position within the lane's slice [0, L) where the maximum resides
__device__ __forceinline__
void recompute_local_max(const float* topk_dists_base, int L, int lane_id,
                         float &local_max_val, int &local_max_pos)
{
    // Scan this lane's slice: [lane_id * L, lane_id * L + L)
    int start = lane_id * L;
    float mval = topk_dists_base[start];
    int mpos = 0;
#pragma unroll
    for (int i = 1; i < 32; ++i) {
        if (i >= L) break;
        float v = topk_dists_base[start + i];
        if (v > mval) { mval = v; mpos = i; }
    }
    local_max_val = mval;
    local_max_pos = mpos;
}

// Device helper: warp-synchronous bitonic sort over the warp's private top-k arrays in shared memory.
// Sorts in ascending order by distance.
// Each warp owns a contiguous segment [warp_base, warp_base + k) in shared memory.
// Each lane handles a contiguous slice of length L = k / 32.
// This is a standard in-place bitonic sorting network implemented with warp-synchronous barriers.
__device__ __forceinline__
void warp_bitonic_sort_shared(float* topk_dists, int* topk_index, int warp_base, int k, int L, int lane_id)
{
    unsigned mask = 0xffffffffu;

    // Bitonic sort on k elements. k is guaranteed to be a power of two.
    for (int size = 2; size <= k; size <<= 1) {
        // direction alternates per subsequence; computed by (i & size) == 0
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes its L items
#pragma unroll
            for (int li = 0; li < 32; ++li) {
                if (li >= L) break;

                int local_idx = lane_id * L + li;             // [0, k)
                int partner   = local_idx ^ stride;           // partner index within [0, k)

                if (partner > local_idx && partner < k) {
                    bool up = ((local_idx & size) == 0);      // true: ascending; false: descending

                    int idx_a = warp_base + local_idx;
                    int idx_b = warp_base + partner;

                    float a = topk_dists[idx_a];
                    float b = topk_dists[idx_b];
                    int   ia = topk_index[idx_a];
                    int   ib = topk_index[idx_b];

                    // Compare-and-swap according to direction
                    if ((a > b) == up) {
                        topk_dists[idx_a] = b;
                        topk_dists[idx_b] = a;
                        topk_index[idx_a] = ib;
                        topk_index[idx_b] = ia;
                    }
                }
            }
            // Synchronize within warp to ensure the next phase sees updated values
            __syncwarp(mask);
        }
    }
}

// Main kernel: one warp per query.
__global__ void knn_kernel_warp32(const float2* __restrict__ query, int query_count,
                                  const float2* __restrict__ data, int data_count,
                                  std::pair<int, float>* __restrict__ result,
                                  int k, int tile_capacity_points)
{
    // Shared memory layout:
    // [ float2 tile[tile_capacity_points] ]
    // [ float  topk_dists[KNN_WARPS_PER_BLOCK * k] ]
    // [   int  topk_index[KNN_WARPS_PER_BLOCK * k] ]
    extern __shared__ unsigned char smem_raw[];
    float2* __restrict__ tile = reinterpret_cast<float2*>(smem_raw);
    size_t tile_bytes = static_cast<size_t>(tile_capacity_points) * sizeof(float2);
    float* __restrict__ topk_dists = reinterpret_cast<float*>(smem_raw + tile_bytes);
    int*   __restrict__ topk_index = reinterpret_cast<int*>(topk_dists + KNN_WARPS_PER_BLOCK * k);

    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x >> 5; // 0..KNN_WARPS_PER_BLOCK-1
    const int warps_per_block = KNN_WARPS_PER_BLOCK;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;

    if (global_warp_id >= query_count) {
        return;
    }

    // Each warp processes one query
    const float2 q = query[global_warp_id];

    // Per-warp base pointers into shared top-k arrays
    const int warp_base = warp_id_in_block * k;

    // L = k / 32 (guaranteed integer because k is power of two and >= 32)
    const int L = k >> 5;

    // Initialize per-warp top-k arrays with +infinity / -1
    const float INF = CUDART_INF_F;
    for (int i = lane_id; i < k; i += WARP_SIZE) {
        topk_dists[warp_base + i] = INF;
        topk_index[warp_base + i] = -1;
    }
    __syncwarp();

    // Initialize local maxima for each lane's slice
    float local_max_val = INF;
    int   local_max_pos = 0;

    // Compute initial warp-wide threshold (worst distance among current top-k)
    // Initially it's +INF; we set owner lane to 0 and pos 0. We'll maintain/upate incrementally.
    float thr_val = INF;
    int   thr_lane = 0;
    int   thr_local_pos = 0;

    // Process data in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_capacity_points) {
        int tile_count = data_count - tile_start;
        if (tile_count > tile_capacity_points) tile_count = tile_capacity_points;

        // Load tile into shared memory cooperatively by the entire block
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            tile[i] = data[tile_start + i];
        }
        __syncthreads(); // Ensure tile is fully loaded before processing

        // Iterate over points in the tile in steps of warp size
        for (int base = 0; base < tile_count; base += WARP_SIZE) {
            int idx_in_tile = base + lane_id;

            // Compute squared L2 distance for this lane's candidate (if valid)
            float cand_dist = INF;
            int   cand_index = -1;
            bool  valid = (idx_in_tile < tile_count);
            if (valid) {
                float2 p = tile[idx_in_tile];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                // Fused multiply-add can help precision and performance; compiler will usually generate FMA.
                cand_dist = dx * dx + dy * dy;
                cand_index = tile_start + idx_in_tile;
            }

            unsigned warp_mask = 0xffffffffu;
            unsigned valid_mask = __ballot_sync(warp_mask, valid);

            // Fast path: if no valid lanes, continue
            if (valid_mask == 0u) continue;

            // Determine which lanes' candidates pass the current threshold
            unsigned accept_mask = __ballot_sync(warp_mask, (valid && (cand_dist < thr_val)));

            // Insert accepted candidates one by one in lane order, updating the threshold incrementally.
            while (accept_mask) {
                int leader = __ffs(accept_mask) - 1;

                // Broadcast candidate from leader lane
                float lead_dist = __shfl_sync(warp_mask, cand_dist, leader);
                int   lead_idx  = __shfl_sync(warp_mask, cand_index, leader);

                // Re-check against (possibly updated) threshold to ensure correctness
                if (lead_dist < thr_val) {
                    // Replace the current worst element with this candidate
                    // Compute the global index of the worst element: warp_base + thr_lane * L + thr_local_pos
                    int worst_global_pos = warp_base + thr_lane * L + thr_local_pos;

                    // Perform the replacement by a single lane (lane 0) to avoid write conflicts
                    if (lane_id == 0) {
                        topk_dists[worst_global_pos] = lead_dist;
                        topk_index[worst_global_pos] = lead_idx;
                    }
                    __syncwarp(warp_mask);

                    // Only the lane that owns the replaced slice needs to recompute its local maximum
                    if (lane_id == thr_lane) {
                        recompute_local_max(topk_dists + warp_base, L, lane_id, local_max_val, local_max_pos);
                    }
                    __syncwarp(warp_mask);

                    // Recompute the warp-wide maximum (new threshold) using the per-lane maxima.
                    float max_val_lane0 = 0.0f;
                    int   max_lane_lane0 = 0;
                    warp_argmax_lane0(local_max_val, lane_id, max_val_lane0, max_lane_lane0);

                    // Broadcast the threshold value and owner lane to all lanes
                    thr_val  = __shfl_sync(warp_mask, max_val_lane0, 0);
                    thr_lane = __shfl_sync(warp_mask, max_lane_lane0, 0);

                    // Broadcast the local position of the max within the owner lane to all lanes
                    thr_local_pos = __shfl_sync(warp_mask, local_max_pos, thr_lane);
                }

                // Remove the processed leader lane from the acceptance mask
                accept_mask &= (accept_mask - 1u);

                // Optionally tighten: Early filter with updated threshold for remaining candidates
                // (Re-evaluate whether remaining candidates still pass after threshold update)
                unsigned still_ok = __ballot_sync(warp_mask, (valid && (cand_dist < thr_val)));
                accept_mask &= still_ok;
            }
        }

        // Ensure all warps are done reading the tile before loading the next one
        __syncthreads();
    }

    // After processing all tiles, the per-warp top-k is complete but unsorted.
    // Sort in ascending order within the warp's segment using a warp-synchronous bitonic sort.
    // First ensure local maxima values correspond to current top-k; not needed for sorting.
    __syncwarp();

    warp_bitonic_sort_shared(topk_dists, topk_index, warp_base, k, L, lane_id);

    // Write out the sorted results to global memory in row-major order per query.
    // Each lane writes its L entries.
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (i >= L) break;
        int local_pos = lane_id * L + i;       // position within [0, k)
        int global_pos = warp_base + local_pos;
        int out_pos = global_warp_id * k + local_pos;

        result[out_pos].first  = topk_index[global_pos];
        result[out_pos].second = topk_dists[global_pos];
    }
}

// Host interface: run_knn
// - query: pointer to device array of float2 (size query_count)
// - data:  pointer to device array of float2 (size data_count)
// - result: pointer to device array of std::pair<int,float> (size query_count * k)
// - k: power of two between 32 and 1024 inclusive
/// @FIXED
/// extern "C"

void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Basic parameter checks (assumed valid as per problem statement, left as debug guards)
    if (query_count <= 0 || data_count <= 0 || k < 32) return;

    // Determine dynamic shared memory requirements.
    // We allocate:
    // - tile_capacity_points * sizeof(float2)
    // - plus KNN_WARPS_PER_BLOCK * k * (sizeof(float) + sizeof(int)) for per-warp top-k arrays
    // Choose an initial tile capacity; adjust down if exceeding device limits.
    int tile_capacity_points = 4096; // initial choice; multiple of 32 for perfect coalescing
    if (tile_capacity_points < WARP_SIZE) tile_capacity_points = WARP_SIZE;
    // Query device's maximum opt-in shared memory per block
    int device_id = 0;
    cudaGetDevice(&device_id);
    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);

    size_t topk_bytes = static_cast<size_t>(KNN_WARPS_PER_BLOCK) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));
    // Ensure at least enough shared memory for top-k arrays; tile size may need to be reduced accordingly.
    size_t smem_needed = topk_bytes + static_cast<size_t>(tile_capacity_points) * sizeof(float2);

    // If needed, reduce tile size to fit the opt-in limit.
    if (max_smem_optin > 0 && smem_needed > static_cast<size_t>(max_smem_optin)) {
        size_t remaining = static_cast<size_t>(max_smem_optin) - topk_bytes;
        if (static_cast<long long>(remaining) < static_cast<long long>(sizeof(float2))) {
            // Still try to run with minimal tile size of 32; kernel will run but shared memory may be tight.
            tile_capacity_points = WARP_SIZE;
            smem_needed = topk_bytes + static_cast<size_t>(tile_capacity_points) * sizeof(float2);
        } else {
            tile_capacity_points = static_cast<int>(remaining / sizeof(float2));
            // round down to a multiple of warp size for coalescing
            tile_capacity_points = (tile_capacity_points / WARP_SIZE) * WARP_SIZE;
            if (tile_capacity_points < WARP_SIZE) tile_capacity_points = WARP_SIZE;
            smem_needed = topk_bytes + static_cast<size_t>(tile_capacity_points) * sizeof(float2);
        }
    }

    // Set the kernel's dynamic shared memory size attribute (opt-in for >48KB per block on modern GPUs)
    // If this fails, we proceed; the kernel launch may fail if smem exceeds allowed size.
    cudaFuncSetAttribute(knn_kernel_warp32, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_needed));

    // Compute grid configuration: one warp per query
    const int threads_per_block = KNN_THREADS_PER_BLOCK;
    const int warps_per_block = KNN_WARPS_PER_BLOCK;
    int total_warps = (query_count + warps_per_block - 1) / warps_per_block;
    int blocks = total_warps;
    if (blocks <= 0) blocks = 1;

    // Launch kernel
    knn_kernel_warp32<<<blocks, threads_per_block, smem_needed>>>(
        query, query_count, data, data_count, result, k, tile_capacity_points
    );

    // Optionally synchronize or check for errors here as needed by the caller.
    // cudaDeviceSynchronize();
}
