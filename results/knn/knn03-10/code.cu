#include <cuda_runtime.h>
#include <utility>

// Optimized k-NN for 2D points using warp-per-query, shared-memory tiling, and warp-synchronous updates.
// - Each warp processes one query point and maintains a private top-k buffer in shared memory.
// - The data points are processed in tiles cached in shared memory by the entire block.
// - Within each tile, each warp computes distances for its query and updates its top-k set by replacing the current maximum when needed.
// - The top-k buffer is stored unsorted during accumulation; after the full scan, it is sorted in ascending order via a bitonic sort and written to the output.
//
// Assumptions:
// - k is a power of two between 32 and 1024.
// - data_count >= k.
// - query_count is typically large (thousands), data_count is very large (millions).
// - Memory for query, data, and result is allocated by cudaMalloc.
// - No dynamic device allocations; only shared memory is used within the kernel.

#ifndef KNNCUDA_UTILS
#define KNNCUDA_UTILS

// Tunables selected to balance shared memory usage and occupancy.
// TILE_POINTS * sizeof(float2) = 4096 * 8 = 32 KB
// Per-warp top-k storage (distance+index): k * 8 bytes. For k=1024 -> 8 KB per warp.
// With WARPS_PER_BLOCK=8 -> 64 KB for top-k buffers.
// Total shared memory per block = 32 KB + 64 KB = 96 KB (fits typical 96 KB per-block budget).
static constexpr int TILE_POINTS = 4096;
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

static __device__ __forceinline__ float sqr(const float x) { return x * x; }

// Recompute the maximum value and its index across an array arr[0..k-1] cooperatively within a warp.
// Each lane scans a strided subset, then a warp-wide reduction selects the maximum and its index.
// The resulting max_val and max_pos are broadcast to all lanes via shuffles.
static __device__ __forceinline__
void warp_recompute_max(const float* __restrict__ arr, int k, int lane, float &max_val, int &max_pos)
{
    float local_max = -CUDART_INF_F;
    int local_idx = -1;

    // Strided scan across k entries: each lane covers indices i = lane, lane+32, ...
    for (int i = lane; i < k; i += 32) {
        float v = arr[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp-wide reduction to find the global maximum and its position.
    // Ties are broken by favoring the larger index for determinism (optional).
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(mask, local_max, offset);
        int other_idx = __shfl_down_sync(mask, local_idx, offset);
        if (other_val > local_max || (other_val == local_max && other_idx > local_idx)) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }
    // Broadcast the final max value and index from lane 0 to all lanes in the warp.
    max_val = __shfl_sync(mask, local_max, 0);
    max_pos = __shfl_sync(mask, local_idx, 0);
}

// Warp-level argmin among current candidates; ineligible lanes set their value to +INF.
// Returns the minimum value and the lane that holds it (broadcast to all lanes).
static __device__ __forceinline__
void warp_argmin(float my_val, bool eligible, float &min_val, int &min_lane)
{
    unsigned mask = 0xffffffffu;
    float v = eligible ? my_val : CUDART_INF_F;
    int lane = threadIdx.x & 31;
    int idx = lane;

    // Warp-wide reduction for minimum.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(mask, v, offset);
        int other_idx = __shfl_down_sync(mask, idx, offset);
        if (other_val < v) {
            v = other_val;
            idx = other_idx;
        }
    }
    min_val = __shfl_sync(mask, v, 0);
    min_lane = __shfl_sync(mask, idx, 0);
}

// Bitonic sort of k elements (power-of-two) in ascending order across the warp, operating on shared memory arrays.
// Each lane processes elements in a strided fashion; __syncwarp is used to ensure ordering between stages.
static __device__ __forceinline__
void warp_bitonic_sort(float* __restrict__ dist, int* __restrict__ idx, int k)
{
    unsigned mask = 0xffffffffu;
    int lane = threadIdx.x & 31;

    // Bitonic sort network: k must be a power of two.
    for (int size = 2; size <= k; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < k; i += 32) {
                int partner = i ^ stride;
                if (partner > i) { // ensure each pair is handled once
                    bool ascending = ((i & size) == 0);
                    float a = dist[i];
                    float b = dist[partner];
                    int ai = idx[i];
                    int bi = idx[partner];

                    bool do_swap = (a > b) == ascending;
                    if (do_swap) {
                        dist[i] = b; dist[partner] = a;
                        idx[i]  = bi; idx[partner]  = ai;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

#endif // KNNCUDA_UTILS

// CUDA kernel implementing k-NN for 2D points.
__global__ void knn2d_kernel(const float2* __restrict__ query,
                              int query_count,
                              const float2* __restrict__ data,
                              int data_count,
                              std::pair<int, float>* __restrict__ result,
                              int k)
{
    // Warp identification
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5; // 0..WARPS_PER_BLOCK-1
    const int warps_per_block = blockDim.x >> 5; // should be WARPS_PER_BLOCK
    const int warp_global = blockIdx.x * warps_per_block + warp_in_block;

    // Per-warp activity flag. Even inactive warps must participate in __syncthreads for safe shared-memory tiling.
    const bool warp_active = (warp_global < query_count);

    // Dynamic shared memory layout:
    // [ float2 tile[TILE_POINTS] ][ float dist[WARPS_PER_BLOCK * k] ][ int idx[WARPS_PER_BLOCK * k] ]
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_ptr = smem_raw;

    // Shared tile for data points
    float2* s_tile = reinterpret_cast<float2*>(smem_ptr);
    size_t tile_bytes = static_cast<size_t>(TILE_POINTS) * sizeof(float2);
    smem_ptr += tile_bytes;

    // Align next pointers to 16 bytes for safety
    size_t misalign = reinterpret_cast<size_t>(smem_ptr) & 0xF;
    if (misalign) smem_ptr += (16 - misalign);

    float* s_topk_dist = reinterpret_cast<float*>(smem_ptr);
    size_t dist_bytes = static_cast<size_t>(warps_per_block) * k * sizeof(float);
    smem_ptr += dist_bytes;

    misalign = reinterpret_cast<size_t>(smem_ptr) & 0xF;
    if (misalign) smem_ptr += (16 - misalign);

    int* s_topk_idx = reinterpret_cast<int*>(smem_ptr);
    // size_t idx_bytes = static_cast<size_t>(warps_per_block) * k * sizeof(int); // not needed further

    // Pointers to this warp's top-k buffers within shared memory
    float* topk_dist = s_topk_dist + warp_in_block * k;
    int*   topk_idx  = s_topk_idx + warp_in_block * k;

    // Load this warp's query point into registers (broadcast from lane 0).
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        if (lane == 0) {
            float2 q = query[warp_global];
            qx = q.x;
            qy = q.y;
        }
        unsigned mask = 0xffffffffu;
        qx = __shfl_sync(mask, qx, 0);
        qy = __shfl_sync(mask, qy, 0);
    }

    // State for maintaining the top-k buffer.
    int filled = 0;             // number of initial elements filled
    float cur_max_val = CUDART_INF_F;
    int cur_max_pos = -1;

    // Process data in tiles
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS) {
        int tile_count = data_count - tile_base;
        if (tile_count > TILE_POINTS) tile_count = TILE_POINTS;

        // Cooperative load of the tile into shared memory by the whole block.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_tile[i] = data[tile_base + i];
        }
        __syncthreads();

        if (warp_active) {
            // Stage 1: Fill initial K items into the buffer without selective replacement.
            int remaining_to_fill = k - filled;
            int fill_now = (remaining_to_fill > 0) ? ((remaining_to_fill < tile_count) ? remaining_to_fill : tile_count) : 0;

            // Each lane fills strided positions of the initial block [filled, filled + fill_now)
            for (int i = lane; i < fill_now; i += 32) {
                float2 p = s_tile[i];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float d = fmaf(dy, dy, dx * dx);
                topk_dist[filled + i] = d;
                topk_idx [filled + i] = tile_base + i;
            }
            // Ensure the fills are visible to other lanes in this warp before recomputing max.
            __syncwarp();

            filled += fill_now;

            // If we've just completed the initial fill, compute the current max and its position.
            if (filled == k && cur_max_pos < 0) {
                warp_recompute_max(topk_dist, k, lane, cur_max_val, cur_max_pos);
            }

            // Stage 2: For the remainder of this tile, perform selective replacements.
            int start = fill_now; // start index in this tile after initial fill
            // Iterate over the tile in chunks of 32 so each lane considers one candidate at a time.
            for (int base = start; base < tile_count; base += 32) {
                int pos = base + lane;
                bool valid = (pos < tile_count);
                float my_dist = CUDART_INF_F;
                int my_idx = -1;

                if (valid) {
                    float2 p = s_tile[pos];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    my_dist = fmaf(dy, dy, dx * dx);
                    my_idx = tile_base + pos;
                }

                // If initial fill not complete yet (possible if k > tile_count and more tiles needed),
                // keep filling buffer from these candidates instead of selective replacement.
                if (filled < k) {
                    // We need to place up to (k - filled) items from this 'base' chunk.
                    int remaining = k - filled;
                    unsigned mask = 0xffffffffu;

                    // We will select up to 'remaining' best among the current 32 candidates (or fewer if many invalid).
                    // Do so by repeatedly picking the minimum and writing it into the next slot.
                    for (int take = 0; take < 32 && remaining > 0; ++take) {
                        // Compute the minimum among currently valid candidates (my_dist) and consume it.
                        float best_val;
                        int best_lane;
                        warp_argmin(my_dist, valid, best_val, best_lane);

                        // If no valid candidates remain in this chunk, break.
                        if (!isfinite(best_val)) break;

                        // Extract the winning candidate's index.
                        int best_idx = __shfl_sync(mask, my_idx, best_lane);

                        // Store into the next slot in top-k buffer.
                        if (lane == 0) {
                            topk_dist[filled] = best_val;
                            topk_idx [filled] = best_idx;
                        }
                        __syncwarp(mask);
                        filled++;
                        remaining--;

                        // Mark the winning lane's candidate as consumed so it won't be picked again.
                        if (lane == best_lane) {
                            valid = false;
                            my_dist = CUDART_INF_F;
                        }

                        if (filled == k) {
                            // Compute initial maximum once the buffer is fully filled.
                            warp_recompute_max(topk_dist, k, lane, cur_max_val, cur_max_pos);
                            break; // proceed to selective replacement for remaining candidates (if any)
                        }
                    }
                    // If still not filled, continue to next 32-candidate block; selective replacement not yet active.
                    if (filled < k) {
                        continue;
                    }
                    // Else, fall through to selective replacement for any leftover candidates in this block.
                }

                // Selective replacement: repeatedly replace the current maximum with the smallest candidate
                // that is less than the current maximum, until no candidate in this set qualifies.
                if (cur_max_pos >= 0) {
                    unsigned mask = 0xffffffffu;
                    while (true) {
                        bool eligible = valid && (my_dist < cur_max_val);
                        unsigned any = __ballot_sync(mask, eligible);
                        if (any == 0u) break; // no qualifying candidates remain in this 32-candidate set

                        // Among eligible candidates, pick the minimum distance to replace the current maximum.
                        float best_val;
                        int best_lane;
                        warp_argmin(my_dist, eligible, best_val, best_lane);

                        // Retrieve index of the winning lane's candidate.
                        int best_idx = __shfl_sync(mask, my_idx, best_lane);

                        // Replace current maximum in the top-k buffer with the new candidate.
                        if (lane == 0) {
                            topk_dist[cur_max_pos] = best_val;
                            topk_idx [cur_max_pos] = best_idx;
                        }
                        __syncwarp(mask);

                        // Recompute the new maximum across the top-k buffer.
                        warp_recompute_max(topk_dist, k, lane, cur_max_val, cur_max_pos);

                        // Mark the winning lane as consumed for this block (so it won't be reconsidered here).
                        if (lane == best_lane) {
                            valid = false;
                        }
                    }
                }
            } // end per-32-candidate chunk within tile
        } // end if warp_active

        __syncthreads();
    } // end tiles loop

    // After processing all tiles, for active warps, sort the warp's top-k buffer in ascending order and write results.
    if (warp_active) {
        // Sort ascending by distance
        warp_bitonic_sort(topk_dist, topk_idx, k);

        // Write sorted results to global memory
        for (int i = lane; i < k; i += 32) {
            int out_pos = warp_global * k + i;
            result[out_pos].first  = topk_idx[i];
            result[out_pos].second = topk_dist[i];
        }
    }
}

// Host interface: launches the kernel.
// query: device pointer to float2 queries, count q
// data:  device pointer to float2 data points, count n
// result: device pointer to std::pair<int,float> of size q*k
/// @FIXED
/// extern "C"

void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Launch configuration
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Compute dynamic shared memory size:
    // tile + per-warp top-k distances + per-warp top-k indices (with alignment padding)
    size_t smem = 0;
    smem += static_cast<size_t>(TILE_POINTS) * sizeof(float2); // tile
    // Align to 16 bytes
    smem = (smem + 15) & ~static_cast<size_t>(15);
    smem += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float); // distances
    // Align to 16 bytes
    smem = (smem + 15) & ~static_cast<size_t>(15);
    smem += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int);   // indices

    // Launch kernel
    knn2d_kernel<<<grid, block, smem>>>(query, query_count, data, data_count, result, k);
}
