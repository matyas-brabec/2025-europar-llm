#include <cuda_runtime.h>
#include <utility>
#include <cfloat>
#include <cstdio>

/*
Optimized k-NN (2D, squared L2) with one warp (32 threads) per query.

Key design:
- Each warp processes one query end-to-end.
- The block loads the dataset in tiles into shared memory; all warps in the block reuse it.
- Each lane maintains a small local top-L list per tile (L = k/32). After each tile, all lanes pack
  only the promising candidates (those that improve the warp’s current top-k threshold) into a per-warp
  shared buffer of size k. The union of previous top-k and new candidates (<= 2k elements) is then
  sorted via an in-warp bitonic sort network to retain the new top-k.
- Per-warp intermediate results (indices and distances) are kept in shared memory throughout and
  are written to global memory at the end.

Implementation notes:
- k is assumed to be a power-of-two between 32 and 1024 inclusive.
- We allocate per-warp candidate buffers of size 2k (distances + indices) in shared memory to support merging.
- Shared memory layout per block:
    [tile of float2 data points] + [warps_per_block * 2k floats] + [warps_per_block * 2k ints]
- We choose a default of 256 threads/block (8 warps). If shared memory is tight for a given k, we reduce
  warps per block to fit.
- Warp-scope synchronization uses __syncwarp(mask) with a mask that includes only lanes belonging to
  warps assigned a valid query. Block-wide synchronization (__syncthreads) wraps tile loads.

This kernel targets modern data center GPUs (A100/H100) and uses dynamic shared memory sized at launch.
*/

// Constants
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Device utility: Warp inclusive sum for int with mask.
// Returns the sum across lanes in 'mask'; each participating lane gets the same final sum if desired via shuffles.
static __device__ __forceinline__ int warp_inclusive_sum(int v, unsigned mask) {
    // Butterfly reduction using shfl with mask
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int n = __shfl_up_sync(mask, v, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
    }
    return v;
}

// Device utility: Warp exclusive prefix sum for int with mask.
// Returns the exclusive prefix sum (sum of values from lanes lower than the current lane within 'mask').
static __device__ __forceinline__ int warp_exclusive_prefix_sum(int v, unsigned mask) {
    int inclusive = warp_inclusive_sum(v, mask);
    return inclusive - v;
}

// Device utility: Warp-wide sum reduction (returns total sum to all lanes in mask)
static __device__ __forceinline__ int warp_sum_all(int v, unsigned mask) {
    // Convert inclusive scan at last active lane to broadcast: reduce with shfl_down
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        int n = __shfl_down_sync(mask, v, offset);
        v += n;
    }
    // After this, the value is valid only in lane 0 of the active mask; broadcast it
    int total = __shfl_sync(mask, v, 0);
    return total;
}

// In-warp bitonic sort of N pairs (dist, idx) stored in shared memory.
// N must be a power of two. The sort order is ascending by 'dist'.
// All lanes in 'mask' must call this with the same arguments. Lanes not in mask may call it too (they will not wait).
static __device__ __forceinline__ void warp_bitonic_sort_pairs(float* dist, int* idx, int N, unsigned mask) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    // Standard bitonic sort network using XOR based indexing.
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each lane processes multiple indices strided by warp size.
            for (int i = lane; i < N; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    float di = dist[i];
                    float dj = dist[ixj];
                    int   ii = idx[i];
                    int   ij = idx[ixj];
                    // If ascending, swap if di > dj; if descending, swap if di < dj
                    bool do_swap = ascending ? (di > dj) : (di < dj);
                    if (do_swap) {
                        dist[i]   = dj; dist[ixj] = di;
                        idx[i]    = ij; idx[ixj]  = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

static __device__ __forceinline__ float sqr_distance2(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Kernel implementing the warp-per-query k-NN with tiled shared-memory caching.
__global__ void knn_kernel_warp_per_query(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    std::pair<int, float>* __restrict__ result,
    int k, int tile_points)
{
    extern __shared__ unsigned char smem_raw[];
    // Shared memory layout:
    // [tile float2 array] [warps_per_block*2k floats] [warps_per_block*2k ints]
    float2* shTile = reinterpret_cast<float2*>(smem_raw);

    // Compute warps per block and per-warp offsets
    const int warps_per_block = blockDim.x / WARP_SIZE;
    // Offsets to per-warp candidate arrays start after tile
    unsigned char* after_tile = smem_raw + static_cast<size_t>(tile_points) * sizeof(float2);

    float* shWarpDists = reinterpret_cast<float*>(after_tile);
    int*   shWarpIdx   = reinterpret_cast<int*>(shWarpDists + static_cast<size_t>(warps_per_block) * (2 * k));

    const int warp_id       = threadIdx.x / WARP_SIZE;
    const int lane          = threadIdx.x & (WARP_SIZE - 1);
    const int warp_global   = blockIdx.x * warps_per_block + warp_id;
    const bool has_query    = (warp_global < query_count);
    // Mask of all lanes belonging to warps with valid queries in this block
    unsigned full_mask = __activemask();
    // Build a mask containing only lanes in the current warp that have a query
    unsigned query_mask = __ballot_sync(full_mask, has_query);

    // Per-warp base pointers for the 2k candidate arrays in shared memory
    float* wdist = shWarpDists + static_cast<size_t>(warp_id) * (2 * k);
    int*   widx  = shWarpIdx   + static_cast<size_t>(warp_id) * (2 * k);

    // Each warp loads its query point, broadcast to all lanes in the warp
    float2 q = make_float2(0.f, 0.f);
    if (lane == 0 && has_query) {
        q = query[warp_global];
    }
    // Broadcast q from lane 0; lanes not in query_mask will get undefined, but they won't use it
    q.x = __shfl_sync(query_mask, q.x, 0);
    q.y = __shfl_sync(query_mask, q.y, 0);

    // Initialize per-warp top-k arrays to +inf and invalid index (-1).
    // Only warps with queries need to initialize their regions.
    if (has_query) {
        const float INF = FLT_MAX;
        // Initialize first k
        for (int i = lane; i < k; i += WARP_SIZE) {
            wdist[i] = INF;
            widx[i]  = -1;
        }
        // Initialize second half (k..2k-1) to INF/-1 once (will be overwritten per tile as needed, but zero-cost safety)
        for (int i = lane; i < k; i += WARP_SIZE) {
            wdist[k + i] = INF;
            widx[k + i]  = -1;
        }
    }
    __syncwarp(query_mask);

    // Number of local slots per lane
    const int L = k / WARP_SIZE;

    // Iterate over the dataset in tiles cached in shared memory
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        const int count = min(tile_points, data_count - tile_start);

        // Block-cooperative load of the current tile into shared memory
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            shTile[i] = data[tile_start + i];
        }
        __syncthreads();

        if (has_query) {
            // Local per-lane top-L lists (kept unsorted). We keep only entries with distance < warpWorst.
            float localDist[32]; // MAX_L = 1024/32 = 32
            int   localIdx[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                if (i < L) { localDist[i] = FLT_MAX; localIdx[i] = -1; }
            }
            float localWorst = FLT_MAX;
            int   localWorstPos = 0;

            // Read current warp threshold (worst top-k distance). The top-k is kept sorted after each merge, so worst is at k-1.
            float threshold = wdist[k - 1];

            // Each lane processes a strided subset of points in the tile
            for (int t = lane; t < count; t += WARP_SIZE) {
                float d = sqr_distance2(q, shTile[t]);
                if (d >= threshold) continue; // Prune if not improving current top-k
                // Accept into local top-L if better than current local worst
                if (d < localWorst) {
                    localDist[localWorstPos] = d;
                    localIdx[localWorstPos]  = tile_start + t;
                    // Recompute local worst among L slots (only when a replacement happens)
                    float wv = -1.0f;
                    int   wp = 0;
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        if (i < L) {
                            float vi = localDist[i];
                            if (vi > wv) { wv = vi; wp = i; }
                        }
                    }
                    localWorst = wv;
                    localWorstPos = wp;
                }
            }

            // Pack only the local entries that beat the (possibly old) threshold
            int packCount = 0;
            // First, count how many valid improvements we have to compute prefix sum positions
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                if (i < L) {
                    if (localIdx[i] >= 0 && localDist[i] < threshold) {
                        ++packCount;
                    }
                }
            }
            // Compute exclusive prefix sum to place packed items contiguously in [k .. k + totalNew)
            int myOffset = warp_exclusive_prefix_sum(packCount, query_mask);
            int totalNew = warp_sum_all(packCount, query_mask);

            // Pre-fill the second half with INF/-1 (to ensure unused entries are clean)
            for (int i = lane; i < k; i += WARP_SIZE) {
                wdist[k + i] = FLT_MAX;
                widx[k + i]  = -1;
            }
            __syncwarp(query_mask);

            // Write our packed items into shared memory at positions [k + myOffset, ...)
            int written = 0;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                if (i < L) {
                    if (localIdx[i] >= 0 && localDist[i] < threshold) {
                        int pos = k + myOffset + written;
                        if (pos < 2 * k) { // safety
                            wdist[pos] = localDist[i];
                            widx[pos]  = localIdx[i];
                        }
                        ++written;
                    }
                }
            }
            __syncwarp(query_mask);

            // If we have any improvements, sort the union of previous top-k and new candidates (size 2k), keep the best k
            if (totalNew > 0) {
                warp_bitonic_sort_pairs(wdist, widx, 2 * k, query_mask);
                __syncwarp(query_mask);
                // After sorting ascending, wdist[0..k-1] holds the current best k in order; wdist[k-1] is the new threshold.
            }
        }

        __syncthreads(); // Ensure all warps are done before loading the next tile
    }

    // Write out the final top-k for each query
    if (has_query) {
        for (int i = lane; i < k; i += WARP_SIZE) {
            int out_idx = warp_global * k + i;
            // Assign to std::pair members directly
            result[out_idx].first  = widx[i];
            result[out_idx].second = wdist[i];
        }
    }
}

// Host-side launcher matching the required interface.
// Chooses block/grid sizes and dynamic shared memory layout to balance per-warp candidate storage (2k entries)
// and the shared-memory tile of data points.
// Uses up to the device's opt-in maximum shared memory per block to maximize tile size.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Default: 256 threads per block (8 warps). Adjust downward if shared memory is tight.
    int desired_threads_per_block = 256;
    int desired_warps_per_block   = desired_threads_per_block / WARP_SIZE;

    // Query device attributes for dynamic shared memory limits
    int device = 0;
    cudaGetDevice(&device);
    int max_smem_optin = 0;
    int max_smem_default = 0;
    cudaDeviceGetAttribute(&max_smem_optin,   cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaDeviceGetAttribute(&max_smem_default, cudaDevAttrMaxSharedMemoryPerBlock, device);
    // Use the larger of default and opt-in as a target budget
    int max_dynamic_smem = max(max_smem_optin, max_smem_default);
    if (max_dynamic_smem <= 0) {
        // Sensible fallback for very old devices (should not happen on A100/H100)
        max_dynamic_smem = 96 * 1024;
    }

    // Per-warp candidate storage (2k pairs: float + int)
    size_t bytes_per_warp_candidates = static_cast<size_t>(2) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    // Find the maximum number of warps per block that fits into shared memory (keeping at least a small tile)
    // We'll require at least 1 point in the tile; tile needs sizeof(float2) bytes.
    int max_warps_by_smem = max(1, (max_dynamic_smem - static_cast<int>(sizeof(float2))) / static_cast<int>(bytes_per_warp_candidates));
    int warps_per_block = min(desired_warps_per_block, max_warps_by_smem);
    warps_per_block = max(1, warps_per_block); // ensure at least one warp

    int threads_per_block = warps_per_block * WARP_SIZE;

    // With chosen warps_per_block, compute remaining shared mem for tile
    size_t cand_bytes_per_block = static_cast<size_t>(warps_per_block) * bytes_per_warp_candidates;
    size_t remaining_for_tile = (cand_bytes_per_block < static_cast<size_t>(max_dynamic_smem))
                                ? (static_cast<size_t>(max_dynamic_smem) - cand_bytes_per_block)
                                : 0;

    // Tile size in points (float2)
    int tile_points = 0;
    if (remaining_for_tile >= sizeof(float2)) {
        tile_points = static_cast<int>(remaining_for_tile / sizeof(float2));
    } else {
        tile_points = 1; // minimal fallback; performance will be reduced but correctness preserved
    }

    // Limit tile_points to a reasonable upper bound to avoid extremely long per-tile loops;
    // A large tile is beneficial but does not need to exceed the dataset size.
    if (tile_points > data_count) tile_points = data_count;

    // Compute the actual dynamic shared memory size for the kernel launch
    size_t dynamic_smem_bytes = static_cast<size_t>(tile_points) * sizeof(float2) + cand_bytes_per_block;

    // Opt-in to using the required amount of dynamic shared memory per block if the device supports it
    // (On A100/H100 this enables using up to ~164KB).
    cudaFuncSetAttribute(knn_kernel_warp_per_query,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(dynamic_smem_bytes));

    // Grid size: one warp per query
    int warps_needed = (query_count + 0) / 1; // one query per warp
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch the kernel
    dim3 block(threads_per_block);
    dim3 grid(blocks);
    knn_kernel_warp_per_query<<<grid, block, dynamic_smem_bytes>>>(
        query, query_count, data, data_count, result, k, tile_points
    );
    // Optional: consider synchronizing or checking errors outside this function as needed by the caller.
}