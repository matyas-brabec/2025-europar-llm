#include <cuda_runtime.h>
#include <utility>

// Optimized k-NN for 2D points using warp-per-query, shared-memory tiling, and warp-cooperative merges.
// - Each warp (32 threads) handles one query.
// - Data points are processed in shared-memory tiles cached by the entire block.
// - Each warp maintains:
//   * An intermediate result of size k (indices and distances), stored in shared memory, sorted ascending.
//   * A candidate buffer of size k (indices and distances), stored in shared memory, unsorted.
//   * A shared counter for candidates (atomicAdd for increment).
//   * A shared max_distance = current k-th smallest distance in the intermediate result.
//   * A shared flush_flag used to coordinate warp-cooperative merges when candidate buffer is full.
// - When the candidate buffer fills, the warp cooperatively merges it with its intermediate result using an
//   in-place bitonic sort over 2*k items executed by the warp across the two arrays without allocating extra memory.
// - After processing all tiles, any remaining candidates are merged as well.
//
// Assumptions and choices:
// - k is a power of two in [32, 1024].
// - No additional device allocations are performed; only dynamic shared memory is used.
// - Threads per block = 4 warps = 128 threads.
// - Tile size = 2048 data points per block (fits comfortably into shared memory on A100/H100).
// - Max dynamic shared memory is requested via cudaFuncSetAttribute, but the kernel can run within default limits.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunables (chosen to balance shared memory usage and occupancy on A100/H100).
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int TILE_POINTS = 2048; // Shared-memory tile size (float2) per block

// Utility: lane and warp indices within a block
static __device__ __forceinline__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
static __device__ __forceinline__ int warp_id_in_block() { return threadIdx.x >> 5; }

// Accessor that represents two contiguous "virtual" segments of total length 2*k:
// positions [0, k-1] map to the "intermediate" arrays; positions [k, 2k-1] map to the "candidate" arrays.
// We implement compare-exchanges across these two physical arrays without extra memory.
struct TwoSegAccessor {
    float* inter_dist;
    int*   inter_idx;
    float* cand_dist;
    int*   cand_idx;
    int    k;

    __device__ __forceinline__ void load(int pos, float& d, int& i) const {
        if (pos < k) {
            d = inter_dist[pos];
            i = inter_idx[pos];
        } else {
            int m = pos - k;
            d = cand_dist[m];
            i = cand_idx[m];
        }
    }
    __device__ __forceinline__ void store(int pos, float d, int i) const {
        if (pos < k) {
            inter_dist[pos] = d;
            inter_idx[pos]  = i;
        } else {
            int m = pos - k;
            cand_dist[m] = d;
            cand_idx[m]  = i;
        }
    }
};

// Warp-cooperative in-place bitonic sort of 2*k items accessed via TwoSegAccessor.
// After completion, elements at virtual positions [0, 2k-1] are globally sorted ascending.
// Consequently, the first k items (positions [0, k-1]) reside in the intermediate arrays in ascending order.
static __device__ __forceinline__ void warp_bitonic_sort_2k(const TwoSegAccessor& acc, int k) {
    const int N = 2 * k;
    const int lane = lane_id();
    // Standard bitonic sort network using (size, stride) loops, executed warp-synchronously.
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += WARP_SIZE) {
                int j = i ^ stride;
                if (i < j) {
                    bool ascending = ((i & size) == 0);
                    float di, dj;
                    int ii, ij;
                    acc.load(i, di, ii);
                    acc.load(j, dj, ij);
                    // Compare-exchange
                    if (ascending) {
                        if (di > dj) {
                            acc.store(i, dj, ij);
                            acc.store(j, di, ii);
                        } else {
                            acc.store(i, di, ii);
                            acc.store(j, dj, ij);
                        }
                    } else {
                        if (di < dj) {
                            acc.store(i, dj, ij);
                            acc.store(j, di, ii);
                        } else {
                            acc.store(i, di, ii);
                            acc.store(j, dj, ij);
                        }
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge candidate buffer into the intermediate result for a single warp.
// - Pads candidate buffer with +inf distances up to size k.
// - Runs a warp-cooperative bitonic sort across the 2*k items, mapping them over the two arrays in place.
// - Writes new max_distance = inter_dist[k-1], resets candidate count to 0, clears flush flag.
static __device__ __forceinline__ void warp_merge_candidates(
    float* inter_dist, int* inter_idx,
    float* cand_dist,  int* cand_idx,
    int* cand_count,   float* max_distance,
    volatile int* flush_flag,
    int k)
{
    const int lane = lane_id();
    // Snapshot current candidate count (<= k due to guarded insertion).
    int count = *cand_count;
    // Pad remaining candidate slots with +inf so they never enter top-k.
    for (int t = lane + count; t < k; t += WARP_SIZE) {
        cand_dist[t] = CUDART_INF_F;
        cand_idx[t]  = -1;
    }
    __syncwarp();

    TwoSegAccessor acc{ inter_dist, inter_idx, cand_dist, cand_idx, k };
    warp_bitonic_sort_2k(acc, k);

    __syncwarp();
    if (lane == 0) {
        *max_distance = inter_dist[k - 1];
        *cand_count   = 0;
        *flush_flag   = 0;
    }
    __syncwarp();
}

// Attempt to insert a candidate (idx, dist) into the warp's shared candidate buffer.
// - Uses atomicAdd on cand_count to get a position; if full, reverts with atomicSub and triggers a cooperative merge.
// - Joins any in-progress merges initiated by other threads (via flush_flag).
// - Returns only once the candidate has been inserted successfully.
static __device__ __forceinline__ void warp_push_candidate_or_merge(
    int idx, float dist,
    float* inter_dist, int* inter_idx,
    float* cand_dist,  int* cand_idx,
    int* cand_count,   float* max_distance,
    volatile int* flush_flag,
    int k)
{
    // Busy-wait loop: attempt insertion; if full, trigger and join merge, then retry.
    while (true) {
        // If a merge is already requested/ongoing by another lane, join it before inserting.
        if (*flush_flag) {
            warp_merge_candidates(inter_dist, inter_idx, cand_dist, cand_idx, cand_count, max_distance, flush_flag, k);
        }
        int pos = atomicAdd(cand_count, 1);
        if (pos < k) {
            // We got a slot.
            cand_dist[pos] = dist;
            cand_idx[pos]  = idx;
            return;
        } else {
            // Buffer was full; revert increment and request a merge.
            atomicSub(cand_count, 1);
            atomicExch((int*)flush_flag, 1);
            // Cooperatively merge and then retry insertion.
            warp_merge_candidates(inter_dist, inter_idx, cand_dist, cand_idx, cand_count, max_distance, flush_flag, k);
        }
    }
}

// Kernel: Each warp computes k-NN for one query. Data points are processed in shared-memory tiles.
// Dynamic shared memory layout per block:
// - First: TILE_POINTS of float2 for the data tile.
// - Then, for each warp (WARPS_PER_BLOCK of them), the per-warp region containing:
//     * inter_dist[k], inter_idx[k], cand_dist[k], cand_idx[k], cand_count, max_distance, flush_flag
__global__ void knn_kernel_2d_warpperquery(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    std::pair<int, float>* __restrict__ result,
    int k)
{
    extern __shared__ unsigned char shared[];
    const int lane = lane_id();
    const int warp_in_block = warp_id_in_block();
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (warp_global >= query_count) return;

    // Shared memory partitioning
    // Block-wide data tile
    float2* smem_tile = reinterpret_cast<float2*>(shared);
    size_t tile_bytes = sizeof(float2) * TILE_POINTS;
    // Align next region to 8 bytes
    size_t offset = (tile_bytes + 7) & ~size_t(7);

    // Per-warp region sizes
    const size_t dist_bytes = sizeof(float) * k;
    const size_t idx_bytes  = sizeof(int)   * k;
    const size_t warp_region_bytes_unaligned = dist_bytes + idx_bytes + dist_bytes + idx_bytes + sizeof(int) + sizeof(float) + sizeof(int);
    const size_t warp_region_bytes = (warp_region_bytes_unaligned + 7) & ~size_t(7);

    unsigned char* warp_region_base = shared + offset + warp_in_block * warp_region_bytes;

    // Pointers into per-warp region
    float* inter_dist = reinterpret_cast<float*>(warp_region_base);
    int*   inter_idx  = reinterpret_cast<int*>(warp_region_base + dist_bytes);
    float* cand_dist  = reinterpret_cast<float*>(warp_region_base + dist_bytes + idx_bytes);
    int*   cand_idx   = reinterpret_cast<int*>(warp_region_base + dist_bytes + idx_bytes + dist_bytes);
    int*   cand_count = reinterpret_cast<int*>(warp_region_base + dist_bytes + idx_bytes + dist_bytes + idx_bytes);
    float* max_distance = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(cand_count) + sizeof(int));
    volatile int* flush_flag = reinterpret_cast<volatile int*>(reinterpret_cast<unsigned char*>(max_distance) + sizeof(float));

    // Initialize per-warp structures
    for (int i = lane; i < k; i += WARP_SIZE) {
        inter_dist[i] = CUDART_INF_F;
        inter_idx[i]  = -1;
    }
    if (lane == 0) {
        *cand_count   = 0;
        *max_distance = CUDART_INF_F;
        *flush_flag   = 0;
    }
    __syncwarp();

    // Load query point into registers
    const float2 q = query[warp_global];

    // Iterate over data in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        int tile_count = data_count - tile_start;
        if (tile_count > TILE_POINTS) tile_count = TILE_POINTS;

        // Cooperative load of tile by entire block
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            smem_tile[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes this tile
        for (int i = lane; i < tile_count; i += WARP_SIZE) {
            // Join any pending merge requested by other lanes in this warp.
            if (*flush_flag) {
                warp_merge_candidates(inter_dist, inter_idx, cand_dist, cand_idx, cand_count, max_distance, flush_flag, k);
            }

            float2 p = smem_tile[i];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dist = dx * dx + dy * dy;

            // Read current max_distance; filtering distances no larger than k-th best so far.
            float maxd = *max_distance;
            if (dist < maxd) {
                // Push candidate; may trigger merge if buffer is full.
                warp_push_candidate_or_merge(tile_start + i, dist,
                                             inter_dist, inter_idx,
                                             cand_dist,  cand_idx,
                                             cand_count, max_distance,
                                             flush_flag, k);
            }
        }
        __syncwarp();   // Ensure warp finished using this tile before block advances tile
        __syncthreads();
    }

    // After all tiles, if there are remaining candidates, merge them.
    if (lane == 0 && *cand_count > 0) {
        atomicExch((int*)flush_flag, 1);
    }
    if (*flush_flag) {
        warp_merge_candidates(inter_dist, inter_idx, cand_dist, cand_idx, cand_count, max_distance, flush_flag, k);
    }

    // Write final results for this query to global memory; inter_dist/inter_idx contain top-k sorted ascending.
    // Each lane writes a strided subset to maximize coalescing across queries.
    const int out_base = warp_global * k;
    for (int j = lane; j < k; j += WARP_SIZE) {
        result[out_base + j].first  = inter_idx[j];
        result[out_base + j].second = inter_dist[j];
    }
}

// Host entry point. Chooses launch configuration and dynamic shared memory size, then launches the kernel.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Grid/block configuration
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Compute dynamic shared memory size required:
    // - TILE_POINTS * sizeof(float2) for the block-wide tile
    // - Per warp: inter_dist[k], inter_idx[k], cand_dist[k], cand_idx[k], cand_count, max_distance, flush_flag
    size_t tile_bytes = sizeof(float2) * TILE_POINTS;
    size_t dist_bytes = sizeof(float) * k;
    size_t idx_bytes  = sizeof(int)   * k;
    size_t warp_region_bytes_unaligned = dist_bytes + idx_bytes + dist_bytes + idx_bytes + sizeof(int) + sizeof(float) + sizeof(int);
    size_t warp_region_bytes = (warp_region_bytes_unaligned + 7) & ~size_t(7);
    size_t dynamic_smem_bytes = ((tile_bytes + 7) & ~size_t(7)) + WARPS_PER_BLOCK * warp_region_bytes;

    // Opt-in to larger dynamic shared memory if available (A100/H100 allow up to ~164KB per block).
    cudaFuncSetAttribute(knn_kernel_2d_warpperquery, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dynamic_smem_bytes);
    cudaFuncSetAttribute(knn_kernel_2d_warpperquery, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    // Launch kernel
    knn_kernel_2d_warpperquery<<<grid, block, dynamic_smem_bytes>>>(query, query_count, data, data_count, result, k);

    // Optionally: synchronize or check for launch errors (omitted by design per instruction constraints).
}