#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <utility>

// Optimized CUDA KNN for 2D points (squared Euclidean distance).
// Design notes:
// - One warp (32 threads) processes one query point.
// - The block loads data points in tiles into shared memory.
// - Each warp maintains its own "intermediate" top-k list (private per query) in shared memory.
// - Each warp also owns a shared candidate buffer of size k and a shared integer count updated with atomicAdd.
// - When a candidate buffer becomes full, the warp merges it with its intermediate result using a warp-cooperative bitonic sort over 2k elements.
// - After the final tile, any remaining candidates are merged.
// - Merging and shared-memory usage are carefully designed to satisfy the constraints and maximize performance on modern NVIDIA GPUs.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Choose number of warps per block. 4 warps => 128 threads per block.
// This balances shared memory footprint and occupancy for k up to 1024.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

// Output pair type with the same layout as std::pair<int, float>.
struct PairIF {
    int   first;
    float second;
};

// Align up helper (host/device)
static inline __host__ __device__ size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

// Warp-cooperative bitonic sort over pairs (distance asc; tie-breaker by index asc for determinism).
// The arrays d[0..n-1], idx[0..n-1] reside in shared memory and are private to the calling warp.
// n must be a power of two (we use n = 2*k; k is guaranteed power of two).
__device__ inline void warp_bitonic_sort_pairs(float* d, int* idx, int n, unsigned warp_mask) {
    // Standard bitonic network: O(n log^2 n).
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int lane = threadIdx.x & (WARP_SIZE - 1);
            for (int i = lane; i < n; i += WARP_SIZE) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = ((i & size) == 0); // true = ascending region
                    float di = d[i];
                    float dj = d[j];
                    int   ii = idx[i];
                    int   ij = idx[j];
                    // Ascending: keep smaller at low index; Descending: keep larger at low index.
                    bool swap = (di > dj);
                    if (di == dj) { // tie-breaker (optional)
                        swap = (ii > ij);
                    }
                    if (swap == up) {
                        d[i] = dj; d[j] = di;
                        idx[i] = ij; idx[j] = ii;
                    }
                }
            }
            __syncwarp(warp_mask);
        }
    }
}

// Merge a warp's candidate buffer into its intermediate top-k using a warp-cooperative bitonic sort.
// Shared memory layout is passed via pointers; all arrays are disjoint per warp.
// Steps:
//  1. Copy current top-k (k) and current candidates (<=k) into a work buffer of size 2k,
//     padding with +inf for missing entries.
//  2. Sort the 2k pairs (ascending by distance).
//  3. Write back the first k pairs as the new top-k; update max_distance and topk_size.
//  4. Reset candidate count to 0.
__device__ inline void warp_merge_candidates(
    int warp_id,
    int k,
    float* topk_dist, int* topk_idx,
    float* cand_dist, int* cand_idx,
    volatile int* cand_count,  // updated with atomics
    float* work_dist, int* work_idx,
    float* max_dist, int* topk_size,
    unsigned warp_mask)
{
    const int topk_base = warp_id * k;
    const int cand_base = warp_id * k;
    const int work_base = warp_id * (2 * k);
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Clamp candidate count to k for copying; extra increments beyond k are ignored here.
    int ccount = cand_count[warp_id];
    if (ccount < 0) ccount = 0;
    if (ccount > k) ccount = k;

    // Copy current top-k into work[0..k-1]
    for (int i = lane; i < k; i += WARP_SIZE) {
        work_dist[work_base + i] = topk_dist[topk_base + i];
        work_idx[work_base + i]  = topk_idx[topk_base + i];
    }
    // Copy candidates into work[k..2k-1], padding with +inf
    for (int i = lane; i < k; i += WARP_SIZE) {
        if (i < ccount) {
            work_dist[work_base + k + i] = cand_dist[cand_base + i];
            work_idx[work_base + k + i]  = cand_idx[cand_base + i];
        } else {
            work_dist[work_base + k + i] = CUDART_INF_F;
            work_idx[work_base + k + i]  = -1;
        }
    }
    __syncwarp(warp_mask);

    // Sort 2k elements (ascending).
    warp_bitonic_sort_pairs(work_dist + work_base, work_idx + work_base, 2 * k, warp_mask);

    // Write back first k as new top-k.
    for (int i = lane; i < k; i += WARP_SIZE) {
        topk_dist[topk_base + i] = work_dist[work_base + i];
        topk_idx[topk_base + i]  = work_idx[work_base + i];
    }

    // Update metadata.
    if (lane == 0) {
        int new_size = topk_size[warp_id] + ccount;
        if (new_size > k) new_size = k;
        topk_size[warp_id] = new_size;
        max_dist[warp_id] = topk_dist[topk_base + (k - 1)];
        // Reset candidate count to 0 after merge.
        cand_count[warp_id] = 0;
    }
    __syncwarp(warp_mask);
}

// Kernel to compute k-NN for 2D points. One warp per query.
// Parameters:
//  - query: device pointer to float2 query points
//  - query_count: number of queries
//  - data: device pointer to float2 data points
//  - data_count: number of data points
//  - out: device pointer to output PairIF (index, distance)
//  - k: number of neighbors (power of two, between 32 and 1024 inclusive)
//  - tile_points: number of data points per shared-memory tile (chosen at launch)
__global__ void knn2d_warp_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    PairIF* __restrict__ out,
    int k,
    int tile_points)
{
    extern __shared__ unsigned char shmem[];
    size_t off = 0;

    // Shared memory layout:
    // 1) Shared tile of data points
    off = align_up(off, 8);
    float2* s_points = reinterpret_cast<float2*>(shmem + off);
    off += static_cast<size_t>(tile_points) * sizeof(float2);

    // 2) Per-warp metadata arrays
    off = align_up(off, 4);
    volatile int* s_cand_count = reinterpret_cast<volatile int*>(shmem + off);
    off += WARPS_PER_BLOCK * sizeof(int);

    off = align_up(off, 4);
    int* s_topk_size = reinterpret_cast<int*>(shmem + off);
    off += WARPS_PER_BLOCK * sizeof(int);

    off = align_up(off, 4);
    float* s_max_dist = reinterpret_cast<float*>(shmem + off);
    off += WARPS_PER_BLOCK * sizeof(float);

    // 3) Per-warp arrays: topk (k), candidates (k), and merge work buffer (2k)
    off = align_up(off, 4);
    int* s_topk_idx = reinterpret_cast<int*>(shmem + off);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int);

    off = align_up(off, 4);
    float* s_topk_dist = reinterpret_cast<float*>(shmem + off);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float);

    off = align_up(off, 4);
    int* s_cand_idx = reinterpret_cast<int*>(shmem + off);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int);

    off = align_up(off, 4);
    float* s_cand_dist = reinterpret_cast<float*>(shmem + off);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float);

    off = align_up(off, 4);
    int* s_work_idx = reinterpret_cast<int*>(shmem + off);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * 2 * k * sizeof(int);

    off = align_up(off, 4);
    float* s_work_dist = reinterpret_cast<float*>(shmem + off);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * 2 * k * sizeof(float);

    // Warp and query mapping
    const int lane     = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active = (global_warp < query_count);
    const unsigned warp_mask = __activemask();

    // Initialize per-warp metadata and arrays.
    if (lane == 0) {
        s_cand_count[warp_id] = 0;
        s_topk_size[warp_id] = 0;
        s_max_dist[warp_id] = CUDART_INF_F;
    }
    // Set intermediate top-k to +inf distances and invalid indices.
    const int topk_base = warp_id * k;
    for (int i = lane; i < k; i += WARP_SIZE) {
        s_topk_dist[topk_base + i] = CUDART_INF_F;
        s_topk_idx[topk_base + i]  = -1;
    }
    __syncwarp(warp_mask);

    // Load query point and broadcast within warp.
    float qx = 0.0f, qy = 0.0f;
    if (active) {
        if (lane == 0) {
            float2 q = query[global_warp];
            qx = q.x; qy = q.y;
        }
        qx = __shfl_sync(warp_mask, qx, 0);
        qy = __shfl_sync(warp_mask, qy, 0);
    }

    // Process data in tiles: load by the block, consume by each active warp.
    for (int base = 0; base < data_count; base += tile_points) {
        const int count = min(tile_points, data_count - base);

        // Block-cooperative load into shared memory
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            s_points[i] = data[base + i];
        }
        __syncthreads();

        if (active) {
            // Each warp iterates over the tile, one element per lane per iteration.
            for (int t = lane; t < count; t += WARP_SIZE) {
                float2 p = s_points[t];
                float dx = p.x - qx;
                float dy = p.y - qy;
                // Squared Euclidean distance
                float dist = fmaf(dx, dx, dy * dy);

                // Threshold: accept if we haven't filled k yet or dist < max_distance.
                // Read shared metadata.
                int   tk_sz  = s_topk_size[warp_id];
                float thr    = (tk_sz < k) ? CUDART_INF_F : s_max_dist[warp_id];
                bool  accept = (tk_sz < k) || (dist < thr);

                // Try to append to candidate buffer using atomicAdd to get a position.
                bool pending = false;
                int  pend_idx = 0;
                float pend_dist = 0.0f;

                if (accept) {
                    int pos = atomicAdd((int*)&s_cand_count[warp_id], 1);
                    if (pos < k) {
                        s_cand_idx[topk_base + pos]  = base + t;
                        s_cand_dist[topk_base + pos] = dist;
                    } else {
                        // Buffer overflowed; keep this candidate pending to be re-attempted after a merge.
                        pending = true;
                        pend_idx = base + t;
                        pend_dist = dist;
                    }
                }

                // If any lane overflowed or the buffer reached capacity, merge and retry pending candidates.
                while (__any_sync(warp_mask, pending) || (s_cand_count[warp_id] >= k)) {
                    // Merge current candidates into top-k.
                    warp_merge_candidates(
                        warp_id, k,
                        s_topk_dist, s_topk_idx,
                        s_cand_dist, s_cand_idx,
                        s_cand_count,
                        s_work_dist, s_work_idx,
                        s_max_dist, s_topk_size,
                        warp_mask);

                    // Try to (re-)insert pending candidate if it still qualifies against the updated threshold.
                    if (pending) {
                        int   tk_sz2 = s_topk_size[warp_id];
                        float thr2   = (tk_sz2 < k) ? CUDART_INF_F : s_max_dist[warp_id];
                        if ((tk_sz2 < k) || (pend_dist < thr2)) {
                            int pos2 = atomicAdd((int*)&s_cand_count[warp_id], 1);
                            if (pos2 < k) {
                                s_cand_idx[topk_base + pos2]  = pend_idx;
                                s_cand_dist[topk_base + pos2] = pend_dist;
                                pending = false;
                            } else {
                                // Still full; will merge again and retry.
                                pending = true;
                            }
                        } else {
                            // No longer qualifies under the new (tighter) threshold.
                            pending = false;
                        }
                    }
                    __syncwarp(warp_mask);
                }
            }
            __syncwarp(warp_mask);

            // At tile end, if there are leftover candidates, merge them as well.
            if (s_cand_count[warp_id] > 0) {
                warp_merge_candidates(
                    warp_id, k,
                    s_topk_dist, s_topk_idx,
                    s_cand_dist, s_cand_idx,
                    s_cand_count,
                    s_work_dist, s_work_idx,
                    s_max_dist, s_topk_size,
                    warp_mask);
            }
        }

        __syncthreads(); // Ensure all warps are done with this tile before loading the next.
    }

    // After all tiles processed, ensure any residual candidates are merged (safety).
    if (active && s_cand_count[warp_id] > 0) {
        warp_merge_candidates(
            warp_id, k,
            s_topk_dist, s_topk_idx,
            s_cand_dist, s_cand_idx,
            s_cand_count,
            s_work_dist, s_work_idx,
            s_max_dist, s_topk_size,
            warp_mask);
    }

    // Write results for this query (ascending by distance).
    if (active) {
        PairIF* out_q = out + static_cast<size_t>(global_warp) * k;
        for (int i = lane; i < k; i += WARP_SIZE) {
            PairIF p;
            p.first  = s_topk_idx[topk_base + i];
            p.second = s_topk_dist[topk_base + i];
            out_q[i] = p;
        }
    }
}

// Helper to compute dynamic shared memory size required for given k and tile_points.
static inline size_t knn_shmem_bytes(int k, int tile_points) {
    size_t off = 0;
    // tile of float2
    off = align_up(off, 8);
    off += static_cast<size_t>(tile_points) * sizeof(float2);

    // per-warp metadata
    off = align_up(off, 4);
    off += WARPS_PER_BLOCK * sizeof(int);   // s_cand_count

    off = align_up(off, 4);
    off += WARPS_PER_BLOCK * sizeof(int);   // s_topk_size

    off = align_up(off, 4);
    off += WARPS_PER_BLOCK * sizeof(float); // s_max_dist

    // per-warp arrays
    off = align_up(off, 4);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int);   // s_topk_idx

    off = align_up(off, 4);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float); // s_topk_dist

    off = align_up(off, 4);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int);   // s_cand_idx

    off = align_up(off, 4);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float); // s_cand_dist

    off = align_up(off, 4);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * 2 * k * sizeof(int);   // s_work_idx

    off = align_up(off, 4);
    off += static_cast<size_t>(WARPS_PER_BLOCK) * 2 * k * sizeof(float); // s_work_dist

    return off;
}

// Host wrapper that configures and launches the kernel.
// The result array is std::pair<int, float> but we reinterpret it to PairIF for device writes.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Choose threads/block and number of blocks.
    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE; // 128
    const int num_warps_total = (query_count + 0); // one warp per query
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Determine maximum dynamic shared memory supported and choose tile size accordingly.
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // On A100/H100, opt-in shared memory per block can reach 163 KB or more. Query and set it.
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_optin <= 0) {
        max_optin = prop.sharedMemPerBlock; // Fallback to default
    }

    // Choose the largest tile_points that fits within the shared memory limit.
    // Start with a preferred tile size and reduce if necessary.
    int tile_points = 2048; // initial guess
    size_t shmem_needed = knn_shmem_bytes(k, tile_points);
    while (shmem_needed > static_cast<size_t>(max_optin) && tile_points > 0) {
        tile_points >>= 1; // reduce by half
        if (tile_points < WARP_SIZE) tile_points = WARP_SIZE;
        shmem_needed = knn_shmem_bytes(k, tile_points);
        if (tile_points == WARP_SIZE) break;
    }

    // Opt-in to larger dynamic shared memory if needed.
    cudaFuncSetAttribute(knn2d_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_needed));

    // Launch kernel.
    PairIF* out_pairs = reinterpret_cast<PairIF*>(result);
    knn2d_warp_kernel<<<blocks, threads_per_block, shmem_needed>>>(
        query, query_count, data, data_count, out_pairs, k, tile_points
    );

    // Synchronize to ensure completion (optional depending on caller expectations).
    cudaDeviceSynchronize();
}