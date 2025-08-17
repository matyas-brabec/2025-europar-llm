#include <cuda_runtime.h>
#include <stdint.h>

// This implementation computes k-nearest neighbors (k-NN) for 2D points using a single warp (32 threads) per query.
// Key design points:
// - Each warp processes one query and maintains its own intermediate top-k result in shared memory.
// - Data points are processed in batches that are cooperatively loaded into shared memory by the whole thread block.
// - Each warp filters candidates using a warp-private max_distance (distance of the current k-th nearest neighbor).
// - Candidates passing the filter are appended into a per-warp candidate buffer using atomicAdd on a shared counter.
// - When the candidate buffer is full (or overflows), the warp merges it into the intermediate result using a warp-cooperative bitonic sort on the union.
// - After finishing all batches, any remaining candidates are merged.
// - The final result for each query is written as k pairs (index, distance) sorted by ascending distance.
//
// Assumptions and constraints fulfilled:
// - Target GPUs are modern data center GPUs (A100/H100), compiled with latest CUDA toolkit.
// - k is a power of two between 32 and 1024 inclusive.
// - data_count >= k.
// - query_count and data_count are large enough to benefit from GPU parallelism.
// - No additional device memory is allocated by the kernel (only uses dynamically allocated shared memory).
// - Thread block configuration: 4 warps per block (128 threads). This balances shared memory footprint and occupancy.
// - Data tile size: 2048 points per batch (16KB shared memory for float2 tile), fitting within SM shared memory capacity along with per-warp structures.

// Helper pair type matching the layout of std::pair<int, float>.
// We use this on device side to avoid depending on libstdc++ in device code.
struct PairIF {
    int   first;
    float second;
};

// Constants for block configuration and tiling
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Number of warps per block (blockDim.x = WARPS_PER_BLOCK * WARP_SIZE)
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)
#endif

// Number of data points per tile loaded into shared memory by a block.
// 2048 points = 16KB shared memory for float2 tile.
#ifndef TILE_POINTS
#define TILE_POINTS 2048
#endif

// Utility: round up x to the next multiple of a (alignment-friendly)
__host__ __device__ __forceinline__ size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

// Warp-cooperative in-shared-memory bitonic sort for (distance, index) pairs.
// - Sorts 'n' elements (n must be a power of two) in ascending order by distance.
// - Operates on shared memory arrays d[] and idx[] (warp-private regions).
// - Uses warp-level striding; threads synchronize with __syncwarp(mask).
__device__ __forceinline__ void warp_bitonic_sort_pairs(float* d, int* idx, int n, unsigned mask) {
    // Classic bitonic sort network: O(n log^2 n)
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncwarp(mask);
            for (int i = int(threadIdx.x & (WARP_SIZE - 1)); i < n; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    float di = d[i];
                    float dj = d[ixj];
                    int   ii = idx[i];
                    int   ij = idx[ixj];
                    // If ordering is wrong for the current sequence direction, swap
                    if ((di > dj) == up) {
                        d[i]    = dj;
                        d[ixj]  = di;
                        idx[i]  = ij;
                        idx[ixj]= ii;
                    }
                }
            }
        }
    }
    __syncwarp(mask);
}

// Warp-cooperative merge of candidate buffer into the intermediate top-k result.
// - resd/resi: current result arrays (size k) for the warp
// - p_res_count: pointer to current valid result count (0..k)
// - candd/candi: candidate arrays (size k) for the warp
// - p_cand_count: pointer to current candidate count (0..k or more; we only use up to k)
// - ud/ui: union arrays (size 2k) for temporary merging and sorting
// - p_maxd: pointer to the warp's max_distance (distance of the current k-th)
// - k: desired number of neighbors
// - mask: warp active mask
__device__ __forceinline__ void warp_merge_candidates_into_result(
    float* resd, int* resi, int* p_res_count,
    float* candd, int* candi, int* p_cand_count,
    float* ud,    int* ui,
    float* p_maxd,
    int k,
    unsigned mask)
{
    const int lane = int(threadIdx.x & (WARP_SIZE - 1));

    // Read counts (only lane 0 touches shared variables; broadcast via shfl)
    int rc = 0, cc = 0;
    if (lane == 0) {
        rc = *p_res_count;
        cc = *p_cand_count;
        if (cc > k) cc = k; // Only the first k candidates are validly stored in the buffer
    }
    rc = __shfl_sync(mask, rc, 0);
    cc = __shfl_sync(mask, cc, 0);

    // Union size and sort span size (next power-of-two)
    int union_n = rc + cc;
    if (union_n == 0) {
        // Nothing to merge
        if (lane == 0) {
            // Reset candidate count; max_distance unchanged
            *p_cand_count = 0;
        }
        __syncwarp(mask);
        return;
    }

    int sort_n = 1;
    while (sort_n < union_n) sort_n <<= 1; // sort_n <= 2k (k is power of two)

    // Copy existing results into union arrays
    for (int i = lane; i < rc; i += WARP_SIZE) {
        ud[i] = resd[i];
        ui[i] = resi[i];
    }
    // Copy candidates into union arrays (placed after existing results)
    for (int i = lane; i < cc; i += WARP_SIZE) {
        ud[rc + i] = candd[i];
        ui[rc + i] = candi[i];
    }
    // Fill the padding with sentinel values (INF distance, invalid index)
    for (int i = lane + union_n; i < sort_n; i += WARP_SIZE) {
        ud[i] = CUDART_INF_F;
        ui[i] = -1;
    }
    __syncwarp(mask);

    // Sort the union in ascending order by distance
    warp_bitonic_sort_pairs(ud, ui, sort_n, mask);

    // Keep the first min(union_n, k) elements as the new result
    int new_rc = union_n < k ? union_n : k;
    for (int i = lane; i < new_rc; i += WARP_SIZE) {
        resd[i] = ud[i];
        resi[i] = ui[i];
    }
    // If fewer than k elements, fill the rest with INF/-1
    for (int i = lane + new_rc; i < k; i += WARP_SIZE) {
        resd[i] = CUDART_INF_F;
        resi[i] = -1;
    }
    __syncwarp(mask);

    if (lane == 0) {
        *p_res_count  = new_rc;
        *p_cand_count = 0; // Reset candidate buffer after merging
        // Update max_distance: distance of the k-th neighbor if we have k; else INF
        *p_maxd = (new_rc == k) ? resd[k - 1] : CUDART_INF_F;
    }
    __syncwarp(mask);
}

// Kernel computing k-NN for 2D points.
// Each warp processes one query point. The block cooperatively loads tiles of data points into shared memory.
// Per-warp shared memory regions hold:
// - Intermediate result arrays (k pairs)
// - Candidate buffer arrays (k pairs) + candidate count
// - Union arrays for merging (2k pairs)
// - Result count and max_distance
__global__ void knn2d_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    PairIF* __restrict__ out,
    int k)
{
    // Compute warp and lane identifiers
    const int lane             = int(threadIdx.x & (WARP_SIZE - 1));
    const int warp_id_in_block = int(threadIdx.x) / WARP_SIZE;
    const int warps_per_block  = WARPS_PER_BLOCK;
    const int global_warp_id   = blockIdx.x * warps_per_block + warp_id_in_block;
    const bool warp_active     = (global_warp_id < query_count);
    const unsigned warp_mask   = __activemask();

    // Dynamic shared memory layout
    extern __shared__ unsigned char smem_bytes[];
    unsigned char* smem = smem_bytes;
    size_t offset = 0;

    // Shared tile of data points for the entire block
    offset = align_up(offset, alignof(float2));
    float2* smem_tile = reinterpret_cast<float2*>(smem + offset);
    offset += TILE_POINTS * sizeof(float2);

    // Per-warp control scalars
    offset = align_up(offset, alignof(int));
    int* smem_cand_counts = reinterpret_cast<int*>(smem + offset);
    offset += warps_per_block * sizeof(int);

    offset = align_up(offset, alignof(int));
    int* smem_res_counts = reinterpret_cast<int*>(smem + offset);
    offset += warps_per_block * sizeof(int);

    offset = align_up(offset, alignof(float));
    float* smem_max_dists = reinterpret_cast<float*>(smem + offset);
    offset += warps_per_block * sizeof(float);

    // Per-warp arrays: result (k), candidate (k), union (2k)
    // Result distances
    offset = align_up(offset, alignof(float));
    float* smem_res_dists = reinterpret_cast<float*>(smem + offset);
    offset += size_t(warps_per_block) * size_t(k) * sizeof(float);

    // Result indices
    offset = align_up(offset, alignof(int));
    int* smem_res_indices = reinterpret_cast<int*>(smem + offset);
    offset += size_t(warps_per_block) * size_t(k) * sizeof(int);

    // Candidate distances
    offset = align_up(offset, alignof(float));
    float* smem_cand_dists = reinterpret_cast<float*>(smem + offset);
    offset += size_t(warps_per_block) * size_t(k) * sizeof(float);

    // Candidate indices
    offset = align_up(offset, alignof(int));
    int* smem_cand_indices = reinterpret_cast<int*>(smem + offset);
    offset += size_t(warps_per_block) * size_t(k) * sizeof(int);

    // Union distances (2k)
    offset = align_up(offset, alignof(float));
    float* smem_union_dists = reinterpret_cast<float*>(smem + offset);
    offset += size_t(warps_per_block) * size_t(2 * k) * sizeof(float);

    // Union indices (2k)
    offset = align_up(offset, alignof(int));
    int* smem_union_indices = reinterpret_cast<int*>(smem + offset);
    offset += size_t(warps_per_block) * size_t(2 * k) * sizeof(int);

    // Warp-private base pointers for per-warp arrays
    float* res_dists  = smem_res_dists   + warp_id_in_block * k;
    int*   res_idx    = smem_res_indices + warp_id_in_block * k;
    float* cand_dists = smem_cand_dists  + warp_id_in_block * k;
    int*   cand_idx   = smem_cand_indices+ warp_id_in_block * k;
    float* un_dists   = smem_union_dists + warp_id_in_block * (2 * k);
    int*   un_idx     = smem_union_indices+warp_id_in_block * (2 * k);
    int*   cand_count = smem_cand_counts + warp_id_in_block;
    int*   res_count  = smem_res_counts  + warp_id_in_block;
    float* max_dist   = smem_max_dists   + warp_id_in_block;

    // Initialize per-warp state
    if (warp_active) {
        if (lane == 0) {
            *cand_count = 0;
            *res_count  = 0;
            *max_dist   = CUDART_INF_F;
        }
        // Initialize result arrays to INF / -1 (optional but helps determinism before first merge)
        for (int i = lane; i < k; i += WARP_SIZE) {
            res_dists[i] = CUDART_INF_F;
            res_idx[i]   = -1;
        }
    }
    __syncwarp(warp_mask);

    // Load the query point into registers and broadcast within the warp
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        float2 q;
        if (lane == 0) {
            q = query[global_warp_id];
        }
        qx = __shfl_sync(warp_mask, q.x, 0);
        qy = __shfl_sync(warp_mask, q.y, 0);
    }

    // Process data points in tiles
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        // Block loads a tile of data points into shared memory
        int tile_count = data_count - base;
        if (tile_count > TILE_POINTS) tile_count = TILE_POINTS;

        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            smem_tile[i] = data[base + i];
        }
        __syncthreads(); // Ensure tile is fully loaded before any warp reads it

        // Warp computes distances to its query and filters by current max_distance
        if (warp_active) {
            // Pending candidate for overflow case
            int   pending_idx = -1;
            float pending_dist= 0.0f;
            int   pending_flag= 0;

            for (int i = lane; i < tile_count; i += WARP_SIZE) {
                float2 p = smem_tile[i];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = fmaf(dx, dx, dy * dy); // squared L2 distance

                float md = *max_dist;
                // Only consider points closer than current max_distance
                if (dist < md) {
                    // Append to candidate buffer; use atomicAdd to get the position
                    int pos = atomicAdd(cand_count, 1);
                    if (pos < k) {
                        cand_dists[pos] = dist;
                        cand_idx[pos]   = base + i;
                    } else {
                        // Buffer overflow: hold this candidate temporarily and trigger a merge
                        pending_idx  = base + i;
                        pending_dist = dist;
                        pending_flag = 1;
                    }
                }

                // If any lane overflows the buffer, merge candidates into results now
                unsigned overflow_mask = __ballot_sync(warp_mask, pending_flag != 0);
                // Also merge if the buffer just became full (equals k), even without overflow
                int cc_snapshot = 0;
                if (lane == 0) cc_snapshot = *cand_count;
                cc_snapshot = __shfl_sync(warp_mask, cc_snapshot, 0);
                int just_full = (cc_snapshot >= k);

                if (overflow_mask || just_full) {
                    __syncwarp(warp_mask);
                    // Merge current candidate buffer into the intermediate result
                    warp_merge_candidates_into_result(
                        res_dists, res_idx, res_count,
                        cand_dists, cand_idx, cand_count,
                        un_dists, un_idx,
                        max_dist,
                        k,
                        warp_mask);

                    // After merging, re-try to append the pending candidate if it still passes the updated max_distance.
                    if (pending_flag) {
                        float md2 = *max_dist;
                        if (pending_dist < md2) {
                            int pos2 = atomicAdd(cand_count, 1);
                            // Guaranteed pos2 < k because buffer was just reset and k >= WARP_SIZE
                            if (pos2 < k) {
                                cand_dists[pos2] = pending_dist;
                                cand_idx[pos2]   = pending_idx;
                            }
                        }
                        pending_flag = 0;
                    }
                    __syncwarp(warp_mask);
                }
            }

            // End of tile loop for this warp: if any candidates remain, merge them to update the threshold early
            int cc_final = 0;
            if (lane == 0) cc_final = *cand_count;
            cc_final = __shfl_sync(warp_mask, cc_final, 0);
            if (cc_final > 0) {
                __syncwarp(warp_mask);
                warp_merge_candidates_into_result(
                    res_dists, res_idx, res_count,
                    cand_dists, cand_idx, cand_count,
                    un_dists, un_idx,
                    max_dist,
                    k,
                    warp_mask);
                __syncwarp(warp_mask);
            }
        }

        __syncthreads(); // Ensure all warps are done reading this tile before loading the next one
    }

    // After processing all tiles, if any candidates remain (shouldn't, but safe), merge them
    if (warp_active) {
        int cc_rem = 0;
        if (lane == 0) cc_rem = *cand_count;
        cc_rem = __shfl_sync(warp_mask, cc_rem, 0);
        if (cc_rem > 0) {
            __syncwarp(warp_mask);
            warp_merge_candidates_into_result(
                res_dists, res_idx, res_count,
                cand_dists, cand_idx, cand_count,
                un_dists, un_idx,
                max_dist,
                k,
                warp_mask);
            __syncwarp(warp_mask);
        }

        // Write final k nearest neighbors for this query to global memory (ascending by distance)
        int out_base = global_warp_id * k;
        for (int j = lane; j < k; j += WARP_SIZE) {
            PairIF pr;
            pr.first  = res_idx[j];
            pr.second = res_dists[j];
            out[out_base + j] = pr;
        }
    }
}

// Host API: launch configuration and shared memory sizing
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Configure grid: one warp per query
    const int warps_per_block   = WARPS_PER_BLOCK;           // 4
    const int threads_per_block = THREADS_PER_BLOCK;         // 128
    const int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Compute dynamic shared memory size per block
    // Layout per block:
    // - TILE_POINTS * sizeof(float2)
    // - warps_per_block * (sizeof(int) cand_count + sizeof(int) res_count + sizeof(float) max_dist)
    // - warps_per_block * [ (k * (sizeof(float) + sizeof(int)) for result)
    //                       + (k * (sizeof(float) + sizeof(int)) for candidates)
    //                       + (2k * (sizeof(float) + sizeof(int)) for union) ]
    size_t smem_bytes = 0;
    smem_bytes = align_up(smem_bytes, alignof(float2));
    smem_bytes += size_t(TILE_POINTS) * sizeof(float2);

    smem_bytes = align_up(smem_bytes, alignof(int));
    smem_bytes += size_t(warps_per_block) * sizeof(int);   // cand_counts
    smem_bytes = align_up(smem_bytes, alignof(int));
    smem_bytes += size_t(warps_per_block) * sizeof(int);   // res_counts
    smem_bytes = align_up(smem_bytes, alignof(float));
    smem_bytes += size_t(warps_per_block) * sizeof(float); // max_dists

    // Per-warp arrays
    size_t per_warp_pairs_bytes = 0;
    per_warp_pairs_bytes  = size_t(k) * (sizeof(float) + sizeof(int));   // result
    per_warp_pairs_bytes += size_t(k) * (sizeof(float) + sizeof(int));   // candidates
    per_warp_pairs_bytes += size_t(2 * k) * (sizeof(float) + sizeof(int)); // union

    smem_bytes = align_up(smem_bytes, alignof(float));
    smem_bytes += size_t(warps_per_block) * size_t(k) * sizeof(float); // res dists
    smem_bytes = align_up(smem_bytes, alignof(int));
    smem_bytes += size_t(warps_per_block) * size_t(k) * sizeof(int);   // res idx
    smem_bytes = align_up(smem_bytes, alignof(float));
    smem_bytes += size_t(warps_per_block) * size_t(k) * sizeof(float); // cand dists
    smem_bytes = align_up(smem_bytes, alignof(int));
    smem_bytes += size_t(warps_per_block) * size_t(k) * sizeof(int);   // cand idx
    smem_bytes = align_up(smem_bytes, alignof(float));
    smem_bytes += size_t(warps_per_block) * size_t(2 * k) * sizeof(float); // union dists
    smem_bytes = align_up(smem_bytes, alignof(int));
    smem_bytes += size_t(warps_per_block) * size_t(2 * k) * sizeof(int);   // union idx

    // Launch kernel
    knn2d_kernel<<<blocks, threads_per_block, smem_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<PairIF*>(result),
        k
    );
}
