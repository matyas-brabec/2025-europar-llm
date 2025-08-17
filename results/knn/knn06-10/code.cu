#include <cuda_runtime.h>
#include <stdint.h>

// This implementation assigns exactly one warp (32 threads) per query point.
// Each warp maintains three per-query arrays in shared memory:
//  - best_k:    the current top-k nearest neighbors (indices and distances), sorted ascending by distance
//  - cand_buf:  a buffer of up to k candidate neighbors (indices and distances) collected from recent data batches
//  - merge_out: temporary output buffer used when merging cand_buf into best_k
//
// All warps in the block cooperatively cache the current batch (tile) of data points into shared memory.
// Each active warp then processes the tile to generate candidate neighbors. The warp uses warp-cooperative
// reservation of slots in its candidate buffer using atomicAdd on a per-warp shared counter. When the
// candidate buffer is full (no capacity left), the warp merges cand_buf into best_k. After the final tile,
// any remaining candidates are also merged.
//
// The best_k is always kept sorted ascending; max_distance is the current k-th (worst) distance in best_k.
// Only distances strictly less than max_distance are inserted into cand_buf.
//
// Hyper-parameters chosen for modern data center GPUs (A100/H100):
//  - Warps per block: 6 (192 threads per block). This provides high warp occupancy while leaving
//    sufficient shared memory for large k (up to 1024).
//  - Tile size: 2048 points (16 KB). Fits into shared memory together with the per-warp buffers at k=1024.
//
// Notes:
//  - k is a power of two between 32 and 1024 inclusive.
//  - data_count >= k.
//  - query_count and data_count are large; the kernel is designed for throughput.
//  - The kernel does not allocate device memory; it uses shared memory and per-thread registers only.

struct PairIF { int first; float second; };

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable parameters
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 6
#endif
#ifndef TILE_POINTS
#define TILE_POINTS 2048
#endif

// Utility: squared L2 distance for 2D points
static __device__ __forceinline__ float sqr_dist2(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // FMA for better throughput
    return fmaf(dx, dx, dy * dy);
}

// Reserve up to 'request' slots in [0, capacity) using atomicAdd on shared counter.
// Returns base index (via ref) and the number of slots actually reserved (via ref).
// If no space available, alloc=0. This uses atomicAdd (and atomicSub for overflow correction)
// to meet the requirement of using atomicAdd to update the number of stored candidates.
static __device__ __forceinline__ void reserve_slots_atomicAdd(int* count_ptr, int capacity, int request, int &base, int &alloc) {
    // One caller at a time per warp uses this function; others wait for broadcast results.
    int pos = atomicAdd(count_ptr, request);
    int avail = capacity - pos;
    if (avail <= 0) {
        // No space at all; revert the addition to keep the counter unchanged.
        atomicSub(count_ptr, request);
        base = 0;
        alloc = 0;
        return;
    }
    // Some space available
    alloc = (avail < request) ? avail : request;
    base  = pos;
    if (alloc < request) {
        // Only a portion was reserved; revert the unallocated part
        atomicSub(count_ptr, request - alloc);
    }
}

// Bitonic sort (ascending) for key-value pairs stored in shared memory.
// Sorts the first n elements. n must be <= capacity; the algorithm sorts up to n_p2,
// the next power-of-two >= n, and assumes that keys[i] = +INF for i in [n, n_p2).
static __device__ __forceinline__ void warp_bitonic_sort_pairs(float* keys, int* vals, int n, int capacity, unsigned mask) {
    if (n <= 1) return;

    // Compute next power of two >= n, but not exceeding capacity
    int n_p2 = 1;
    while (n_p2 < n) n_p2 <<= 1;
    if (n_p2 > capacity) n_p2 = capacity;

    // Pad the tail [n, n_p2) with +INF and a dummy index, so sort network behaves correctly.
    const float INF_F = CUDART_INF_F;
    for (int i = threadIdx.x & (WARP_SIZE - 1); i < n_p2; i += WARP_SIZE) {
        if (i >= n) {
            keys[i] = INF_F;
            vals[i] = -1;
        }
    }
    __syncwarp(mask);

    // Standard bitonic sort network
    for (int k = 2; k <= n_p2; k <<= 1) {
        // Merge bitonic sequences of length k
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x & (WARP_SIZE - 1); i < n_p2; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0); // Determine sort direction for this subsequence
                    float key_i = keys[i];
                    float key_j = keys[ixj];
                    int val_i = vals[i];
                    int val_j = vals[ixj];
                    bool comp = (key_i > key_j);
                    if (!ascending) comp = !comp; // reverse condition for descending
                    if (comp) {
                        // Swap
                        keys[i] = key_j;
                        keys[ixj] = key_i;
                        vals[i] = val_j;
                        vals[ixj] = val_i;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
    // After sorting, the first n elements are in ascending order; padded elements are +INF at the end.
}

// Merge (k and c elements, both ascending) into top-k smallest (ascending).
// - best_k_dist/best_k_idx: input sorted ascending, length k
// - cand_dist/cand_idx: input sorted ascending, length c
// - out_dist/out_idx: output buffer of length k (ascending)
// This uses a warp-cooperative merge path partition; each lane produces out elements at stride WARP_SIZE.
static __device__ __forceinline__ void warp_merge_topk_sorted(
    const float* __restrict__ best_k_dist,
    const int*   __restrict__ best_k_idx,
    int k,
    const float* __restrict__ cand_dist,
    const int*   __restrict__ cand_idx,
    int c,
    float* __restrict__ out_dist,
    int*   __restrict__ out_idx,
    unsigned mask)
{
    // Handle trivial cases
    if (c <= 0) {
        for (int t = threadIdx.x & (WARP_SIZE - 1); t < k; t += WARP_SIZE) {
            out_dist[t] = best_k_dist[t];
            out_idx[t]  = best_k_idx[t];
        }
        __syncwarp(mask);
        return;
    }
    // Merge-path per output position t in [0, k)
    const float INF_F = CUDART_INF_F;
    for (int t = threadIdx.x & (WARP_SIZE - 1); t < k; t += WARP_SIZE) {
        int low  = max(0, t - c);
        int high = min(t, k);
        // Binary search on diagonal t: find i such that best[i-1] <= cand[t-i] and cand[t-i-1] < best[i]
        while (low < high) {
            int mid = (low + high) >> 1;
            int j = t - mid;
            float left_best = (mid > 0) ? best_k_dist[mid - 1] : -CUDART_INF_F;
            float left_cand = (j   > 0) ? cand_dist[j - 1]     : -CUDART_INF_F;
            if (left_best <= left_cand) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        int i = low;
        int j = t - i;
        float next_best = (i < k) ? best_k_dist[i] : INF_F;
        float next_cand = (j < c) ? cand_dist[j]   : INF_F;
        if (next_best <= next_cand) {
            out_dist[t] = next_best;
            out_idx[t]  = best_k_idx[i];
        } else {
            out_dist[t] = next_cand;
            out_idx[t]  = cand_idx[j];
        }
    }
    __syncwarp(mask);
}

// Merge cand_buf into best_k (both in shared memory), update max_distance and reset cand_count to 0.
// - best_k_* are always kept sorted ascending.
// - cand_buf_* are sorted (ascending) here before merging.
// - merge_out_* are temporary buffers to hold the merged top-k before copying back to best_k_*.
static __device__ __forceinline__ void warp_flush_and_merge(
    float* best_k_dist, int* best_k_idx,
    float* cand_dist,   int* cand_idx,
    float* merge_out_dist, int* merge_out_idx,
    int k,
    int* cand_count_ptr,
    float* max_dist_ptr,
    unsigned mask)
{
    // All threads in this warp must call this function together
    __syncwarp(mask);

    int c;
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        c = *cand_count_ptr;
    }
    c = __shfl_sync(mask, c, 0);
    if (c <= 0) {
        return;
    }

    // Sort candidate buffer (ascending). Only need first c entries.
    warp_bitonic_sort_pairs(cand_dist, cand_idx, c, k, mask);

    // Merge best_k (k ascending) with cand (c ascending) into top-k (ascending)
    warp_merge_topk_sorted(best_k_dist, best_k_idx, k, cand_dist, cand_idx, c, merge_out_dist, merge_out_idx, mask);

    // Copy merged result back into best_k
    for (int i = threadIdx.x & (WARP_SIZE - 1); i < k; i += WARP_SIZE) {
        best_k_dist[i] = merge_out_dist[i];
        best_k_idx[i]  = merge_out_idx[i];
    }
    __syncwarp(mask);

    // Update max_distance = k-th (worst) distance
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        *max_dist_ptr = best_k_dist[k - 1];
        // Reset candidate count
        *cand_count_ptr = 0;
    }
    __syncwarp(mask);
}

static __global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    PairIF* __restrict__ result,
    int k)
{
    // Warp and lane identifiers
    const int lane       = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_blk= threadIdx.x >> 5; // 0..WARPS_PER_BLOCK-1
    const int warps_per_block = WARPS_PER_BLOCK;
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Global warp id maps to query id
    const int global_warp_id = blockIdx.x * warps_per_block + warp_in_blk;
    const bool warp_active = (global_warp_id < query_count);

    // Dynamic shared memory layout:
    extern __shared__ unsigned char smem[];
    unsigned char* ptr = smem;

    // Shared tile for data points (cached by the block)
    float2* tile = reinterpret_cast<float2*>(ptr);
    ptr += sizeof(float2) * TILE_POINTS;

    // Per-warp arrays stored contiguously across the block
    // Layout for all warps in block: [best_dist][best_idx][cand_dist][cand_idx][merge_out_dist][merge_out_idx][cand_counts][max_dists]
    float* best_all_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * warps_per_block * k;

    int*   best_all_idx  = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int)   * warps_per_block * k;

    float* cand_all_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * warps_per_block * k;

    int*   cand_all_idx  = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int)   * warps_per_block * k;

    float* merge_all_dist= reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * warps_per_block * k;

    int*   merge_all_idx = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int)   * warps_per_block * k;

    int*   cand_counts   = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int)   * warps_per_block;

    float* max_dists     = reinterpret_cast<float*>(ptr);
    // ptr += sizeof(float) * warps_per_block; // not needed further

    // Pointers to this warp's per-query buffers in shared memory
    float* best_k_dist   = best_all_dist   + warp_in_blk * k;
    int*   best_k_idx    = best_all_idx    + warp_in_blk * k;
    float* cand_dist     = cand_all_dist   + warp_in_blk * k;
    int*   cand_idx      = cand_all_idx    + warp_in_blk * k;
    float* merge_out_dist= merge_all_dist  + warp_in_blk * k;
    int*   merge_out_idx = merge_all_idx   + warp_in_blk * k;
    int*   cand_count_ptr= cand_counts     + warp_in_blk;
    float* max_dist_ptr  = max_dists       + warp_in_blk;

    // Initialize per-warp structures
    if (warp_active) {
        // best_k initialized to +INF distances and invalid indices; kept sorted ascending (trivially)
        for (int i = lane; i < k; i += WARP_SIZE) {
            best_k_dist[i] = CUDART_INF_F;
            best_k_idx[i]  = -1;
        }
        if (lane == 0) {
            *cand_count_ptr = 0;
            *max_dist_ptr   = CUDART_INF_F;
        }
    } else {
        // Still initialize shared structures to avoid undefined behavior
        for (int i = lane; i < k; i += WARP_SIZE) {
            best_k_dist[i] = CUDART_INF_F;
            best_k_idx[i]  = -1;
        }
        if (lane == 0) {
            *cand_count_ptr = 0;
            *max_dist_ptr   = CUDART_INF_F;
        }
    }
    __syncwarp(FULL_MASK);

    // Load the query point for this warp and broadcast
    float2 q = make_float2(0.f, 0.f);
    if (lane == 0 && warp_active) {
        q = query[global_warp_id];
    }
    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    // Iterate over data points in tiles
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        int tile_count = data_count - base;
        if (tile_count > TILE_POINTS) tile_count = TILE_POINTS;

        // Block-wide cooperative load of data into shared memory tile
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            tile[i] = data[base + i];
        }
        __syncthreads(); // ensure tile is ready

        // Process tile: each warp computes distances for its query and collects candidates
        if (warp_active) {
            float current_maxd = *max_dist_ptr;

            for (int ti = lane; ti < tile_count; ti += WARP_SIZE) {
                // Compute squared distance for this data point
                float2 p = tile[ti];
                float d  = sqr_dist2(p, q);
                int   idx = base + ti;

                // Consider as candidate only if strictly better than current k-th best
                bool is_pending = (d < current_maxd);

                // Attempt to insert this element into cand_buf; if buffer full, flush and retry if still qualifies
                while (true) {
                    unsigned pending_mask = __ballot_sync(FULL_MASK, is_pending);
                    int need = __popc(pending_mask);
                    if (need == 0) break;

                    int base_pos = 0, alloc = 0;
                    if (lane == 0) {
                        reserve_slots_atomicAdd(cand_count_ptr, k, need, base_pos, alloc);
                    }
                    base_pos = __shfl_sync(FULL_MASK, base_pos, 0);
                    alloc    = __shfl_sync(FULL_MASK, alloc,    0);

                    if (alloc > 0) {
                        // Compute position among pending lanes
                        int my_rank = __popc(pending_mask & ((1u << lane) - 1));
                        if (is_pending && my_rank < alloc) {
                            int wpos = base_pos + my_rank;
                            // Insert into candidate buffer
                            cand_dist[wpos] = d;
                            cand_idx[wpos]  = idx;
                            // Mark as inserted
                            is_pending = false;
                        }
                        // If some pending were not inserted due to insufficient space, they will loop again
                    } else {
                        // No capacity at all; flush/merge and update current_maxd, then re-evaluate pending
                        warp_flush_and_merge(best_k_dist, best_k_idx,
                                             cand_dist, cand_idx,
                                             merge_out_dist, merge_out_idx,
                                             k, cand_count_ptr, max_dist_ptr, FULL_MASK);
                        current_maxd = *max_dist_ptr;
                        // Re-evaluate if this element still qualifies after the threshold decreased
                        is_pending = (d < current_maxd);
                    }
                }
            }

            // After finishing the tile, flush remaining candidates (if any)
            int remaining = 0;
            if (lane == 0) remaining = *cand_count_ptr;
            remaining = __shfl_sync(FULL_MASK, remaining, 0);
            if (remaining > 0) {
                warp_flush_and_merge(best_k_dist, best_k_idx,
                                     cand_dist, cand_idx,
                                     merge_out_dist, merge_out_idx,
                                     k, cand_count_ptr, max_dist_ptr, FULL_MASK);
            }
        }

        __syncthreads(); // ensure all warps are done with this tile before overwriting it
    }

    // Write out final results for this query
    if (warp_active) {
        // best_k_dist/best_k_idx are sorted ascending
        int out_base = global_warp_id * k;
        // Each lane writes a strided subset
        for (int i = lane; i < k; i += WARP_SIZE) {
            PairIF r; r.first = best_k_idx[i]; r.second = best_k_dist[i];
            result[out_base + i] = r;
        }
    }
}

// Host API
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Reinterpret result pointer as a trivial pair struct to avoid device-side std::pair dependencies
    PairIF* d_result = reinterpret_cast<PairIF*>(result);

    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int num_warps = (query_count + warps_per_block - 1) / warps_per_block;
    const dim3 grid(num_warps);
    const dim3 block(threads_per_block);

    // Compute dynamic shared memory size:
    // - TILE_POINTS float2:  TILE_POINTS * 8 bytes
    // - Per-warp arrays (for all warps in the block):
    //     best_dist:      w * k * 4
    //     best_idx:       w * k * 4
    //     cand_dist:      w * k * 4
    //     cand_idx:       w * k * 4
    //     merge_out_dist: w * k * 4
    //     merge_out_idx:  w * k * 4
    //     cand_counts:    w * 4
    //     max_dists:      w * 4
    size_t shmem_size = 0;
    shmem_size += sizeof(float2) * TILE_POINTS;
    shmem_size += (size_t)warps_per_block * k * sizeof(float); // best_dist
    shmem_size += (size_t)warps_per_block * k * sizeof(int);   // best_idx
    shmem_size += (size_t)warps_per_block * k * sizeof(float); // cand_dist
    shmem_size += (size_t)warps_per_block * k * sizeof(int);   // cand_idx
    shmem_size += (size_t)warps_per_block * k * sizeof(float); // merge_out_dist
    shmem_size += (size_t)warps_per_block * k * sizeof(int);   // merge_out_idx
    shmem_size += (size_t)warps_per_block * sizeof(int);       // cand_counts
    shmem_size += (size_t)warps_per_block * sizeof(float);     // max_dists

    // Launch kernel
    knn_kernel<<<grid, block, shmem_size>>>(query, query_count, data, data_count, d_result, k);
    // The caller is responsible for checking/handling CUDA errors and synchronization if desired.
}