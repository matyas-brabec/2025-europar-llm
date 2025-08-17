#include <cuda_runtime.h>
#include <utility>

// This implementation performs k-NN for 2D points on the GPU.
// - One warp (32 threads) handles one query point.
// - Input data points are processed in shared-memory batches ("tiles") loaded by the entire block.
// - Each warp maintains:
//   * A private intermediate result of k nearest neighbors (sorted ascending by distance) in shared memory.
//   * A candidate buffer (size k) in shared memory with a shared counter updated via atomicAdd.
//   * A "max_distance" shared variable storing the current k-th (largest) distance in the intermediate result.
// - When the warp's candidate buffer becomes full, it is sorted (Bitonic sort), merged with the intermediate result
//   (into a bitonic sequence using pairwise minima), and sorted again (Bitonic sort) to update the intermediate result.
// - After processing all tiles, any remaining candidates in the buffer are merged in the same way.
// - Distances are squared Euclidean distances; ties are resolved arbitrarily.
//
// Design choices:
// - Threads per block: 128 (4 warps). This balances shared memory usage (for k up to 1024) and occupancy on A100/H100.
// - Tile size (data points cached per block in shared memory): 4096 points (32 KB). Combined with per-warp buffers of
//   3*k pairs (intermediate, buffer, merged), this fits within typical shared memory limits for A100/H100 when k <= 1024.
// - Bitonic sort is implemented in parallel across the 32 lanes of the warp, operating on shared memory with warp-scope
//   synchronization via __syncwarp().
//
// Notes on correctness and synchronization:
// - The candidate buffer's count is always updated using atomicAdd as required. Appends are performed warp-cooperatively:
//   the warp aggregates accepted candidates for a 32-point group, reserves space with a single atomicAdd, writes those
//   that fit, flushes when needed, and retries any leftovers after the flush. This ensures no candidate is lost.
// - When flushing, the buffer is sorted (size k), merged with the current intermediate result, and the merged result is
//   sorted to produce the updated intermediate result. max_distance is updated accordingly.

#ifndef KNN_THREADS_PER_BLOCK
#define KNN_THREADS_PER_BLOCK 128  // 4 warps per block
#endif

#ifndef KNN_TILE_POINTS
#define KNN_TILE_POINTS 4096       // points per shared-memory tile (float2 => 8 bytes each; 4096 -> 32KB)
#endif

// Simple POD pair (index, distance) used on the device side.
// This has the same memory footprint as std::pair<int,float> (8 bytes).
struct PairIF {
    int   first;   // index in the data array
    float second;  // squared distance
};

// Internal candidate representation for sorts/merges.
struct Candidate {
    float dist;
    int   idx;
};

static __device__ __forceinline__ Candidate make_cand(float d, int i) {
    Candidate c; c.dist = d; c.idx = i; return c;
}

// Warp-parallel Bitonic Sort (ascending) on shared-memory array 'arr' of size 'n' (power of two).
// - The 32 lanes cooperatively execute the compare-and-swap network.
// - Each lane handles indices i = lane, lane+32, lane+64, ... looping across the array.
// - Synchronization between stages is via __syncwarp(warp_mask).
static __device__ __forceinline__ void warp_bitonic_sort_asc(Candidate* arr, int n, unsigned warp_mask) {
    // Bitonic sort network as per the given pseudocode, parallelized by distributing 'i' across lanes.
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each thread processes multiple i values, striding by 32.
            int lane = threadIdx.x & 31;
            for (int i = lane; i < n; i += 32) {
                int l = i ^ j;
                if (l > i) {
                    bool ascending = ((i & k) == 0);
                    Candidate ai = arr[i];
                    Candidate al = arr[l];
                    // Compare by distance; swap according to bitonic ordering rule.
                    bool cmp = (ascending ? (ai.dist > al.dist) : (ai.dist < al.dist));
                    if (cmp) {
                        arr[i] = al;
                        arr[l] = ai;
                    }
                }
            }
            __syncwarp(warp_mask);
        }
    }
}

// Merge buffer and intermediate into 'merged' via pairwise minima to form a bitonic sequence, then sort.
// Precondition:
// - 'inter' is sorted ascending (invariant).
// - 'buf' is sorted ascending.
// - 'merged' is a temporary array (size k).
// Postcondition:
// - 'inter' updated to the sorted ascending top-k result from buf ∪ inter.
// - 'max_dist' updated to inter[k-1].dist.
// - 'buf_count' reset to 0.
static __device__ __forceinline__ void warp_merge_flush(
    Candidate* inter, Candidate* buf, Candidate* merged,
    int k, int lane, unsigned warp_mask,
    volatile int* buf_count_ptr, volatile float* max_dist_ptr)
{
    int buf_count = *buf_count_ptr;
    if (buf_count <= 0) return;

    // If buffer is not full, pad with +INF so that Bitonic Sort (size k) is valid.
    if (buf_count < k) {
        for (int i = lane + buf_count; i < k; i += 32) {
            buf[i] = make_cand(CUDART_INF_F, -1);
        }
        __syncwarp(warp_mask);
    }

    // 1) Sort the buffer ascending.
    warp_bitonic_sort_asc(buf, k, warp_mask);

    // 2) Merge buffer and intermediate via pairwise minima to produce a bitonic sequence.
    for (int i = lane; i < k; i += 32) {
        Candidate a = buf[i];
        Candidate b = inter[k - 1 - i];
        merged[i] = (a.dist <= b.dist) ? a : b;
    }
    __syncwarp(warp_mask);

    // 3) Sort the merged result ascending.
    warp_bitonic_sort_asc(merged, k, warp_mask);

    // Update the intermediate result and max distance.
    for (int i = lane; i < k; i += 32) {
        inter[i] = merged[i];
    }
    __syncwarp(warp_mask);
    if (lane == 0) {
        *max_dist_ptr = inter[k - 1].dist;
        *buf_count_ptr = 0;
    }
    __syncwarp(warp_mask);
}

// Kernel implementing k-NN using one warp per query.
__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    PairIF* __restrict__ result,
    int k)
{
    constexpr int WARPS_PER_BLOCK = KNN_THREADS_PER_BLOCK / 32;
    const int lane        = threadIdx.x & 31;
    const int warp_local  = threadIdx.x >> 5; // warp index within block
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_local;

    // Shared memory layout (dynamic):
    // [tile_data (KNN_TILE_POINTS float2)] [inter_all (WARPS*k Candidates)] [buf_all (WARPS*k)] [merged_all (WARPS*k)]
    extern __shared__ unsigned char smem[];
    unsigned char* smem_ptr = smem;
    size_t offset = 0;

    // Tile buffer shared by the entire block
    float2* tile_data = reinterpret_cast<float2*>(smem_ptr + offset);
    offset += static_cast<size_t>(KNN_TILE_POINTS) * sizeof(float2);
    // Align to 8-byte boundary
    offset = (offset + 7) & ~static_cast<size_t>(7);

    // Per-warp arrays
    Candidate* inter_all  = reinterpret_cast<Candidate*>(smem_ptr + offset);
    offset += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(Candidate);
    Candidate* buf_all    = reinterpret_cast<Candidate*>(smem_ptr + offset);
    offset += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(Candidate);
    Candidate* merged_all = reinterpret_cast<Candidate*>(smem_ptr + offset);
    offset += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(Candidate);

    Candidate* inter  = inter_all  + warp_local * k;
    Candidate* buf    = buf_all    + warp_local * k;
    Candidate* merged = merged_all + warp_local * k;

    // Shared control variables per warp
    __shared__ int   s_buf_count[WARPS_PER_BLOCK];
    __shared__ float s_max_dist[WARPS_PER_BLOCK];

    // Initialize per-warp state
    if (lane == 0) {
        s_buf_count[warp_local] = 0;
        s_max_dist[warp_local]  = CUDART_INF_F;
    }
    // Initialize the intermediate result to +INF (sorted by construction)
    for (int i = lane; i < k; i += 32) {
        inter[i] = make_cand(CUDART_INF_F, -1);
    }
    __syncwarp();

    // Warp mask for synchronization among active lanes in the warp
    const unsigned warp_mask = 0xffffffffu;

    // Preload the query point for this warp (if it corresponds to a valid query)
    float qx = 0.0f, qy = 0.0f;
    if (warp_global < query_count) {
        float2 q;
        if (lane == 0) q = query[warp_global];
        qx = __shfl_sync(warp_mask, q.x, 0);
        qy = __shfl_sync(warp_mask, q.y, 0);
    }

    // Process data in tiles loaded into shared memory
    for (int tile_base = 0; tile_base < data_count; tile_base += KNN_TILE_POINTS) {
        const int tile_count = min(KNN_TILE_POINTS, data_count - tile_base);

        // Load the tile cooperatively by all threads in the block
        for (int idx = threadIdx.x; idx < tile_count; idx += blockDim.x) {
            tile_data[idx] = data[tile_base + idx];
        }
        __syncthreads();

        if (warp_global < query_count) {
            // Iterate over the tile in warp-sized groups
            for (int t = 0; t < tile_count; t += 32) {
                // Each lane computes at most one candidate in this group
                const int local_idx = t + lane;
                float dist = CUDART_INF_F;
                int   gidx = -1;
                if (local_idx < tile_count) {
                    float2 p = tile_data[local_idx];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    dist = dx * dx + dy * dy;
                    gidx = tile_base + local_idx;
                }

                // Check against current max distance to filter
                const float maxd = s_max_dist[warp_local];
                const bool  valid = (local_idx < tile_count) && (dist < maxd);
                unsigned    mask  = __ballot_sync(warp_mask, valid);
                if (mask == 0) continue; // no candidates in this group

                int num_valid = __popc(mask);

                // Reserve space in the candidate buffer with a single atomicAdd
                int base = 0;
                if (lane == 0) {
                    base = atomicAdd(&s_buf_count[warp_local], num_valid);
                }
                base = __shfl_sync(warp_mask, base, 0);
                int space_left = max(0, k - base);
                // Compute per-lane rank among valid lanes
                unsigned lt_mask = (lane == 0) ? 0u : (0xffffffffu >> (32 - lane));
                int rank = __popc(mask & lt_mask);

                // Write the portion that fits
                bool will_write = valid && (rank < space_left);
                if (will_write) {
                    int pos = base + rank;
                    buf[pos] = make_cand(dist, gidx);
                }
                __syncwarp(warp_mask);

                // If overflow occurred, flush buffer and retry leftovers
                if (num_valid > space_left) {
                    // Cap buffer count at k before flushing
                    if (lane == 0) {
                        s_buf_count[warp_local] = k;
                    }
                    __syncwarp(warp_mask);

                    // Flush (sort buffer, merge, sort merged)
                    warp_merge_flush(inter, buf, merged, k, lane, warp_mask,
                                     &s_buf_count[warp_local], &s_max_dist[warp_local]);

                    // Retry leftovers (those with rank >= space_left) against updated max distance
                    const float new_maxd = s_max_dist[warp_local];
                    bool leftover = valid && (rank >= space_left) && (dist < new_maxd);
                    unsigned mask2 = __ballot_sync(warp_mask, leftover);
                    int num2 = __popc(mask2);
                    if (num2 > 0) {
                        int base2 = 0;
                        if (lane == 0) {
                            base2 = atomicAdd(&s_buf_count[warp_local], num2);
                        }
                        base2 = __shfl_sync(warp_mask, base2, 0);
                        unsigned lt_mask2 = (lane == 0) ? 0u : (0xffffffffu >> (32 - lane));
                        int rank2 = __popc(mask2 & lt_mask2);
                        if (leftover) {
                            int pos2 = base2 + rank2;
                            // This always fits since num2 <= 32 and k >= 32 and buffer was empty after flush.
                            buf[pos2] = make_cand(dist, gidx);
                        }
                        __syncwarp(warp_mask);
                    }
                }
            }
        }

        __syncthreads(); // Ensure all warps are done with this tile before loading the next one
    }

    // Final flush for any remaining candidates
    if (warp_global < query_count) {
        int count = s_buf_count[warp_local];
        if (count > 0) {
            // Pad and flush merge
            warp_merge_flush(inter, buf, merged, k, lane, warp_mask,
                             &s_buf_count[warp_local], &s_max_dist[warp_local]);
        }

        // Write out the final top-k results for this query
        for (int i = lane; i < k; i += 32) {
            Candidate c = inter[i];
            result[warp_global * k + i] = PairIF{ c.idx, c.dist };
        }
    }
}

// Host-side launcher matching the requested interface.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Configuration
    constexpr int threads_per_block = KNN_THREADS_PER_BLOCK;
    constexpr int warps_per_block   = threads_per_block / 32;
    const int total_warps           = query_count;
    const int blocks                = (total_warps + warps_per_block - 1) / warps_per_block;

    // Dynamic shared memory size:
    // tile_data + per-warp (inter + buf + merged), each of size k Candidates (8 bytes each)
    const size_t smem_tile_bytes    = static_cast<size_t>(KNN_TILE_POINTS) * sizeof(float2);
    const size_t smem_perwarp_bytes = static_cast<size_t>(k) * sizeof(Candidate) * 3u;
    const size_t smem_bytes         = smem_tile_bytes + static_cast<size_t>(warps_per_block) * smem_perwarp_bytes;

    // Reinterpret the result pointer to a POD pair type with the same memory layout.
    PairIF* result_alias = reinterpret_cast<PairIF*>(result);

    // Launch kernel
    knn_kernel<<<blocks, threads_per_block, smem_bytes>>>(
        query, query_count, data, data_count, result_alias, k
    );
}