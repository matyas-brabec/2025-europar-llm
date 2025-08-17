#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>
#include <utility>

// This implementation assigns one warp (32 threads) to each query point.
// The block cooperatively loads input data points into shared memory tiles.
// Each warp maintains:
//  - An intermediate result (its current top-k neighbors) privately in registers (distributed across threads).
//  - A per-warp candidate buffer of capacity k in shared memory, plus a shared counter updated via atomicAdd.
//
// The algorithm processes the data in batches (tiles) cached in shared memory.
// Each warp computes distances from its query to all points in a tile and filters by current max_distance
// (distance of the k-th neighbor). Points closer than max_distance are buffered as candidates.
// When the per-warp buffer fills up, the warp merges the buffered candidates into its intermediate result.
// After processing all tiles, the warp merges any remaining candidates, then gathers and sorts its top-k
// results (ascending by distance) in shared memory and writes them to the output.
//
// Notes:
//  - k is a power of two in [32, 1024]. We use k/32 elements per thread to store the warp's top-k list.
//  - Shared memory usage per block (worst-case with k=1024, 8 warps and tile of 4096 points) is ~96 KB.
//  - The kernel uses warp-level intrinsics (__shfl_sync, __ballot_sync, __syncwarp) for communication.
//  - The output is written into a std::pair<int,float> array; we use a POD alias with identical layout.

struct PairIF { int first; float second; };

// Utilities
static inline __device__ unsigned lane_id() { return threadIdx.x & 31; }

template <typename T>
static inline __device__ T warp_bcast(T v, int src_lane, unsigned mask = 0xFFFFFFFFu) {
    return __shfl_sync(mask, v, src_lane);
}

static inline __device__ float warp_reduce_max(float v, unsigned mask = 0xFFFFFFFFu) {
    // Full-warp reduction to maximum
    for (int offset = 16; offset > 0; offset >>= 1) {
        float oth = __shfl_down_sync(mask, v, offset);
        v = fmaxf(v, oth);
    }
    // Broadcast max to all lanes (in lane 0 after reduction)
    float maxv = __shfl_sync(mask, v, 0);
    return maxv;
}

// Compute warp-wide argmax of values, returning the lane that holds the maximum value.
// Tie-breaker: smaller lane index wins to keep behavior deterministic.
static inline __device__ int warp_argmax_lane(float local_val, int lane, unsigned mask = 0xFFFFFFFFu) {
    int max_lane = lane;
    float max_val = local_val;
    // Reduce by comparing against values from other lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        float oth_val  = __shfl_down_sync(mask, max_val,  offset);
        int   oth_lane = __shfl_down_sync(mask, max_lane, offset);
        bool take_other = (oth_val > max_val) || ((oth_val == max_val) && (oth_lane < max_lane));
        max_val  = take_other ? oth_val  : max_val;
        max_lane = take_other ? oth_lane : max_lane;
    }
    // Broadcast winner lane to all lanes
    int winner_lane = __shfl_sync(mask, max_lane, 0);
    return winner_lane;
}

// Merge the candidate buffer (cand_count entries) into the top-k list stored in registers (best_dist/best_idx).
// - k: total top-k size
// - kpt: number of elements per thread (k/32)
// - best_dist/idx: per-thread arrays of size kpt
// - cand_dist/idx: per-warp arrays in shared memory of size k
// - warp_maxdist: in/out distance of the current k-th neighbor (global worst among the top-k)
template <int KPT_MAX>
static inline __device__ void merge_candidates_warp(
    int k,
    int cand_count,
    const float* __restrict__ cand_dist,
    const int*   __restrict__ cand_idx,
    float (&best_dist)[KPT_MAX],
    int   (&best_idx)[KPT_MAX],
    int kpt,
    float &warp_maxdist
) {
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = lane_id();

    if (cand_count <= 0) return;

    // Process candidates sequentially with warp cooperation
    for (int c = 0; c < cand_count; ++c) {
        float cd = cand_dist[c];
        int   ci = cand_idx[c];

        // Only attempt to insert if strictly better than current k-th
        if (cd < warp_maxdist) {
            // Each lane computes its local maximum among its kpt entries and remembers the slot index
            float local_max = -CUDART_INF_F;
            int local_slot = 0;
            #pragma unroll
            for (int s = 0; s < KPT_MAX; ++s) {
                if (s < kpt) {
                    float v = best_dist[s];
                    if (v > local_max) {
                        local_max = v;
                        local_slot = s;
                    }
                }
            }

            // Find which lane currently holds the global maximum
            int max_owner_lane = warp_argmax_lane(local_max, lane, full_mask);
            float max_val = warp_bcast(local_max, max_owner_lane, full_mask);

            // If the candidate is better than the current global worst, replace it
            if (cd < max_val) {
                // Only the owner lane updates its own arrays at the recorded slot
                if (lane == max_owner_lane) {
                    best_dist[local_slot] = cd;
                    best_idx[local_slot]  = ci;
                }
                // Recompute the new global worst (warp_maxdist) after replacement
                float new_local_max = -CUDART_INF_F;
                #pragma unroll
                for (int s = 0; s < KPT_MAX; ++s) {
                    if (s < kpt) {
                        new_local_max = fmaxf(new_local_max, best_dist[s]);
                    }
                }
                warp_maxdist = warp_reduce_max(new_local_max, full_mask);
            }
        }
    }
}

// Bitonic sort (ascending by distance) over per-warp arrays in shared memory.
// length == k is a power of two (32..1024).
static inline __device__ void bitonic_sort_pairs_shared(float* dist, int* idx, int length) {
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = lane_id();
    for (int size = 2; size <= length; size <<= 1) {
        // "size" is the size of the bitonic sequence
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < length; i += 32) {
                int partner = i ^ stride;
                if (partner > i) {
                    bool up = ((i & size) == 0); // ascending in the first half
                    float a = dist[i];
                    float b = dist[partner];
                    int ia = idx[i];
                    int ib = idx[partner];
                    bool swap = up ? (a > b) : (a < b);
                    if (swap) {
                        dist[i]      = b;
                        dist[partner]= a;
                        idx[i]       = ib;
                        idx[partner] = ia;
                    }
                }
            }
            __syncwarp(full_mask);
        }
    }
}

template<int WARPS_PER_BLOCK, int TILE_POINTS>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    int k,
    PairIF* __restrict__ result
) {
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = lane_id();
    const int warp_in_block = threadIdx.x >> 5;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    const bool warp_active = (global_warp_id < query_count);
    const int kpt = k >> 5; // k / 32
    const int KPT_MAX = 1024 / 32; // 32

    extern __shared__ unsigned char smem_raw[];
    auto align8 = [](size_t x) { return (x + 7) & ~size_t(7); };

    // Lay out shared memory:
    size_t smem_off = 0;

    // Tile of data points
    float2* tile = reinterpret_cast<float2*>(smem_raw + smem_off);
    smem_off += TILE_POINTS * sizeof(float2);
    smem_off = align8(smem_off);

    // Per-warp candidate buffers (indices and distances)
    int* cand_idx_base   = reinterpret_cast<int*>(smem_raw + smem_off);
    smem_off += WARPS_PER_BLOCK * size_t(k) * sizeof(int);
    smem_off = align8(smem_off);

    float* cand_dist_base = reinterpret_cast<float*>(smem_raw + smem_off);
    smem_off += WARPS_PER_BLOCK * size_t(k) * sizeof(float);
    smem_off = align8(smem_off);

    int* cand_count_base = reinterpret_cast<int*>(smem_raw + smem_off);
    smem_off += WARPS_PER_BLOCK * sizeof(int);
    // No further align needed

    // Pointers to this warp's candidate buffer
    int*   cand_idx  = cand_idx_base  + warp_in_block * k;
    float* cand_dist = cand_dist_base + warp_in_block * k;
    volatile int* cand_count = cand_count_base;

    // Initialize per-warp candidate count
    if (lane == 0) cand_count[warp_in_block] = 0;

    // Private intermediate result in registers (distributed across lanes)
    int   best_idx[32];   // KPT_MAX = 32 (max k/32)
    float best_dist[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (i < kpt) {
            best_idx[i] = -1;
            best_dist[i] = CUDART_INF_F;
        }
    }

    // Load query point for this warp
    float2 q = make_float2(0.0f, 0.0f);
    if (warp_active) {
        q = query[global_warp_id];
    }

    // Initialize warp-level max distance (k-th neighbor distance)
    float warp_maxdist = CUDART_INF_F;

    // Process data in tiles
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        int tile_size = data_count - base;
        if (tile_size > TILE_POINTS) tile_size = TILE_POINTS;

        // Block-cooperative load of tile into shared memory
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[base + i];
        }
        __syncthreads();

        if (warp_active) {
            // Each warp processes all points in the tile
            for (int j = lane; j < tile_size; j += 32) {
                float2 p = tile[j];
                float dx = q.x - p.x;
                float dy = q.y - p.y;
                float dist = dx * dx + dy * dy;

                // Determine which lanes have candidates closer than current max
                bool is_active = (dist < warp_maxdist);
                unsigned active_mask = __ballot_sync(full_mask, is_active);
                if (active_mask) {
                    int leader = __ffs(active_mask) - 1;

                    int local_rank = 0;
                    if (is_active) {
                        unsigned prior = active_mask & ((1u << lane) - 1u);
                        local_rank = __popc(prior);
                    }
                    int num_active = __popc(active_mask);

                    int base_pos = 0;
                    if (lane == leader) {
                        base_pos = atomicAdd((int*)&cand_count[warp_in_block], num_active);
                    }
                    base_pos = __shfl_sync(active_mask, base_pos, leader);

                    int remaining = k - base_pos;
                    int take = remaining > 0 ? (remaining < num_active ? remaining : num_active) : 0;
                    int take_bcast = __shfl_sync(active_mask, take, leader);

                    // Write the first 'take' candidates into the buffer
                    if (is_active && local_rank < take_bcast) {
                        int pos = base_pos + local_rank;
                        cand_idx[pos]  = base + j;
                        cand_dist[pos] = dist;
                    }

                    // If buffer is full, merge it with intermediate result
                    bool need_merge = (base_pos + num_active) >= k;
                    unsigned merge_mask = __ballot_sync(full_mask, need_merge);
                    if (merge_mask) {
                        __syncwarp(full_mask);
                        if (lane == 0) {
                            // Clamp count to k (should be >= k now)
                            int cc = cand_count[warp_in_block];
                            if (cc > k) cand_count[warp_in_block] = k;
                        }
                        __syncwarp(full_mask);

                        int cc = cand_count[warp_in_block];
                        merge_candidates_warp<32>(k, cc, cand_dist, cand_idx, best_dist, best_idx, kpt, warp_maxdist);

                        if (lane == 0) {
                            cand_count[warp_in_block] = 0;
                        }
                        __syncwarp(full_mask);

                        // Handle leftover candidates from this iteration that didn't fit before merge
                        bool had_leftover = is_active && (local_rank >= take_bcast) && (dist < warp_maxdist);
                        unsigned leftover_mask = __ballot_sync(full_mask, had_leftover);
                        if (leftover_mask) {
                            int leader2 = __ffs(leftover_mask) - 1;
                            int n2 = __popc(leftover_mask);
                            int local_rank2 = 0;
                            if (had_leftover) {
                                unsigned prior2 = leftover_mask & ((1u << lane) - 1u);
                                local_rank2 = __popc(prior2);
                            }
                            int base2 = 0;
                            if (lane == leader2) {
                                base2 = atomicAdd((int*)&cand_count[warp_in_block], n2);
                            }
                            base2 = __shfl_sync(leftover_mask, base2, leader2);
                            if (had_leftover) {
                                int pos2 = base2 + local_rank2;
                                cand_idx[pos2]  = base + j;
                                cand_dist[pos2] = dist;
                            }
                        }
                    }
                }
            }
        }

        __syncthreads(); // ensure all warps done with the tile before loading next
    }

    // After processing all tiles, merge any remaining candidates
    if (warp_active) {
        int cc = cand_count[warp_in_block];
        if (cc > 0) {
            merge_candidates_warp<32>(k, cc, cand_dist, cand_idx, best_dist, best_idx, kpt, warp_maxdist);
            if (lane == 0) cand_count[warp_in_block] = 0;
            __syncwarp(full_mask);
        }

        // Gather the top-k from registers into the per-warp shared buffer (reuse it for sorting)
        // Layout: position pos = s * 32 + lane for s in [0, kpt)
        #pragma unroll
        for (int s = 0; s < 32; ++s) {
            if (s < kpt) {
                int pos = s * 32 + lane;
                cand_idx[pos]  = best_idx[s];
                cand_dist[pos] = best_dist[s];
            }
        }
        __syncwarp(full_mask);

        // Sort ascending by distance using bitonic sort
        bitonic_sort_pairs_shared(cand_dist, cand_idx, k);

        // Write results to global memory
        PairIF* out = result + global_warp_id * k;
        for (int pos = lane; pos < k; pos += 32) {
            out[pos].first  = cand_idx[pos];
            out[pos].second = cand_dist[pos];
        }
    }
}

// Host interface
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Choose algorithm parameters
    constexpr int WARPS_PER_BLOCK = 8;   // 8 warps -> 256 threads per block
    constexpr int TILE_POINTS     = 4096; // 4096 points per shared-memory tile (32 KB)

    // Grid configuration
    int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 block_dim(WARPS_PER_BLOCK * 32, 1, 1);
    dim3 grid_dim(blocks, 1, 1);

    // Compute dynamic shared memory size
    auto align8 = [](size_t x) { return (x + 7) & ~size_t(7); };
    size_t smem_size = 0;
    smem_size += TILE_POINTS * sizeof(float2);
    smem_size = align8(smem_size);
    smem_size += WARPS_PER_BLOCK * size_t(k) * sizeof(int);
    smem_size = align8(smem_size);
    smem_size += WARPS_PER_BLOCK * size_t(k) * sizeof(float);
    smem_size = align8(smem_size);
    smem_size += WARPS_PER_BLOCK * sizeof(int);

    // Set attribute to allow large dynamic shared memory usage if needed
    cudaFuncSetAttribute(knn_kernel<WARPS_PER_BLOCK, TILE_POINTS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_size));

    // Launch kernel
    PairIF* result_if = reinterpret_cast<PairIF*>(result);
    knn_kernel<WARPS_PER_BLOCK, TILE_POINTS><<<grid_dim, block_dim, smem_size>>>(
        query, query_count, data, data_count, k, result_if);
}