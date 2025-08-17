#include <cuda_runtime.h>
#include <stdint.h>
#include <utility>
#include <limits>

// This implementation computes k-NN for 2D points using one warp (32 threads) per query.
// Each warp maintains an intermediate top-k result in registers, with each thread
// storing k/32 consecutive elements. Candidates are collected into a per-warp buffer
// in shared memory and merged into the intermediate result when the buffer is full.
// Distance computations are processed in batches: each block cooperatively loads a
// batch of data points into shared memory, then each warp computes distances from its
// own query to all cached data points.
//
// Sorting and merging of the distributed k-size register arrays use a warp-level
// bitonic sort network. Cross-lane exchanges rely on __shfl_xor_sync; within-lane
// exchanges swap thread-local registers.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Maximum k supported by problem statement is 1024 => k/32 <= 32 entries per lane.
#define MAX_KPW 32

// Structure for candidate pairs in shared memory (AoS form: 8 bytes each)
struct Candidate {
    float dist;
    int   idx;
};

// Utility: get lane id and warp id within the block
static inline __device__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
static inline __device__ int warp_id_in_block() { return threadIdx.x >> 5; }

// Utility: fast squared L2 distance between two float2 points
static inline __device__ float squared_l2(const float2 &a, const float2 &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Warp-level bitonic sort over k = KPW * 32 elements distributed so that lane L stores
// local arrays arr[0..KPW-1] corresponding to global indices i = L*KPW + t.
// The algorithm follows the standard bitonic sort network:
// - For j < KPW, partner is within the same thread; swap thread-local registers.
// - For j >= KPW, partner is in another lane partner_lane = lane ^ (j / KPW), but the
//   register index t is the same in both lanes. Use __shfl_xor_sync to exchange.
// Sorting is ascending by distance, with index as payload moved alongside.
static inline __device__ void warp_bitonic_sort_asc(float dist_local[MAX_KPW],
                                                    int   idx_local[MAX_KPW],
                                                    int   KPW, int k)
{
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = lane_id();

    // Outer stages: kk is the size of the subsequences being merged (2,4,8,...,k)
    for (int kk = 2; kk <= k; kk <<= 1) {
        // Inner stages: jj halves at each iteration
        for (int jj = kk >> 1; jj > 0; jj >>= 1) {
            if (jj >= KPW) {
                // Cross-lane compare-exchange: partner is in another lane, same local index
                int partner_delta = jj / KPW; // XOR distance across lanes
#pragma unroll
                for (int t = 0; t < MAX_KPW; ++t) {
                    if (t >= KPW) break;

                    float a = dist_local[t];
                    int   ia = idx_local[t];

                    // Fetch partner lane's value at the same local index t
                    float b = __shfl_xor_sync(full_mask, a, partner_delta);
                    int   ib = __shfl_xor_sync(full_mask, ia, partner_delta);

                    // Compute global index i for this element
                    int i_global = lane * KPW + t;

                    // Direction of the bitonic compare (ascending in lower half of kk)
                    bool ascending = ((i_global & kk) == 0);
                    // Lower half of the jj-pair (bit jj cleared)
                    bool lower = ((i_global & jj) == 0);

                    // Select min or max depending on ascending and position
                    // If ascending and lower -> take min; ascending and upper -> take max
                    // If descending and lower -> take max; descending and upper -> take min
                    bool take_min = (ascending == lower);

                    // Compare a and b
                    bool a_le_b = (a <= b);
                    float sel_d = take_min ? (a_le_b ? a : b) : (a_le_b ? b : a);
                    int   sel_i = take_min ? (a_le_b ? ia : ib) : (a_le_b ? ib : ia);

                    dist_local[t] = sel_d;
                    idx_local[t]  = sel_i;
                }
                __syncwarp();
            } else {
                // Within-lane compare-exchange: pairs of thread-local registers
#pragma unroll
                for (int t = 0; t < MAX_KPW; ++t) {
                    if (t >= KPW) break;
                    // Process each pair only once: only when (t & jj) == 0
                    if ((t & jj) == 0) {
                        int partner = t ^ jj;

                        float a = dist_local[t];
                        int   ia = idx_local[t];
                        float b = dist_local[partner];
                        int   ib = idx_local[partner];

                        int i_global = lane * KPW + t;
                        bool ascending = ((i_global & kk) == 0);

                        // Compute min/max of the pair
                        bool a_le_b = (a <= b);
                        float min_d = a_le_b ? a : b;
                        int   min_i = a_le_b ? ia : ib;
                        float max_d = a_le_b ? b : a;
                        int   max_i = a_le_b ? ib : ia;

                        if (ascending) {
                            dist_local[t]       = min_d;
                            idx_local[t]        = min_i;
                            dist_local[partner] = max_d;
                            idx_local[partner]  = max_i;
                        } else {
                            dist_local[t]       = max_d;
                            idx_local[t]        = max_i;
                            dist_local[partner] = min_d;
                            idx_local[partner]  = min_i;
                        }
                    }
                }
                __syncwarp();
            }
        }
    }
}

// Merge function: merge the per-warp candidate buffer (in shared memory) into the
// intermediate result stored in registers, following the specification:
// 0) Intermediate result in registers is sorted ascending.
// 1) Swap content of the buffer and the intermediate result so that the buffer is in registers:
//    - Load up to k candidates from shared memory into buf_* registers (pad with +inf if needed).
//    - Store the current intermediate result from registers into the shared buffer region (k elements).
// 2) Sort the buffer (buf_*) in ascending order using warp bitonic sort.
// 3) Construct a bitonic sequence in registers: for global index i, take
//    min(buf[i], shared_result[k-1-i]) into res registers. This yields the k best of the union.
// 4) Sort the resulting bitonic sequence ascending in registers to update the intermediate result.
// The function returns the updated max_distance (distance of the k-th neighbor).
static inline __device__ float merge_buffer_into_result(
    Candidate *warp_buf, int buf_count, // shared memory buffer and existing candidate count
    float res_d[MAX_KPW], int res_i[MAX_KPW], // intermediate result in registers (sorted)
    float buf_d[MAX_KPW], int buf_i[MAX_KPW], // scratch registers for buffer
    int KPW, int k)
{
    const int lane = lane_id();

    // Step 1: Swap: load buffer into buf_* regs (pad with INF), store current result into shared buffer
#pragma unroll
    for (int t = 0; t < MAX_KPW; ++t) {
        if (t >= KPW) break;
        int i_global = lane * KPW + t;

        // Load candidate or +inf if beyond buf_count
        float d = (i_global < buf_count) ? warp_buf[i_global].dist : CUDART_INF_F;
        int   idx = (i_global < buf_count) ? warp_buf[i_global].idx  : -1;
        buf_d[t] = d;
        buf_i[t] = idx;

        // Store current result into shared buffer (swap)
        warp_buf[i_global].dist = res_d[t];
        warp_buf[i_global].idx  = res_i[t];
    }
    __syncwarp();

    // Step 2: Sort buffer in ascending order
    warp_bitonic_sort_asc(buf_d, buf_i, KPW, k);

    // Step 3: Merge via bitonic min pairing: res[i] = min(buf[i], shared_res[k-1-i])
#pragma unroll
    for (int t = 0; t < MAX_KPW; ++t) {
        if (t >= KPW) break;

        int i_global = lane * KPW + t;
        int j_global = k - 1 - i_global;

        // Read the swapped-out intermediate result from shared memory at reversed index
        float od = warp_buf[j_global].dist;
        int   oi = warp_buf[j_global].idx;

        // Take the minimum across the pair (ties arbitrarily resolved)
        float bd = buf_d[t];
        int   bi = buf_i[t];

        if (od < bd) {
            res_d[t] = od;
            res_i[t] = oi;
        } else {
            res_d[t] = bd;
            res_i[t] = bi;
        }
    }
    __syncwarp();

    // Step 4: Final sort of the bitonic sequence in ascending order
    warp_bitonic_sort_asc(res_d, res_i, KPW, k);

    // Update and return max_distance (the last element of the sorted result, i.e., index k-1)
    // The last element resides in lane 31 at local index KPW-1. Broadcast from lane 31.
    float local_last = res_d[KPW - 1];
    float max_dist = __shfl_sync(0xFFFFFFFFu, local_last, WARP_SIZE - 1);
    return max_dist;
}

// CUDA kernel. Each warp processes one query point.
__global__ void knn2d_kernel(const float2 * __restrict__ query, int query_count,
                             const float2 * __restrict__ data,  int data_count,
                             std::pair<int, float> * __restrict__ result,
                             int k, int tile_points)
{
    extern __shared__ unsigned char smem_bytes[];
    unsigned char *smem_ptr = smem_bytes;

    // Layout of shared memory:
    // [0 .. tile_points-1] float2 cached points
    float2 *smem_points = reinterpret_cast<float2*>(smem_ptr);
    smem_ptr += sizeof(float2) * (size_t)tile_points;

    // Candidate buffers per warp: warps_per_block * k entries
    int warps_per_block = blockDim.x / WARP_SIZE;
    Candidate *smem_cand = reinterpret_cast<Candidate*>(smem_ptr);
    smem_ptr += sizeof(Candidate) * (size_t)warps_per_block * (size_t)k;

    // Per-warp counters (number of candidates currently stored)
    int *smem_counts = reinterpret_cast<int*>(smem_ptr);
    // No further shared memory needed after this.

    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = lane_id();
    const int warp_in_block = warp_id_in_block();
    const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
    const int query_idx = global_warp;

    if (query_idx >= query_count) return;

    // Determine KPW given k (power of two, between 32 and 1024)
    const int KPW = k / WARP_SIZE;

    // Pointers to this warp's candidate buffer and its counter in shared memory
    Candidate *warp_buf = smem_cand + (size_t)warp_in_block * (size_t)k;
    volatile int &warp_count = reinterpret_cast<volatile int*>(smem_counts)[warp_in_block];

    // Load query point into registers; lane 0 loads and broadcasts
    float2 q;
    if (lane == 0) q = query[query_idx];
    q.x = __shfl_sync(full_mask, q.x, 0);
    q.y = __shfl_sync(full_mask, q.y, 0);

    // Initialize per-warp candidate count
    if (lane == 0) warp_count = 0;
    __syncwarp();

    // Per-warp intermediate result in registers: k elements distributed across lanes.
    // Initialize with +inf distances and indices = -1; this array is kept sorted.
    float res_d[MAX_KPW];
    int   res_i[MAX_KPW];

#pragma unroll
    for (int t = 0; t < MAX_KPW; ++t) {
        if (t >= KPW) break;
        res_d[t] = CUDART_INF_F;
        res_i[t] = -1;
    }
    // Initial max_distance is +inf
    float max_distance = CUDART_INF_F;

    // Scratch buffer in registers for merging (holds buffer values during swap/sort)
    float buf_d[MAX_KPW];
    int   buf_i[MAX_KPW];

    // Process data in batches: each block loads tile_points points into shared memory
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        int n_this_tile = data_count - tile_start;
        if (n_this_tile > tile_points) n_this_tile = tile_points;

        // Block-wide cooperative load of the tile into shared memory
        for (int i = threadIdx.x; i < n_this_tile; i += blockDim.x) {
            smem_points[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes all points in the tile
        for (int base = 0; base < n_this_tile; base += WARP_SIZE) {
            int idx_in_tile = base + lane;

            // Load a point and compute distance if within bounds
            float dist = CUDART_INF_F;
            int   gidx = -1;
            if (idx_in_tile < n_this_tile) {
                float2 p = smem_points[idx_in_tile];
                dist = squared_l2(q, p);
                gidx = tile_start + idx_in_tile;
            }

            // Filter by current max_distance. Use ballot to count the number of candidates.
            unsigned mask = __ballot_sync(full_mask, (dist < max_distance));

            int n_cands = __popc(mask);
            if (n_cands > 0) {
                // Before inserting, ensure buffer has enough space; if not, flush (merge).
                if (lane == 0) {
                    if (warp_count + n_cands > k) {
                        // Merge current buffer into result and reset count to 0
                        // Note: other lanes wait on this merge
                    }
                }
                __syncwarp();

                if (warp_count + n_cands > k) {
                    // Perform merge (Step 0..4)
                    max_distance = merge_buffer_into_result(warp_buf, warp_count,
                                                            res_d, res_i, buf_d, buf_i,
                                                            KPW, k);
                    if (lane == 0) warp_count = 0;
                    __syncwarp();

                    // Refilter with potentially updated max_distance
                    mask = __ballot_sync(full_mask, (dist < max_distance));
                    n_cands = __popc(mask);
                }

                if (n_cands > 0) {
                    // Reserve space in the buffer: lane 0 updates the counter, broadcasts old value
                    int old_count = 0;
                    if (lane == 0) {
                        old_count = warp_count;
                        warp_count = warp_count + n_cands;
                    }
                    old_count = __shfl_sync(full_mask, old_count, 0);

                    // Compute position for each selected lane using prefix count within the warp
#if __CUDACC_VER_MAJOR__ >= 9
                    unsigned lane_mask_lt = __lanemask_lt();
#else
                    unsigned lane_mask_lt = (1u << lane) - 1u;
#endif
                    int pos_in_warp = __popc(mask & lane_mask_lt);

                    if ((mask >> lane) & 1u) {
                        int store_pos = old_count + pos_in_warp;
                        warp_buf[store_pos].dist = dist;
                        warp_buf[store_pos].idx  = gidx;
                    }
                }

                // If buffer is exactly full after insertion, flush now
                if (lane == 0 && warp_count == k) {
                    // Merge current buffer into result and reset count to 0
                }
                __syncwarp();

                if (warp_count == k) {
                    max_distance = merge_buffer_into_result(warp_buf, warp_count,
                                                            res_d, res_i, buf_d, buf_i,
                                                            KPW, k);
                    if (lane == 0) warp_count = 0;
                    __syncwarp();
                }
            }
        }

        __syncthreads(); // ensure tile is not used anymore before next load
    }

    // After all data processed, if candidates remain in buffer, merge them
    if (warp_count > 0) {
        max_distance = merge_buffer_into_result(warp_buf, warp_count,
                                                res_d, res_i, buf_d, buf_i,
                                                KPW, k);
        if (lane == 0) warp_count = 0;
        __syncwarp();
    }

    // Write out the final top-k for this query (sorted ascending). Each lane writes its KPW entries.
#pragma unroll
    for (int t = 0; t < MAX_KPW; ++t) {
        if (t >= KPW) break;
        int i_global = lane * KPW + t;
        int out_index = query_idx * k + i_global;
        // Guaranteed in-bounds since result has query_count * k elements
        result[out_index] = std::pair<int, float>(res_i[t], res_d[t]);
    }
}

// Host-side runner that configures the kernel launch parameters, shared memory usage,
// and invokes the kernel.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose number of warps per block and tile size based on available dynamic shared memory.
    // Default target: 8 warps per block (256 threads), adjust if necessary.
    int device = 0;
    cudaGetDevice(&device);

    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_smem_optin == 0) {
        // Fallback to legacy per-block limit
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        max_smem_optin = prop.sharedMemPerBlock;
    }

    // Try configurations with decreasing warps per block to fit shared memory comfortably.
    int best_warps = 8; // start with 8 warps per block
    int best_tile_points = 0;
    size_t best_smem_bytes = 0;

    for (int warps = 8; warps >= 1; warps >>= 1) {
        // Candidate buffer bytes for all warps in block
        size_t cand_bytes = (size_t)warps * (size_t)k * sizeof(Candidate);
        size_t count_bytes = (size_t)warps * sizeof(int);

        // Leave some headroom; compute maximum tile points we can fit
        size_t overhead = cand_bytes + count_bytes;
        if (overhead >= (size_t)max_smem_optin) continue;

        size_t bytes_left = (size_t)max_smem_optin - overhead;
        int tile_points = (int)(bytes_left / sizeof(float2));

        // Ensure tile_points is a reasonable multiple of warp size
        if (tile_points > 0) {
            // Clamp tile points to at most data_count (no need to exceed), but keep a good chunk
            if (tile_points > 8192) tile_points = 8192; // avoid over-allocating per block
            // Round down to multiple of WARP_SIZE
            tile_points = (tile_points / WARP_SIZE) * WARP_SIZE;
        }

        if (tile_points >= WARP_SIZE) {
            best_warps = warps;
            best_tile_points = tile_points;
            best_smem_bytes = overhead + (size_t)tile_points * sizeof(float2);
            break;
        }
    }

    if (best_tile_points == 0) {
        // Extremely constrained case; fall back to minimal configuration: 1 warp, 32 points tile
        best_warps = 1;
        best_tile_points = WARP_SIZE;
        best_smem_bytes = (size_t)best_warps * (size_t)k * sizeof(Candidate)
                        + (size_t)best_warps * sizeof(int)
                        + (size_t)best_tile_points * sizeof(float2);
    }

    // Configure kernel launch
    dim3 block(best_warps * WARP_SIZE, 1, 1);
    int num_warps_total = (query_count + 0) / 1; // one warp per query
    int grid_x = (query_count + best_warps - 1) / best_warps;
    dim3 grid(grid_x, 1, 1);

    // Set kernel attribute to allow use of larger dynamic shared memory if needed
    cudaFuncSetAttribute(knn2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)best_smem_bytes);

    // Launch kernel
    knn2d_kernel<<<grid, block, best_smem_bytes>>>(query, query_count, data, data_count, result, k, best_tile_points);

    // Optional: synchronize or leave it to the caller
    // cudaDeviceSynchronize();
}