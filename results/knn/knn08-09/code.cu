#include <cuda_runtime.h>
#include <utility>

/*
Optimized warp-cooperative k-NN for 2D points (squared Euclidean distance).

High-level overview (see inline comments in code for details):
- Each warp (32 threads) handles one query point.
- The block cooperatively loads a tile of data points into shared memory.
- Each warp computes distances from its query to the cached data points and uses a warp ballot to add the closer-than-max_distance points into a per-warp candidate buffer in shared memory.
- When the buffer is full (k elements), the warp merges the buffer with its intermediate result (maintained in registers) using:
    1) Swap buffer and intermediate result (buffer -> registers, result -> shared memory)
    2) Sort the buffer (registers) with warp-level Bitonic Sort
    3) Merge (min) buffer elements with the reversed intermediate result (shared memory) to form a bitonic sequence
    4) Sort the bitonic sequence (registers) with Bitonic Sort -> new intermediate result
- After all tiles, any residual candidates are merged similarly (padding with +inf).
- Each thread keeps k/32 consecutive neighbors in registers. During bitonic sort:
    - For j >= (k/32), swaps are across lanes at the same per-thread index via warp shuffles.
    - For j < (k/32), swaps are within the same thread between its registers.
- The final k neighbors per query are stored in ascending order into the output.

Assumptions per prompt:
- Hardware: NVIDIA A100/H100-class GPUs.
- k is a power-of-two, 32 <= k <= 1024.
- data_count >= k.
- Inputs are large enough to benefit from parallelization.
- Device pointers are pre-allocated via cudaMalloc.

Hyperparameters (chosen to fit within 96KB shared memory per block, while keeping decent tile size and parallelism):
- WARPS_PER_BLOCK = 8 (256 threads per block)
- TILE_POINTS = 3072 (each point is float2 => 8 bytes; 3072*8 = 24KB tile)
- Per-warp candidate buffer: k pairs (index + distance). For k=1024 -> 8KB per warp.
  Total per block: 8 warps * 8KB = 64KB candidates + 24KB tile ≈ 88KB (< 96KB).
*/

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)
#define TILE_POINTS 3072
#define FULL_MASK 0xFFFFFFFFu

// Device utility: squared L2 distance for 2D points
__device__ __forceinline__ float sqr_dist_2d(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Warp-level broadcast of the current max_distance (k-th element).
// Each thread holds E = k/32 values in registers, with consecutive positions.
// The last element (k-1) is stored in lane 31, at local index E-1.
__device__ __forceinline__ float warp_broadcast_max_distance(const float* reg_dist, int E, int lane) {
    float last = (lane == WARP_SIZE - 1) ? reg_dist[E - 1] : 0.0f;
    return __shfl_sync(FULL_MASK, last, WARP_SIZE - 1);
}

// Warp-level Bitonic Sort over k = 32 * E elements.
// Each thread stores E consecutive elements in registers: reg_dist[0..E-1], reg_idx[0..E-1].
// The global linear index of reg element (lane, pos) is i = lane * E + pos.
// - For j >= E: partner is in a different lane but has the same local index (pos). Use __shfl_xor_sync with lane_mask = j / E.
// - For j < E: partner is in the same thread at local index pos ^ j. Swap within registers if pos^j > pos (to avoid double swaps).
__device__ __forceinline__ void warp_bitonic_sort(float* reg_dist, int* reg_idx, int E, int lane) {
    const int K = E * WARP_SIZE;

    // Outer loop: size of subsequences being merged into bitonic sequences
    for (int ksize = 2; ksize <= K; ksize <<= 1) {
        // Inner loop: distance between compared pairs
        for (int j = ksize >> 1; j > 0; j >>= 1) {
            if (j >= E) {
                // Cross-lane compare-exchange at the same local position
                int lane_mask = j / E; // partner lane = lane ^ lane_mask
                for (int p = 0; p < E; ++p) {
                    float v = reg_dist[p];
                    int   id = reg_idx[p];
                    float vp = __shfl_xor_sync(FULL_MASK, v, lane_mask);
                    int   ip = __shfl_xor_sync(FULL_MASK, id, lane_mask);

                    int i_linear = lane * E + p;
                    int l_linear = i_linear ^ j;
                    bool up = ((i_linear & ksize) == 0); // sort direction for this pair

                    float minv = (v < vp) ? v : vp;
                    float maxv = (v < vp) ? vp : v;
                    int   mini = (v < vp) ? id : ip;
                    int   maxi = (v < vp) ? ip : id;

                    bool lower = (i_linear < l_linear);
                    if (up) {
                        reg_dist[p] = lower ? minv : maxv;
                        reg_idx[p]  = lower ? mini : maxi;
                    } else {
                        reg_dist[p] = lower ? maxv : minv;
                        reg_idx[p]  = lower ? maxi : mini;
                    }
                }
            } else {
                // Intra-lane compare-exchange between positions p and p^j
                for (int p = 0; p < E; ++p) {
                    int q = p ^ j;
                    if (q > p) {
                        float a = reg_dist[p];
                        float b = reg_dist[q];
                        int   ia = reg_idx[p];
                        int   ib = reg_idx[q];

                        int i_linear = lane * E + p;
                        bool up = ((i_linear & ksize) == 0);
                        // Compare-swap depending on direction
                        if (a < b) {
                            if (!up) {
                                // descending: swap to place larger at p
                                reg_dist[p] = b; reg_idx[p] = ib;
                                reg_dist[q] = a; reg_idx[q] = ia;
                            }
                            // else ascending: already ok
                        } else {
                            if (up) {
                                // ascending: swap to place smaller at p
                                reg_dist[p] = b; reg_idx[p] = ib;
                                reg_dist[q] = a; reg_idx[q] = ia;
                            }
                            // else descending: already ok
                        }
                    }
                }
            }
        }
    }
}

// Flush and merge the per-warp candidate buffer with the intermediate result stored in registers.
// Steps:
//   0) The current intermediate result in registers is sorted ascending.
//   1) Swap: bring the candidate buffer (shared memory) into registers; write the register result into the buffer.
//   2) Sort the register buffer ascending via bitonic sort.
//   3) Merge-by-min into the registers: reg[i] = min(reg[i], buffer_shared[k-1-i]) to form a bitonic sequence of length k.
//   4) Sort the registers ascending via bitonic sort to obtain the updated intermediate result.
// After merge, the per-warp candidate count is reset to 0 and max_distance is updated.
//
// Arguments:
//   s_buf_dist, s_buf_idx: base pointers to shared memory candidate buffers for this warp (length k each)
//   cand_count: pointer to the per-warp candidate count in shared memory
//   reg_dist, reg_idx: per-thread register arrays (length E = k/32) storing the intermediate result (sorted ascending)
//   k: number of neighbors
//   E: elements per thread (k/32)
//   lane: lane id within the warp
//   max_distance: reference to per-thread copy of the current max_distance (broadcasted across warp at the end)
//
// Note: This function is warp-synchronous and uses __syncwarp() for intra-warp ordering.
__device__ __forceinline__ void warp_flush_and_merge(float* s_buf_dist, int* s_buf_idx, int* cand_count,
                                                     float* reg_dist, int* reg_idx,
                                                     int k, int E, int lane, float &max_distance) {
    int cc = *cand_count;

    // Pad the remaining buffer slots with +inf and invalid indices (-1)
    if (cc < k) {
        for (int t = lane; t < (k - cc); t += WARP_SIZE) {
            s_buf_dist[cc + t] = CUDART_INF_F;
            s_buf_idx[cc + t] = -1;
        }
    }
    __syncwarp();

    // 1) Swap: buffer (shared) <-> intermediate result (registers)
    //    After this, reg_* holds the candidate buffer and s_buf_* holds the previous intermediate result.
    for (int p = 0; p < E; ++p) {
        int i_linear = lane * E + p;
        float tmpd = s_buf_dist[i_linear];
        int   tmpi = s_buf_idx[i_linear];
        s_buf_dist[i_linear] = reg_dist[p];
        s_buf_idx[i_linear]  = reg_idx[p];
        reg_dist[p] = tmpd;
        reg_idx[p]  = tmpi;
    }
    __syncwarp();

    // 2) Sort the buffer (now in registers) ascending
    warp_bitonic_sort(reg_dist, reg_idx, E, lane);
    __syncwarp();

    // 3) Merge-by-min into registers: reg[i] = min(reg[i], inter_shared[k-1-i])
    for (int p = 0; p < E; ++p) {
        int i_linear = lane * E + p;
        int j_linear = (k - 1) - i_linear;
        float otherd = s_buf_dist[j_linear];
        int   otheri = s_buf_idx[j_linear];
        if (otherd < reg_dist[p]) {
            reg_dist[p] = otherd;
            reg_idx[p]  = otheri;
        }
    }
    __syncwarp();

    // 4) Sort the merged bitonic sequence (registers) ascending
    warp_bitonic_sort(reg_dist, reg_idx, E, lane);
    __syncwarp();

    // Reset candidate count
    if (lane == 0) *cand_count = 0;
    __syncwarp();

    // Update and broadcast max_distance (k-th neighbor distance, i.e., last element)
    max_distance = warp_broadcast_max_distance(reg_dist, E, lane);
    __syncwarp();
}

// Kernel: each warp processes a query. The block loads tiles of data into shared memory.
__global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                           const float2* __restrict__ data, int data_count,
                           std::pair<int, float>* __restrict__ result, int k) {

    // Shared memory layout:
    // [ tile_data (TILE_POINTS) ][ cand_dists (WARPS_PER_BLOCK*k) ][ cand_idxs (WARPS_PER_BLOCK*k) ][ cand_counts (WARPS_PER_BLOCK) ]
    extern __shared__ unsigned char smem[];
    float2* s_data = reinterpret_cast<float2*>(smem);
    float* s_cand_dists = reinterpret_cast<float*>(s_data + TILE_POINTS);
    int*   s_cand_idxs  = reinterpret_cast<int*>(s_cand_dists + WARPS_PER_BLOCK * k);
    int*   s_cand_counts = reinterpret_cast<int*>(s_cand_idxs + WARPS_PER_BLOCK * k);

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_in_block = tid >> 5; // 0..WARPS_PER_BLOCK-1

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_global = (blockIdx.x * warps_per_block) + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;

    const int E = k >> 5; // Elements per thread

    // Base pointers for this warp's candidate buffer within shared memory
    float* s_buf_dist = s_cand_dists + warp_in_block * k;
    int*   s_buf_idx  = s_cand_idxs  + warp_in_block * k;
    int*   s_buf_count = s_cand_counts + warp_in_block;

    // Initialize per-warp candidate count
    if (lane == 0) *s_buf_count = 0;
    __syncwarp();

    // Local registers for intermediate result (sorted ascending invariant)
    float reg_dist[32]; // E <= 32 (since k <= 1024)
    int   reg_idx[32];
    #pragma unroll
    for (int p = 0; p < 32; ++p) {
        if (p < E) {
            reg_dist[p] = CUDART_INF_F;
            reg_idx[p]  = -1;
        }
    }
    __syncwarp();

    // Grid-stride over query warps
    for (int qwarp = warp_global; qwarp < query_count; qwarp += total_warps) {

        // Load query point for this warp
        float2 q;
        if (qwarp < query_count) {
            q = query[qwarp];
        }

        // Reset candidate buffer and intermediate result for this query
        if (lane == 0) *s_buf_count = 0;
        __syncwarp();
        #pragma unroll
        for (int p = 0; p < 32; ++p) {
            if (p < E) {
                reg_dist[p] = CUDART_INF_F;
                reg_idx[p]  = -1;
            }
        }
        __syncwarp();

        // Initialize max_distance to +inf (broadcast)
        float max_distance = warp_broadcast_max_distance(reg_dist, E, lane);

        // Process data in tiles
        for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS) {
            int tile_size = data_count - tile_base;
            if (tile_size > TILE_POINTS) tile_size = TILE_POINTS;

            // Load tile into shared memory cooperatively (whole block)
            for (int i = tid; i < tile_size; i += blockDim.x) {
                s_data[i] = data[tile_base + i];
            }
            __syncthreads(); // ensure tile is loaded before any warp consumes it

            // Each warp processes the tile against its query
            if (qwarp < query_count) {
                // Iterate points in the tile in groups of 32 so that each lane handles one point per iteration
                for (int i = lane; i < tile_size; i += WARP_SIZE) {
                    // Compute squared distance to query
                    float2 d = s_data[i];
                    float dist = sqr_dist_2d(q, d);
                    int   gidx = tile_base + i;

                    // Check candidate against current max_distance
                    int is_cand = (dist < max_distance) ? 1 : 0;

                    // Warp ballot of candidates
                    unsigned mask = __ballot_sync(FULL_MASK, is_cand);
                    int n_cand = __popc(mask);

                    if (n_cand > 0) {
                        // Read current count
                        int cc = 0;
                        if (lane == 0) cc = *s_buf_count;
                        cc = __shfl_sync(FULL_MASK, cc, 0);

                        // If adding all candidates would overflow the buffer, flush-merge first.
                        if (cc + n_cand > k) {
                            // Merge the existing buffer with the intermediate result.
                            warp_flush_and_merge(s_buf_dist, s_buf_idx, s_buf_count,
                                                 reg_dist, reg_idx, k, E, lane, max_distance);
                            // Re-evaluate candidacy with updated max_distance.
                            is_cand = (dist < max_distance) ? 1 : 0;
                            mask = __ballot_sync(FULL_MASK, is_cand);
                            n_cand = __popc(mask);
                            if (n_cand == 0) {
                                // No longer a candidate
                                continue;
                            }
                            if (lane == 0) cc = *s_buf_count;
                            cc = __shfl_sync(FULL_MASK, cc, 0);
                        }

                        // Insert candidates into the buffer at positions [cc, cc + n_cand)
                        int my_offset = __popc(mask & ((1u << lane) - 1));
                        int base = cc;
                        base = __shfl_sync(FULL_MASK, base, 0);
                        if (is_cand) {
                            int pos = base + my_offset;
                            s_buf_dist[pos] = dist;
                            s_buf_idx[pos]  = gidx;
                        }
                        __syncwarp();
                        if (lane == 0) {
                            *s_buf_count = cc + n_cand;
                        }
                        __syncwarp();

                        // If the buffer is now full, flush-merge it immediately.
                        int now_cc = 0;
                        if (lane == 0) now_cc = *s_buf_count;
                        now_cc = __shfl_sync(FULL_MASK, now_cc, 0);
                        if (now_cc == k) {
                            warp_flush_and_merge(s_buf_dist, s_buf_idx, s_buf_count,
                                                 reg_dist, reg_idx, k, E, lane, max_distance);
                        }
                    }
                }
            }

            __syncthreads(); // ensure all warps finished with the tile before loading the next
        }

        // After the last tile, if there are residual candidates, flush-merge them.
        if (qwarp < query_count) {
            int cc_final = 0;
            if (lane == 0) cc_final = *s_buf_count;
            cc_final = __shfl_sync(FULL_MASK, cc_final, 0);
            if (cc_final > 0) {
                warp_flush_and_merge(s_buf_dist, s_buf_idx, s_buf_count,
                                     reg_dist, reg_idx, k, E, lane, max_distance);
            }

            // Write out the final k nearest neighbors for this query.
            // Each thread writes its E consecutive elements.
            int out_base = qwarp * k + (lane * E);
            #pragma unroll
            for (int p = 0; p < 32; ++p) {
                if (p < E) {
                    result[out_base + p].first  = reg_idx[p];
                    result[out_base + p].second = reg_dist[p];
                }
            }
        }

        __syncthreads(); // synchronize before the next query warp (to keep shared memory usage predictable)
    }
}

// Host entry point: launches the kernel.
// - query, data, result are device pointers (allocated via cudaMalloc)
// - k is a power of two, 32 <= k <= 1024
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Launch configuration
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    // Number of blocks to cover all query warps; grid-stride loop will handle excess.
    int warps_needed = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int max_blocks = 65535;
    dim3 grid(warps_needed < max_blocks ? warps_needed : max_blocks, 1, 1);

    // Dynamic shared memory size:
    // tile + per-warp candidates (dists + idxs) + per-warp candidate counts
    size_t smem_size = 0;
    smem_size += sizeof(float2) * TILE_POINTS;
    smem_size += WARPS_PER_BLOCK * k * sizeof(float); // distances
    smem_size += WARPS_PER_BLOCK * k * sizeof(int);   // indices
    smem_size += WARPS_PER_BLOCK * sizeof(int);       // counts

    // Launch
    knn_kernel<<<grid, block, smem_size>>>(query, query_count, data, data_count, result, k);
}