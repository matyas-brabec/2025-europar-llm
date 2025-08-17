#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>

// This implementation uses one warp (32 threads) per query to compute k-NN over 2D points.
// It processes the dataset in tiles cached in shared memory. Each warp maintains:
// - a private, distributed intermediate top-k list in registers (k/32 entries per thread)
// - a per-warp candidate buffer in shared memory with capacity k (indices and distances) and an atomic counter
// When the candidate buffer fills, the warp merges the buffer with its intermediate result
// using a cooperative, warp-parallel bitonic sort on 2k elements in shared memory, then updates max_distance.
// After all tiles are processed, any remaining candidates are merged, and final sorted k-NN are written out.
//
// Assumptions:
// - k is a power of two in [32, 1024]
// - data_count >= k
// - Arrays 'query', 'data', and 'result' are allocated with cudaMalloc
// - H100/A100-class GPU, latest CUDA toolkit
//
// Hyperparameters:
// - Warps per block: 4 (128 threads)
// - Tile size: 2048 points cached in shared memory (16KB)
// - Dynamic shared memory is used for tile cache and per-warp buffers/workspaces

#ifndef FULL_WARP_MASK
#define FULL_WARP_MASK 0xFFFFFFFFu
#endif

// Tunables
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
static constexpr int TILE_POINTS = 2048;
static constexpr int MAX_K = 1024;
static constexpr int MAX_L_PER_THREAD = MAX_K / 32;

// Compute squared Euclidean distance between 2D points
__device__ __forceinline__ float squared_l2(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // Fused multiply-add for accuracy and throughput
    return fmaf(dx, dx, dy * dy);
}

// Warp-wide reduction: maximum
__device__ __forceinline__ float warpReduceMax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(FULL_WARP_MASK, v, offset));
    }
    return v;
}

// In-warp parallel bitonic sort on shared arrays of length 'N' (must be power-of-two).
// Sorts ascending by distance; keeps indices aligned to distances.
// Each warp cooperatively sorts its own window. Threads process elements in strides of 32.
__device__ __forceinline__ void warp_bitonic_sort_asc(float* dist, int* idx, int N) {
    // Standard bitonic network over N elements. We assume exactly one warp participates.
    for (int size = 2; size <= N; size <<= 1) {
        // Merge bitonic sequences of length 'size'
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x & 31; i < N; i += 32) {
                int j = i ^ stride;
                if (j > i) {
                    // Determine sort direction for this pair
                    bool ascending = ((i & size) == 0);
                    float ai = dist[i];
                    float aj = dist[j];
                    int ii = idx[i];
                    int ij = idx[j];
                    // Compare-swap
                    bool swap = (ai > aj) == ascending;
                    if (swap) {
                        dist[i] = aj; dist[j] = ai;
                        idx[i]  = ij; idx[j]  = ii;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge per-warp candidate buffer (cand_dist/cand_idx, count) with the warp's current
// intermediate top-k held in registers (top_dist/top_idx, L entries per thread).
// Uses a bitonic sort on 2k elements in shared memory, then takes the k smallest.
// Updates the warp's max_distance and clears the candidate count.
__device__ void warp_merge_buffer_with_topk(
    const int lane,
    const int warp_local_id,
    const int k,
    const int L, // k/32
    // Per-thread distributed top-k
    float top_dist[MAX_L_PER_THREAD],
    int   top_idx [MAX_L_PER_THREAD],
    // Shared memory workspace for this warp
    float* sort_dist, // length 2k
    int*   sort_idx,  // length 2k
    // Candidate buffer for this warp in shared memory
    volatile float* cand_dist, // length k
    volatile int*   cand_idx,  // length k
    // Candidate count and max distance arrays in shared memory
    volatile int* s_cand_count,
    volatile float* s_max_dist
) {
    // Read current candidate count (safe via atomicAdd with 0)
    int cand_count = 0;
    if (lane == 0) {
        cand_count = atomicAdd((int*)&s_cand_count[warp_local_id], 0);
    }
    cand_count = __shfl_sync(FULL_WARP_MASK, cand_count, 0);

    // Prepare combined array of size 2k: first k = current top-k, next cand_count = candidates, rest = +INF
    // Fill top-k part directly from per-thread registers
    for (int i = 0; i < L; ++i) {
        int pos = i * 32 + lane;
        // pos in [0, k-1]
        sort_dist[pos] = top_dist[i];
        sort_idx[pos]  = top_idx[i];
    }
    __syncwarp();

    // Fill candidates into positions [k .. k + cand_count - 1]
    for (int i = lane; i < cand_count; i += 32) {
        sort_dist[k + i] = cand_dist[i];
        sort_idx [k + i] = cand_idx [i];
    }
    __syncwarp();

    // Fill remaining positions [k + cand_count .. 2k-1] with +INF (ignored after sort)
    const int totalN = 2 * k;
    for (int i = k + cand_count + lane; i < totalN; i += 32) {
        sort_dist[i] = CUDART_INF_F;
        sort_idx [i] = -1;
    }
    __syncwarp();

    // Bitonic sort ascending
    warp_bitonic_sort_asc(sort_dist, sort_idx, totalN);

    // Take first k elements as new top-k and distribute back to registers (sorted ascending)
    for (int i = 0; i < L; ++i) {
        int pos = i * 32 + lane; // 0..k-1
        top_dist[i] = sort_dist[pos];
        top_idx [i] = sort_idx [pos];
    }

    // Update max_distance to the (k-1)-th element (last of top-k)
    float new_max = 0.0f;
    if (lane == 0) {
        new_max = sort_dist[k - 1];
        s_max_dist[warp_local_id] = new_max;
        // Reset candidate count to zero
        s_cand_count[warp_local_id] = 0;
    }
    __syncwarp();
}

// Main kernel
__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    std::pair<int, float>* __restrict__ result,
    int k
) {
    // One warp per query
    const int lane = threadIdx.x & 31;
    const int warp_id_in_block = threadIdx.x >> 5;
    const int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    if (global_warp_id >= query_count) return;

    // Number of (index, distance) pairs per thread in the warp's distributed top-k
    const int L = k >> 5; // k / 32; since k is a power of two >= 32

    // Shared memory layout:
    // [0 .. TILE_POINTS-1]: float2 tile cache
    // Then per-warp regions of size perWarpBytes = 24*k bytes:
    //   cand_idx[k], cand_dist[k], sort_idx[2k], sort_dist[2k]
    extern __shared__ unsigned char smem[];
    float2* s_points = reinterpret_cast<float2*>(smem);
    unsigned char* s_after_points = reinterpret_cast<unsigned char*>(s_points + TILE_POINTS);

    const size_t perWarpBytes =
        (size_t)k * sizeof(int) + (size_t)k * sizeof(float) + // candidate buffer
        (size_t)(2 * k) * sizeof(int) + (size_t)(2 * k) * sizeof(float); // sort workspace

    unsigned char* myWarpBase = s_after_points + (size_t)warp_id_in_block * perWarpBytes;

    // Candidate buffer for this warp
    int*   s_cand_idx  = reinterpret_cast<int*>(myWarpBase);
    float* s_cand_dist = reinterpret_cast<float*>(s_cand_idx + k);

    // Sort workspace for this warp
    int*   s_sort_idx  = reinterpret_cast<int*>(s_cand_dist + k);
    float* s_sort_dist = reinterpret_cast<float*>(s_sort_idx + 2 * k);

    // Per-warp candidate counts and max distance in static shared memory
    __shared__ int   s_candidate_count[WARPS_PER_BLOCK];
    __shared__ float s_max_distance   [WARPS_PER_BLOCK];

    // Initialize per-warp counters and thresholds
    if (lane == 0) {
        s_candidate_count[warp_id_in_block] = 0;
        s_max_distance   [warp_id_in_block] = CUDART_INF_F;
    }
    __syncwarp();

    // Load this warp's query point
    float2 q = query[global_warp_id];

    // Initialize per-thread portion of the intermediate top-k (private copy per warp, distributed across threads)
    float top_dist[MAX_L_PER_THREAD];
    int   top_idx [MAX_L_PER_THREAD];
    for (int i = 0; i < L; ++i) {
        top_dist[i] = CUDART_INF_F;
        top_idx [i] = -1;
    }

    // Process dataset in tiles cached in shared memory
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        const int tile_len = min(TILE_POINTS, data_count - tile_start);

        // Load tile into shared memory cooperatively by the whole block
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            s_points[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes all points in the tile in chunks of 32 (one per lane)
        for (int t = 0; t < tile_len; t += 32) {
            const int local_idx = t + lane;
            const bool valid = (local_idx < tile_len);

            // Snapshot of current threshold for this warp to filter candidates
            float max_threshold = s_max_distance[warp_id_in_block];

            // Compute distance to candidate point
            float d = CUDART_INF_F;
            int   gidx = -1;
            if (valid) {
                float2 p = s_points[local_idx];
                d = squared_l2(p, q);
                gidx = tile_start + local_idx;
            }

            // Filter using current threshold
            const bool qualifies = valid && (d < max_threshold);

            // Warp ballot to compact qualifying candidates in this chunk
            unsigned mask = __ballot_sync(FULL_WARP_MASK, qualifies);
            int qcount = __popc(mask);
            if (qcount > 0) {
                // Position among qualifiers in this chunk
                unsigned lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
                int my_offset = __popc(mask & lane_mask);

                // Reserve space in candidate buffer for all qualifiers in this chunk
                int base = 0;
                if (lane == 0) {
                    base = atomicAdd(&s_candidate_count[warp_id_in_block], qcount);
                }
                base = __shfl_sync(FULL_WARP_MASK, base, 0);

                // How many can we insert before reaching capacity k?
                int allowed = 0;
                if (base < k) {
                    int remaining = k - base;
                    allowed = remaining < qcount ? remaining : qcount;
                }

                // Insert the first 'allowed' qualifiers
                if (qualifies && my_offset < allowed) {
                    int pos = base + my_offset;
                    s_cand_dist[pos] = d;
                    s_cand_idx [pos] = gidx;
                }
                __syncwarp();

                // If overflow occurred, merge buffer with intermediate result and insert leftovers
                if (qcount > allowed) {
                    // Merge now (buffer is full)
                    warp_merge_buffer_with_topk(
                        lane, warp_id_in_block, k, L,
                        top_dist, top_idx,
                        s_sort_dist, s_sort_idx,
                        s_cand_dist, s_cand_idx,
                        s_candidate_count, s_max_distance
                    );
                    __syncwarp();

                    // Prepare leftover mask by removing the first 'allowed' set bits from mask
                    unsigned leftover_mask = mask;
                    if (lane == 0) {
                        int to_skip = allowed;
                        while (to_skip-- > 0) {
                            // Clear least significant set bit
                            leftover_mask &= (leftover_mask - 1u);
                        }
                    }
                    leftover_mask = __shfl_sync(FULL_WARP_MASK, leftover_mask, 0);
                    int leftover_count = qcount - allowed;

                    // Reserve space for leftovers (fresh buffer, count=0 after merge)
                    int base2 = 0;
                    if (lane == 0) {
                        base2 = atomicAdd(&s_candidate_count[warp_id_in_block], leftover_count);
                    }
                    base2 = __shfl_sync(FULL_WARP_MASK, base2, 0);

                    // Insert leftovers contiguously
                    bool is_leftover = ((leftover_mask >> lane) & 1u) != 0u;
                    if (is_leftover) {
                        int my_off2 = __popc(leftover_mask & lane_mask);
                        int pos2 = base2 + my_off2;
                        s_cand_dist[pos2] = d;
                        s_cand_idx [pos2] = gidx;
                    }
                    __syncwarp();

                    // If the buffer is exactly full after inserting leftovers, merge immediately
                    int cnt_after = 0;
                    if (lane == 0) {
                        cnt_after = atomicAdd(&s_candidate_count[warp_id_in_block], 0);
                    }
                    cnt_after = __shfl_sync(FULL_WARP_MASK, cnt_after, 0);
                    if (cnt_after >= k) {
                        warp_merge_buffer_with_topk(
                            lane, warp_id_in_block, k, L,
                            top_dist, top_idx,
                            s_sort_dist, s_sort_idx,
                            s_cand_dist, s_cand_idx,
                            s_candidate_count, s_max_distance
                        );
                    }
                }
            }

            __syncwarp();
        }

        __syncthreads(); // ensure all warps finished reading this tile before loading the next
    }

    // After all tiles, if there are remaining candidates, merge them
    int final_count = 0;
    if (lane == 0) {
        final_count = atomicAdd(&s_candidate_count[warp_id_in_block], 0);
    }
    final_count = __shfl_sync(FULL_WARP_MASK, final_count, 0);
    if (final_count > 0) {
        warp_merge_buffer_with_topk(
            lane, warp_id_in_block, k, L,
            top_dist, top_idx,
            s_sort_dist, s_sort_idx,
            s_cand_dist, s_cand_idx,
            s_candidate_count, s_max_distance
        );
    }

    // Write final results for this query in ascending order of distance
    // Each thread writes its L entries at positions [i*32 + lane]
    int out_base = global_warp_id * k;
    for (int i = 0; i < L; ++i) {
        int pos = i * 32 + lane;
        int out_idx = out_base + pos;
        result[out_idx].first  = top_idx[i];
        result[out_idx].second = top_dist[i];
    }
}

// Host interface
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Launch parameters
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Dynamic shared memory size: tile + per-warp buffers/workspaces
    size_t tile_bytes = (size_t)TILE_POINTS * sizeof(float2);
    size_t per_warp_bytes =
        (size_t)k * sizeof(int) + (size_t)k * sizeof(float) +   // candidate buffer
        (size_t)(2 * k) * sizeof(int) + (size_t)(2 * k) * sizeof(float); // sort workspace
    size_t shared_bytes = tile_bytes + WARPS_PER_BLOCK * per_warp_bytes;

    // Opt-in to larger dynamic shared memory if necessary
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);

    // Launch
    knn_kernel<<<grid, block, shared_bytes>>>(query, query_count, data, data_count, result, k);
}