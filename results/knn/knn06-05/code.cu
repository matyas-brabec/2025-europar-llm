#include <cuda_runtime.h>
#include <utility>
#include <cfloat>
#include <cstdint>

// This implementation assigns one warp (32 threads) to each query point.
// The data points are processed in tiles that are cached in shared memory.
// For each query (warp), we keep:
//   - An intermediate result: k best distances + indices (top-k neighbors).
//   - A candidate buffer in shared memory that stores up to k candidates.
//   - A shared integer counter that tracks candidate buffer size.
// When the candidate buffer is full, we merge it with the intermediate
// result using a warp-wide bitonic sort on the union of size 2*k and keep
// the k smallest distances.
//
// The kernel uses atomicAdd on a shared counter to assign positions in the
// per-warp candidate buffer. Distance filtering uses a max_distance
// variable equal to the current k-th (worst) distance in the intermediate
// result. Threads communicate within a warp using __shfl_sync and
// __syncwarp. Data tiling uses __syncthreads across the block.

using Pair = std::pair<int, float>;

constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_BLOCK  = 4;   // 4 warps -> 128 threads per block
constexpr int BLOCK_SIZE       = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int TILE_POINTS      = 2048; // number of data points cached per tile

// Warp-wide merge of candidate buffer into intermediate top-k using bitonic sort.
// Each warp operates only on its own regions in shared memory.
__device__ __forceinline__
void warp_merge_topk(
    int lane,                 // thread's lane ID within the warp (0..31)
    int k,                    // number of neighbors
    int *cand_count_ptr,      // pointer to shared candidate count for this warp
    int *best_indices,        // shared array [k] of current best indices
    float *best_dists,        // shared array [k] of current best distances
    int *cand_indices,        // shared array [k] of candidate indices
    float *cand_dists,        // shared array [k] of candidate distances
    int *merge_indices,       // shared array [2*k] scratch indices
    float *merge_dists        // shared array [2*k] scratch distances
) {
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    int cand_count = 0;
    if (lane == 0) {
        cand_count = *cand_count_ptr;
    }
    cand_count = __shfl_sync(FULL_MASK, cand_count, 0);

    if (cand_count == 0) {
        // Nothing to merge.
        return;
    }

    // Build union of current top-k and candidates into merge_* arrays.
    // First [0..k-1] = current best, then [k..k+cand_count-1] = candidates,
    // remaining slots up to 2*k are padded with FLT_MAX so they end up at
    // the end of the sorted sequence.
    for (int i = lane; i < k; i += WARP_SIZE) {
        merge_dists[i]   = best_dists[i];
        merge_indices[i] = best_indices[i];
    }

    for (int i = lane; i < cand_count; i += WARP_SIZE) {
        merge_dists[k + i]   = cand_dists[i];
        merge_indices[k + i] = cand_indices[i];
    }

    for (int i = lane + k + cand_count; i < 2 * k; i += WARP_SIZE) {
        merge_dists[i]   = FLT_MAX;
        merge_indices[i] = -1;
    }

    __syncwarp();

    // Bitonic sort on 2*k elements in ascending order of distance.
    const int N = 2 * k;
    for (int size = 2; size <= N; size <<= 1) {
        int halfSize = size >> 1;
        for (int stride = halfSize; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += WARP_SIZE) {
                int partner = i ^ stride;
                if (partner > i) {
                    float dist_i = merge_dists[i];
                    float dist_j = merge_dists[partner];
                    int idx_i    = merge_indices[i];
                    int idx_j    = merge_indices[partner];

                    // Determine sorting direction for this subsequence.
                    bool ascending = ((i & size) == 0);
                    bool do_swap   = (dist_i > dist_j) == ascending;

                    if (do_swap) {
                        // Swap distances
                        merge_dists[i]       = dist_j;
                        merge_dists[partner] = dist_i;
                        // Swap indices
                        merge_indices[i]       = idx_j;
                        merge_indices[partner] = idx_i;
                    }
                }
            }
            __syncwarp();
        }
    }

    // First k entries now hold the smallest distances; copy back to best_*.
    for (int i = lane; i < k; i += WARP_SIZE) {
        best_dists[i]   = merge_dists[i];
        best_indices[i] = merge_indices[i];
    }

    if (lane == 0) {
        *cand_count_ptr = 0;
    }
    __syncwarp();
}

__global__
void knn_kernel(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    Pair * __restrict__ result,
    int k
) {
    extern __shared__ unsigned char smem[];

    // Layout dynamic shared memory:
    // [0]                    float2 sh_data[TILE_POINTS]
    // [aligned]              int   candidate_counts[WARPS_PER_BLOCK]
    // [next]                 int   best_indices[WARPS_PER_BLOCK][k]
    // [next]                 float best_dists  [WARPS_PER_BLOCK][k]
    // [next]                 int   cand_indices[WARPS_PER_BLOCK][k]
    // [next]                 float cand_dists  [WARPS_PER_BLOCK][k]
    // [next]                 int   merge_indices[WARPS_PER_BLOCK][2*k]
    // [next]                 float merge_dists  [WARPS_PER_BLOCK][2*k]

    unsigned char *ptr = smem;

    // Shared tile of data points.
    float2 *sh_data = reinterpret_cast<float2*>(ptr);
    ptr += TILE_POINTS * sizeof(float2);

    // Align to 16 bytes for the remaining arrays.
    uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
    iptr = (iptr + 15u) & ~uintptr_t(15u);
    ptr = reinterpret_cast<unsigned char*>(iptr);

    int *candidate_counts = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * sizeof(int);

    int *best_indices = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(int);

    float *best_dists = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(float);

    int *cand_indices = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(int);

    float *cand_dists = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(float);

    int *merge_indices = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * 2 * k * sizeof(int);

    float *merge_dists = reinterpret_cast<float*>(ptr);
    // ptr += WARPS_PER_BLOCK * 2 * k * sizeof(float); // not needed further

    const int tid   = threadIdx.x;
    const int lane  = tid & (WARP_SIZE - 1);
    const int warp  = tid / WARP_SIZE; // warp index within block

    const int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool active     = (global_warp < query_count);
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Per-warp views into shared arrays.
    int   *warp_cand_count  = &candidate_counts[warp];
    int   *warp_best_indices = best_indices   + warp * k;
    float *warp_best_dists   = best_dists     + warp * k;
    int   *warp_cand_indices = cand_indices   + warp * k;
    float *warp_cand_dists   = cand_dists     + warp * k;
    int   *warp_merge_indices = merge_indices + warp * 2 * k;
    float *warp_merge_dists   = merge_dists   + warp * 2 * k;

    // Initialize per-warp intermediate result and candidate count.
    if (active) {
        // Initialize best distances to +inf and indices to -1.
        for (int i = lane; i < k; i += WARP_SIZE) {
            warp_best_dists[i]   = FLT_MAX;
            warp_best_indices[i] = -1;
        }
        if (lane == 0) {
            *warp_cand_count = 0;
        }
    }

    // Ensure all shared memory initialization is visible before using sh_data.
    __syncthreads();

    // Load query point for this warp.
    float2 q;
    if (active) {
        q = query[global_warp];
    }

    // Iterate over data in tiles.
    for (int data_start = 0; data_start < data_count; data_start += TILE_POINTS) {
        int remaining = data_count - data_start;
        int tile_size = remaining < TILE_POINTS ? remaining : TILE_POINTS;

        // Load tile into shared memory cooperatively by the whole block.
        for (int idx = tid; idx < tile_size; idx += blockDim.x) {
            sh_data[idx] = data[data_start + idx];
        }

        __syncthreads();

        if (active) {
            // Each warp processes the tile in groups of 32 points (one per lane).
            for (int base = 0; base < tile_size; base += WARP_SIZE) {
                int idx_in_tile = base + lane;
                bool in_range   = (idx_in_tile < tile_size);

                float dist      = 0.0f;
                int data_index  = -1;

                if (in_range) {
                    float2 p = sh_data[idx_in_tile];
                    float dx = p.x - q.x;
                    float dy = p.y - q.y;
                    dist = dx * dx + dy * dy;
                    data_index = data_start + idx_in_tile;
                }

                // max_distance: distance of current k-th nearest neighbor
                // (k-th = worst in sorted top-k array).
                float max_distance = warp_best_dists[k - 1];

                // Determine which lanes have candidates closer than max_distance.
                unsigned int candidate_mask =
                    __ballot_sync(FULL_MASK, in_range && dist < max_distance);
                int num_new = __popc(candidate_mask);

                if (num_new > 0) {
                    // Ensure that adding num_new candidates does not overflow
                    // the candidate buffer. If it would, flush and merge first.
                    while (true) {
                        int cur_count = 0;
                        if (lane == 0) {
                            cur_count = *warp_cand_count;
                        }
                        cur_count = __shfl_sync(FULL_MASK, cur_count, 0);

                        if (cur_count + num_new <= k) {
                            // Enough space for all new candidates.
                            break;
                        }

                        // Candidate buffer full: merge it with the intermediate result.
                        warp_merge_topk(
                            lane,
                            k,
                            warp_cand_count,
                            warp_best_indices,
                            warp_best_dists,
                            warp_cand_indices,
                            warp_cand_dists,
                            warp_merge_indices,
                            warp_merge_dists
                        );

                        // Update max_distance and recompute candidate mask with
                        // the new threshold, since it may have decreased.
                        max_distance = warp_best_dists[k - 1];
                        candidate_mask =
                            __ballot_sync(FULL_MASK, in_range && dist < max_distance);
                        num_new = __popc(candidate_mask);

                        if (num_new == 0) {
                            break;
                        }
                    }

                    if (num_new > 0) {
                        // Reserve a contiguous block in the candidate buffer using atomicAdd.
                        int base_pos = 0;
                        if (lane == 0) {
                            base_pos = atomicAdd(warp_cand_count, num_new);
                        }
                        base_pos = __shfl_sync(FULL_MASK, base_pos, 0);

                        // Compute per-lane offset among the new candidates.
                        unsigned int mask_before =
                            candidate_mask & ((1u << lane) - 1u);
                        int rank = __popc(mask_before);

                        if (in_range && dist < max_distance) {
                            int pos = base_pos + rank;
                            warp_cand_indices[pos] = data_index;
                            warp_cand_dists[pos]   = dist;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    if (active) {
        // Final merge for any remaining candidates in the buffer.
        int cand_count = 0;
        if (lane == 0) {
            cand_count = *warp_cand_count;
        }
        cand_count = __shfl_sync(FULL_MASK, cand_count, 0);

        if (cand_count > 0) {
            warp_merge_topk(
                lane,
                k,
                warp_cand_count,
                warp_best_indices,
                warp_best_dists,
                warp_cand_indices,
                warp_cand_dists,
                warp_merge_indices,
                warp_merge_dists
            );
        }

        // After all merges, warp_best_dists/indices contain the final k-NN
        // for this query in ascending order of distance.
        for (int i = lane; i < k; i += WARP_SIZE) {
            int   idx  = warp_best_indices[i];
            float dist = warp_best_dists[i];

            Pair &out = result[global_warp * k + i];
            out.first  = idx;
            out.second = dist;
        }
    }
}

// Host interface as requested.
void run_knn(
    const float2 *query,   // device pointer
    int query_count,
    const float2 *data,    // device pointer
    int data_count,
    Pair *result,          // device pointer
    int k
) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Compute required dynamic shared memory size.
    size_t shmem_size = 0;

    shmem_size += TILE_POINTS * sizeof(float2);

    // Align to 16 bytes.
    const size_t align = 16;
    shmem_size = (shmem_size + align - 1) / align * align;

    shmem_size += WARPS_PER_BLOCK * sizeof(int);            // candidate_counts
    shmem_size += WARPS_PER_BLOCK * k * sizeof(int);        // best_indices
    shmem_size += WARPS_PER_BLOCK * k * sizeof(float);      // best_dists
    shmem_size += WARPS_PER_BLOCK * k * sizeof(int);        // cand_indices
    shmem_size += WARPS_PER_BLOCK * k * sizeof(float);      // cand_dists
    shmem_size += WARPS_PER_BLOCK * 2 * k * sizeof(int);    // merge_indices
    shmem_size += WARPS_PER_BLOCK * 2 * k * sizeof(float);  // merge_dists

    // Ensure the kernel can use the requested dynamic shared memory size
    // on modern GPUs (A100/H100 support up to 164 KB per block).
    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_size
    );

    knn_kernel<<<grid, block, shmem_size>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k
    );

    cudaDeviceSynchronize();
}