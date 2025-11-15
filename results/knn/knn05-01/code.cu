// CUDA k-NN implementation for 2D points.
// - Each query is processed by a single warp (32 threads).
// - The block cooperatively loads data points into shared memory in tiles.
// - Each warp maintains a private intermediate top-k result distributed
//   across its threads (per-thread register arrays).
// - For each query, there is a per-warp candidate buffer in shared memory
//   that stores up to k candidate neighbors before merging.
// - When the candidate buffer is full (or at the end), it is merged with
//   the intermediate top-k using a warp-parallel bitonic sort on a
//   shared-memory array of size up to 2*k.
// - Distances are squared Euclidean distances in 2D.
//
// Assumptions:
// - Target GPU: modern data-center GPU (e.g., A100, H100).
// - k is a power of two between 32 and 1024 inclusive.
// - data_count >= k.
// - query, data, and result pointers refer to device memory allocated by cudaMalloc.

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>
#include <cmath>

// Warp size used throughout the implementation.
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Maximum k supported by this implementation (as per problem statement).
static constexpr int MAX_K = 1024;

// Number of warps per block. Each warp processes one query.
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

// Number of data points loaded per tile into shared memory.
static constexpr int TILE_SIZE = 4096;

// Maximum number of top-k entries stored by a single thread (distributed
// across the 32 threads of a warp). This is ceil(MAX_K / WARP_SIZE).
static constexpr int MAX_ITEMS_PER_THREAD = (MAX_K + WARP_SIZE - 1) / WARP_SIZE;

// Warp-wide maximum reduction.
__device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffffu;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Compute the current worst (largest) distance in the distributed top-k
// structure of a warp.
//
// The top-k entries are stored per-thread in best_dists[], with
// items_per_thread entries per thread. The mapping from global
// top-k index g (0..k-1) to thread and local index is:
//   thread lane: g % WARP_SIZE
//   local index: g / WARP_SIZE
__device__ __forceinline__
float compute_worst_distance(int k, int items_per_thread, int lane,
                             const float* best_dists) {
    float local_max = -FLT_MAX;
    for (int i = 0; i < items_per_thread; ++i) {
        int global_pos = i * WARP_SIZE + lane;
        if (global_pos < k) {
            local_max = fmaxf(local_max, best_dists[i]);
        }
    }
    float warp_max = warp_reduce_max(local_max);
    unsigned mask = 0xffffffffu;
    // Broadcast from lane 0 so that all threads in the warp have the same threshold.
    return __shfl_sync(mask, warp_max, 0);
}

// Merge the current per-warp candidate buffer with the intermediate top-k
// result for a single query.
//
// Parameters:
// - k:               number of neighbors to keep
// - cand_count:      number of valid candidates currently in shared_dists[0..cand_count-1]
// - items_per_thread: number of top-k entries stored per thread
// - lane:            lane ID in the warp (0..31)
// - best_dists:      per-thread array of top-k distances (size MAX_ITEMS_PER_THREAD)
// - best_indices:    per-thread array of top-k indices (size MAX_ITEMS_PER_THREAD)
// - shared_dists:    per-warp shared buffer (size 2*MAX_K), used both for
//                    candidate buffer and merge workspace
// - shared_indices:  same as shared_dists, but for indices
//
// Operation:
//   1. Move candidate entries from [0, cand_count) to [k, k + cand_count).
//   2. Copy current top-k from registers into [0, k).
//   3. Fill the rest up to next power-of-two size with sentinel (FLT_MAX, -1).
//   4. Perform a warp-parallel bitonic sort on [0, pow2N) by distance.
//   5. Copy the first k entries back into distributed per-thread storage.
__device__ __forceinline__
void warp_merge_topk(int k, int cand_count, int items_per_thread, int lane,
                     float* best_dists, int* best_indices,
                     float* shared_dists, int* shared_indices) {
    const unsigned mask = 0xffffffffu;

    // Step 1: move candidates from [0, cand_count) to [k, k + cand_count)
    for (int i = lane; i < cand_count; i += WARP_SIZE) {
        shared_dists[k + i]   = shared_dists[i];
        shared_indices[k + i] = shared_indices[i];
    }
    __syncwarp(mask);

    // Step 2: copy current top-k from registers into [0, k)
    for (int i = 0; i < items_per_thread; ++i) {
        int global_pos = i * WARP_SIZE + lane;
        if (global_pos < k) {
            shared_dists[global_pos]   = best_dists[i];
            shared_indices[global_pos] = best_indices[i];
        }
    }
    __syncwarp(mask);

    // Total number of elements to sort: current top-k + candidates
    int N = k + cand_count;

    // Next power-of-two >= N for bitonic sort
    int pow2N = 1;
    while (pow2N < N) {
        pow2N <<= 1;
    }

    // Step 3: fill tail [N, pow2N) with sentinel values
    for (int i = lane + N; i < pow2N; i += WARP_SIZE) {
        shared_dists[i]   = FLT_MAX;
        shared_indices[i] = -1;
    }
    __syncwarp(mask);

    // Step 4: bitonic sort on [0, pow2N), ascending by distance
    for (int size = 2; size <= pow2N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int tid = lane; tid < pow2N; tid += WARP_SIZE) {
                int j = tid ^ stride;
                if (j > tid) {
                    bool up = ((tid & size) == 0);
                    float vi = shared_dists[tid];
                    float vj = shared_dists[j];
                    int   ii = shared_indices[tid];
                    int   ij = shared_indices[j];
                    bool swap = up ? (vi > vj) : (vi < vj);
                    if (swap) {
                        shared_dists[tid]   = vj;
                        shared_dists[j]     = vi;
                        shared_indices[tid] = ij;
                        shared_indices[j]   = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }

    // Step 5: copy the first k elements back into distributed per-thread storage
    for (int i = 0; i < items_per_thread; ++i) {
        int global_pos = i * WARP_SIZE + lane;
        if (global_pos < k) {
            best_dists[i]   = shared_dists[global_pos];
            best_indices[i] = shared_indices[global_pos];
        }
    }
    __syncwarp(mask);
}

// Kernel computing k-NN for 2D points.
// Each warp processes one query point.
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                std::pair<int, float>* __restrict__ result,
                int k) {
    // Shared memory:
    // - s_data:        tiled data points
    // - s_shared_dists/s_shared_indices: per-warp buffers for candidates and merges
    __shared__ float2 s_data[TILE_SIZE];
    __shared__ float  s_shared_dists[WARPS_PER_BLOCK][2 * MAX_K];
    __shared__ int    s_shared_indices[WARPS_PER_BLOCK][2 * MAX_K];

    const int tid        = threadIdx.x;
    const int warp_local = tid / WARP_SIZE;      // warp index within block
    const int lane       = tid & (WARP_SIZE - 1); // lane index within warp
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_local;

    const bool active = (warp_global < query_count);

    // Per-warp candidate buffer pointers in shared memory.
    float* cand_dists   = s_shared_dists[warp_local];
    int*   cand_indices = s_shared_indices[warp_local];

    // Per-thread portion of the intermediate top-k (stored in registers).
    float best_dists[MAX_ITEMS_PER_THREAD];
    int   best_indices[MAX_ITEMS_PER_THREAD];

    int   items_per_thread = 0;
    float threshold        = FLT_MAX; // k-th nearest distance (worst among current top-k)
    int   cand_count       = 0;       // number of candidates currently stored in cand_dists[0..cand_count-1]
    float2 q;

    if (active) {
        items_per_thread = (k + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < items_per_thread; ++i) {
            best_dists[i]   = FLT_MAX;
            best_indices[i] = -1;
        }
        // Each thread in the warp loads the same query point; the load is cached/broadcasted.
        q = query[warp_global];
        threshold  = FLT_MAX;
        cand_count = 0;
    }

    // Process data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_count = data_count - tile_start;
        if (tile_count > TILE_SIZE) {
            tile_count = TILE_SIZE;
        }

        // Block-level cooperative load of the current tile into shared memory.
        for (int i = tid; i < tile_count; i += blockDim.x) {
            s_data[i] = data[tile_start + i];
        }

        __syncthreads(); // ensure tile is fully loaded before use by any warp

        if (active) {
            const unsigned full_mask = 0xffffffffu;

            // Each warp processes the tile for its own query point.
            for (int idx = lane; idx < tile_count; idx += WARP_SIZE) {
                float2 p = s_data[idx];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                // Squared Euclidean distance in 2D.
                float dist = fmaf(dx, dx, dy * dy);

                bool is_candidate = (dist < threshold);

                // Warp-aggregated insertion into the candidate buffer, with
                // on-demand merging when the buffer would overflow.
                while (true) {
                    unsigned ballot = __ballot_sync(full_mask, is_candidate);
                    int warp_candidates = __popc(ballot);
                    if (warp_candidates == 0) {
                        // No lane in this warp has a candidate for this data point.
                        break;
                    }

                    int new_count = cand_count + warp_candidates;
                    if (new_count > k) {
                        // Candidate buffer would overflow; merge existing candidates
                        // with the intermediate top-k first.
                        warp_merge_topk(k, cand_count, items_per_thread, lane,
                                        best_dists, best_indices,
                                        cand_dists, cand_indices);
                        cand_count = 0;
                        // Recompute threshold (k-th nearest distance).
                        threshold = compute_worst_distance(k, items_per_thread, lane, best_dists);
                        // Re-evaluate this data point with the new threshold.
                        is_candidate = (dist < threshold);
                        // Loop again: new ballot and potential insertion.
                        continue;
                    }

                    // We have enough space in the candidate buffer for all warp_candidates.
                    int prefix = __popc(ballot & ((1u << lane) - 1));

                    int base = 0;
                    if (lane == 0) {
                        base       = cand_count;
                        cand_count = new_count;
                    }
                    // Broadcast base and updated cand_count from lane 0.
                    base       = __shfl_sync(full_mask, base, 0);
                    cand_count = __shfl_sync(full_mask, cand_count, 0);

                    if (is_candidate) {
                        int pos = base + prefix;
                        cand_dists[pos]   = dist;
                        cand_indices[pos] = tile_start + idx; // global index of data point
                    }
                    break;
                }
            }
        }

        __syncthreads(); // ensure all warps are done with this tile before reusing s_data
    }

    if (active) {
        // Merge any remaining candidates in the buffer after processing all tiles.
        if (cand_count > 0) {
            warp_merge_topk(k, cand_count, items_per_thread, lane,
                            best_dists, best_indices,
                            cand_dists, cand_indices);
            cand_count = 0;
            // No further use for threshold.
        }

        // Write the final top-k results for this query to global memory.
        // The top-k set is sorted by distance in ascending order.
        size_t base = static_cast<size_t>(warp_global) * static_cast<size_t>(k);
        for (int i = 0; i < items_per_thread; ++i) {
            int global_pos = i * WARP_SIZE + lane;
            if (global_pos < k) {
                int   nn_idx  = best_indices[i];
                float nn_dist = best_dists[i];
                result[base + global_pos].first  = nn_idx;
                result[base + global_pos].second = nn_dist;
            }
        }
    }
}

// Host interface: launch the k-NN kernel.
//
// query, data, and result must be device pointers allocated via cudaMalloc.
// The result is written entirely on the device; the caller is responsible
// for copying it back to host memory if needed.
void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    // Sanity check for k against the implementation's maximum.
    if (k <= 0 || k > MAX_K) {
        // For production code, better error handling is recommended.
        return;
    }

    int num_warps  = query_count;
    int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(num_blocks);

    knn_kernel<<<grid_dim, block_dim>>>(query, query_count,
                                        data, data_count,
                                        result, k);

    // Synchronize to ensure kernel completion before the function returns.
    cudaDeviceSynchronize();
}