#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <limits>
#include <algorithm>

// This implementation computes k-NN for 2D points with one warp (32 threads) per query.
// It uses shared memory tiling for data points and per-warp shared memory buffers:
// - best[k]: the current k nearest neighbors (sorted ascending by distance)
// - cand[k]: a buffer of candidates collected from the current/previous tiles
// - out[k]:  a temporary buffer used during merging
// The candidate buffer is filled until it reaches k entries; then it is sorted (bitonic sort)
// and merged with the current best using a parallel merge-path algorithm across the warp.
//
// Design and optimization notes:
// - Each warp handles one query and uses __ballot_sync and __shfl_sync to coordinate writes
//   into its candidate buffer without atomics.
// - Tiling of the data points into shared memory provides cache locality and reduces global
//   memory traffic. The entire block cooperatively loads tiles; then each warp traverses the
//   tile for its query.
// - Both "best" and "cand" are kept in shared memory. "best" is always sorted. "cand" is sorted
//   only when flushed (full, or final flush). Merging two sorted arrays leverages "merge-path"
//   partitioning across warp lanes to parallelize merging, keeping only the first k results.
// - Distances use squared L2 norm. Ties are resolved arbitrarily.
// - k is assumed to be a power of two in [32, 1024], inclusive.
// - The kernel uses dynamically allocated shared memory; run_knn() computes the appropriate
//   shared memory size and sets the corresponding kernel attribute on supported hardware
//   (e.g., A100/H100) to allow up to ~164KB or more of dynamic shared memory per block.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Choose number of warps per block (tuneable). Using 4 warps/block is a good trade-off
// between shared memory capacity (for larger k) and occupancy on A100/H100.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

// Pair container used internally for (distance, index).
struct Candidate {
    float dist;
    int   index;
};
static_assert(sizeof(Candidate) == 8, "Candidate must be 8 bytes");

// Utility: squared Euclidean distance between two float2 points.
__forceinline__ __device__ float squared_distance(const float2& a, const float2& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // FMA for potential precision/perf improvement
    return fmaf(dx, dx, dy * dy);
}

// Warp-scope bitonic sort on shared-memory array 'arr' of length 'n' (power of two).
// Sorts ascending by Candidate.dist. Uses only warp-scope sync (no __syncthreads).
__forceinline__ __device__ void warp_bitonic_sort(Candidate* arr, int n) {
    const unsigned full_mask = 0xFFFFFFFFu;
    // Bitonic sort network: O(n log^2 n).
    for (int size = 2; size <= n; size <<= 1) {
        // The 'size' stage sets the sort direction for subsequences.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes multiple indices spaced by warp size.
            for (int i = threadIdx.x & (WARP_SIZE - 1); i < n; i += WARP_SIZE) {
                int j = i ^ stride;
                if (j > i) {
                    bool ascending = ((i & size) == 0);
                    Candidate a = arr[i];
                    Candidate b = arr[j];
                    // Compare-exchange for ascending/descending order.
                    bool swapNeeded = ascending ? (a.dist > b.dist) : (a.dist < b.dist);
                    if (swapNeeded) {
                        arr[i] = b;
                        arr[j] = a;
                    }
                }
            }
            __syncwarp(full_mask);
        }
    }
}

// Merge-path binary search: find i in [max(0, diag - bCount), min(diag, aCount)]
// such that A[i-1] <= B[diag-i] and B[diag-i-1] < A[i]. We compare by .dist.
__forceinline__ __device__ int merge_path_search(const Candidate* A, int aCount,
                                                 const Candidate* B, int bCount,
                                                 int diag) {
    int low  = max(0, diag - bCount);
    int high = min(diag, aCount);
    while (low < high) {
        int mid = (low + high) >> 1;
        int i = mid;
        int j = diag - i;

        float a_i     = (i < aCount) ? A[i].dist : CUDART_INF_F;
        float b_jm1   = (j > 0)      ? B[j - 1].dist : -CUDART_INF_F;

        if (a_i < b_jm1) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Parallel merge of two sorted arrays A[0..k-1], B[0..k-1] into out[0..k-1] (top-k only).
// Each warp lane is assigned a contiguous output range [d0, d1).
__forceinline__ __device__ void warp_merge_topk(const Candidate* A, const Candidate* B,
                                                Candidate* out, int k) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Partition the first k outputs among warp lanes.
    int chunk = (k + WARP_SIZE - 1) / WARP_SIZE; // ceil(k / 32)
    int d0 = min(lane * chunk, k);
    int d1 = min(d0 + chunk, k);

    int i0 = merge_path_search(A, k, B, k, d0);
    int j0 = d0 - i0;

    int i1 = merge_path_search(A, k, B, k, d1);
    int j1 = d1 - i1;

    int i = i0;
    int j = j0;

    for (int out_pos = d0; out_pos < d1; ++out_pos) {
        float va = (i < k) ? A[i].dist : CUDART_INF_F;
        float vb = (j < k) ? B[j].dist : CUDART_INF_F;
        bool takeA = (va <= vb);
        if (takeA) {
            out[out_pos] = A[i++];
        } else {
            out[out_pos] = B[j++];
        }
    }
}

// Flush candidate buffer by sorting and merging into 'best'.
// Requires: 'best' is sorted ascending and has length k.
// - If candCount == k: sort cand[0..k-1], merge with best, write result to out, then copy out->best.
// - If candCount  < k: pad cand[candCount..k-1] with +inf, then sort, merge, etc.
// After flushing, candCount is set to 0, and kthDist is updated to best[k-1].dist.
__forceinline__ __device__ void warp_flush_candidates(Candidate* best,
                                                      Candidate* cand,
                                                      Candidate* out,
                                                      int k,
                                                      int& candCount,
                                                      float& kthDist) {
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Pad with +inf if not full.
    int cc = __shfl_sync(full_mask, candCount, 0);
    if (cc < k) {
        for (int i = lane + cc; i < k; i += WARP_SIZE) {
            cand[i].dist = CUDART_INF_F;
            cand[i].index = -1;
        }
        __syncwarp(full_mask);
    }

    // Sort candidate buffer and merge with current best.
    warp_bitonic_sort(cand, k);
    __syncwarp(full_mask);

    warp_merge_topk(best, cand, out, k);
    __syncwarp(full_mask);

    // Copy merged top-k back to best.
    for (int i = lane; i < k; i += WARP_SIZE) {
        best[i] = out[i];
    }
    __syncwarp(full_mask);

    if (lane == 0) {
        candCount = 0;
        kthDist = best[k - 1].dist;
    }
    __syncwarp(full_mask);
}

// Kernel: each warp handles one query. The block cooperatively loads data tiles into shared memory.
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k,
                           int tile_capacity_points) {
    extern __shared__ unsigned char shared_bytes[];

    // Layout of shared memory:
    // [ float2 tile[tile_capacity_points] ][ Candidate best[WARPS_PER_BLOCK * k] ]
    // [ Candidate cand[WARPS_PER_BLOCK * k] ][ Candidate out[WARPS_PER_BLOCK * k] ]
    float2* tile = reinterpret_cast<float2*>(shared_bytes);
    size_t tile_bytes = static_cast<size_t>(tile_capacity_points) * sizeof(float2);
    Candidate* smem_candidates = reinterpret_cast<Candidate*>(shared_bytes + tile_bytes);

    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned full_mask = 0xFFFFFFFFu;

    Candidate* best_base = smem_candidates;                                   // size: warps_per_block * k
    Candidate* cand_base = best_base + warps_per_block * k;                   // size: warps_per_block * k
    Candidate* out_base  = cand_base + warps_per_block * k;                   // size: warps_per_block * k

    Candidate* best = best_base + warp_id_in_block * k;
    Candidate* cand = cand_base + warp_id_in_block * k;
    Candidate* out  = out_base  + warp_id_in_block * k;

    const int warp_global_id = blockIdx.x * warps_per_block + warp_id_in_block;
    const bool warp_active = (warp_global_id < query_count);

    // Initialize best with +inf distances so it's a valid sorted list.
    if (warp_active) {
        for (int i = lane; i < k; i += WARP_SIZE) {
            best[i].dist = CUDART_INF_F;
            best[i].index = -1;
        }
    }
    __syncwarp(full_mask);

    float2 q = make_float2(0.f, 0.f);
    if (warp_active && lane == 0) {
        q = query[warp_global_id];
    }
    // Broadcast query components to all lanes of the warp.
    if (warp_active) {
        q.x = __shfl_sync(full_mask, q.x, 0);
        q.y = __shfl_sync(full_mask, q.y, 0);
    }

    int candCount = 0;
    float kthDist = CUDART_INF_F;

    // Process the dataset in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_capacity_points) {
        int tile_count = min(tile_capacity_points, data_count - tile_start);

        // Block-wide cooperative load of the current tile into shared memory.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            tile[i] = data[tile_start + i];
        }
        __syncthreads();

        if (warp_active) {
            // Local snapshot of kthDist for gating candidates between flushes.
            float local_kth = kthDist;

            // Each warp lane iterates over a strided subset of the tile.
            for (int i = lane; i < tile_count; i += WARP_SIZE) {
                float2 p = tile[i];
                int    idx = tile_start + i;
                float  dist = squared_distance(q, p);

                // Accept only if strictly better than current k-th.
                bool valid = (dist < local_kth);

                // Warp-scope packing loop: insert as many as possible into cand[],
                // flushing when full and re-evaluating against the updated threshold.
                while (true) {
                    unsigned mask = __ballot_sync(full_mask, valid);
                    if (mask == 0u) break; // no more lanes to insert this round

                    int cc = __shfl_sync(full_mask, candCount, 0);
                    int free_slots = k - cc;

                    if (free_slots == 0) {
                        // Flush: sort cand[0..k-1], merge with best, update kthDist and reset candCount.
                        warp_flush_candidates(best, cand, out, k, candCount, kthDist);
                        // Broadcast updated threshold and re-evaluate validity.
                        local_kth = __shfl_sync(full_mask, kthDist, 0);
                        valid = (dist < local_kth);
                        continue;
                    }

                    int accepted = __popc(mask);
                    int take = (accepted < free_slots) ? accepted : free_slots;

                    int base = cc;
                    if (lane == 0) {
                        candCount = cc + take;
                    }

                    int rank = __popc(mask & ((1u << lane) - 1));
                    bool do_write = valid && (rank < take);
                    if (do_write) {
                        int pos = base + rank;
                        cand[pos].dist = dist;
                        cand[pos].index = idx;
                    }

                    // If wrote, clear valid; otherwise remain valid for next iteration (after a flush).
                    valid = valid && !(do_write);
                }
            } // end per-tile work
        }

        __syncthreads();
    } // end tiles loop

    // Final flush if candidate buffer is non-empty.
    if (warp_active) {
        int cc = __shfl_sync(full_mask, candCount, 0);
        if (cc > 0) {
            warp_flush_candidates(best, cand, out, k, candCount, kthDist);
        }

        // Write the result for this query.
        size_t out_base_idx = static_cast<size_t>(warp_global_id) * static_cast<size_t>(k);
        for (int i = lane; i < k; i += WARP_SIZE) {
            int idx = best[i].index;
            float dist = best[i].dist;
            result[out_base_idx + i].first = idx;
            result[out_base_idx + i].second = dist;
        }
    }
}

// Host function to run the k-NN kernel. Computes dynamic shared memory size
// based on k and the chosen number of warps per block, opts into larger shared
// memory if available, and launches the kernel.
void run_knn(const float2* query, int query_count,
             const float2* data, int data_count,
             std::pair<int, float>* result, int k) {
    // Choose launch parameters.
    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    // Shared memory per warp: three k-sized arrays of Candidate (best, cand, out).
    const size_t per_warp_bytes = static_cast<size_t>(3) * static_cast<size_t>(k) * sizeof(Candidate);
    const size_t per_block_arrays_bytes = per_warp_bytes * static_cast<size_t>(warps_per_block);

    // Query device shared memory capability.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Use the opt-in maximum dynamic shared memory size if available (A100/H100).
    size_t max_dynamic_smem = (prop.sharedMemPerBlockOptin > 0) ? prop.sharedMemPerBlockOptin
                                                                : prop.sharedMemPerBlock;

    // Compute how many float2 points fit in the remaining shared memory for the tile.
    size_t tile_bytes_budget = (per_block_arrays_bytes < max_dynamic_smem) ?
                               (max_dynamic_smem - per_block_arrays_bytes) : 0;
    int tile_capacity_points = static_cast<int>(tile_bytes_budget / sizeof(float2));
    if (tile_capacity_points < 1) tile_capacity_points = 1;

    // Total dynamic shared memory per block to request.
    size_t dynamic_smem_bytes = per_block_arrays_bytes +
                                static_cast<size_t>(tile_capacity_points) * sizeof(float2);

    // Opt-in to the required dynamic shared memory size (may increase per-block limit).
    cudaFuncSetAttribute(knn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(dynamic_smem_bytes));

    // Configure grid.
    int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch kernel.
    knn_kernel<<<num_blocks, threads_per_block, dynamic_smem_bytes>>>(query, query_count,
                                                                      data, data_count,
                                                                      result, k,
                                                                      tile_capacity_points);
}