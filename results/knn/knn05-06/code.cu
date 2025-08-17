#include <cuda_runtime.h>
#include <stdint.h>

// This implementation computes k-NN for 2D points using one warp (32 threads) per query.
// It processes the data in shared-memory tiles. Each warp maintains two per-query buffers
// in shared memory:
//   - intermediate result (k best pairs so far)
//   - candidate buffer (up to k new candidate pairs)
// When the candidate buffer fills (or at the end), the warp merges the intermediate
// result and candidate buffer by performing an in-place bitonic sort over the combined
// 2k elements in shared memory, then keeps the first k elements as the new intermediate result.
//
// The kernel uses only per-block dynamic shared memory (no additional device allocations).
// It assumes k is a power of two between 32 and 1024 inclusive, and data_count >= k.

struct IdxDist {
    int   idx;
    float dist;
};

static __device__ __forceinline__ float sqr(float x) { return x * x; }

static __device__ __forceinline__ bool pairGreater(int ai, float ad, int bi, float bd) {
    // Return true if (ad, ai) > (bd, bi) lexicographically (distance primary, index secondary)
    return (ad > bd) || ((ad == bd) && (ai > bi));
}

static __device__ __forceinline__ unsigned lane_id() {
    return threadIdx.x & 31;
}

static __device__ __forceinline__ unsigned warp_id_in_block() {
    return threadIdx.x >> 5;
}

static __device__ __forceinline__ unsigned full_mask() {
    return 0xFFFFFFFFu;
}

// Accessors for the combined array of size 2k, where positions [0..k-1] are in res buffers
// and positions [k..2k-1] are in cand buffers. This lets us run a bitonic sort
// across 2k elements without allocating an additional array.
static __device__ __forceinline__
void get_pair_2k(int pos, int k,
                 const int* __restrict__ res_idx, const float* __restrict__ res_dist,
                 const int* __restrict__ cand_idx, const float* __restrict__ cand_dist,
                 int &idx, float &dist) {
    if (pos < k) {
        idx = res_idx[pos];
        dist = res_dist[pos];
    } else {
        int p = pos - k;
        idx = cand_idx[p];
        dist = cand_dist[p];
    }
}

static __device__ __forceinline__
void set_pair_2k(int pos, int k,
                 int* __restrict__ res_idx, float* __restrict__ res_dist,
                 int* __restrict__ cand_idx, float* __restrict__ cand_dist,
                 int idx, float dist) {
    if (pos < k) {
        res_idx[pos]  = idx;
        res_dist[pos] = dist;
    } else {
        int p = pos - k;
        cand_idx[p]  = idx;
        cand_dist[p] = dist;
    }
}

// In-place bitonic sort of the combined array of length N = 2*k stored across two buffers.
// After completion, res buffers [0..k-1] contain the k smallest pairs in ascending order.
// This uses 32 threads of the current warp cooperatively; threads operate on strided indices.
static __device__ __forceinline__
void bitonic_sort_combined_2k(int k,
                              int* __restrict__ res_idx, float* __restrict__ res_dist,
                              int* __restrict__ cand_idx, float* __restrict__ cand_dist) {
    const int N = 2 * k;
    const unsigned lane = lane_id();

    // Standard bitonic sort network for N elements (N is power of two).
    // Using warp-synchronous programming with __syncwarp barriers.
    for (int size = 2; size <= N; size <<= 1) {
        // The direction for this size block is ascending when ((i & size) == 0)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += warpSize) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = ((i & size) == 0);
                    int ai, bi;
                    float ad, bd;
                    get_pair_2k(i, k, res_idx, res_dist, cand_idx, cand_dist, ai, ad);
                    get_pair_2k(j, k, res_idx, res_dist, cand_idx, cand_dist, bi, bd);

                    bool swap_needed = up ? pairGreater(ai, ad, bi, bd) : pairGreater(bi, bd, ai, ad);
                    if (swap_needed) {
                        // Swap pairs at i and j
                        set_pair_2k(i, k, res_idx, res_dist, cand_idx, cand_dist, bi, bd);
                        set_pair_2k(j, k, res_idx, res_dist, cand_idx, cand_dist, ai, ad);
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Flush the candidate buffer by merging it with the intermediate result using a
// bitonic sort over the 2k combined elements. Any unused candidate slots should
// already be padded with (idx=-1, dist=INF) by the caller if cand_count < k.
static __device__ __forceinline__
void flush_and_merge_warp(int k,
                          int* __restrict__ res_idx, float* __restrict__ res_dist,
                          int* __restrict__ cand_idx, float* __restrict__ cand_dist) {
    bitonic_sort_combined_2k(k, res_idx, res_dist, cand_idx, cand_dist);
    // After sort, res buffers contain the k smallest elements in ascending order.
}

// Kernel: one warp handles one query.
__global__ void knn_warp_kernel(const float2* __restrict__ query,
                                int query_count,
                                const float2* __restrict__ data,
                                int data_count,
                                IdxDist* __restrict__ result,
                                int k,
                                int tile_points,
                                int warps_per_block) {
    extern __shared__ unsigned char smem_raw[];

    // Layout shared memory:
    // [tile_points * sizeof(float2)] tile buffer for data points (shared by the entire block)
    // Then per-warp private regions:
    //   res_idx[warps_per_block][k]
    //   res_dist[warps_per_block][k]
    //   cand_idx[warps_per_block][k]
    //   cand_dist[warps_per_block][k]
    //   cand_count[warps_per_block]
    //
    // Total dynamic shared memory is set accordingly by the host.
    unsigned char* ptr = smem_raw;

    // Tile of data points for the whole block
    float2* tile = reinterpret_cast<float2*>(ptr);
    ptr += sizeof(float2) * tile_points;

    // Align to 16 bytes for good measure
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    addr = (addr + 15u) & ~uintptr_t(15u);
    ptr = reinterpret_cast<unsigned char*>(addr);

    // Per-warp buffers
    int*   res_idx_all  = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * warps_per_block * k;

    float* res_dist_all = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * warps_per_block * k;

    int*   cand_idx_all = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * warps_per_block * k;

    float* cand_dist_all = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * warps_per_block * k;

    int* cand_count_all = reinterpret_cast<int*>(ptr);
    // No need to advance ptr further

    const unsigned lane   = lane_id();
    const unsigned warp_b = warp_id_in_block();
    const unsigned warp_g = (blockIdx.x * warps_per_block) + warp_b;
    const bool warp_active = (warp_g < static_cast<unsigned>(query_count));

    // Per-warp base pointers into shared memory buffers
    int*   res_idx  = res_idx_all  + warp_b * k;
    float* res_dist = res_dist_all + warp_b * k;
    int*   cand_idx  = cand_idx_all  + warp_b * k;
    float* cand_dist = cand_dist_all + warp_b * k;
    int&   cand_count = cand_count_all[warp_b];

    // Initialize per-warp structures
    if (lane == 0) {
        cand_count = 0;
    }
    // Initialize intermediate result with +INF distances so any real candidate is better.
    if (warp_active) {
        for (int i = lane; i < k; i += warpSize) {
            res_idx[i]  = -1;
            res_dist[i] = CUDART_INF_F;
        }
    }
    __syncwarp();

    // Load the query point for this warp
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        float2 q = query[warp_g];
        qx = q.x; qy = q.y;
    }

    // Current threshold: the k-th smallest distance so far (last element of res_dist).
    float threshold = CUDART_INF_F;

    // Process data in tiles
    for (int base = 0; base < data_count; base += tile_points) {
        const int count = min(tile_points, data_count - base);

        // Load data tile into shared memory cooperatively by the whole block
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            tile[i] = data[base + i];
        }
        __syncthreads();

        // Each active warp processes its query against all points in the tile
        if (warp_active) {
            for (int t = lane; t < count; t += warpSize) {
                float2 p = tile[t];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float d2 = fmaf(dx, dx, dy * dy); // dx*dx + dy*dy (use FMA for better perf)

                // Fast reject by current threshold
                int global_idx = base + t;
                unsigned keep_mask = __ballot_sync(full_mask(), d2 < threshold);
                if (keep_mask == 0u) {
                    continue; // no lane keeps this iteration
                }

                int adds = __popc(keep_mask);

                // If buffer would overflow, flush it first, then re-evaluate with updated threshold
                int local_cand_count = 0;
                if (lane == 0) local_cand_count = cand_count;
                local_cand_count = __shfl_sync(full_mask(), local_cand_count, 0);

                if (local_cand_count + adds > k) {
                    // Pad remaining candidate slots with +INF so that combined size is exactly k
                    for (int i = lane; i < k; i += warpSize) {
                        if (i >= local_cand_count) {
                            cand_idx[i]  = -1;
                            cand_dist[i] = CUDART_INF_F;
                        }
                    }
                    __syncwarp();

                    // Merge candidate buffer with intermediate result
                    flush_and_merge_warp(k, res_idx, res_dist, cand_idx, cand_dist);

                    if (lane == 0) cand_count = 0;
                    __syncwarp();

                    // Update threshold
                    threshold = res_dist[k - 1];

                    // Recompute keep mask under tighter threshold
                    keep_mask = __ballot_sync(full_mask(), d2 < threshold);
                    adds = __popc(keep_mask);
                    if (adds == 0) {
                        continue;
                    }
                }

                // Reserve slots in candidate buffer
                int base_off = 0;
                if (lane == 0) {
                    base_off = cand_count;
                    cand_count = base_off + adds;
                }
                base_off = __shfl_sync(full_mask(), base_off, 0);

                // Compute per-lane offset among kept lanes
                unsigned lt_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
                int my_off = __popc(keep_mask & lt_mask);

                // Write kept candidates
                if (d2 < threshold) {
                    int pos = base_off + my_off;
                    // pos is guaranteed < k due to earlier flush if needed
                    cand_idx[pos]  = global_idx;
                    cand_dist[pos] = d2;
                }
            }
        }

        __syncthreads(); // ensure all warps finished with tile before overwriting it
    }

    // Final flush for remaining candidates (if any)
    if (warp_active) {
        int final_count = 0;
        if (lane == 0) final_count = cand_count;
        final_count = __shfl_sync(full_mask(), final_count, 0);

        if (final_count > 0) {
            // Pad the remaining candidate slots with +INF
            for (int i = lane; i < k; i += warpSize) {
                if (i >= final_count) {
                    cand_idx[i]  = -1;
                    cand_dist[i] = CUDART_INF_F;
                }
            }
            __syncwarp();

            flush_and_merge_warp(k, res_idx, res_dist, cand_idx, cand_dist);
            if (lane == 0) cand_count = 0;
            __syncwarp();
        }

        // Write the final k nearest neighbors for this query
        for (int i = lane; i < k; i += warpSize) {
            int out_index = warp_g * k + i;
            result[out_index].idx  = res_idx[i];
            result[out_index].dist = res_dist[i];
        }
    }
}

// Host entry point. Chooses hyper-parameters and launches the kernel.
// The result pointer is std::pair<int,float>* on the host; we reinterpret it as IdxDist* for device.
// It assumes all device memory allocations and copies are managed by the caller.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Query available dynamic shared memory (opt-in) and choose launch parameters.
    int device = 0;
    cudaGetDevice(&device);

    int maxOptinSmem = 0;
    cudaDeviceGetAttribute(&maxOptinSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (maxOptinSmem <= 0) {
        // Fallback to legacy limit if opt-in not supported
        cudaDeviceGetAttribute(&maxOptinSmem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    }

    // Choose warps per block (default 8) and adjust if needed to fit in shared memory.
    int warps_per_block = 8; // 8 warps -> 256 threads per block, good balance on A100/H100
    const int warp_threads = 32;
    const int threads_per_block = warps_per_block * warp_threads;

    // Per-warp shared memory usage:
    // res_idx[k] + res_dist[k] + cand_idx[k] + cand_dist[k] = 4*k + 4*k + 4*k + 4*k bytes = 16*k bytes
    const size_t per_warp_bytes = static_cast<size_t>(16) * static_cast<size_t>(k);
    // Additional per-warp small overhead for counters (int)
    const size_t per_warp_overhead = sizeof(int);

    // Reserve some safety margin for alignment
    const size_t safety_bytes = 256;

    // Compute per-block bytes for per-warp buffers
    auto block_perwarp_bytes = [&](int warps) -> size_t {
        return warps * (per_warp_bytes + per_warp_overhead);
    };

    // Choose tile_points to maximize cache reuse while fitting in shared memory:
    // tile_bytes = tile_points * sizeof(float2)
    // total_smem = tile_bytes + block_perwarp_bytes + safety <= maxOptinSmem
    int tile_points = 4096; // start with 4K points (32 KB), a good default
    while (true) {
        size_t tile_bytes = static_cast<size_t>(tile_points) * sizeof(float2);
        size_t total = tile_bytes + block_perwarp_bytes(warps_per_block) + safety_bytes;
        if (total <= static_cast<size_t>(maxOptinSmem)) break;

        // If too big, reduce tile size first, but keep it at least 1024
        if (tile_points > 1024) {
            tile_points = max(1024, tile_points / 2);
            continue;
        }
        // If still too big, reduce warps per block and reset tile size
        if (warps_per_block > 1) {
            warps_per_block /= 2;
            tile_points = 4096; // try again with larger tile
            continue;
        }
        // As a last resort, reduce tile even further
        while (true) {
            size_t total2 = static_cast<size_t>(tile_points) * sizeof(float2) + block_perwarp_bytes(warps_per_block) + safety_bytes;
            if (total2 <= static_cast<size_t>(maxOptinSmem) || tile_points <= 32) break;
            tile_points /= 2;
        }
        break;
    }

    // Round tile_points up/down to a multiple of 32 for better coalescing
    tile_points = (tile_points / 32) * 32;
    if (tile_points < 32) tile_points = 32;

    // Compute dynamic shared memory size
    size_t dyn_smem_bytes = sizeof(float2) * static_cast<size_t>(tile_points)
                          + block_perwarp_bytes(warps_per_block)
                          + safety_bytes;

    // Opt-in to use large dynamic shared memory if necessary
    cudaFuncSetAttribute(knn_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(dyn_smem_bytes));

    // Compute grid size: one warp per query
    int warps_total = (query_count + 0) / 1; // one warp per query
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch kernel
    IdxDist* out = reinterpret_cast<IdxDist*>(result);
    dim3 block(threads_per_block);
    dim3 grid(blocks);

    knn_warp_kernel<<<grid, block, dyn_smem_bytes>>>(
        query, query_count, data, data_count, out, k, tile_points, warps_per_block
    );

    // Users of this function are expected to synchronize or use the stream they provided externally.
}