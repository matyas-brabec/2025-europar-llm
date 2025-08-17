#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// Kernel configuration tuned for A100/H100-class GPUs.
// - 256 threads per block (8 warps), one warp processes one query.
// - Data points are processed in tiles cached in shared memory.
// - Each thread keeps a small per-lane buffer of size k/32 in registers.
// - The final k results per query are merged and sorted cooperatively in shared memory.
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / 32)
#define WARP_SIZE 32
// Maximum per-lane buffer size: k is in [32, 1024] and power of two, so k/32 in [1, 32].
#define KLANE_MAX 32

// Utility: warp lane id and warp id within block
static __device__ __forceinline__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
static __device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }

// In-warp swap helper for shared-memory arrays of pairs (distance, index)
static __device__ __forceinline__
void swap_pair(float &a_d, int &a_i, float &b_d, int &b_i) {
    float td = a_d; int ti = a_i;
    a_d = b_d; a_i = b_i;
    b_d = td; b_i = ti;
}

// Bitonic sort (ascending by distance) over a shared-memory array of length K.
// Threads within a warp cooperatively sort sdist[0..K-1] and sidx[0..K-1] in-place.
// K is guaranteed to be a power of two between 32 and 1024.
// Requires: all 32 threads in the warp participate; synchronization via __syncwarp().
static __device__ void warp_bitonic_sort(float* sdist, int* sidx, int K) {
    const unsigned mask = 0xffffffffu;
    // Outer loop controls the sequence length of bitonic merges.
    for (int size = 2; size <= K; size <<= 1) {
        // Inner loop controls the compare distance within the current sequence.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each lane handles a strided set of indices.
            for (int i = lane_id(); i < K; i += WARP_SIZE) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    bool up = ((i & size) == 0);
                    float a = sdist[i];
                    float b = sdist[ixj];
                    // Note: For ties (a == b) we leave as-is; stability is not required.
                    if ((a > b) == up) {
                        // Swap both distance and index
                        swap_pair(sdist[i], sidx[i], sdist[ixj], sidx[ixj]);
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// CUDA kernel: one warp processes one query.
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int K,
                           int tile_elems)
{
    extern __shared__ unsigned char smem[];
    // Shared memory layout per block:
    // [0 .. tile_elems*sizeof(float2))               -> cached data tile as float2[]
    // [tile_elems*sizeof(float2) .. + warps*K*4)     -> per-warp distances (float[])
    // [.. + warps*K*4 .. + warps*K*8)                -> per-warp indices (int[])
    float2* tile = reinterpret_cast<float2*>(smem);
    float*  sdist_base = reinterpret_cast<float*>(tile + tile_elems);
    int*    sidx_base  = reinterpret_cast<int*>(sdist_base + WARPS_PER_BLOCK * K);

    const int lane = lane_id();
    const int wid  = warp_id();
    const int warp_global_query = blockIdx.x * WARPS_PER_BLOCK + wid;

    // If no query in this block at all, we can exit early.
    if (blockIdx.x * WARPS_PER_BLOCK >= query_count) return;

    const bool warp_active = (warp_global_query < query_count);
    float* sdist = sdist_base + wid * K;
    int*   sidx  = sidx_base  + wid * K;

    // Each thread keeps a small local buffer of size Klane = K / 32 (power of two, in [1, 32]).
    const int Klane = K >> 5;
    float local_dist[KLANE_MAX];
    int   local_idx[KLANE_MAX];

    // Initialize per-lane local buffers to +inf distance and invalid index.
    if (warp_active) {
#pragma unroll
        for (int i = 0; i < KLANE_MAX; ++i) {
            if (i < Klane) {
                local_dist[i] = CUDART_INF_F;
                local_idx[i]  = -1;
            }
        }
    }

    // Load the query point once per warp and broadcast to all lanes.
    float qx = 0.f, qy = 0.f;
    if (warp_active) {
        if (lane == 0) {
            float2 q = query[warp_global_query];
            qx = q.x; qy = q.y;
        }
        const unsigned mask = 0xffffffffu;
        qx = __shfl_sync(mask, qx, 0);
        qy = __shfl_sync(mask, qy, 0);
    }

    // Per-lane tracking of current worst (maximum) distance in the local buffer,
    // used as a quick rejection threshold.
    float local_worst = CUDART_INF_F;
    int   local_worst_pos = 0;

    // Iterate over the dataset in tiles cached in shared memory.
    for (int base = 0; base < data_count; base += tile_elems) {
        int tile_n = data_count - base;
        if (tile_n > tile_elems) tile_n = tile_elems;

        // Cooperative load of the current tile into shared memory.
        // All threads in the block participate in the load and in the barrier.
        for (int t = threadIdx.x; t < tile_n; t += blockDim.x) {
            // Use __ldg to hint read-only caching; data[] is declared const.
            tile[t] = __ldg(&data[base + t]);
        }
        __syncthreads();

        // Each active warp processes the cached tile for its query.
        if (warp_active) {
            // Strided loop over tile elements: each lane processes tile indices t = lane, lane+32, ...
            for (int t = lane; t < tile_n; t += WARP_SIZE) {
                float2 p = tile[t];
                float dx = p.x - qx;
                float dy = p.y - qy;
                // Squared Euclidean distance
                float d = fmaf(dx, dx, dy * dy);
                const int gidx = base + t;

                // If candidate is better than current worst in local buffer, insert it.
                if (d < local_worst) {
                    // Replace the current worst element.
                    local_dist[local_worst_pos] = d;
                    local_idx[local_worst_pos]  = gidx;

                    // Recompute the worst element and its position among the Klane entries.
                    // This is O(Klane) and happens only on successful insert.
                    float w = local_dist[0];
                    int   wp = 0;
#pragma unroll
                    for (int i = 1; i < KLANE_MAX; ++i) {
                        if (i < Klane) {
                            float vi = local_dist[i];
                            if (vi > w) { w = vi; wp = i; }
                        }
                    }
                    local_worst = w;
                    local_worst_pos = wp;
                }
            }
        }

        __syncthreads(); // Ensure the tile region can be safely overwritten in the next iteration.
    }

    // Active warps now cooperatively merge their per-lane buffers into a per-warp buffer in shared memory.
    if (warp_active) {
        // Gather: each lane writes its Klane elements contiguously into the per-warp segment.
#pragma unroll
        for (int i = 0; i < KLANE_MAX; ++i) {
            if (i < Klane) {
                int pos = lane * Klane + i;  // unique in [0, K)
                sdist[pos] = local_dist[i];
                sidx[pos]  = local_idx[i];
            }
        }
        __syncwarp();

        // Final cooperative sort across the K entries to produce ordered k-NN for this query.
        warp_bitonic_sort(sdist, sidx, K);

        // Write back the results in sorted order. Each lane writes a strided subset.
        std::pair<int, float>* out = result + (size_t)warp_global_query * (size_t)K;
        for (int i = lane; i < K; i += WARP_SIZE) {
            out[i].first  = sidx[i];
            out[i].second = sdist[i];
        }
    }
}

// Host interface: launches the kernel.
// - query: device pointer to float2 query points
// - query_count: number of queries
// - data: device pointer to float2 data points
// - data_count: number of data points
// - result: device pointer to std::pair<int,float> output buffer of size query_count * k
// - k: number of neighbors; must be power of two in [32, 1024]
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Launch configuration
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Determine dynamic shared memory requirements and choose a tile size.
    // Target a tile of 4096 points (32 KiB) when possible.
    int tile_elems = 4096;

    // Compute top-k shared memory bytes per block.
    size_t topk_bytes = size_t(WARPS_PER_BLOCK) * size_t(k) * (sizeof(float) + sizeof(int));
    size_t tile_bytes = size_t(tile_elems) * sizeof(float2);
    size_t smem_bytes = topk_bytes + tile_bytes;

    // Query the device limit for opt-in dynamic shared memory and adjust tile size if needed.
    int device = 0;
    cudaGetDevice(&device);
    int max_optin_smem = 0;
    cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    if (smem_bytes > (size_t)max_optin_smem) {
        // Reduce tile_elems to fit within the available shared memory.
        size_t max_tile_bytes = (size_t)max_optin_smem > topk_bytes ? (size_t)max_optin_smem - topk_bytes : 0;
        size_t feasible_tile_elems = max_tile_bytes / sizeof(float2);
        if (feasible_tile_elems == 0) {
            // As a last resort (should not happen on A100/H100 with given k range),
            // clamp to the minimal tile size of 32 to keep the kernel functional.
            feasible_tile_elems = 32;
        } else if (feasible_tile_elems > (size_t)tile_elems) {
            feasible_tile_elems = tile_elems;
        }
        tile_elems = (int)feasible_tile_elems;
        tile_bytes = size_t(tile_elems) * sizeof(float2);
        smem_bytes = topk_bytes + tile_bytes;
    }

    // Request the required dynamic shared memory size for this kernel.
    // This enables usage beyond the default per-block limit on A100/H100.
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

    // Launch the kernel
    knn_kernel<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k, tile_elems);
}