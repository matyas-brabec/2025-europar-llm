#include <cuda_runtime.h>
#include <utility>

// Optimized k-NN for 2D points using warp-per-query, shared-memory tiling, and a per-warp max-heap.
// - Each warp (32 threads) processes one query.
// - Data points are processed in shared-memory tiles to maximize cache reuse.
// - Each warp maintains its private top-k as a max-heap in shared memory.
// - Distance is squared Euclidean.
// - At the end, each warp heap-sorts its heap into ascending order and writes results.
//
// Hyperparameters:
// - WARPS_PER_BLOCK = 8 (block size 256 threads)
// - TILE_POINTS = 4096 (32 KB tile for float2)
// Shared memory usage per block:
// - Tile: 32 KB
// - Per-warp top-k buffers: WARPS_PER_BLOCK * k * (4 + 4) bytes
// - Per-warp control ints: WARPS_PER_BLOCK * 2 * 4 bytes
// For k=1024, WARPS_PER_BLOCK=8 => 96 KB + 64 bytes control ~= 96 KB total, fits A100 (opt-in) and H100.

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif

#ifndef TILE_POINTS
#define TILE_POINTS 4096
#endif

// Lane id within a warp
static __device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }
// Warp id within a block
static __device__ __forceinline__ int warp_id_in_block() { return threadIdx.x >> 5; }

// Squared L2 distance between q=(qx,qy) and p=(px,py)
static __device__ __forceinline__ float sq_l2(const float qx, const float qy, const float2 p) {
    float dx = qx - p.x;
    float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

// Max-heap sift-down (heap stored in dist/idx arrays)
// Maintains max-heap property for [0..length-1] starting at index 'start'
static __device__ __forceinline__ void heap_sift_down(float* dist, int* idx, int start, int length) {
    int root = start;
    while (true) {
        int left = (root << 1) + 1;
        if (left >= length) break;
        int right = left + 1;
        int swap_idx = root;
        if (dist[swap_idx] < dist[left]) swap_idx = left;
        if (right < length && dist[swap_idx] < dist[right]) swap_idx = right;
        if (swap_idx == root) break;
        float td = dist[root];
        dist[root] = dist[swap_idx];
        dist[swap_idx] = td;
        int ti = idx[root];
        idx[root] = idx[swap_idx];
        idx[swap_idx] = ti;
        root = swap_idx;
    }
}

// Build max-heap from arbitrary array contents
static __device__ __forceinline__ void heap_build(float* dist, int* idx, int length) {
    for (int i = (length >> 1) - 1; i >= 0; --i) {
        heap_sift_down(dist, idx, i, length);
    }
}

// Insert candidate (d, id) into max-heap of size k if better than current worst
static __device__ __forceinline__ void heap_insert_if_better(float* dist, int* idx, int k, float d, int id) {
    // Root holds current worst (maximum distance)
    if (d < dist[0]) {
        dist[0] = d;
        idx[0] = id;
        heap_sift_down(dist, idx, 0, k);
    }
}

// In-place heapsort on max-heap (dist/idx), produces ascending order by distance
static __device__ __forceinline__ void heap_sort_ascending(float* dist, int* idx, int length) {
    for (int end = length - 1; end > 0; --end) {
        // Move current max to the end
        float td = dist[0];
        dist[0] = dist[end];
        dist[end] = td;
        int ti = idx[0];
        idx[0] = idx[end];
        idx[end] = ti;
        // Restore heap property on reduced heap
        heap_sift_down(dist, idx, 0, end);
    }
    // After this, dist[0..length-1] is ascending
}

extern "C" __global__ void knn_kernel(const float2* __restrict__ query,
                                      int query_count,
                                      const float2* __restrict__ data,
                                      int data_count,
                                      std::pair<int, float>* __restrict__ result,
                                      int k)
{
    // Dynamic shared memory layout:
    // [ tile (TILE_POINTS float2) ]
    // [ all_idx (WARPS_PER_BLOCK * k int) ]
    // [ all_dist (WARPS_PER_BLOCK * k float) ]
    // [ warp_fill (WARPS_PER_BLOCK int) ]
    // [ warp_heap_built (WARPS_PER_BLOCK int) ]
    extern __shared__ unsigned char smem_raw[];
    size_t smem_off = 0;

    // Tile buffer
    float2* tile = reinterpret_cast<float2*>(smem_raw + smem_off);
    smem_off += size_t(TILE_POINTS) * sizeof(float2);

    // Align to int
    smem_off = (smem_off + alignof(int) - 1) & ~(size_t(alignof(int)) - 1);
    int* all_idx = reinterpret_cast<int*>(smem_raw + smem_off);
    smem_off += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(int);

    // Align to float
    smem_off = (smem_off + alignof(float) - 1) & ~(size_t(alignof(float)) - 1);
    float* all_dist = reinterpret_cast<float*>(smem_raw + smem_off);
    smem_off += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(float);

    // Per-warp control variables
    smem_off = (smem_off + alignof(int) - 1) & ~(size_t(alignof(int)) - 1);
    int* warp_fill = reinterpret_cast<int*>(smem_raw + smem_off);
    smem_off += size_t(WARPS_PER_BLOCK) * sizeof(int);

    smem_off = (smem_off + alignof(int) - 1) & ~(size_t(alignof(int)) - 1);
    int* warp_heap_built = reinterpret_cast<int*>(smem_raw + smem_off);
    smem_off += size_t(WARPS_PER_BLOCK) * sizeof(int);

    const int lane = lane_id();
    const int warp = warp_id_in_block();
    const int warp_in_grid = blockIdx.x * WARPS_PER_BLOCK + warp;

    // Pointers to this warp's private top-k buffers
    int* w_idx = all_idx + warp * k;
    float* w_dist = all_dist + warp * k;

    // Initialize control variables
    if (threadIdx.x < WARPS_PER_BLOCK) {
        warp_fill[threadIdx.x] = 0;
        warp_heap_built[threadIdx.x] = 0;
    }
    __syncthreads();

    // Load query point and broadcast within warp
    float qx = 0.0f, qy = 0.0f;
    bool warp_active = (warp_in_grid < query_count);
    if (warp_active && lane == 0) {
        float2 q = query[warp_in_grid];
        qx = q.x;
        qy = q.y;
    }
    unsigned warp_mask = __activemask();
    qx = __shfl_sync(warp_mask, qx, 0);
    qy = __shfl_sync(warp_mask, qy, 0);

    // Process data in tiles
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS) {
        int tile_count = data_count - tile_base;
        if (tile_count > TILE_POINTS) tile_count = TILE_POINTS;

        // Load tile cooperatively by the whole block
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            tile[i] = data[tile_base + i];
        }
        __syncthreads();

        // Each warp processes the tile for its query
        if (warp_active) {
            for (int t = 0; t < tile_count; t += warpSize) {
                int i = t + lane;
                bool lane_active = (i < tile_count);
                float cand_dist = 0.0f;
                int cand_idx = tile_base + i;

                if (lane_active) {
                    float2 p = tile[i];
                    cand_dist = sq_l2(qx, qy, p);
                }

                // Initialization phase: fill first k elements (unsorted), then build heap once
                if (!warp_heap_built[warp]) {
                    if (lane_active) {
                        int pos = atomicAdd(&warp_fill[warp], 1);
                        if (pos < k) {
                            // Write directly into buffer
                            w_dist[pos] = cand_dist;
                            w_idx[pos] = cand_idx;
                        }
                        // Collect overflow candidates that arrive after k has been filled
                        unsigned overflow_mask = __ballot_sync(warp_mask, pos >= k);
                        __syncwarp(warp_mask);
                        // Build heap once when enough items collected
                        if (lane == 0 && !warp_heap_built[warp] && warp_fill[warp] >= k) {
                            // Clamp fill to exactly k and build max-heap
                            warp_fill[warp] = k;
                            heap_build(w_dist, w_idx, k);
                            warp_heap_built[warp] = 1;
                        }
                        __syncwarp(warp_mask);

                        // After heap is built, process overflow candidates within this step
                        if (warp_heap_built[warp]) {
                            if (overflow_mask != 0 && lane == 0) {
                                // For each overflowing lane, fetch its candidate via shuffles and insert
                                unsigned mask_iter = overflow_mask;
                                while (mask_iter) {
                                    int src_lane = __ffs(mask_iter) - 1;
                                    mask_iter &= (mask_iter - 1);
                                    float d = __shfl_sync(warp_mask, cand_dist, src_lane);
                                    int id = __shfl_sync(warp_mask, cand_idx, src_lane);
                                    heap_insert_if_better(w_dist, w_idx, k, d, id);
                                }
                            }
                            __syncwarp(warp_mask);
                        }
                    } else {
                        // Lane inactive; still need to participate in syncs
                        __syncwarp(warp_mask);
                        if (lane == 0 && !warp_heap_built[warp] && warp_fill[warp] >= k) {
                            warp_fill[warp] = k;
                            heap_build(w_dist, w_idx, k);
                            warp_heap_built[warp] = 1;
                        }
                        __syncwarp(warp_mask);
                    }
                } else {
                    // Heap phase: threshold filter then sequential insert by lane 0
                    float thresh = w_dist[0]; // current worst (max)
                    unsigned accept_mask = __ballot_sync(warp_mask, lane_active && (cand_dist < thresh));
                    if (lane == 0 && accept_mask) {
                        unsigned mask_iter = accept_mask;
                        while (mask_iter) {
                            int src_lane = __ffs(mask_iter) - 1;
                            mask_iter &= (mask_iter - 1);
                            float d = __shfl_sync(warp_mask, cand_dist, src_lane);
                            int id = __shfl_sync(warp_mask, cand_idx, src_lane);
                            heap_insert_if_better(w_dist, w_idx, k, d, id);
                        }
                    }
                    __syncwarp(warp_mask);
                }
            } // end tile scan
        }

        __syncthreads(); // Ensure tile is no longer in use before loading next
    } // end tiles

    // Finalize results: heap-sort to ascending and write out
    if (warp_active && lane == 0) {
        // Ensure heap built (data_count >= k by assumption, but guard anyway)
        if (!warp_heap_built[warp]) {
            int fill = warp_fill[warp];
            // If fill < k (shouldn't happen), pad with +inf to size k, then build heap
            for (int i = fill; i < k; ++i) {
                w_dist[i] = CUDART_INF_F;
                w_idx[i] = -1;
            }
            heap_build(w_dist, w_idx, k);
            warp_heap_built[warp] = 1;
        }
        heap_sort_ascending(w_dist, w_idx, k);

        // Write results for this query
        size_t base = size_t(warp_in_grid) * size_t(k);
        for (int j = 0; j < k; ++j) {
            result[base + j].first = w_idx[j];
            result[base + j].second = w_dist[j];
        }
    }
}

// Host interface: launches the kernel
extern "C" void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Configure launch
    dim3 block(WARPS_PER_BLOCK * 32, 1, 1);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    // Dynamic shared memory size
    size_t tile_bytes = size_t(TILE_POINTS) * sizeof(float2);
    size_t perwarp_topk_bytes = size_t(k) * (sizeof(int) + sizeof(float));
    size_t topk_bytes = size_t(WARPS_PER_BLOCK) * perwarp_topk_bytes;
    size_t control_bytes = size_t(WARPS_PER_BLOCK) * 2 * sizeof(int);
    size_t shmem_bytes = tile_bytes + topk_bytes + control_bytes;

    // Opt-in larger dynamic shared memory if needed
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_bytes);

    // Launch
    knn_kernel<<<grid, block, shmem_bytes>>>(query, query_count, data, data_count, result, k);
}