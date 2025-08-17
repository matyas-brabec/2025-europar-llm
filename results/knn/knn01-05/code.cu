#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <utility>

// Optimized k-NN for 2D points using a block that processes multiple queries (one warp per query).
// - Each block contains WARPS_PER_BLOCK warps (default 8), thus 256 threads.
// - The block iterates over the data in tiles loaded into shared memory to maximize reuse across warps.
// - Each warp maintains its own top-k as a max-heap in shared memory. Insertions/replacements are
//   serialized within the warp via ballot/ffs to avoid races. Heap operations are O(log k).
// - At the end, each warp performs an in-place heap sort to produce ascending nearest neighbors,
//   then writes results [index, squared distance] for its assigned query.
//
// Notes:
// - Distances are squared Euclidean distances (no sqrt).
// - k is power-of-two in [32, 1024], and data_count >= k.
// - The kernel uses only shared memory (no additional global allocations).
// - For large k, we opt-in to larger dynamic shared memory (A100/H100 support up to ~164 KB).
// - Hyperparameters chosen for H100/A100-class GPUs.
//
// You can tune WARPS_PER_BLOCK and BASE_TILE_POINTS below if needed.

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)
#endif

// A reasonable base tile size. Actual tile_points used at runtime may be reduced to fit shared memory.
#ifndef BASE_TILE_POINTS
#define BASE_TILE_POINTS 2048
#endif

// Device utility: swap two entries in parallel arrays (distance and index).
__device__ __forceinline__ void swap_pair(float &ad, int &ai, float &bd, int &bi) {
    float td = ad;
    int ti = ai;
    ad = bd; ai = bi;
    bd = td; bi = ti;
}

// Device utility: sift-up operation for max-heap (by distance) at position pos.
__device__ __forceinline__ void heap_sift_up(float* __restrict__ hd, int* __restrict__ hi, int pos) {
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        if (hd[parent] >= hd[pos]) break;
        swap_pair(hd[parent], hi[parent], hd[pos], hi[pos]);
        pos = parent;
    }
}

// Device utility: sift-down operation for max-heap (by distance) starting at position pos with heap size n.
__device__ __forceinline__ void heap_sift_down(float* __restrict__ hd, int* __restrict__ hi, int n, int pos) {
    while (true) {
        int left = (pos << 1) + 1;
        if (left >= n) break;
        int right = left + 1;
        int largest = left;
        if (right < n && hd[right] > hd[left]) largest = right;
        if (hd[pos] >= hd[largest]) break;
        swap_pair(hd[pos], hi[pos], hd[largest], hi[largest]);
        pos = largest;
    }
}

// Device utility: in-place heap sort (ascending) given a max-heap.
// After completion, hd[0..n-1] and hi[0..n-1] are sorted ascending by distance.
__device__ __forceinline__ void heap_sort_ascending(float* __restrict__ hd, int* __restrict__ hi, int n) {
    for (int end = n - 1; end > 0; --end) {
        swap_pair(hd[0], hi[0], hd[end], hi[end]);
        heap_sift_down(hd, hi, end, 0);
    }
}

// The main CUDA kernel.
// Shared memory layout (dynamic):
//   [0 .. tile_points-1]                               -> float2 tile data points (shared across warps in the block)
//   [tile_points .. tile_points + warps*k - 1]         -> float warp_dists (per-warp heaps concatenated)
//   [.. + warps*k .. + warps*k + warps*k - 1]          -> int   warp_indices (per-warp heaps concatenated)
//   [.. + warps*k + warps*k .. + warps*k + warps*k + warps - 1] -> int warp_heap_size (one per warp)
__global__ void knn2d_kernel(const float2* __restrict__ query,
                             int query_count,
                             const float2* __restrict__ data,
                             int data_count,
                             std::pair<int, float>* __restrict__ result,
                             int k,
                             int tile_points) {
    extern __shared__ unsigned char smem[];
    unsigned char* smem_ptr = smem;

    // Shared memory pointers setup
    float2* sm_tile = reinterpret_cast<float2*>(smem_ptr);
    smem_ptr += static_cast<size_t>(tile_points) * sizeof(float2);

    float* sm_warp_dists = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(float);

    int* sm_warp_indices = reinterpret_cast<int*>(smem_ptr);
    smem_ptr += static_cast<size_t>(WARPS_PER_BLOCK) * k * sizeof(int);

    int* sm_warp_heap_size = reinterpret_cast<int*>(smem_ptr);
    // smem_ptr += WARPS_PER_BLOCK * sizeof(int); // not needed further

    const int warp_id = threadIdx.x >> 5;       // warp index within block
    const int lane_id = threadIdx.x & 31;       // lane index
    const unsigned full_mask = 0xFFFFFFFFu;

    // This block handles WARPS_PER_BLOCK queries: global query index for this warp:
    const int qidx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool warp_active = (qidx < query_count);

    // Per-warp pointers into shared memory for the heap arrays.
    float* __restrict__ heap_d = sm_warp_dists + warp_id * k;
    int*   __restrict__ heap_i = sm_warp_indices + warp_id * k;
    int*   __restrict__ heap_size_ptr = sm_warp_heap_size + warp_id;

    // Initialize per-warp heap size.
    if (lane_id == 0) {
        *heap_size_ptr = 0;
    }

    // Broadcast the query point to all lanes in the warp.
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        if (lane_id == 0) {
            float2 q = query[qidx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(full_mask, qx, 0);
        qy = __shfl_sync(full_mask, qy, 0);
    }

    // Iterate over the data set in tiles.
    for (int base = 0; base < data_count; base += tile_points) {
        int count = data_count - base;
        if (count > tile_points) count = tile_points;

        // Cooperative load of the data tile into shared memory by all threads in the block.
        for (int t = threadIdx.x; t < count; t += blockDim.x) {
            sm_tile[t] = data[base + t];
        }

        __syncthreads();

        // Each active warp processes this tile and updates its heap.
        if (warp_active) {
            for (int i = lane_id; i < count; i += 32) {
                float2 p = sm_tile[i];
                float dx = qx - p.x;
                float dy = qy - p.y;
                float dist = fmaf(dx, dx, dy * dy);
                int   didx = base + i;

                // Warp-serialized cooperative heap update.
                // Lanes with candidates that should be inserted/replaced will be processed one at a time.
                while (true) {
                    // Read current heap size and threshold (root). Avoid reading heap_d[0] if empty.
                    int hsz = *heap_size_ptr;
                    float thr = (hsz > 0 ? heap_d[0] : CUDART_INF_F);

                    // Decide if this lane wants to attempt an update.
                    int want = (hsz < k) || (dist < thr);
                    unsigned mask = __ballot_sync(full_mask, want);
                    if (mask == 0u) break;

                    int leader = __ffs(mask) - 1;  // 0..31
                    if (lane_id == leader) {
                        int cur_hsz = *heap_size_ptr;
                        if (cur_hsz < k) {
                            // Insert new element at the end, then sift-up.
                            int pos = cur_hsz;
                            heap_d[pos] = dist;
                            heap_i[pos] = didx;
                            *heap_size_ptr = cur_hsz + 1;
                            heap_sift_up(heap_d, heap_i, pos);
                        } else {
                            // Replace root if better, then sift-down.
                            if (dist < heap_d[0]) {
                                heap_d[0] = dist;
                                heap_i[0] = didx;
                                heap_sift_down(heap_d, heap_i, k, 0);
                            }
                        }
                    }
                    __syncwarp(full_mask);
                    // Re-evaluate; if this lane's candidate is no longer desirable, it will naturally stop.
                }
            }
        }

        __syncthreads();
    }

    // Finalize: each active warp sorts its heap ascending and writes out results.
    if (warp_active) {
        if (lane_id == 0) {
            int hsz = *heap_size_ptr;
            // hsz should be k (since data_count >= k). If not, heap_sort still works on hsz items.
            if (hsz > 1) {
                // The array is already a max-heap by construction; run in-place heap sort to ascending.
                heap_sort_ascending(heap_d, heap_i, hsz);
            }
            // Write results in ascending order: j-th neighbor for query qidx.
            int out_base = qidx * k;
            // If hsz < k (shouldn't happen given data_count >= k), fill the rest with dummy values.
            for (int j = 0; j < hsz; ++j) {
                result[out_base + j].first  = heap_i[j];
                result[out_base + j].second = heap_d[j];
            }
            for (int j = hsz; j < k; ++j) {
                // Should not occur under given constraints, but kept for safety.
                result[out_base + j].first  = -1;
                result[out_base + j].second = CUDART_INF_F;
            }
        }
    }
}

// Host launcher: selects tile size to fit shared memory, sets opt-in shared memory size, and launches the kernel.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k) {
    // Hyperparameters tuned for A100/H100-class GPUs.
    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = THREADS_PER_BLOCK;

    // Compute how much shared memory is needed as a function of tile_points.
    auto smem_required = [&](int tile_points) -> size_t {
        size_t tile_bytes = static_cast<size_t>(tile_points) * sizeof(float2);
        size_t heaps_bytes = static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));
        size_t control_bytes = static_cast<size_t>(warps_per_block) * sizeof(int); // heap sizes
        return tile_bytes + heaps_bytes + control_bytes;
    };

    // Query the device's opt-in shared memory limit and the default per-block limit.
    int device = 0;
    cudaGetDevice(&device);
    int max_optin_smem = 0;
    cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    int default_smem = 0;
    cudaDeviceGetAttribute(&default_smem, cudaDevAttrMaxSharedMemoryPerBlock, device);

    // Base tile size. We may reduce it if needed to fit shared memory constraints.
    int tile_points = BASE_TILE_POINTS;

    // Compute minimum shared memory required without the tile to ensure k/warps fit.
    size_t min_required_wo_tile = static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int))
                                + static_cast<size_t>(warps_per_block) * sizeof(int);

    // Choose the maximum available shared memory (opt-in if supported).
    int max_available_smem = max_optin_smem > 0 ? max_optin_smem : default_smem;

    // Clamp tile_points to fit within available shared memory.
    if (smem_required(tile_points) > static_cast<size_t>(max_available_smem)) {
        // Reduce tile_points to the largest value that fits.
        size_t available_for_tile = static_cast<size_t>(max_available_smem) - min_required_wo_tile;
        int max_tile_points_fit = static_cast<int>(available_for_tile / sizeof(float2));
        if (max_tile_points_fit < 1) {
            // As a last resort, force tile_points to 1 (should not happen for target GPUs and k<=1024).
            tile_points = 1;
        } else {
            tile_points = max_tile_points_fit;
        }
    }

    size_t dynamic_smem_bytes = smem_required(tile_points);

    // Opt-in to larger dynamic shared memory if supported.
    if (max_optin_smem > 0 && dynamic_smem_bytes > static_cast<size_t>(default_smem)) {
        cudaFuncSetAttribute(knn2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(dynamic_smem_bytes));
    }

    // Grid sizing: one warp per query, WARPS_PER_BLOCK warps per block.
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch kernel.
    knn2d_kernel<<<blocks, threads_per_block, dynamic_smem_bytes>>>(
        query, query_count, data, data_count, result, k, tile_points
    );
}