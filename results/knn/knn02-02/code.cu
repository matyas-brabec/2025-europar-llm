#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>

// This implementation assigns exactly one warp (32 threads) to each query point.
// Each warp maintains a private max-heap of size k in shared memory to track the current top-k nearest neighbors.
// The dataset is processed in tiles cached in shared memory for coalesced global loads and re-use within the block.
// The heap is updated in a warp-synchronous manner: within each group of 32 candidate distances, only those
// that are smaller than the current heap maximum are considered. Candidates are inserted serially by lane 0,
// using a standard heap "sift-down" in shared memory with O(log k) time per insertion. Other lanes cooperate
// via warp shuffles and ballots to pick the current best candidate(s) to insert, minimizing the number of heap updates.
//
// Notes:
// - Distances are squared Euclidean (no sqrt).
// - k is a power of two in [32, 1024]. One warp per query; multiple warps per block.
// - Initial heap is built from the first k data points, loaded directly from global memory for simplicity.
// - Remaining points are processed in shared-memory tiles.
// - At the end, the heap is heap-sorted in-place to ascending order and results are stored to the output array.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Warp full mask for 32-thread warps
#define FULL_MASK 0xFFFFFFFFu

// Device inline: squared L2 distance between two float2 points
static __device__ __forceinline__ float l2_sq(const float2 a, const float2 b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    // Use fused multiply-add for accuracy and throughput: dx*dx + dy*dy
    return fmaf(dy, dy, dx * dx);
}

// Heap utilities operating on shared-memory arrays (max-heap, key = distance)
// The heap is per-warp and is updated only by a single thread (lane 0) at any time.

static __device__ __forceinline__ void heap_sift_down(float* __restrict__ hDist,
                                                      int* __restrict__ hIdx,
                                                      int heap_size,
                                                      int start_idx) {
    int i = start_idx;
    while (true) {
        int left = (i << 1) + 1;
        if (left >= heap_size) break;
        int right = left + 1;
        int largest = left;
        if (right < heap_size && hDist[right] > hDist[left]) {
            largest = right;
        }
        if (hDist[i] < hDist[largest]) {
            // swap i and largest
            float td = hDist[i];
            hDist[i] = hDist[largest];
            hDist[largest] = td;

            int ti = hIdx[i];
            hIdx[i] = hIdx[largest];
            hIdx[largest] = ti;

            i = largest;
        } else {
            break;
        }
    }
}

static __device__ __forceinline__ void heap_build(float* __restrict__ hDist,
                                                  int* __restrict__ hIdx,
                                                  int heap_size) {
    // Build a max-heap from arbitrary array contents: O(heap_size)
    for (int i = (heap_size >> 1) - 1; i >= 0; --i) {
        heap_sift_down(hDist, hIdx, heap_size, i);
    }
}

static __device__ __forceinline__ void heap_replace_root(float* __restrict__ hDist,
                                                         int* __restrict__ hIdx,
                                                         int heap_size,
                                                         float new_dist,
                                                         int new_idx) {
    // Replace root with new value and restore heap property
    hDist[0] = new_dist;
    hIdx[0] = new_idx;
    heap_sift_down(hDist, hIdx, heap_size, 0);
}

static __device__ __forceinline__ void heap_heapsort_ascending(float* __restrict__ hDist,
                                                               int* __restrict__ hIdx,
                                                               int heap_size) {
    // Standard heapsort on a max-heap array: After completion, array is sorted ascending by distance.
    for (int end = heap_size - 1; end > 0; --end) {
        // swap root (max) with end
        float td = hDist[0];
        hDist[0] = hDist[end];
        hDist[end] = td;

        int ti = hIdx[0];
        hIdx[0] = hIdx[end];
        hIdx[end] = ti;

        // sift-down root, with heap_size reduced by 1
        int i = 0;
        int size = end;
        while (true) {
            int left = (i << 1) + 1;
            if (left >= size) break;
            int right = left + 1;
            int largest = left;
            if (right < size && hDist[right] > hDist[left]) largest = right;
            if (hDist[i] < hDist[largest]) {
                float td2 = hDist[i];
                hDist[i] = hDist[largest];
                hDist[largest] = td2;

                int ti2 = hIdx[i];
                hIdx[i] = hIdx[largest];
                hIdx[largest] = ti2;

                i = largest;
            } else {
                break;
            }
        }
    }
}

// Warp reduction to find lane index of minimal value among the 32 candidates.
// Inactive lanes should pass val = +inf so they never win. The minimal value and its lane index
// will become available in lane 0 after this function.
static __device__ __forceinline__ void warp_find_min_lane(float& val, int& lane_of_val) {
    // Track lane indices alongside values to extract the winning lane
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(FULL_MASK, val, offset);
        int other_lane = __shfl_down_sync(FULL_MASK, lane_of_val, offset);
        // Choose strictly smaller; tie-breaker: smaller lane id wins for determinism
        if ((other_val < val) || (other_val == val && other_lane < lane_of_val)) {
            val = other_val;
            lane_of_val = other_lane;
        }
    }
}

// Kernel implementing k-NN for 2D points with one warp per query.
// Shared memory layout per block (dynamic):
// [0 ... tile_points-1]                        : float2 tile buffer (shared across warps in block)
// [tile_points ... tile_points + W*K - 1]     : int topIdx for each warp, contiguous segments of length K
// [next ... next + W*K - 1]                   : float topDist for each warp, contiguous segments of length K
__global__ void knn2d_warp_kernel(const float2* __restrict__ query,
                                  int query_count,
                                  const float2* __restrict__ data,
                                  int data_count,
                                  std::pair<int, float>* __restrict__ result,
                                  int k,
                                  int tile_points)
{
    extern __shared__ unsigned char smem_raw[];
    float2* const tile = reinterpret_cast<float2*>(smem_raw);

    // Compute number of warps per block and identifiers
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int thread_in_block = threadIdx.x;
    const int lane = thread_in_block & (WARP_SIZE - 1);
    const int warp_in_block = thread_in_block >> 5;
    const int warp_global = warps_per_block * blockIdx.x + warp_in_block;

    if (warp_global >= query_count) {
        return;
    }

    // Pointers to per-warp top-k storage in shared memory
    // Compute base pointers after tile buffer
    // Align to 4-byte boundary (already aligned due to float2 usage, but be explicit)
    unsigned char* p = reinterpret_cast<unsigned char*>(tile + tile_points);
    // Integer indices
    int* const topIdxBase = reinterpret_cast<int*>(p);
    // Advance by warps_per_block * k ints
    float* const topDistBase = reinterpret_cast<float*>(topIdxBase + warps_per_block * k);
    // Per-warp slices
    int* const heapIdx = topIdxBase + warp_in_block * k;
    float* const heapDist = topDistBase + warp_in_block * k;

    // Load query point into registers (all lanes get the same data)
    const float2 q = query[warp_global];

    // Phase 1: Initialize heap from the first k data points (global memory direct loads).
    // Each lane processes m = k / 32 elements (k is power of 2, >= 32 => divisible by 32).
    const int m = k / WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < m; ++i) {
        const int idx = i * WARP_SIZE + lane; // 0..k-1
        // data_count >= k guaranteed; no bounds check needed
        const float2 pnt = data[idx];
        heapIdx[idx] = idx;
        heapDist[idx] = l2_sq(pnt, q);
    }
    __syncwarp(); // ensure initial values are stored before building the heap

    // Build max-heap from initial k distances using lane 0
    if (lane == 0) {
        heap_build(heapDist, heapIdx, k);
    }
    __syncwarp();

    // Phase 2: Process remaining data in tiles cached in shared memory.
    // Start from index = k because first k were used for heap initialization.
    for (int base = k; base < data_count; base += tile_points) {
        const int count = min(tile_points, data_count - base);

        // Block-wide cooperative load into shared memory
        for (int t = thread_in_block; t < count; t += blockDim.x) {
            tile[t] = data[base + t];
        }
        __syncthreads();

        // Each warp scans the tile in strides of 32
        for (int off = 0; off < count; off += WARP_SIZE) {
            const int idx_in_tile = off + lane;
            const int gidx = base + idx_in_tile;

            // Compute distance for this candidate; if out of bounds, set to +inf
            float candDist = CUDART_INF_F;
            int candIdx = -1;
            if (idx_in_tile < count) {
                const float2 pnt = tile[idx_in_tile];
                candDist = l2_sq(pnt, q);
                candIdx = gidx;
            }

            // Snapshot of current heap threshold (max distance in top-k) from lane 0
            float tau = 0.0f;
            if (lane == 0) {
                tau = heapDist[0];
            }
            tau = __shfl_sync(FULL_MASK, tau, 0);

            // Mark active if this candidate might improve the heap
            bool active = (candDist < tau);

            // Process all improving candidates from this warp-group of 32 elements.
            // At each iteration, the best candidate among active lanes is inserted into the heap by lane 0.
            while (true) {
                unsigned int active_mask = __ballot_sync(FULL_MASK, active);
                if (active_mask == 0u) break;

                // Reduction to find lane with minimal candidate distance
                float v = active ? candDist : CUDART_INF_F;
                int min_lane = lane;
                warp_find_min_lane(v, min_lane);

                // Broadcast the winning lane and distance to lane 0
                const int winner_lane = __shfl_sync(FULL_MASK, min_lane, 0);
                const float winner_dist = __shfl_sync(FULL_MASK, v, 0);
                const int winner_idx = __shfl_sync(FULL_MASK, candIdx, winner_lane);

                // Insert into heap (lane 0 only), then update tau
                if (lane == 0) {
                    heap_replace_root(heapDist, heapIdx, k, winner_dist, winner_idx);
                }
                // Broadcast updated tau from lane 0 to all lanes
                float new_tau = 0.0f;
                if (lane == 0) {
                    new_tau = heapDist[0];
                }
                new_tau = __shfl_sync(FULL_MASK, new_tau, 0);

                // Update active flags: the winner is no longer active; others remain active only if they still beat new_tau
                active = active && (lane != winner_lane) && (candDist < new_tau);
            }
            // Next group of 32 candidates
        }
        __syncthreads(); // ensure tile buffer is not read while being overwritten by other blocks
    }

    // Phase 3: Sort the heap into ascending order and write results
    if (lane == 0) {
        heap_heapsort_ascending(heapDist, heapIdx, k);
    }
    __syncwarp();

    // Store results to global memory: each lane writes a strided subset for coalescing
    const int out_base = warp_global * k;
    for (int i = lane; i < k; i += WARP_SIZE) {
        result[out_base + i].first = heapIdx[i];
        result[out_base + i].second = heapDist[i];
    }
}

// Host helper to round down to a multiple
static inline int round_down(int x, int multiple) {
    return (x / multiple) * multiple;
}

// Host function to choose block size, shared memory sizes, and launch the kernel.
void run_knn(const float2* query, int query_count,
             const float2* data, int data_count,
             std::pair<int, float>* result, int k)
{
    // Choose warps per block and threads per block. Start with 8 warps (256 threads),
    // reduce if shared memory budget requires.
    int warps_per_block = 8;
    int threads_per_block = warps_per_block * WARP_SIZE;

    // Determine maximum dynamically allocated shared memory per block (opt-in)
    int device = 0;
    cudaGetDevice(&device);
    int max_optin_smem = 0;
    cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_optin_smem == 0) {
        // Fallback to the default maximum if opt-in not supported
        cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    }

    // Compute shared memory needed for per-warp top-k storage
    // Each entry is an int index and a float distance
    size_t per_warp_topk_bytes = static_cast<size_t>(k) * (sizeof(int) + sizeof(float));
    // Adjust warps per block if necessary to leave room for the tile
    while (warps_per_block > 0) {
        size_t topk_bytes = per_warp_topk_bytes * static_cast<size_t>(warps_per_block);
        if (topk_bytes < static_cast<size_t>(max_optin_smem)) break;
        warps_per_block >>= 1;
    }
    if (warps_per_block == 0) {
        // As a last resort, run with a single warp per block (should not happen with given constraints)
        warps_per_block = 1;
    }
    threads_per_block = warps_per_block * WARP_SIZE;

    // Compute tile size so that total shared memory fits within the opt-in limit.
    // tile_bytes = max_optin_smem - topk_bytes
    size_t topk_bytes = per_warp_topk_bytes * static_cast<size_t>(warps_per_block);
    size_t max_tile_bytes = (max_optin_smem > static_cast<int>(topk_bytes)) ? (max_optin_smem - topk_bytes) : 0;

    // Each tile element is a float2
    int tile_points = 0;
    if (max_tile_bytes >= sizeof(float2)) {
        tile_points = static_cast<int>(max_tile_bytes / sizeof(float2));
    }
    // Round tile_points down to a multiple of blockDim (for easy coalesced loads), but keep at least one warp
    tile_points = round_down(tile_points, threads_per_block);
    if (tile_points < WARP_SIZE) {
        // Ensure at least one warp's worth of points in the tile
        tile_points = WARP_SIZE;
    }
    // Also cap tile_points to data_count-k to avoid allocating beyond what's needed (optional)
    if (tile_points > data_count) tile_points = data_count;

    // Final dynamic shared memory size for the kernel launch
    size_t dyn_smem_size = static_cast<size_t>(tile_points) * sizeof(float2) + topk_bytes;

    // Set kernel attribute to allow using large dynamic shared memory if needed
    // Use the max opt-in to be safe across different k
    cudaFuncSetAttribute(knn2d_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin_smem);

    // Compute grid size: one warp per query
    int grid_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch kernel
    dim3 grid(grid_blocks);
    dim3 block(threads_per_block);
    knn2d_warp_kernel<<<grid, block, dyn_smem_size>>>(query, query_count, data, data_count, result, k, tile_points);

    // Optionally, synchronize or check errors here in production code
    // cudaDeviceSynchronize();
}