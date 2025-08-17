#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <utility>

// Device-side POD equivalent of std::pair<int, float> to avoid depending on libstdc++ device annotations.
// We will reinterpret_cast the std::pair<int,float>* pointer to this type inside run_knn.
struct PairIF {
    int first;
    float second;
};

// Utility: check for CUDA runtime errors in host code paths. This is used only in the host function below.
static inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), file, line);
        abort();
    }
}
#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

// Warp-level constants
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Max threads per block we plan to use. We adapt the actual warps per block at runtime,
// but never launch more than MAX_WARPS_PER_BLOCK warps.
#ifndef MAX_WARPS_PER_BLOCK
#define MAX_WARPS_PER_BLOCK 8
#endif

// Device utility: swap two elements in the heap arrays.
__device__ __forceinline__ void heap_swap(float* __restrict__ hd, int* __restrict__ hi, int a, int b) {
    float td = hd[a]; hd[a] = hd[b]; hd[b] = td;
    int ti = hi[a]; hi[a] = hi[b]; hi[b] = ti;
}

// Device utility: sift-up operation for a max-heap.
// Moves the element at position 'pos' up until the max-heap property is restored.
__device__ __forceinline__ void heap_sift_up(float* __restrict__ hd, int* __restrict__ hi, int pos) {
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        if (hd[parent] >= hd[pos]) break;
        heap_swap(hd, hi, parent, pos);
        pos = parent;
    }
}

// Device utility: sift-down operation for a max-heap of size 'heap_size'.
// Restores the max-heap property starting from the root at position 0.
__device__ __forceinline__ void heap_sift_down(float* __restrict__ hd, int* __restrict__ hi, int heap_size) {
    int pos = 0;
    while (true) {
        int left = (pos << 1) + 1;
        if (left >= heap_size) break;
        int right = left + 1;
        int largest = left;
        if (right < heap_size && hd[right] > hd[left]) {
            largest = right;
        }
        if (hd[pos] >= hd[largest]) break;
        heap_swap(hd, hi, pos, largest);
        pos = largest;
    }
}

// Device utility: push a new element (d, id) into the max-heap if heap_size < k.
// Post-condition: heap_size is incremented and heap property maintained.
__device__ __forceinline__ void heap_push(float* __restrict__ hd, int* __restrict__ hi, int& heap_size, float d, int id) {
    int pos = heap_size;
    hd[pos] = d;
    hi[pos] = id;
    heap_size++;
    heap_sift_up(hd, hi, pos);
}

// Device utility: replace root with (d, id) and sift-down to maintain heap property.
__device__ __forceinline__ void heap_replace_root(float* __restrict__ hd, int* __restrict__ hi, int heap_size, float d, int id) {
    hd[0] = d;
    hi[0] = id;
    heap_sift_down(hd, hi, heap_size);
}

// Device utility: pop the root from the heap (max element) and write it to (out_d, out_i).
// Decreases heap_size and maintains heap property.
__device__ __forceinline__ void heap_pop(float* __restrict__ hd, int* __restrict__ hi, int& heap_size, float& out_d, int& out_i) {
    out_d = hd[0];
    out_i = hi[0];
    int last = heap_size - 1;
    hd[0] = hd[last];
    hi[0] = hi[last];
    heap_size = last;
    if (heap_size > 0) {
        heap_sift_down(hd, hi, heap_size);
    }
}

// Each query is processed by a single warp (32 threads). Data points are processed in tiles cached in shared memory.
// Per-warp, we maintain a private max-heap of size k in shared memory (distances + indices).
// This kernel is designed for large data_count and many queries, with k in [32, 1024] (power of two).
__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data, int data_count,
    PairIF* __restrict__ result, int k,
    int warps_per_block, int tile_points)
{
    extern __shared__ unsigned char smem[];
    // Shared memory layout:
    // [tile_points * sizeof(float2)] + [warps_per_block * k * sizeof(float)] + [warps_per_block * k * sizeof(int)] + [warps_per_block * sizeof(int)]
    float2* tile = reinterpret_cast<float2*>(smem);
    size_t tile_bytes = size_t(tile_points) * sizeof(float2);
    unsigned char* ptr = smem + tile_bytes;

    float* all_heap_d = reinterpret_cast<float*>(ptr);
    ptr += size_t(warps_per_block) * size_t(k) * sizeof(float);

    int* all_heap_i = reinterpret_cast<int*>(ptr);
    ptr += size_t(warps_per_block) * size_t(k) * sizeof(int);

    int* heap_sizes = reinterpret_cast<int*>(ptr);
    // ptr += size_t(warps_per_block) * sizeof(int); // Not needed further

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_in_block = tid >> 5;
    int global_warp = blockIdx.x * warps_per_block + warp_in_block;

    bool warp_active = (global_warp < query_count);

    // Per-warp heap pointers into shared memory
    float* heap_d = all_heap_d + size_t(warp_in_block) * size_t(k);
    int*   heap_i = all_heap_i + size_t(warp_in_block) * size_t(k);

    // Initialize heap size to 0 for active warps
    if (lane == 0) {
        heap_sizes[warp_in_block] = 0;
    }
    __syncwarp();

    // Broadcast query point to all lanes in this warp
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        if (lane == 0) {
            float2 q = query[global_warp];
            qx = q.x;
            qy = q.y;
        }
        // Broadcast qx, qy from lane 0 to all lanes of the warp
        unsigned mask = 0xffffffffu;
        qx = __shfl_sync(mask, qx, 0);
        qy = __shfl_sync(mask, qy, 0);
    }

    // Process data in tiles cached in shared memory
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        int count = data_count - tile_start;
        if (count > tile_points) count = tile_points;

        // Block-level cooperative load of the tile into shared memory.
        // Use a contiguous, coalesced access pattern.
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            tile[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each active warp processes the tile; inactive warps just participate in barriers.
        if (warp_active) {
            // Stride over tile indices by warp size so each lane handles a separate subset.
            for (int t = lane; t < count; t += WARP_SIZE) {
                // Load point from shared memory
                float2 p = tile[t];
                float dx = qx - p.x;
                float dy = qy - p.y;
                // Squared Euclidean distance (use FMA for throughput)
                float d = fmaf(dx, dx, dy * dy);
                int id = tile_start + t;

                // We will perform at most one insertion per lane per processed element.
                // Identify candidates that might need an insertion into the heap.
                // We snapshot heap size to avoid reading invalid heap root when heap is not full yet.
                int local_hs = heap_sizes[warp_in_block];
                bool want_insert = (local_hs < k);
                if (!want_insert) {
                    // Now it's safe to read heap root (heap_d[0]) because heap_size >= 1
                    float worst = heap_d[0];
                    want_insert = (d < worst);
                }

                // Ballot lanes that want to insert; then handle them one-by-one in lane order.
                unsigned active = __ballot_sync(0xffffffffu, want_insert);
                while (active) {
                    int leader = __ffs(active) - 1; // index [0..31] of first set bit
                    if (lane == leader) {
                        // Re-check against up-to-date heap state
                        int& hs = heap_sizes[warp_in_block];
                        if (hs < k) {
                            // Heap not full yet: push the candidate
                            heap_push(heap_d, heap_i, hs, d, id);
                        } else {
                            // Heap full: replace root if this candidate is better (smaller distance)
                            if (d < heap_d[0]) {
                                heap_replace_root(heap_d, heap_i, hs, d, id);
                            }
                        }
                    }
                    __syncwarp();
                    // Clear processed bit and continue
                    active &= active - 1;
                }
            }
        }
        __syncthreads();
    }

    // After processing all tiles, lane 0 of each active warp extracts the k nearest neighbors
    // from the max-heap into the result array in ascending order of distance.
    if (warp_active && lane == 0) {
        int hs = heap_sizes[warp_in_block];
        // It is possible (only theoretically) that data_count < k; the problem statement guarantees data_count >= k.
        // But to be robust, clamp hs to k.
        if (hs > k) hs = k;

        // Extract from heap: popping gives elements in non-increasing order (largest first).
        // We store them into result in reverse so that result[j] is j-th nearest neighbor (ascending distances).
        PairIF* out = result + size_t(global_warp) * size_t(k);
        for (int pos = hs - 1; pos >= 0; --pos) {
            float d;
            int id;
            heap_pop(heap_d, heap_i, hs, d, id);
            out[pos].first = id;
            out[pos].second = d;
        }
        // If for any reason hs < k (should not happen as data_count >= k), pad the remaining
        // entries with invalid indices and +INF distances to maintain invariants.
        for (int pos = 0; pos < k - hs; ++pos) {
            out[pos].first = -1;
            out[pos].second = CUDART_INF_F;
        }
    }
}

// Host helper: compute an efficient launch configuration based on available shared memory.
// We adapt warps per block and tile size to fit within the device's max dynamic shared memory limit.
// - warps_per_block in {8, 4, 2, 1}
// - tile_points chosen to use the remaining shared memory after accounting for per-warp top-k storage
static inline void choose_launch_config(int k, int query_count, int data_count,
                                        int& warps_per_block, dim3& blocks, dim3& threads,
                                        int& tile_points, size_t& smem_bytes) {
    // Query device properties
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Maximum dynamic shared memory per block available (opt-in for Ampere/Hopper)
    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    int max_smem_default = prop.sharedMemPerBlock; // typically 48/64/96 KB
    int max_smem = (max_smem_optin > 0 ? max_smem_optin : max_smem_default);

    // Start with a preferred number of warps per block and reduce if necessary to fit smem.
    int candidate_warps[] = { MAX_WARPS_PER_BLOCK, 4, 2, 1 };
    warps_per_block = 1;
    tile_points = 0;
    smem_bytes = 0;

    for (int w : candidate_warps) {
        if (w <= 0) continue;
        // Per-warp top-k storage: distances (float) + indices (int) + heap_size (int per warp)
        size_t bytes_topk = size_t(w) * size_t(k) * (sizeof(float) + sizeof(int)) + size_t(w) * sizeof(int);
        // We need at least one blockDim.x worth of points in the tile to get a perfectly coalesced load.
        int block_threads = w * WARP_SIZE;
        size_t bytes_min_tile = size_t(block_threads) * sizeof(float2); // at least one per thread
        if (bytes_topk + bytes_min_tile > size_t(max_smem)) {
            // Not enough shared memory for this number of warps per block; try fewer warps.
            continue;
        }
        // Use as many points as remaining shared memory allows.
        size_t bytes_tile = size_t(max_smem) - bytes_topk;
        int tp = int(bytes_tile / sizeof(float2));
        // Round tile_points down to a multiple of block_threads for coalesced loads
        tp = (tp / block_threads) * block_threads;
        // Ensure at least one stride
        if (tp < block_threads) tp = block_threads;
        // Upper bound tile_points by data_count to avoid overprovisioning dynamic shared memory (not necessary, but reasonable)
        if (tp > data_count) tp = ((data_count + block_threads - 1) / block_threads) * block_threads;
        // Compute shared memory bytes for this configuration
        size_t smem_needed = bytes_topk + size_t(tp) * sizeof(float2);
        if (smem_needed <= size_t(max_smem)) {
            warps_per_block = w;
            tile_points = tp;
            smem_bytes = smem_needed;
            break;
        }
    }

    // Fallback safety: if the above loop fails (should not), pick 1 warp/block with minimal tile.
    if (tile_points <= 0 || smem_bytes == 0) {
        warps_per_block = 1;
        int block_threads = warps_per_block * WARP_SIZE;
        size_t bytes_topk = size_t(warps_per_block) * size_t(k) * (sizeof(float) + sizeof(int)) + size_t(warps_per_block) * sizeof(int);
        size_t bytes_tile = size_t(max_smem) - bytes_topk;
        int tp = int(bytes_tile / sizeof(float2));
        tp = (tp / block_threads) * block_threads;
        if (tp < block_threads) tp = block_threads;
        tile_points = tp;
        smem_bytes = bytes_topk + size_t(tp) * sizeof(float2);
    }

    // Compute launch grid/block dimensions
    threads = dim3(warps_per_block * WARP_SIZE);
    int blocks_needed = (query_count + warps_per_block - 1) / warps_per_block;
    blocks = dim3(blocks_needed, 1, 1);
}

// Entry point: launch the optimized CUDA kernel.
// - query: pointer to device memory array of float2, size query_count
// - data: pointer to device memory array of float2, size data_count
// - result: pointer to device memory array of std::pair<int,float>, size query_count * k
// - k: number of neighbors (power of two between 32 and 1024), with data_count >= k
void run_knn(const float2* query, int query_count, const float2* data, int data_count, std::pair<int, float>* result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Choose launch configuration based on device shared memory capacity and inputs
    int warps_per_block = 0;
    int tile_points = 0;
    size_t smem_bytes = 0;
    dim3 blocks, threads;
    choose_launch_config(k, query_count, data_count, warps_per_block, blocks, threads, tile_points, smem_bytes);

    // Opt-in to use the maximum dynamic shared memory available on the device for this kernel
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (max_smem_optin > 0) {
        CUDA_CHECK(cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_optin));
    }

    // Launch kernel. We pass result as POD PairIF* to be writable by device code.
    knn_kernel<<<blocks, threads, smem_bytes>>>(
        query, query_count, data, data_count,
        reinterpret_cast<PairIF*>(result), k,
        warps_per_block, tile_points
    );
    CUDA_CHECK(cudaGetLastError());
}