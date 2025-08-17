#include <cuda_runtime.h>
#include <utility>
#include <algorithm>
#include <cmath>

// This implementation computes k-NN for 2D points using a single warp (32 threads) per query.
// The algorithm processes the data in shared-memory tiles. Each warp maintains:
// - An intermediate result of size k (sorted ascending by distance).
// - A candidate buffer of size k (unsorted; sorted on flush).
// - A temporary array of size k for merging/sorting.
// - A shared counter for the number of candidates currently stored in the buffer.
// When the candidate buffer becomes full (>= k), the warp flushes it:
//   1) Sort the buffer with bitonic sort (ascending).
//   2) Merge the buffer and the intermediate result into a bitonic sequence using pairwise min between
//      buffer[i] and intermediate[k-1-i].
//   3) Sort the merged result with bitonic sort (ascending) to get the updated intermediate result.
// The warp maintains a max_distance value equal to the distance of the current k-th neighbor (last element
// of the intermediate result). New distances >= max_distance are ignored at insertion time.
//
// The kernel uses dynamic shared memory and cooperatively loads tiles of data points per block.
// The host function selects warps per block and tile size to fit into the device shared memory budget,
// and opts-in to use the maximum dynamic shared memory when available on modern GPUs (A100/H100).

// Helpers
#define CUDA_FULL_MASK 0xFFFFFFFFu
static constexpr int WARP_SIZE = 32;

struct KDPair {
    float dist;
    int   idx;
};

// Align an offset up to the specified alignment (which is a power of two)
__host__ __device__ static inline size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

// Warp-synchronous bitonic sort on shared-memory array "arr" of length n (power of two).
// Sorts ascending by distance. Threads participate cooperatively with strided indices.
__device__ __forceinline__ void warp_bitonic_sort(KDPair* arr, int n) {
    // Bitonic sort with XOR-based partner indexing, using the standard nested-loop structure.
    for (int kstage = 2; kstage <= n; kstage <<= 1) {
        for (int j = kstage >> 1; j > 0; j >>= 1) {
            // Each lane processes multiple indices i in [0, n) with stride WARP_SIZE
            for (int i = threadIdx.x & (WARP_SIZE - 1); i < n; i += WARP_SIZE) {
                int l = i ^ j;
                if (l > i) {
                    bool up = ((i & kstage) == 0);
                    KDPair ai = arr[i];
                    KDPair al = arr[l];
                    bool swap_cond = up ? (ai.dist > al.dist) : (ai.dist < al.dist);
                    if (swap_cond) {
                        arr[i] = al;
                        arr[l] = ai;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Fill arr[n] with value v cooperatively by a warp (strided by lane).
__device__ __forceinline__ void warp_fill(KDPair* arr, int n, const KDPair& v) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    for (int i = lane; i < n; i += WARP_SIZE) {
        arr[i] = v;
    }
    __syncwarp();
}

// Copy src[n] to dst[n] cooperatively by a warp (strided by lane).
__device__ __forceinline__ void warp_copy(KDPair* dst, const KDPair* src, int n) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    for (int i = lane; i < n; i += WARP_SIZE) {
        dst[i] = src[i];
    }
    __syncwarp();
}

// Merge two sorted arrays 'buf_sorted' (ascending) and 'inter_sorted' (ascending) into 'merged' (size n)
// via the bitonic "min" trick: merged[i] = min(buf_sorted[i], inter_sorted[n-1-i]) by distance.
// Result is a bitonic sequence which is then sorted with bitonic sort to obtain ascending order.
__device__ __forceinline__ void warp_bitonic_min_merge(KDPair* merged, const KDPair* buf_sorted, const KDPair* inter_sorted, int n) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    for (int i = lane; i < n; i += WARP_SIZE) {
        KDPair a = buf_sorted[i];
        KDPair b = inter_sorted[n - 1 - i];
        merged[i] = (a.dist < b.dist) ? a : b;
    }
    __syncwarp();
}

// Flush the candidate buffer into the intermediate result.
// Steps:
// 1) Pad the buffer (positions [count, k)) with +inf distances.
// 2) Sort the buffer with warp_bitonic_sort (ascending).
// 3) Merge buffer and intermediate with warp_bitonic_min_merge into 'merged'.
// 4) Sort merged with warp_bitonic_sort (ascending).
// 5) Copy merged to intermediate.
// 6) Reset candidate count and update max_distance (return it).
__device__ __forceinline__ float warp_flush_buffer(KDPair* buf, KDPair* inter, KDPair* merged,
                                                   int k, int* candidate_count_ptr) {
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Read current candidate count (the buffer might have logically more candidates than we physically store).
    int count = 0;
    if (lane == 0) count = *candidate_count_ptr;
    count = __shfl_sync(CUDA_FULL_MASK, count, 0);

    // We only physically store the first min(count, k) entries; pad the rest with +inf.
    int effective = count < k ? count : k;

    // Pad [effective, k) with inf distances and invalid indices.
    KDPair infv; infv.dist = CUDART_INF_F; infv.idx = -1;
    for (int i = lane + effective; i < k; i += WARP_SIZE) {
        buf[i] = infv;
    }
    __syncwarp();

    // Sort buffer ascending
    warp_bitonic_sort(buf, k);

    // Merge with intermediate using bitonic min trick
    warp_bitonic_min_merge(merged, buf, inter, k);

    // Sort merged ascending
    warp_bitonic_sort(merged, k);

    // Copy back to intermediate
    warp_copy(inter, merged, k);

    // Reset candidate count to 0
    if (lane == 0) *candidate_count_ptr = 0;
    __syncwarp();

    // Update and return max_distance = inter[k-1].dist
    float max_d = 0.0f;
    if (lane == 0) max_d = inter[k - 1].dist;
    max_d = __shfl_sync(CUDA_FULL_MASK, max_d, 0);
    return max_d;
}

// Kernel
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k,
                           int tile_points,          // number of data points per shared-memory tile
                           int warps_per_block) {    // warps per block = blockDim.x / 32
    extern __shared__ unsigned char smem_raw[];
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Shared memory layout:
    // [ tile (float2[tile_points]) ]
    // [ padding to KDPair alignment ]
    // [ inter  (KDPair[warps_per_block * k]) ]
    // [ buf    (KDPair[warps_per_block * k]) ]
    // [ merged (KDPair[warps_per_block * k]) ]
    // [ counters (int[warps_per_block]) ]

    size_t offset = 0;
    float2* tile = reinterpret_cast<float2*>(smem_raw + offset);
    offset += static_cast<size_t>(tile_points) * sizeof(float2);
    offset = align_up(offset, alignof(KDPair));

    KDPair* inter_base  = reinterpret_cast<KDPair*>(smem_raw + offset);
    offset += static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * sizeof(KDPair);

    KDPair* buf_base    = reinterpret_cast<KDPair*>(smem_raw + offset);
    offset += static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * sizeof(KDPair);

    KDPair* merged_base = reinterpret_cast<KDPair*>(smem_raw + offset);
    offset += static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * sizeof(KDPair);

    offset = align_up(offset, alignof(int));
    int* counters = reinterpret_cast<int*>(smem_raw + offset);
    // No further offset needed

    // Per-warp slices
    KDPair* inter  = inter_base  + static_cast<size_t>(warp_id) * static_cast<size_t>(k);
    KDPair* buf    = buf_base    + static_cast<size_t>(warp_id) * static_cast<size_t>(k);
    KDPair* merged = merged_base + static_cast<size_t>(warp_id) * static_cast<size_t>(k);
    int* candidate_count_ptr = &counters[warp_id];

    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    const int warp_stride = gridDim.x * warps_per_block;

    for (int q = global_warp_id; q < query_count; q += warp_stride) {
        // Initialize intermediate result to +inf distances.
        KDPair infv; infv.dist = CUDART_INF_F; infv.idx = -1;
        warp_fill(inter, k, infv);

        if (lane == 0) *candidate_count_ptr = 0;
        __syncwarp();

        // Load query point into registers
        float2 qpt = query[q];
        float max_distance = CUDART_INF_F;

        // Process data in tiles
        for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
            int tile_len = data_count - tile_start;
            if (tile_len > tile_points) tile_len = tile_points;

            // Load the tile cooperatively by the whole block
            for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
                tile[t] = data[tile_start + t];
            }
            __syncthreads();

            // Each warp processes the tile for its query
            for (int t = lane; t < tile_len; t += WARP_SIZE) {
                float2 p = tile[t];
                float dx = p.x - qpt.x;
                float dy = p.y - qpt.y;
                float d = dx * dx + dy * dy;

                if (d < max_distance) {
                    int pos = atomicAdd(candidate_count_ptr, 1);
                    if (pos < k) {
                        buf[pos].dist = d;
                        buf[pos].idx  = tile_start + t;
                    }
                }

                __syncwarp();
                // If buffer is full, flush and merge
                bool do_flush = false;
                if (lane == 0) do_flush = (*candidate_count_ptr >= k);
                do_flush = __shfl_sync(CUDA_FULL_MASK, do_flush, 0);
                if (do_flush) {
                    max_distance = warp_flush_buffer(buf, inter, merged, k, candidate_count_ptr);
                }
            }

            __syncwarp();
            __syncthreads();
        }

        // Flush any remaining candidates
        int remaining = 0;
        if (lane == 0) remaining = *candidate_count_ptr;
        remaining = __shfl_sync(CUDA_FULL_MASK, remaining, 0);
        if (remaining > 0) {
            max_distance = warp_flush_buffer(buf, inter, merged, k, candidate_count_ptr);
        }

        // Write out the final k nearest neighbors for this query (inter is sorted ascending)
        for (int i = lane; i < k; i += WARP_SIZE) {
            const KDPair e = inter[i];
            std::pair<int, float>& out = result[static_cast<size_t>(q) * static_cast<size_t>(k) + i];
            out.first  = e.idx;
            out.second = e.dist;
        }
        __syncwarp();
    }
}

// Host function: selects block configuration and shared memory size, then launches the kernel.
void run_knn(const float2* query, int query_count,
             const float2* data, int data_count,
             std::pair<int, float>* result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Determine device and shared memory capabilities
    int device = 0;
    cudaGetDevice(&device);

    int max_blocks = 0;
    cudaDeviceGetAttribute(&max_blocks, cudaDevAttrMultiProcessorCount, device);

    int max_shmem_optin = 0;
    cudaDeviceGetAttribute(&max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    int max_shmem_default = 0;
    cudaDeviceGetAttribute(&max_shmem_default, cudaDevAttrMaxSharedMemoryPerBlock, device);

    // We'll try to use the opt-in maximum shared memory if available; otherwise default limit.
    int max_shmem_available = (max_shmem_optin > 0) ? max_shmem_optin : max_shmem_default;

    // Choose warps per block to fit shared memory usage. Try from 6 down to 1 to allow flexibility.
    // Each warp needs 3*k KDPair entries + 1 int for the counter.
    // Additionally, we need tile_points * sizeof(float2) bytes.
    // We'll select tile_points based on remaining shared memory after per-warp allocations.
    int warps_per_block = 4; // start target
    const int max_try_warps = 6;
    for (int try_warps = std::min(max_try_warps, (int)((max_shmem_available) / std::max(1, 3 * k * (int)sizeof(KDPair) + (int)sizeof(int)))); try_warps >= 1; --try_warps) {
        size_t per_warp_bytes = static_cast<size_t>(3) * static_cast<size_t>(k) * sizeof(KDPair) + sizeof(int);
        size_t warp_region = per_warp_bytes * try_warps;
        if ((int)warp_region >= max_shmem_available) continue; // leaves no room for tile
        // leave at least space for a minimal tile (at least one warp's worth of points)
        size_t remaining = static_cast<size_t>(max_shmem_available) - warp_region;
        if (remaining >= sizeof(float2) * 32) { // at least one warp of points
            warps_per_block = try_warps;
            break;
        }
    }

    // Compute tile_points as large as possible given shared memory budget
    size_t per_warp_bytes = static_cast<size_t>(3) * static_cast<size_t>(k) * sizeof(KDPair) + sizeof(int);
    size_t warp_region = per_warp_bytes * warps_per_block;
    size_t remaining = (max_shmem_available > (int)warp_region) ? (static_cast<size_t>(max_shmem_available) - warp_region) : 0;
    int tile_points = 0;
    if (remaining >= sizeof(float2)) {
        tile_points = static_cast<int>(remaining / sizeof(float2));
        // cap tile_points to a reasonable number to avoid over-large tiles; 8192 is a good balance
        tile_points = std::min(tile_points, 8192);
        // ensure at least one warp worth of work in each tile
        tile_points = std::max(tile_points, warps_per_block * WARP_SIZE);
    } else {
        // Fallback minimal tile
        tile_points = warps_per_block * WARP_SIZE;
    }

    // Total dynamic shared memory required
    size_t shmem_bytes = 0;
    shmem_bytes += static_cast<size_t>(tile_points) * sizeof(float2);
    shmem_bytes = align_up(shmem_bytes, alignof(KDPair));
    shmem_bytes += static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * sizeof(KDPair); // inter
    shmem_bytes += static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * sizeof(KDPair); // buf
    shmem_bytes += static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * sizeof(KDPair); // merged
    shmem_bytes = align_up(shmem_bytes, alignof(int));
    shmem_bytes += static_cast<size_t>(warps_per_block) * sizeof(int); // counters

    // If opt-in is available, request it to allow large dynamic shared memory
    if (max_shmem_optin > 0) {
        cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_bytes);
    }

    // Launch configuration
    int threads_per_block = warps_per_block * WARP_SIZE;
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;
    // Avoid launching more blocks than necessary; but grid can be large.
    // Optionally cap to a multiple of SMs to avoid oversubscription:
    int max_active_blocks = max_blocks * 8; // heuristic cap
    if (blocks > max_active_blocks) blocks = max_active_blocks;
    if (blocks < 1) blocks = 1;

    knn_kernel<<<blocks, threads_per_block, shmem_bytes>>>(query, query_count, data, data_count, result, k, tile_points, warps_per_block);

    // Synchronize to ensure completion (optional error check can be added if desired)
    cudaDeviceSynchronize();
}