#include <cuda_runtime.h>
#include <cuda.h>
#include <utility>
#include <limits>
#include <float.h>

// This code implements a k-NN (k nearest neighbors) CUDA kernel for 2D points.
// Each query is processed by a single warp (32 threads). The kernel processes
// the dataset in shared-memory tiles and collects candidate neighbors that
// are closer than the current max-distance (distance to the k-th nearest neighbor).
// Candidates are stored in a per-warp shared buffer and, whenever that buffer
// becomes full (size >= k), it is merged with the warp's intermediate result
// via bitonic sorts as described in the problem statement.

// Device-side pair representation to match std::pair<int,float> layout.
struct PairIF {
    int first;
    float second;
};
static_assert(sizeof(PairIF) == sizeof(std::pair<int,float>), "PairIF must match std::pair<int,float> size");

// Compile-time parameters
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Number of warps per block and derived threads per block.
// Adjusting BLOCK_WARPS impacts shared memory usage per block. We choose 4 to
// balance shared memory footprint and occupancy on A100/H100.
#ifndef BLOCK_WARPS
#define BLOCK_WARPS 4
#endif

#ifndef BLOCK_THREADS
#define BLOCK_THREADS (BLOCK_WARPS * WARP_SIZE)
#endif

// Utility: align an offset up to the given alignment (power of two).
__device__ __forceinline__ size_t align_up(size_t offset, size_t alignment) {
    return (offset + (alignment - 1)) & ~(alignment - 1);
}

// Warp-scope bitonic sort on length-n array (n is power-of-two).
// Sorts in ascending order by distance. Index array is permuted accordingly.
// All 32 threads of the warp cooperate to sort n elements resident in shared memory.
// Each thread processes indices i = lane, lane+WARP_SIZE, ...
__device__ __forceinline__ void warp_bitonic_sort(float* dist, int* idx, int n, unsigned mask) {
    // The standard serial bitonic sort pseudocode is parallelized by having
    // each lane handle a strided subset of indices. The XOR pairing pattern
    // ensures disjoint compare-exchange pairs, so there are no write conflicts.
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int lane = threadIdx.x & (WARP_SIZE - 1);
            for (int i = lane; i < n; i += WARP_SIZE) {
                int l = i ^ j;
                if (l > i) {
                    bool up = ((i & k) == 0);
                    float di = dist[i];
                    float dl = dist[l];
                    int ii = idx[i];
                    int il = idx[l];
                    bool do_swap = up ? (di > dl) : (di < dl);
                    if (do_swap) {
                        // swap elements i and l
                        dist[i] = dl;
                        dist[l] = di;
                        idx[i] = il;
                        idx[l] = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// Flush/merge routine for a single warp.
// - Sorts candidate buffer (size k, padding with +inf if current count < k).
// - Merges with the intermediate result via element-wise min of buffer[i] and result[k-1-i].
// - Sorts the merged buffer again (bitonic sort) to restore ascending order.
// - Copies the merged buffer into the intermediate result.
// - Updates max_distance to the new k-th (last) distance.
// - Resets candidate_count to 0.
//
// All operations are warp-scope, using only __syncwarp for synchronization.
__device__ __forceinline__ void warp_flush_merge(
    int k,
    float* res_dist, int* res_idx,        // intermediate result per warp, length k (sorted ascending)
    float* cand_dist, int* cand_idx,      // candidate buffer per warp, length k (unsorted, partially filled)
    volatile int* cand_count_ptr,         // shared counter for candidate buffer
    float* max_distance_ptr,              // shared per-warp max distance (k-th distance)
    unsigned mask                         // active lane mask
) {
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Snapshot and clamp the candidate count. Note: oversubscription due to atomicAdd is possible.
    int count = *cand_count_ptr;
    if (count > k) count = k;

    // Pad remaining candidate slots [count, k) with +inf to make length = k (power of two).
    for (int i = lane + count; i < k; i += WARP_SIZE) {
        cand_dist[i] = FLT_MAX;
        cand_idx[i] = -1;
    }
    __syncwarp(mask);

    // 1. Sort candidate buffer (ascending).
    warp_bitonic_sort(cand_dist, cand_idx, k, mask);
    __syncwarp(mask);

    // 2. Merge: produce bitonic sequence by taking element-wise min of
    //    cand[i] and res[k-1-i]. Store back into cand arrays.
    for (int i = lane; i < k; i += WARP_SIZE) {
        int j = k - 1 - i;
        float cd = cand_dist[i];
        float rd = res_dist[j];
        int ci = cand_idx[i];
        int ri = res_idx[j];
        if (rd < cd) {
            cand_dist[i] = rd;
            cand_idx[i] = ri;
        } else {
            // keep current cand[i]
        }
    }
    __syncwarp(mask);

    // 3. Sort the merged (bitonic) buffer ascending.
    warp_bitonic_sort(cand_dist, cand_idx, k, mask);
    __syncwarp(mask);

    // Copy merged buffer into intermediate result and update max_distance.
    for (int i = lane; i < k; i += WARP_SIZE) {
        res_dist[i] = cand_dist[i];
        res_idx[i] = cand_idx[i];
    }
    __syncwarp(mask);

    if (lane == 0) {
        *cand_count_ptr = 0;
        *max_distance_ptr = res_dist[k - 1];
    }
    __syncwarp(mask);
}

// CUDA kernel: each warp processes one query.
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    PairIF* __restrict__ out,
    int k,
    int tile_points // number of data points cached per tile in shared memory
) {
    // Dynamic shared memory layout:
    // [ tile_points * float2 ] [ BLOCK_WARPS*k * float (cand_dist) ] [ BLOCK_WARPS*k * int (cand_idx) ]
    // [ BLOCK_WARPS*k * float (res_dist) ] [ BLOCK_WARPS*k * int (res_idx) ]
    extern __shared__ unsigned char smem[];
    size_t offset = 0;

    // Shared tile of data points
    float2* s_tile = reinterpret_cast<float2*>(smem + offset);
    offset += size_t(tile_points) * sizeof(float2);
    offset = align_up(offset, alignof(float)); // align for float (4 bytes)

    // Per-warp candidate buffers
    float* s_cand_dist = reinterpret_cast<float*>(smem + offset);
    offset += size_t(BLOCK_WARPS) * size_t(k) * sizeof(float);
    offset = align_up(offset, alignof(int));
    int*   s_cand_idx = reinterpret_cast<int*>(smem + offset);
    offset += size_t(BLOCK_WARPS) * size_t(k) * sizeof(int);
    offset = align_up(offset, alignof(float));

    // Per-warp intermediate result buffers (sorted)
    float* s_res_dist = reinterpret_cast<float*>(smem + offset);
    offset += size_t(BLOCK_WARPS) * size_t(k) * sizeof(float);
    offset = align_up(offset, alignof(int));
    int*   s_res_idx = reinterpret_cast<int*>(smem + offset);
    offset += size_t(BLOCK_WARPS) * size_t(k) * sizeof(int);
    // offset is the end of used dynamic shared memory

    // Per-warp candidate counters and max distances in shared memory.
    __shared__ int   s_cand_count[BLOCK_WARPS];
    __shared__ float s_max_distance[BLOCK_WARPS];

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_in_block = threadIdx.x / WARP_SIZE; // 0..BLOCK_WARPS-1
    int warp_global = blockIdx.x * BLOCK_WARPS + warp_in_block;

    unsigned full_mask = 0xFFFFFFFFu; // all 32 lanes active

    // Offsets into per-warp segments
    float* warp_cand_dist = s_cand_dist + size_t(warp_in_block) * size_t(k);
    int*   warp_cand_idx  = s_cand_idx  + size_t(warp_in_block) * size_t(k);
    float* warp_res_dist  = s_res_dist  + size_t(warp_in_block) * size_t(k);
    int*   warp_res_idx   = s_res_idx   + size_t(warp_in_block) * size_t(k);

    // Initialize per-warp structures
    if (lane == 0) {
        s_cand_count[warp_in_block] = 0;
        s_max_distance[warp_in_block] = FLT_MAX;
    }
    __syncwarp(full_mask);

    // Initialize intermediate result to +inf distances and -1 indices.
    for (int i = lane; i < k; i += WARP_SIZE) {
        warp_res_dist[i] = FLT_MAX;
        warp_res_idx[i]  = -1;
    }
    __syncwarp(full_mask);

    // Load query point and broadcast across warp.
    float qx = 0.0f, qy = 0.0f;
    bool active = (warp_global < query_count);
    if (active) {
        if (lane == 0) {
            float2 q = query[warp_global];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(full_mask, qx, 0);
        qy = __shfl_sync(full_mask, qy, 0);
    }

    // Process data in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        int tile_count = data_count - tile_start;
        if (tile_count > tile_points) tile_count = tile_points;

        // Load tile into shared memory cooperatively by the whole block
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_tile[i] = data[tile_start + i];
        }
        __syncthreads();

        if (active) {
            // Process the tile in warp-sized rounds to allow safe flush when buffer is full.
            for (int base = 0; base < tile_count; base += WARP_SIZE) {
                int i = base + lane;
                if (i < tile_count) {
                    float2 p = s_tile[i];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    float dist = dx * dx + dy * dy;

                    float maxd = s_max_distance[warp_in_block];
                    if (dist < maxd) {
                        int pos = atomicAdd(&s_cand_count[warp_in_block], 1);
                        if (pos < k) {
                            warp_cand_dist[pos] = dist;
                            warp_cand_idx[pos]  = tile_start + i;
                        }
                    }
                }
                __syncwarp(full_mask);

                // If candidate buffer is full (or oversubscribed), flush/merge.
                bool full = false;
                if (lane == 0) {
                    full = (s_cand_count[warp_in_block] >= k);
                }
                full = __shfl_sync(full_mask, full, 0);
                if (full) {
                    warp_flush_merge(
                        k,
                        warp_res_dist, warp_res_idx,
                        warp_cand_dist, warp_cand_idx,
                        &s_cand_count[warp_in_block],
                        &s_max_distance[warp_in_block],
                        full_mask
                    );
                }
                __syncwarp(full_mask);
            }
        }

        __syncthreads();
    }

    // After last tile: if buffer not empty, flush/merge remaining candidates.
    if (active) {
        bool has_leftover = false;
        if (lane == 0) {
            has_leftover = (s_cand_count[warp_in_block] > 0);
        }
        has_leftover = __shfl_sync(full_mask, has_leftover, 0);
        if (has_leftover) {
            warp_flush_merge(
                k,
                warp_res_dist, warp_res_idx,
                warp_cand_dist, warp_cand_idx,
                &s_cand_count[warp_in_block],
                &s_max_distance[warp_in_block],
                full_mask
            );
        }

        // Write final k nearest neighbors to global memory.
        PairIF* out_ptr = out + size_t(warp_global) * size_t(k);
        for (int i = lane; i < k; i += WARP_SIZE) {
            out_ptr[i].first  = warp_res_idx[i];
            out_ptr[i].second = warp_res_dist[i];
        }
    }
}

// Host-side launcher.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Determine device and its shared memory capabilities.
    int device = 0;
    cudaGetDevice(&device);

    int max_optin_smem = 0;
    cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, knn_kernel);

    // Compute per-block dynamic shared memory requirement:
    // Per-warp arrays (candidate + result) = 2 * BLOCK_WARPS * k * (sizeof(float)+sizeof(int))
    size_t per_warp_arrays_bytes = size_t(2) * size_t(BLOCK_WARPS) * size_t(k) * (sizeof(float) + sizeof(int));

    // Choose a tile size subject to dynamic shared memory limits. Start with a target cap (e.g., 8192) and adjust down if needed.
    int target_tile_points = 8192;
    // Available dynamic shared memory = (max_optin_smem - static_smem_used_by_kernel)
    size_t static_smem_used = attr.sharedSizeBytes; // statically declared __shared__ usage
    size_t max_dynamic_allowed = 0;
    if (max_optin_smem > 0 && max_optin_smem > static_smem_used) {
        max_dynamic_allowed = size_t(max_optin_smem) - static_smem_used;
    } else {
        // Fall back to default dynamic shared memory limit (attr.maxDynamicSharedSizeBytes may not be exposed).
        // Use a conservative default of 48 KiB minus static.
        size_t default_limit = 48 * 1024;
        if (default_limit > static_smem_used) max_dynamic_allowed = default_limit - static_smem_used;
        else max_dynamic_allowed = 0;
    }

    // Compute maximum tile_points that fit in dynamic shared memory after accounting for per-warp arrays.
    size_t remaining_for_tile = 0;
    int tile_points = 0;
    // Also account for alignment padding; add a small safety margin (e.g., 64 bytes).
    const size_t safety_margin = 64;

    if (max_dynamic_allowed > per_warp_arrays_bytes + safety_margin + sizeof(float2)) {
        remaining_for_tile = max_dynamic_allowed - per_warp_arrays_bytes - safety_margin;
        tile_points = int(remaining_for_tile / sizeof(float2));
    } else {
        tile_points = 0;
    }

    if (tile_points <= 0) {
        // If we cannot allocate enough for any tile point with current BLOCK_WARPS, decrease target tile size to minimum 1.
        tile_points = 1;
    }
    if (tile_points > target_tile_points) tile_points = target_tile_points;
    if (tile_points > data_count) tile_points = data_count;

    size_t dynamic_smem_bytes = size_t(tile_points) * sizeof(float2)
                              + per_warp_arrays_bytes;

    // Opt-in to maximum dynamic shared memory if available.
    if (max_optin_smem > 0) {
        cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin_smem);
    }

    // Grid/block configuration: one warp per query.
    int warps_needed = (query_count + 0) / 1; // one warp per query
    int blocks = (query_count + BLOCK_WARPS - 1) / BLOCK_WARPS;

    // Launch kernel
    knn_kernel<<<blocks, BLOCK_THREADS, dynamic_smem_bytes>>>(
        query, query_count, data, data_count,
        reinterpret_cast<PairIF*>(const_cast<std::pair<int,float>*>(result)),
        k,
        tile_points
    );
    // The caller can synchronize or check errors as needed outside this function.
}