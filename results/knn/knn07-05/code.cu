#include <cuda_runtime.h>
#include <utility>
#include <algorithm>

// This code implements a k-NN (k-nearest neighbors) search for 2D points using CUDA.
// - Each query is processed by a single warp (32 threads).
// - The block cooperatively loads tiles of the data points into shared memory.
// - Each warp maintains a private intermediate result (top-k) and a candidate buffer (size k) in shared memory.
// - Distances are squared Euclidean distances.
// - Candidate insertion uses warp-aggregated atomicAdd on a per-warp shared counter.
// - When the candidate buffer is full, the warp merges it with the intermediate result:
//   1) Sort candidate buffer with bitonic sort (ascending).
//   2) Merge (bitonic merge trick): merged[i] = min(intermediate[i], candidate[k-1-i]).
//   3) Sort merged (ascending) to form the updated intermediate result.
// - After all tiles, any remaining candidates are merged the same way.
// - Results (indices and distances) are written to the output array in ascending order of distances.

#ifndef KNN_WARPS_PER_BLOCK
#define KNN_WARPS_PER_BLOCK 4  // Number of warps per block (blockDim.x = 32 * KNN_WARPS_PER_BLOCK)
#endif

#ifndef KNN_TILE_CAPACITY
#define KNN_TILE_CAPACITY 8192 // Number of data points per tile cached in shared memory
#endif

// Simple POD pair compatible with std::pair<int,float> memory layout for device writes.
// We reinterpret device output pointer to this layout to avoid relying on libstdc++ in device code.
struct PairIF { int first; float second; };

// Return a lane-local mask of lower lanes: bits set for threads with laneId < this lane.
static __device__ __forceinline__ unsigned lane_mask_lt()
{
#if (__CUDACC_VER_MAJOR__ >= 9)
    // Alternative using asm: but simple arithmetic is sufficient
#endif
    unsigned lane = threadIdx.x & 31;
    return (lane == 0) ? 0u : ((1u << lane) - 1u);
}

// Warp-level bitonic sort on an array of pairs (dist[idx], idx[idx]) stored in shared memory.
// - n must be a power of two (here n == k, guaranteed by problem statement).
// - Sorting is ascending by distance.
// - All 32 threads in the warp cooperate; each thread processes a strided subset of indices.
static __device__ __forceinline__ void warp_bitonic_sort_pairs(float* dist, int* idx, int n, unsigned warp_mask)
{
    // Outer loop controls the size of the bitonic sequence (size = 2, 4, 8, ..., n)
    for (int size = 2; size <= n; size <<= 1)
    {
        // Inner loop controls the distance between compared elements (stride halves each iteration)
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            // Each thread processes indices i = lane, lane+32, lane+64, ...
            for (int i = (threadIdx.x & 31); i < n; i += 32)
            {
                int l = i ^ stride;
                if (l > i)
                {
                    bool up = ((i & size) == 0); // ascending segment if true, descending if false
                    float di = dist[i], dl = dist[l];
                    int   ii = idx[i],  il = idx[l];

                    // Compare-exchange based on direction 'up'
                    // For ascending segments, place smaller at i; for descending, place larger at i.
                    bool cond = (di > dl);
                    if (up ? cond : !cond)
                    {
                        dist[i] = dl; dist[l] = di;
                        idx[i]  = il; idx[l]  = ii;
                    }
                }
            }
            __syncwarp(warp_mask);
        }
    }
}

// Merge the per-warp candidate buffer into the intermediate top-k result using the prescribed method.
// Steps:
//   1) Pad candidate buffer tail with +INF (if not full) and bitonic-sort ascending.
//   2) For i in [0..k): merged[i] = min(inter[i], cand[k-1-i]) -> written into cand[i].
//   3) Bitonic-sort merged (cand) ascending.
//   4) Copy merged back to inter; update max_distance and reset candidate count.
// All operations are warp-synchronous and use shared memory arrays.
static __device__ __forceinline__ void warp_flush_merge(
    float* cand_dist, int* cand_idx,
    float* inter_dist, int* inter_idx,
    volatile int* cand_count_ptr, volatile float* max_dist_ptr,
    int k, unsigned warp_mask)
{
    int cand_count = *cand_count_ptr;

    // Pad unused candidate entries with +INF and invalid index.
    // This ensures bitonic sort behaves correctly for the full k-length array.
    for (int i = (threadIdx.x & 31); i < k; i += 32)
    {
        if (i >= cand_count)
        {
            cand_dist[i] = CUDART_INF_F;
            cand_idx[i]  = -1;
        }
    }
    __syncwarp(warp_mask);

    // Step 1: sort candidate buffer ascending
    warp_bitonic_sort_pairs(cand_dist, cand_idx, k, warp_mask);

    // Step 2: merge with intermediate: merged[i] = min(inter[i], cand[k-1-i])
    // Write merged into cand arrays (OK with warp barrier to avoid read-write hazards).
    for (int i = (threadIdx.x & 31); i < k; i += 32)
    {
        float di = inter_dist[i];
        int   ii = inter_idx[i];
        float dc = cand_dist[k - 1 - i];
        int   ic = cand_idx[k - 1 - i];

        bool take_cand = (dc < di);
        cand_dist[i] = take_cand ? dc : di;
        cand_idx[i]  = take_cand ? ic : ii;
    }
    __syncwarp(warp_mask);

    // Step 3: sort merged result ascending
    warp_bitonic_sort_pairs(cand_dist, cand_idx, k, warp_mask);

    // Step 4: copy to intermediate result and update max distance, reset candidate count
    for (int i = (threadIdx.x & 31); i < k; i += 32)
    {
        inter_dist[i] = cand_dist[i];
        inter_idx[i]  = cand_idx[i];
    }
    __syncwarp(warp_mask);

    if ((threadIdx.x & 31) == 0) {
        *max_dist_ptr = inter_dist[k - 1];
        *cand_count_ptr = 0;
    }
    __syncwarp(warp_mask);
}

// Try to insert a single candidate (index, distance) into the per-warp candidate buffer.
// - pass: whether this thread produced a candidate (distance < max_distance).
// - Uses warp-aggregated atomicAdd to reserve positions in the buffer.
// - If buffer capacity (k) would be exceeded, partially fills remaining slots, flushes (merge), then
//   re-evaluates remaining candidates against updated max_distance and inserts the remainder.
static __device__ __forceinline__ void warp_try_add_candidate(
    bool pass, int index, float distance, int k,
    float* cand_dist, int* cand_idx, volatile int* cand_count_ptr, volatile float* max_dist_ptr,
    float* inter_dist, int* inter_idx,
    unsigned warp_mask)
{
    unsigned mask = __ballot_sync(warp_mask, pass);
    int lane = threadIdx.x & 31;
    if (mask == 0) return;

    int n = __popc(mask);
    int rank = __popc(mask & lane_mask_lt());

    // Read current count atomically (atomicAdd with 0) to satisfy the requirement to use atomicAdd.
    int cur = atomicAdd((int*)cand_count_ptr, 0);
    int space = k - cur;

    if (space >= n)
    {
        // Enough space for all in one go
        int base = atomicAdd((int*)cand_count_ptr, n);
        if (pass)
        {
            int pos = base + rank;
            cand_dist[pos] = distance;
            cand_idx[pos]  = index;
        }
        return;
    }
    else
    {
        // Not enough space: fill remaining, flush, then insert leftovers if still passing threshold
        int to_write = max(0, space);
        if (to_write > 0)
        {
            int base = atomicAdd((int*)cand_count_ptr, to_write);
            if (pass && (rank < to_write))
            {
                int pos = base + rank;
                cand_dist[pos] = distance;
                cand_idx[pos]  = index;
            }
        }
        __syncwarp(warp_mask);

        // Flush (merge) full candidate buffer
        warp_flush_merge(cand_dist, cand_idx, inter_dist, inter_idx, cand_count_ptr, max_dist_ptr, k, warp_mask);
        __syncwarp(warp_mask);

        // Re-evaluate leftover lanes with updated max_distance
        float maxd = *max_dist_ptr;
        bool keep_leftover = pass && (rank >= to_write) && (distance < maxd);
        unsigned rem_mask = __ballot_sync(warp_mask, keep_leftover);
        int rem = __popc(rem_mask);
        if (rem > 0)
        {
            int r2 = __popc(rem_mask & lane_mask_lt());
            int base2 = atomicAdd((int*)cand_count_ptr, rem);
            if (keep_leftover)
            {
                int pos = base2 + r2;
                cand_dist[pos] = distance;
                cand_idx[pos]  = index;
            }
        }
    }
}

// Kernel: Each warp processes one query. The block loads tiles of the data array to shared memory.
__global__ void knn_kernel_2d(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    PairIF* __restrict__ result,
    int k)
{
    // Thread and warp identifiers
    const int lane  = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5; // warp index within the block
    const int warps_per_block = blockDim.x >> 5;
    const unsigned warp_mask = 0xFFFFFFFFu;

    // Global warp index maps one-to-one with query index
    const int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const int query_idx = warp_global;

    // Shared memory layout (dynamic):
    // [0 .. TILE_CAPACITY-1] float2: tile of data points loaded by the whole block
    extern __shared__ unsigned char shared_bytes[];
    float2* s_data = reinterpret_cast<float2*>(shared_bytes);

    // Compute base pointer for per-warp storage after the data tile region
    unsigned char* perwarp_base = shared_bytes + sizeof(float2) * KNN_TILE_CAPACITY;

    // Per-warp storage size (bytes): 4 arrays of k elements (float, int) + int counter + float max_distance
    const size_t per_warp_bytes = ((size_t)k * (sizeof(float) + sizeof(int)) * 2) + sizeof(int) + sizeof(float);

    // Compute this warp's base pointer into its private shared memory region
    unsigned char* wptr = perwarp_base + warp_in_block * per_warp_bytes;

    // Carve per-warp regions
    float* w_cand_dist = reinterpret_cast<float*>(wptr);
    int*   w_cand_idx  = reinterpret_cast<int*>(w_cand_dist + k);
    float* w_inter_dist= reinterpret_cast<float*>(w_cand_idx + k);
    int*   w_inter_idx = reinterpret_cast<int*>(w_inter_dist + k);
    volatile int*   w_cand_count = reinterpret_cast<volatile int*>(w_inter_idx + k);
    volatile float* w_max_dist   = reinterpret_cast<volatile float*>(const_cast<int*>(reinterpret_cast<const int*>(w_cand_count)) + 1);

    // Initialize per-warp structures
    if (query_idx < query_count)
    {
        for (int i = lane; i < k; i += 32)
        {
            w_inter_dist[i] = CUDART_INF_F;
            w_inter_idx[i]  = -1;
        }
    }
    __syncwarp(warp_mask);
    if (lane == 0) {
        *w_cand_count = 0;
        *w_max_dist   = CUDART_INF_F;
    }
    __syncwarp(warp_mask);

    // Load query point into registers and broadcast within warp
    float2 q = make_float2(0.f, 0.f);
    if (lane == 0 && query_idx < query_count)
    {
        q = query[query_idx];
    }
    q.x = __shfl_sync(warp_mask, q.x, 0);
    q.y = __shfl_sync(warp_mask, q.y, 0);

    // Process data in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += KNN_TILE_CAPACITY)
    {
        int tile_size = min(KNN_TILE_CAPACITY, data_count - tile_start);

        // Cooperative load of the tile into shared memory by the entire block
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x)
        {
            s_data[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each active warp processes its query over the cached tile
        if (query_idx < query_count)
        {
            float maxd = *w_max_dist; // Local copy of current max distance for this warp
            for (int t = lane; t < tile_size; t += 32)
            {
                float2 p = s_data[t];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                float d2 = fmaf(dx, dx, dy * dy);

                // Filter by current max_distance; attempt candidate insertion if it's better
                bool pass = (d2 < maxd);
                int  idx  = tile_start + t;

                // Insert candidate using warp-aggregated atomic pattern,
                // flushing (merge) when the candidate buffer fills up.
                if (pass)
                {
                    warp_try_add_candidate(pass, idx, d2, k,
                        w_cand_dist, w_cand_idx, w_cand_count, w_max_dist,
                        w_inter_dist, w_inter_idx, warp_mask);

                    // Update local copy of maxd after potential flush/merge
                    maxd = *w_max_dist;
                }
            }
        }
        __syncthreads(); // Ensure tile is no longer in use before overwriting in next iteration
    }

    // After all tiles, flush remaining candidates if any
    if (query_idx < query_count)
    {
        int cc = atomicAdd((int*)w_cand_count, 0);
        if (cc > 0)
        {
            __syncwarp(warp_mask);
            warp_flush_merge(w_cand_dist, w_cand_idx, w_inter_dist, w_inter_idx, w_cand_count, w_max_dist, k, warp_mask);
            __syncwarp(warp_mask);
        }

        // Write final results: intermediate result holds k nearest neighbors in ascending order
        PairIF* out = result + (size_t)query_idx * (size_t)k;
        for (int i = lane; i < k; i += 32)
        {
            out[i].first  = w_inter_idx[i];
            out[i].second = w_inter_dist[i];
        }
    }
}

// Host-side launcher with sensible defaults for modern data-center GPUs (A100/H100).
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Kernel configuration: 4 warps per block (128 threads/block) by default.
    constexpr int warps_per_block = KNN_WARPS_PER_BLOCK;
    constexpr int threads_per_block = warps_per_block * 32;
    dim3 block(threads_per_block, 1, 1);
    dim3 grid((query_count + warps_per_block - 1) / warps_per_block, 1, 1);

    // Dynamic shared memory size:
    // - Data tile: KNN_TILE_CAPACITY * sizeof(float2)
    // - Per-warp private storage: warps_per_block * (4*k elements + counters)
    size_t tile_bytes = (size_t)KNN_TILE_CAPACITY * sizeof(float2);
    size_t per_warp_bytes = ((size_t)k * (sizeof(float) + sizeof(int)) * 2) + sizeof(int) + sizeof(float);
    size_t smem_bytes = tile_bytes + (size_t)warps_per_block * per_warp_bytes;

    // Opt-in to large dynamic shared memory if needed (beyond 48KB default).
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    int requested = (int)smem_bytes;
    if (requested > max_optin) {
        // If requested exceeds the opt-in max, clamp tile size heuristically by reducing tile capacity.
        // Fallback heuristic: shrink tile to fit within max_optin while keeping per-warp storage intact.
        int64_t available_for_tile = (int64_t)max_optin - (int64_t)warps_per_block * (int64_t)per_warp_bytes;
        int new_tile_cap = (available_for_tile > 0) ? (int)(available_for_tile / (int64_t)sizeof(float2)) : 0;
        // Ensure at least 1024 tile capacity to maintain some caching benefit; if not possible, set to minimal positive.
        if (new_tile_cap < 1024) new_tile_cap = max(1, new_tile_cap);
        // Recompute shared memory size with reduced tile capacity
        tile_bytes = (size_t)new_tile_cap * sizeof(float2);
        smem_bytes = tile_bytes + (size_t)warps_per_block * per_warp_bytes;
        requested = (int)smem_bytes;
    }
    cudaFuncSetAttribute(knn_kernel_2d, cudaFuncAttributeMaxDynamicSharedMemorySize, requested);

    // Launch kernel
    knn_kernel_2d<<<grid, block, smem_bytes>>>(
        query, query_count, data, data_count,
        reinterpret_cast<PairIF*>(result), k);
}