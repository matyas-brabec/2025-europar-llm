#include <cuda_runtime.h>
#include <utility>

// Optimized CUDA k-NN (2D, squared L2) using one warp (32 threads) per query.
// - Processes data in shared-memory batches ("tiles") loaded by the block.
// - Each warp maintains:
//   * An intermediate result of k nearest neighbors (distributed across lanes in registers).
//   * A shared-memory candidate buffer of size k with an atomic counter.
//   * A shared max_distance (distance of the current k-th nearest neighbor).
// - Candidates closer than max_distance are added to the buffer using atomicAdd for position allocation.
// - When the buffer becomes full, it is merged into the intermediate result cooperatively by the warp.
// - After all tiles, any remaining candidates in the buffer are merged, then results are sorted per query and written out.

#ifndef KNN_THREADS_PER_BLOCK
#define KNN_THREADS_PER_BLOCK 256  // 8 warps per block
#endif

#ifndef KNN_TILE_POINTS
#define KNN_TILE_POINTS 4096       // Number of data points cached per block tile (floats2 => 32 KB for 4096)
#endif

static __device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }
static __device__ __forceinline__ int warp_id() { return threadIdx.x >> 5; }
static __device__ __forceinline__ int warps_per_block() { return blockDim.x >> 5; }

static __device__ __forceinline__ unsigned full_mask() { return 0xFFFFFFFFu; }

// Compute the local (per-lane) maximum value and its position within a small array of length chunk (chunk in [1,32])
static __device__ __forceinline__
void local_array_argmax(const float* __restrict__ vals, int chunk, float& out_val, int& out_pos)
{
    float v = vals[0];
    int p = 0;
    #pragma unroll
    for (int i = 1; i < 32; ++i) {
        if (i >= chunk) break;
        float vi = vals[i];
        if (vi > v) { v = vi; p = i; }
    }
    out_val = v;
    out_pos = p;
}

// Warp-wide argmax reduction over (value, lane, pos). Returns in lane 0, but broadcasts to all lanes for convenience.
static __device__ __forceinline__
void warp_argmax(float local_val, int local_pos, int& out_lane, int& out_pos, float& out_val)
{
    unsigned mask = full_mask();
    int l = lane_id();
    float best_val = local_val;
    int best_lane = l;
    int best_pos  = local_pos;

    // Tree reduction using shuffle-down
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(mask, best_val, offset);
        int other_lane  = __shfl_down_sync(mask, best_lane, offset);
        int other_pos   = __shfl_down_sync(mask, best_pos, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_lane = other_lane;
            best_pos = other_pos;
        }
    }
    // Broadcast the result from lane 0 to all lanes
    out_val = __shfl_sync(mask, best_val, 0);
    out_lane = __shfl_sync(mask, best_lane, 0);
    out_pos = __shfl_sync(mask, best_pos, 0);
}

// Warp-wide maximum reduction. Returns the maximum broadcast to all lanes.
static __device__ __forceinline__
float warp_maxf(float v)
{
    unsigned mask = full_mask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset));
    }
    return __shfl_sync(mask, v, 0);
}

// Sort k elements in shared memory (dist/idx) cooperatively by a single warp using bitonic sort (ascending by distance).
static __device__ __forceinline__
void warp_bitonic_sort(float* __restrict__ dist, int* __restrict__ idx, int k)
{
    unsigned mask = full_mask();
    int lane = lane_id();

    // Standard bitonic sort network over shared array indices [0..k)
    for (int size = 2; size <= k; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < k; i += 32) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = ((i & size) == 0);
                    float ai = dist[i];
                    float aj = dist[j];
                    int   bi = idx[i];
                    int   bj = idx[j];
                    // Swap if out of order for this step
                    if ((ai > aj) == up) {
                        dist[i] = aj; dist[j] = ai;
                        idx[i] = bj;  idx[j] = bi;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// Merge the candidate buffer of a warp into its intermediate top-k, cooperatively using all 32 lanes.
// - top_d/top_i are the per-lane register arrays of size 'chunk' that together represent the current k best.
// - candDist/candIdx/candCount are per-warp shared-memory buffers.
// - maxDistSh is the per-warp shared maximum distance in shared memory; updated here after merging.
// Note: This routine assumes that the calling warp is uniform (all lanes participate).
static __device__ __forceinline__
void warp_merge_buffer(
    int k,
    int chunk,
    float* __restrict__ candDist_base, // base pointer for all warps
    int*   __restrict__ candIdx_base,  // base pointer for all warps
    int*   __restrict__ candCount,     // per-warp counters array
    float* __restrict__ maxDistSh,     // per-warp max-distance array
    float* __restrict__ top_d,         // per-lane register array [chunk] of distances
    int*   __restrict__ top_i          // per-lane register array [chunk] of indices
)
{
    unsigned mask = full_mask();
    int wid = warp_id();
    int lane = lane_id();

    float* candDist = candDist_base + wid * k;
    int*   candIdx  = candIdx_base  + wid * k;

    // Read candidate count uniformly
    int c = 0;
    if (lane == 0) c = candCount[wid];
    c = __shfl_sync(mask, c, 0);

    // Current maximum (k-th best) distance across the top-k
    float local_max, curMax;
    int local_pos;
    local_array_argmax(top_d, chunk, local_max, local_pos);
    curMax = warp_maxf(local_max);

    // Sequentially insert candidates; each insertion is warp-cooperative via argmax
    for (int j = 0; j < c; ++j) {
        float cd = candDist[j];
        int   ci = candIdx[j];

        if (cd < curMax) {
            // Find current global argmax across the top-k representation
            local_array_argmax(top_d, chunk, local_max, local_pos);
            int max_lane, max_pos;
            float max_val;
            warp_argmax(local_max, local_pos, max_lane, max_pos, max_val);

            if (cd < max_val) {
                if (lane == max_lane) {
                    top_d[max_pos] = cd;
                    top_i[max_pos] = ci;
                }
                __syncwarp(mask);
                // Update curMax for next iterations
                local_array_argmax(top_d, chunk, local_max, local_pos);
                curMax = warp_maxf(local_max);
            }
        }
    }

    if (lane == 0) candCount[wid] = 0;
    __syncwarp(mask);

    // Write back the updated maximum to shared memory
    local_array_argmax(top_d, chunk, local_max, local_pos);
    curMax = warp_maxf(local_max);
    if (lane == 0) maxDistSh[wid] = curMax;
    __syncwarp(mask);
}

// Kernel implementing k-NN with one warp per query
__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    std::pair<int, float>* __restrict__ result,
    int k
)
{
    // Dynamic shared memory layout:
    // [0]   float2 tile[KNN_TILE_POINTS]
    // [1]   int    candCount[warpsPerBlock]
    // [2]   float  maxDist[warpsPerBlock]
    // [3]   float  candDist[warpsPerBlock * k]
    // [4]   int    candIdx [warpsPerBlock * k]

    extern __shared__ unsigned char smem_raw[];
    unsigned char* ptr = smem_raw;

    float2* tile = reinterpret_cast<float2*>(ptr);
    ptr += sizeof(float2) * KNN_TILE_POINTS;

    // Align to 8 bytes for safety
    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    uptr = (uptr + 7) & ~uintptr_t(7);
    ptr = reinterpret_cast<unsigned char*>(uptr);

    int wpb = warps_per_block();

    int* candCount = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * wpb;

    float* maxDistSh = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * wpb;

    float* candDist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * (size_t)wpb * (size_t)k;

    int* candIdx = reinterpret_cast<int*>(ptr);
    // ptr += sizeof(int) * (size_t)wpb * (size_t)k; // not needed further

    unsigned mask = full_mask();
    int wid  = warp_id();
    int lane = lane_id();

    int global_warp = blockIdx.x * wpb + wid;
    int qid = global_warp; // one query per warp
    bool valid = (qid < query_count);

    // Each lane holds chunk = k/32 elements of the intermediate top-k
    const int chunk = max(1, k >> 5); // k is a power of two >= 32

    // Per-lane intermediate top-k storage in registers
    float top_d[32];
    int   top_i[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (i < chunk) {
            top_d[i] = CUDART_INF_F;
            top_i[i] = -1;
        }
    }

    // Initialize per-warp shared variables
    if (lane == 0) {
        candCount[wid] = 0;
        maxDistSh[wid] = CUDART_INF_F;
    }
    __syncwarp(mask);

    // Load query point to registers
    float qx = 0.0f, qy = 0.0f;
    if (valid) {
        float2 q = query[qid];
        qx = q.x;
        qy = q.y;
    }

    // Iterate over data in tiles
    for (int base = 0; base < data_count; base += KNN_TILE_POINTS) {
        int tileCount = min(KNN_TILE_POINTS, data_count - base);

        // Block-wide load of tile into shared memory
        for (int t = threadIdx.x; t < tileCount; t += blockDim.x) {
            tile[t] = data[base + t];
        }
        __syncthreads();

        if (valid) {
            // Local cached copy of max distance for quick checks
            float curMax = maxDistSh[wid];

            // Process the tile: each lane steps over positions [lane, lane+32, ...]
            for (int j = lane; j < tileCount; j += 32) {
                float2 p = tile[j];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = fmaf(dx, dx, dy * dy); // squared L2

                // Check against current threshold
                int pass = (dist < curMax);
                unsigned passMask = __ballot_sync(mask, pass);
                int n = __popc(passMask);
                if (n) {
                    // One atomicAdd per warp-reservation
                    int basePos = 0;
                    if (lane == 0) basePos = atomicAdd(&candCount[wid], n);
                    basePos = __shfl_sync(mask, basePos, 0);

                    // Rank within passing lanes
                    int rank = __popc(passMask & ((1u << lane) - 1));

                    // Number that fit before reaching capacity
                    int fit = max(0, k - basePos);
                    int toWrite = min(n, fit);

                    if (pass && (rank < toWrite)) {
                        int pos = basePos + rank;
                        candDist[wid * k + pos] = dist;
                        candIdx [wid * k + pos] = base + j;
                    }

                    // If buffer is now full or overflown, merge it
                    bool needFlush = (basePos + n >= k);
                    if (needFlush) {
                        __syncwarp(mask);
                        warp_merge_buffer(k, chunk, candDist, candIdx, candCount, maxDistSh, top_d, top_i);
                        __syncwarp(mask);
                        curMax = maxDistSh[wid];

                        // Handle leftovers (those that didn't fit before flush)
                        int leftover = n - toWrite;
                        if (leftover > 0) {
                            int isLeft = (pass && (rank >= toWrite)) ? 1 : 0;
                            unsigned leftMask = __ballot_sync(mask, isLeft);
                            int leftRank = __popc(leftMask & ((1u << lane) - 1));
                            int base2 = 0;
                            if (lane == 0) base2 = atomicAdd(&candCount[wid], leftover);
                            base2 = __shfl_sync(mask, base2, 0);
                            if (isLeft) {
                                int pos = base2 + leftRank;
                                candDist[wid * k + pos] = dist;
                                candIdx [wid * k + pos] = base + j;
                            }
                            // If the buffer became exactly full again, flush to keep the requirement
                            bool flushAgain = (base2 + leftover >= k);
                            if (flushAgain) {
                                __syncwarp(mask);
                                warp_merge_buffer(k, chunk, candDist, candIdx, candCount, maxDistSh, top_d, top_i);
                                __syncwarp(mask);
                                curMax = maxDistSh[wid];
                            }
                        }
                    }
                }
            }

            // End of tile: flush any remaining candidates
            int c = 0;
            if (lane == 0) c = candCount[wid];
            c = __shfl_sync(mask, c, 0);
            if (c > 0) {
                __syncwarp(mask);
                warp_merge_buffer(k, chunk, candDist, candIdx, candCount, maxDistSh, top_d, top_i);
                __syncwarp(mask);
            }
        }

        __syncthreads(); // ensure tile not reused until all warps finish processing it
    }

    // After processing all tiles: output results
    if (valid) {
        // Reuse candidate buffer as a per-warp temporary array to gather and sort the intermediate top-k
        float* outDist = candDist + wid * k;
        int*   outIdx  = candIdx  + wid * k;

        // Gather from per-lane registers to per-warp shared arrays
        for (int i = 0; i < chunk; ++i) {
            int pos = lane * chunk + i;
            if (pos < k) {
                outDist[pos] = top_d[i];
                outIdx [pos] = top_i[i];
            }
        }
        __syncwarp(mask);

        // Sort k elements (ascending by distance) cooperatively by the warp
        warp_bitonic_sort(outDist, outIdx, k);

        // Write to global memory
        for (int i = lane; i < k; i += 32) {
            int outPos = qid * k + i;
            result[outPos].first  = outIdx[i];
            result[outPos].second = outDist[i];
        }
    }
}

// Host entry point: configures and launches the kernel.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Kernel configuration
    const int threads = KNN_THREADS_PER_BLOCK;
    const int warpsPerBlock = threads / 32;
    const int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Dynamic shared memory size calculation
    size_t smem = 0;
    smem += sizeof(float2) * (size_t)KNN_TILE_POINTS;       // tile
    smem = (smem + 7) & ~size_t(7);                         // alignment
    smem += sizeof(int)   * (size_t)warpsPerBlock;          // candCount
    smem += sizeof(float) * (size_t)warpsPerBlock;          // maxDist
    smem += sizeof(float) * (size_t)warpsPerBlock * (size_t)k; // candDist
    smem += sizeof(int)   * (size_t)warpsPerBlock * (size_t)k; // candIdx

    knn_kernel<<<blocks, threads, smem>>>(query, query_count, data, data_count, result, k);
}