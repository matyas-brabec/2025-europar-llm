#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// Optimized CUDA implementation of k-NN for 2D points using one warp (32 threads) per query.
// The algorithm processes the dataset in shared-memory tiles. Each warp maintains:
// - A private "intermediate result" top-k (indices and distances) kept sorted in ascending order.
// - A per-warp candidate buffer in shared memory with capacity k, and a shared atomic counter.
// New candidates with distance < max_distance (k-th distance in the intermediate result) are pushed
// into the buffer using warp-aggregated atomicAdd. When the buffer gets full, it is sorted with
// Bitonic Sort, merged with the intermediate result using the described min-pairing trick to create
// a bitonic sequence of length k, then sorted again with Bitonic Sort to update the intermediate result.
// At the end, any remaining candidates in the buffer are merged as well.
//
// Notes:
// - k is a power of two in [32, 1024], making Bitonic Sort straightforward.
// - No extra device memory is allocated; all temporary storage is in shared memory.
// - For performance, we use warp-aggregated atomics (one atomicAdd per warp event) and shared memory tiling.
// - The result array is of type std::pair<int, float>, but on device we reinterpret it as a POD struct.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable parameters: number of warps per block and shared-memory tile size for data points.
// WARPS_PER_BLOCK influences shared memory usage per block: each warp needs ~ (4 * k * sizeof(T)) bytes.
// TILE_SIZE * sizeof(float2) is also allocated in shared memory per block.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 2048
#endif

// POD type compatible with std::pair<int, float> memory layout
struct PairIF { int first; float second; };

static __device__ __forceinline__ float sqdist2(const float2& a, const float2& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // Use FMA to improve throughput: dx*dx + dy*dy
    return fmaf(dy, dy, dx * dx);
}

// Warp-scope Bitonic Sort for arrays stored in shared memory.
// - Sorts 'dist[0..n-1]' ascending and applies the same swaps to 'idx'.
// - n must be a power of two; here n == k in [32, 1024], so it's valid.
// - Parallelization: each lane processes indices i = lane + t * WARP_SIZE.
//   We use __syncwarp between stages to ensure memory ordering across the warp.
static __device__ __forceinline__ void warp_bitonic_sort_asc(float* dist, int* idx, int n) {
    unsigned mask = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = lane; i < n; i += WARP_SIZE) {
                int l = i ^ j;
                if (l > i) {
                    bool up = ((i & k) == 0);
                    float di = dist[i];
                    float dl = dist[l];
                    int ii = idx[i];
                    int il = idx[l];
                    // Bitonic compare-and-swap:
                    // if (up  && di > dl) swap
                    // if (!up && di < dl) swap
                    if ( (up  && (di > dl)) || (!up && (di < dl)) ) {
                        dist[i] = dl; dist[l] = di;
                        idx[i]  = il; idx[l]  = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// Merge the full buffer (size k, sorted ascending) with the current intermediate top-k (sorted ascending):
// 1. Create a bitonic sequence by taking element-wise min of (res[i], buf[k-1-i]) into buf[i].
// 2. Sort the resulting bitonic sequence in-place (ascending) using Bitonic Sort.
// 3. Copy back to res[], update max_distance (res[k-1]).
// All arrays are per-warp shared-memory slices.
static __device__ __forceinline__ void warp_merge_buffer_with_result(
    float* bufDist, int* bufIdx,
    float* resDist, int* resIdx,
    int k,
    float* maxDistSharedForWarp // pointer to per-warp shared max_distance
) {
    unsigned mask = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Step 2: pairwise min to form a bitonic sequence (write into buffer)
    for (int i = lane; i < k; i += WARP_SIZE) {
        int j = k - 1 - i;
        float dr = resDist[i];
        float db = bufDist[j];
        int ir = resIdx[i];
        int ib = bufIdx[j];
        if (dr <= db) {
            bufDist[i] = dr; bufIdx[i] = ir;
        } else {
            bufDist[i] = db; bufIdx[i] = ib;
        }
    }
    __syncwarp(mask);

    // Step 3: sort the merged bitonic sequence ascending
    warp_bitonic_sort_asc(bufDist, bufIdx, k);

    // Copy back to result arrays and update max_distance
    for (int i = lane; i < k; i += WARP_SIZE) {
        resDist[i] = bufDist[i];
        resIdx[i]  = bufIdx[i];
    }
    __syncwarp(mask);

    if (lane == 0) {
        *maxDistSharedForWarp = resDist[k - 1];
    }
    __syncwarp(mask);
}

// Perform a full merge when the candidate buffer is full (or padded at the end):
// - Sort the buffer (ascending).
// - Merge with intermediate top-k.
// - Reset candidate count to 0.
static __device__ __forceinline__ void warp_flush_and_merge(
    float* bufDist, int* bufIdx,
    float* resDist, int* resIdx,
    int k,
    int* candCountSharedForWarp,
    float* maxDistSharedForWarp
) {
    // Sort buffer ascending
    warp_bitonic_sort_asc(bufDist, bufIdx, k);
    // Merge into result
    warp_merge_buffer_with_result(bufDist, bufIdx, resDist, resIdx, k, maxDistSharedForWarp);
    // Reset candidate buffer count
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (lane == 0) {
        *candCountSharedForWarp = 0;
    }
    __syncwarp();
}

// Kernel implementing warp-per-query k-NN with shared-memory tiling and buffered merging.
// Shared memory layout:
// [0 .. TILE_SIZE-1] float2 tile of data points
// Followed by per-warp slices:
//   candDist[WARPS_PER_BLOCK][k], candIdx[WARPS_PER_BLOCK][k],
//   resDist[WARPS_PER_BLOCK][k],  resIdx[WARPS_PER_BLOCK][k],
//   candCount[WARPS_PER_BLOCK],   maxDist[WARPS_PER_BLOCK]
__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    PairIF* __restrict__ result,
    int k
) {
    // Identify warp and lane
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpInBlock = threadIdx.x / WARP_SIZE;
    const unsigned fullMask = 0xFFFFFFFFu;

    extern __shared__ unsigned char shared_raw[];
    unsigned char* ptr = shared_raw;

    // Shared tile of data points
    float2* shData = reinterpret_cast<float2*>(ptr);
    ptr += TILE_SIZE * sizeof(float2);

    // Per-warp arrays (laid out consecutively to avoid pointer arrays in shared memory)
    float* candDistBase = reinterpret_cast<float*>(ptr);                   // [WARPS_PER_BLOCK * k]
    ptr += WARPS_PER_BLOCK * k * sizeof(float);
    int*   candIdxBase  = reinterpret_cast<int*>(ptr);                     // [WARPS_PER_BLOCK * k]
    ptr += WARPS_PER_BLOCK * k * sizeof(int);
    float* resDistBase  = reinterpret_cast<float*>(ptr);                   // [WARPS_PER_BLOCK * k]
    ptr += WARPS_PER_BLOCK * k * sizeof(float);
    int*   resIdxBase   = reinterpret_cast<int*>(ptr);                     // [WARPS_PER_BLOCK * k]
    ptr += WARPS_PER_BLOCK * k * sizeof(int);

    int*   candCount    = reinterpret_cast<int*>(ptr);                     // [WARPS_PER_BLOCK]
    ptr += WARPS_PER_BLOCK * sizeof(int);
    float* maxDist      = reinterpret_cast<float*>(ptr);                   // [WARPS_PER_BLOCK]
    // ptr += WARPS_PER_BLOCK * sizeof(float); // End of shared allocation

    // Grid-stride over queries in groups of WARPS_PER_BLOCK per block to safely use __syncthreads for tile loads.
    for (int qStart = blockIdx.x * WARPS_PER_BLOCK; qStart < query_count; qStart += gridDim.x * WARPS_PER_BLOCK) {
        int qIdx = qStart + warpInBlock;

        // Per-warp shared-memory slices
        float* bufDist = candDistBase + warpInBlock * k;
        int*   bufIdx  = candIdxBase  + warpInBlock * k;
        float* resDist = resDistBase  + warpInBlock * k;
        int*   resIdx  = resIdxBase   + warpInBlock * k;
        int*   myCandCount = candCount + warpInBlock;
        float* myMaxDist   = maxDist   + warpInBlock;

        // Initialize per-warp state if this warp has a valid query
        float2 q = make_float2(0.f, 0.f);
        if (qIdx < query_count) {
            q = query[qIdx];
            // Initialize intermediate result arrays to +inf distances and -1 indices
            for (int i = lane; i < k; i += WARP_SIZE) {
                resDist[i] = FLT_MAX;
                resIdx[i]  = -1;
            }
            if (lane == 0) {
                *myCandCount = 0;
                *myMaxDist   = FLT_MAX; // No filtering until the first merge
            }
        }
        __syncwarp(fullMask);

        // Process data in shared-memory tiles
        for (int base = 0; base < data_count; base += TILE_SIZE) {
            int tileLen = data_count - base;
            if (tileLen > TILE_SIZE) tileLen = TILE_SIZE;

            // Entire block loads the current tile into shared memory
            __syncthreads();
            for (int t = threadIdx.x; t < tileLen; t += blockDim.x) {
                shData[t] = data[base + t];
            }
            __syncthreads();

            if (qIdx < query_count) {
                // For each point in the tile, compute distance and push candidate if closer than max_distance
                for (int t = lane; t < tileLen; t += WARP_SIZE) {
                    float d2 = sqdist2(q, shData[t]);

                    // Load current max_distance for this warp (may be updated after merges)
                    float currMax = *myMaxDist;
                    // Determine which threads have candidates
                    int pred = (d2 < currMax) ? 1 : 0;
                    unsigned mask = __ballot_sync(fullMask, pred);

                    if (mask) {
                        int n = __popc(mask);
                        // Warp-aggregated atomicAdd: lane 0 reserves 'n' slots in the candidate buffer count
                        int prior = 0;
                        if (lane == 0) {
                            prior = atomicAdd(myCandCount, n);
                        }
                        prior = __shfl_sync(fullMask, prior, 0);

                        // Compute per-thread rank among the active lanes
                        int rank = __popc(mask & ((1u << lane) - 1));

                        // Determine how many slots are actually available (to avoid writing past k)
                        int granted = 0;
                        if (lane == 0) {
                            int space = k - prior;
                            if (space < 0) space = 0;
                            granted = (n < space) ? n : space;
                        }
                        granted = __shfl_sync(fullMask, granted, 0);

                        // Write granted candidates to buffer
                        if (pred && (rank < granted)) {
                            int pos = prior + rank;
                            bufDist[pos] = d2;
                            bufIdx[pos]  = base + t; // global data index
                        }

                        // If buffer is (logically) full, perform flush+merge.
                        // Note: myCandCount may exceed k due to reservation for 'n' even if not all were written.
                        int needFlush = ((prior + n) >= k) ? 1 : 0;
                        if (needFlush) {
                            __syncwarp(fullMask);
                            // Pad any unwritten entries (if any) with +inf to ensure we have k valid entries to sort/merge.
                            // Already ensured by 'granted' limiting actual writes; we rely on old contents for remaining slots,
                            // but to be robust, explicitly fill any remaining slots when 'prior + granted < k'.
                            for (int i = lane + prior + granted; i < k; i += WARP_SIZE) {
                                bufDist[i] = FLT_MAX;
                                bufIdx[i]  = -1;
                            }
                            __syncwarp(fullMask);

                            // Sort, merge, reset count, update max_distance
                            warp_flush_and_merge(bufDist, bufIdx, resDist, resIdx, k, myCandCount, myMaxDist);

                            // Reattempt insertion for leftover threads from this iteration (those not granted)
                            int leftoverPred = (pred && (rank >= granted)) ? 1 : 0;
                            unsigned leftoverMask = __ballot_sync(fullMask, leftoverPred);
                            int leftoverN = __popc(leftoverMask);
                            if (leftoverN) {
                                int prior2 = 0;
                                if (lane == 0) {
                                    prior2 = atomicAdd(myCandCount, leftoverN);
                                }
                                prior2 = __shfl_sync(fullMask, prior2, 0);
                                int rank2 = __popc(leftoverMask & ((1u << lane) - 1));
                                int granted2 = 0;
                                if (lane == 0) {
                                    int space2 = k - prior2;
                                    if (space2 < 0) space2 = 0;
                                    granted2 = (leftoverN < space2) ? leftoverN : space2;
                                }
                                granted2 = __shfl_sync(fullMask, granted2, 0);

                                if (leftoverPred && (rank2 < granted2)) {
                                    int pos2 = prior2 + rank2;
                                    bufDist[pos2] = d2;
                                    bufIdx[pos2]  = base + t;
                                }

                                // If we filled the buffer again, flush and merge once more
                                int needFlush2 = ((prior2 + leftoverN) >= k) ? 1 : 0;
                                if (needFlush2) {
                                    __syncwarp(fullMask);
                                    for (int i = lane + prior2 + granted2; i < k; i += WARP_SIZE) {
                                        bufDist[i] = FLT_MAX;
                                        bufIdx[i]  = -1;
                                    }
                                    __syncwarp(fullMask);
                                    warp_flush_and_merge(bufDist, bufIdx, resDist, resIdx, k, myCandCount, myMaxDist);
                                }
                            }
                        }
                    }
                } // end tile points loop
            } // end if valid query

            __syncthreads();
        } // end for all tiles

        // After all tiles, if buffer has any remaining candidates, pad and merge.
        if (qIdx < query_count) {
            int count = (lane == 0) ? *myCandCount : 0;
            count = __shfl_sync(fullMask, count, 0);
            if (count > 0) {
                // Pad to k with +inf
                for (int i = lane + count; i < k; i += WARP_SIZE) {
                    bufDist[i] = FLT_MAX;
                    bufIdx[i]  = -1;
                }
                __syncwarp(fullMask);
                // Sort and merge the (possibly partially filled) buffer
                warp_flush_and_merge(bufDist, bufIdx, resDist, resIdx, k, myCandCount, myMaxDist);
            }

            // Write final sorted results (ascending distances) to global memory
            PairIF* out = result + static_cast<size_t>(qIdx) * static_cast<size_t>(k);
            for (int i = lane; i < k; i += WARP_SIZE) {
                out[i].first  = resIdx[i];
                out[i].second = resDist[i];
            }
        }

        __syncthreads();
    } // end query groups
}

void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Kernel launch configuration
    const int warpsPerBlock = WARPS_PER_BLOCK;
    const int threadsPerBlock = warpsPerBlock * WARP_SIZE;
    const int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Dynamic shared memory size calculation:
    // - TILE_SIZE float2s for data tile
    // - For each warp:
    //    candDist[k] + candIdx[k] + resDist[k] + resIdx[k] => (2*k*sizeof(float) + 2*k*sizeof(int))
    //    candCount[1] + maxDist[1] => (sizeof(int) + sizeof(float))
    size_t perWarpBytes = 2ull * k * sizeof(float) + 2ull * k * sizeof(int) + sizeof(int) + sizeof(float);
    size_t sharedBytes = TILE_SIZE * sizeof(float2) + static_cast<size_t>(warpsPerBlock) * perWarpBytes;

    // Launch kernel. Reinterpret std::pair pointer to POD PairIF pointer for device access.
    PairIF* out = reinterpret_cast<PairIF*>(result);
    knn_kernel<<<blocks, threadsPerBlock, sharedBytes>>>(query, query_count, data, data_count, out, k);
}