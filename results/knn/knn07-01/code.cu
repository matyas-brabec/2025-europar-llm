#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <limits>

// This implementation assigns one warp (32 threads) per query point.
// The kernel processes the dataset in batches (tiles) cached in shared memory,
// and uses per-warp shared memory for:
// - intermediate result (k nearest so far), sorted ascending by distance
// - candidate buffer (up to k candidates), unsorted between merges
// - temporary buffer used during merging (k elements)
// - a per-warp shared counter (atomicAdd) to count candidates currently in the buffer
//
// Whenever a buffer fills (candidate count >= k), the warp:
// 1) Sorts the candidate buffer (bitonic sort, ascending)
// 2) Merges the candidate buffer with the intermediate result by taking element-wise minima
//    between buffer[i] and result[k-1-i], forming a bitonic sequence of k elements
// 3) Sorts the merged sequence (bitonic sort, ascending) to obtain the new intermediate result
//
// After the last tile is processed, the warp merges any remaining candidates as above,
// and then writes out the final k nearest neighbor indices and distances for its query.
//
// Notes:
// - Distance is squared Euclidean in 2D.
// - k is a power of two within [32, 1024].
// - Shared memory is dynamically partitioned into per-warp areas and a data tile buffer.
// - We use warp-synchronous programming with __syncwarp for warp coordination.
// - AtomicAdd on the per-warp counter is used with a warp-wide reservation using ballot/popc,
//   which prevents buffer overflow and allows immediate merging when the buffer fills.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Device utility: warp-synchronous bitonic sort for paired arrays (distance + index), ascending by distance.
// - n must be a power-of-two (here, n == k).
// - All threads in the warp participate; each thread processes indices i = lane, i += warpSize.
// - Uses shared memory arrays for dist and idx; operations are in-place.
__device__ __forceinline__
void warp_bitonic_sort_pairs_ascending(int n, float* dist, int* idx, unsigned mask, int lane) {
    // Outer loop: size of bitonic sequences
    for (int k = 2; k <= n; k <<= 1) {
        // Inner loop: stride within sequences
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each thread processes several indices in [0, n)
            for (int i = lane; i < n; i += WARP_SIZE) {
                int l = i ^ j;
                if (l > i) {
                    bool ascending = ((i & k) == 0);
                    float ai = dist[i];
                    float al = dist[l];
                    // Perform compare-and-swap depending on direction
                    bool do_swap = (ascending && ai > al) || (!ascending && ai < al);
                    if (do_swap) {
                        dist[i] = al;
                        dist[l] = ai;
                        int ti = idx[i];
                        idx[i] = idx[l];
                        idx[l] = ti;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
    __syncwarp(mask);
}

// Device utility: merge candidate buffer into the intermediate result.
// Steps:
// 1) Sort buffer (ascending)
// 2) Create bitonic merge array tmp[i] = min(buffer[i], result[k-1-i])
// 3) Sort tmp (ascending) and write back to result
// 4) Update maxDistance and reset candidate counter and buffer contents to +inf/-1
__device__ __forceinline__
void warp_merge_buffer_into_result(
    int k,
    int* bufIdx, float* bufDist,
    int* resIdx, float* resDist,
    int* tmpIdx, float* tmpDist,
    volatile int* candCountPtr,
    unsigned mask, int lane,
    float& maxDistance)
{
    // 1) Sort candidate buffer
    warp_bitonic_sort_pairs_ascending(k, bufDist, bufIdx, mask, lane);

    // 2) Element-wise min between buffer[i] and result[k-1-i] to form a bitonic sequence
    for (int i = lane; i < k; i += WARP_SIZE) {
        float bd = bufDist[i]; int bi = bufIdx[i];
        float rd = resDist[k - 1 - i]; int ri = resIdx[k - 1 - i];
        if (bd < rd) {
            tmpDist[i] = bd; tmpIdx[i] = bi;
        } else {
            tmpDist[i] = rd; tmpIdx[i] = ri;
        }
    }
    __syncwarp(mask);

    // 3) Sort the merged (bitonic) sequence ascending, then write back to result
    warp_bitonic_sort_pairs_ascending(k, tmpDist, tmpIdx, mask, lane);
    for (int i = lane; i < k; i += WARP_SIZE) {
        resDist[i] = tmpDist[i];
        resIdx[i] = tmpIdx[i];
    }
    __syncwarp(mask);

    // 4) Update maxDistance, reset candidate counter and clear buffer (set INF/-1)
    if (lane == 0) {
        maxDistance = resDist[k - 1];
        *candCountPtr = 0;
    }
    maxDistance = __shfl_sync(mask, maxDistance, 0);
    for (int i = lane; i < k; i += WARP_SIZE) {
        bufDist[i] = CUDART_INF_F;
        bufIdx[i] = -1;
    }
    __syncwarp(mask);
}

// Device utility: warp-synchronous buffered insertion with overflow-safe merging.
// Each lane can optionally provide a candidate (cand==true) with (candIdx, candDist).
// The function reserves space in the per-warp candidate buffer using a single atomicAdd,
// then writes successful candidates (within buffer capacity).
// If the reservation causes the buffer to fill (base + nCand >= k), the warp merges the buffer
// into the intermediate result immediately and retries unresolved candidates.
// maxDistance is updated during merges and broadcast to all lanes.
__device__ __forceinline__
void warp_try_push_candidate(
    bool cand, int candIdx, float candDist,
    volatile int* candCountPtr, // shared counter for this warp
    int* bufIdx, float* bufDist,
    int* resIdx, float* resDist,
    int* tmpIdx, float* tmpDist,
    int k, unsigned mask, int lane,
    float& maxDistance)
{
    while (true) {
        unsigned active = __ballot_sync(mask, cand);
        int n = __popc(active);
        if (n == 0) break; // No pending candidates to insert

        // Compute rank among active lanes
        unsigned laneMask = (1u << lane) - 1u;
        int rank = __popc(active & laneMask);

        // Reserve slots in buffer with a single atomicAdd by lane 0
        int base = 0;
        if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
            base = atomicAdd((int*)candCountPtr, n);
        }
        base = __shfl_sync(mask, base, 0);

        int pos = base + rank;
        bool success = cand && (pos < k);

        // Write successful candidates (within capacity)
        if (success) {
            bufIdx[pos] = candIdx;
            bufDist[pos] = candDist;
        }
        __syncwarp(mask);

        // If buffer filled or overflowed, merge now; losers will retry
        bool needMerge = (base + n) >= k;
        if (needMerge) {
            warp_merge_buffer_into_result(k, bufIdx, bufDist, resIdx, resDist,
                                          tmpIdx, tmpDist, candCountPtr, mask, lane, maxDistance);
            cand = cand && !success; // losers remain pending
        } else {
            // No merge needed; all pending candidates were inserted successfully
            break;
        }
    }
}

// Kernel: one warp per query. The whole block loads data tiles into shared memory.
// Dynamic shared memory layout:
// [ per-warp data (W warps): (resIdx[k], resDist[k], bufIdx[k], bufDist[k], tmpIdx[k], tmpDist[k]) ] +
// [ per-warp counters (W ints) ] +
// [ data tile buffer (tilePoints float2) ]
__global__
void knn_kernel_2d_warp(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data, int data_count,
    std::pair<int, float>* __restrict__ out,
    int k, int tilePoints)
{
    extern __shared__ unsigned char smem[];
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warpInBlock = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;
    const int warpGlobal = blockIdx.x * warpsPerBlock + warpInBlock;
    const unsigned fullMask = 0xFFFFFFFFu;

    // Compute per-warp shared memory region
    const size_t perWarpStrideBytes =
        (size_t)k * sizeof(int)   + (size_t)k * sizeof(float) + // resIdx, resDist
        (size_t)k * sizeof(int)   + (size_t)k * sizeof(float) + // bufIdx, bufDist
        (size_t)k * sizeof(int)   + (size_t)k * sizeof(float);  // tmpIdx, tmpDist

    unsigned char* base = smem;
    unsigned char* warpBase = base + perWarpStrideBytes * warpInBlock;

    int*   resIdx = reinterpret_cast<int*>(warpBase);
    float* resDist = reinterpret_cast<float*>(resIdx + k);
    int*   bufIdx = reinterpret_cast<int*>(resDist + k);
    float* bufDist = reinterpret_cast<float*>(bufIdx + k);
    int*   tmpIdx = reinterpret_cast<int*>(bufDist + k);
    float* tmpDist = reinterpret_cast<float*>(tmpIdx + k);

    // Counters area for all warps in block
    unsigned char* ctrBase = base + perWarpStrideBytes * warpsPerBlock;
    int* counters = reinterpret_cast<int*>(ctrBase);
    volatile int* candCountPtr = counters + warpInBlock;

    // Data tile area
    unsigned char* tileBase = reinterpret_cast<unsigned char*>(counters + warpsPerBlock);
    float2* tile = reinterpret_cast<float2*>(tileBase);

    // All threads in the block will participate in tile loads (__syncthreads barriers).
    // Only warps mapped to a valid query will compute k-NN.
    bool warpActive = (warpGlobal < query_count);

    // Initialize per-warp structures
    float qx = 0.0f, qy = 0.0f;
    float maxDistance = CUDART_INF_F;

    if (warpActive) {
        // Load query point and broadcast within warp
        if (lane == 0) {
            float2 q = query[warpGlobal];
            qx = q.x; qy = q.y;
        }
        qx = __shfl_sync(fullMask, qx, 0);
        qy = __shfl_sync(fullMask, qy, 0);

        // Initialize intermediate result with +inf distances and invalid indices
        for (int i = lane; i < k; i += WARP_SIZE) {
            resIdx[i] = -1;
            resDist[i] = CUDART_INF_F;
            bufIdx[i] = -1;
            bufDist[i] = CUDART_INF_F;
            tmpIdx[i] = -1;
            tmpDist[i] = CUDART_INF_F;
        }
        if (lane == 0) {
            *candCountPtr = 0;
        }
        __syncwarp(fullMask);
        maxDistance = CUDART_INF_F;
    }

    // Process data in tiles cached in shared memory
    for (int tileStart = 0; tileStart < data_count; tileStart += tilePoints) {
        int count = data_count - tileStart;
        if (count > tilePoints) count = tilePoints;

        // Load tile into shared memory using the entire block
        for (int t = threadIdx.x; t < count; t += blockDim.x) {
            tile[t] = data[tileStart + t];
        }
        __syncthreads(); // Ensure tile is loaded before any warp computes

        if (warpActive) {
            // Each lane processes a strided subset of the tile
            float localMax = maxDistance; // cache of maxDistance (updated only after merges)
            for (int t = lane; t < count; t += WARP_SIZE) {
                float2 p = tile[t];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float d2 = dx * dx + dy * dy;
                bool isCand = d2 < localMax;

                // Try to push candidate; may trigger merge if buffer fills
                if (isCand) {
                    int dataIdx = tileStart + t;
                    warp_try_push_candidate(true, dataIdx, d2,
                                            candCountPtr, bufIdx, bufDist,
                                            resIdx, resDist, tmpIdx, tmpDist,
                                            k, fullMask, lane, localMax);
                    // Update cached threshold after potential merge
                    maxDistance = localMax;
                }
            }
        }

        __syncthreads(); // Ensure all warps done with this tile before it is overwritten
    }

    // After processing all tiles: if buffer has unused candidates, merge them.
    if (warpActive) {
        int pending = 0;
        if (lane == 0) pending = *candCountPtr;
        pending = __shfl_sync(fullMask, pending, 0);
        if (pending > 0) {
            warp_merge_buffer_into_result(k, bufIdx, bufDist, resIdx, resDist,
                                          tmpIdx, tmpDist, candCountPtr, fullMask, lane, maxDistance);
        }

        // Write out final k nearest neighbors for this query
        int outBase = warpGlobal * k;
        for (int i = lane; i < k; i += WARP_SIZE) {
            out[outBase + i].first = resIdx[i];
            out[outBase + i].second = resDist[i];
        }
    }
}

// Host interface: determines a suitable launch configuration (warps per block, tile size, shared memory),
// launches the kernel, and relies solely on the provided device allocations.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Device properties related to shared memory
    int device = 0;
    cudaGetDevice(&device);

    int maxSharedOptin = 0;
    cudaDeviceGetAttribute(&maxSharedOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    int maxSharedDefault = 0;
    cudaDeviceGetAttribute(&maxSharedDefault, cudaDevAttrMaxSharedMemoryPerBlock, device);

    // Use the largest available shared memory per block
    int maxSharedPerBlock = (maxSharedOptin > 0) ? maxSharedOptin : maxSharedDefault;

    // Per-warp shared memory footprint (bytes)
    size_t perWarpStrideBytes =
        (size_t)k * sizeof(int)   + (size_t)k * sizeof(float) + // resIdx, resDist
        (size_t)k * sizeof(int)   + (size_t)k * sizeof(float) + // bufIdx, bufDist
        (size_t)k * sizeof(int)   + (size_t)k * sizeof(float);  // tmpIdx, tmpDist

    // Choose warps per block and tile size to fit in shared memory
    int bestWarpsPerBlock = 0;
    int bestTilePoints = 0;
    size_t bestSmemBytes = 0;

    // Limit warps per block to not exceed 1024 threads per block
    const int maxWarpsByThreads = 1024 / WARP_SIZE;
    // Reasonable upper bound on warps per block to balance shared memory use and parallelism
    const int maxWarpsConsider = (maxWarpsByThreads < 8) ? maxWarpsByThreads : 8;

    // Prefer a reasonably large tile for global memory efficiency
    const int preferredTilePoints = 4096;

    for (int warps = maxWarpsConsider; warps >= 1; --warps) {
        size_t perBlockWarpArea = perWarpStrideBytes * (size_t)warps;
        size_t countersBytes = sizeof(int) * (size_t)warps;
        // Compute the maximum tile size that fits into shared memory
        size_t leftover = (maxSharedPerBlock > (perBlockWarpArea + countersBytes))
                        ? (maxSharedPerBlock - perBlockWarpArea - countersBytes) : 0;
        int tilePoints = (int)(leftover / sizeof(float2));
        if (tilePoints <= 0) continue;

        // Cap tile size at preferredTilePoints if possible
        if (tilePoints > preferredTilePoints) tilePoints = preferredTilePoints;

        // Align tile size to a multiple of blockDim.x for coalesced loads
        int blockThreads = warps * WARP_SIZE;
        int alignedTile = (tilePoints / blockThreads) * blockThreads;
        if (alignedTile == 0) alignedTile = blockThreads; // at least one per-thread load when possible
        tilePoints = alignedTile;

        size_t totalSmem = perBlockWarpArea + countersBytes + (size_t)tilePoints * sizeof(float2);
        if (totalSmem <= (size_t)maxSharedPerBlock) {
            bestWarpsPerBlock = warps;
            bestTilePoints = tilePoints;
            bestSmemBytes = totalSmem;
            break;
        }
    }

    // Fallback: ensure we have at least one warp and a minimal tile
    if (bestWarpsPerBlock == 0) {
        bestWarpsPerBlock = 1;
        int blockThreads = bestWarpsPerBlock * WARP_SIZE;
        size_t perBlockWarpArea = perWarpStrideBytes * (size_t)bestWarpsPerBlock;
        size_t countersBytes = sizeof(int) * (size_t)bestWarpsPerBlock;
        size_t leftover = (maxSharedPerBlock > (perBlockWarpArea + countersBytes))
                        ? (maxSharedPerBlock - perBlockWarpArea - countersBytes) : 0;
        int tilePoints = (int)(leftover / sizeof(float2));
        if (tilePoints <= 0) tilePoints = blockThreads; // minimal viable tile
        // Align tile size
        int alignedTile = (tilePoints / blockThreads) * blockThreads;
        if (alignedTile == 0) alignedTile = blockThreads;
        bestTilePoints = alignedTile;
        bestSmemBytes = perBlockWarpArea + countersBytes + (size_t)bestTilePoints * sizeof(float2);
    }

    // Configure launch
    dim3 block(bestWarpsPerBlock * WARP_SIZE, 1, 1);
    dim3 grid((query_count + bestWarpsPerBlock - 1) / bestWarpsPerBlock, 1, 1);

    // Opt-in to large dynamic shared memory if available/needed
    cudaFuncSetAttribute(knn_kernel_2d_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bestSmemBytes);

    // Launch kernel
    knn_kernel_2d_warp<<<grid, block, bestSmemBytes>>>(query, query_count, data, data_count, result, k, bestTilePoints);
}