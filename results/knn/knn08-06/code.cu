#include <cuda_runtime.h>
#include <utility>
#include <float.h>

// This implementation computes k-NN for 2D points on the GPU.
// Each query is processed by a single warp (32 threads).
// For each query, the warp maintains:
//   - A sorted intermediate result of k nearest neighbors in registers.
//   - A candidate buffer of up to k elements in shared memory.
//
// The intermediate result is distributed so that each thread stores
// k/32 consecutive elements in its registers. The full k-element array
// is laid out as:
//   arr[0 .. k-1], where element at global position g is stored in
//   thread (lane) t = g / localK and local register index e = g % localK,
//   with localK = k / 32.
//
// The candidate buffer is shared per-warp and uses the same layout
// (contiguous in global index g).
//
// When the buffer is full or needs to be merged, we:
//   1. Pad the buffer to k elements with +inf distances.
//   2. Swap buffer and intermediate result so buffer becomes register-resident.
//   3. Bitonic-sort the buffer.
//   4. Merge buffer and previous result into a bitonic sequence in registers
//      by taking minima of pairs (buffer[i], result[k-1-i]).
//   5. Bitonic-sort the merged sequence to restore global ascending order.
//   6. Update max_distance to the distance of the k-th nearest neighbor.
//
// Distances are squared Euclidean (L2) distances.

constexpr int WARP_SIZE          = 32;
constexpr int MAX_K              = 1024;                       // maximum supported k
constexpr int MAX_K_PER_THREAD   = MAX_K / WARP_SIZE;          // 32 when MAX_K=1024
constexpr int WARPS_PER_BLOCK    = 4;                          // 4 warps => 128 threads per block
constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int DATA_TILE          = 2048;                       // number of data points cached per block

// Warp-cooperative bitonic sort for k elements distributed across 32 threads.
// Each thread stores localK consecutive elements in its registers:
//   global index g = lane * localK + e, where e in [0, localK).
// k is a power of two in [32, 1024]. localK = k / 32 is thus a power of two in [1, 32].
// The arrays 'dist' and 'idx' contain the localK elements for this thread.
__device__ __forceinline__
void bitonic_sort_k(float dist[MAX_K_PER_THREAD],
                    int   idx [MAX_K_PER_THREAD],
                    int localK,
                    int k)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Outer loop over bitonic sequence sizes.
    for (int size = 2; size <= k; size <<= 1) {
        // Inner loop over compare distances within the current size.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {

            if (stride < localK) {
                // Intra-thread comparisons: partner index within the same thread.
                // For stride < localK, XOR toggles only the low bits within [0, localK),
                // so both indices reside in the same thread and have different local indices.
                for (int e = 0; e < localK; ++e) {
                    int partner = e ^ stride;
                    // process each pair only once
                    if (partner > e) {
                        int g = lane * localK + e;
                        bool up = ((g & size) == 0);  // true => ascending, false => descending

                        float a  = dist[e];
                        float b  = dist[partner];
                        int   ia = idx[e];
                        int   ib = idx[partner];

                        bool comp = (a > b);
                        if (comp == up) {
                            // swap
                            dist[e]       = b;
                            idx[e]        = ib;
                            dist[partner] = a;
                            idx[partner]  = ia;
                        }
                    }
                }
            } else {
                // Inter-thread comparisons: partner index resides in another lane.
                // For stride >= localK, stride is a multiple of localK (both are powers of two).
                // XOR affects only the higher bits that encode the lane index, leaving
                // the intra-thread index e unchanged.
                int warpDelta = stride / localK;  // lane XOR distance in [1,16]
                for (int e = 0; e < localK; ++e) {
                    int g = lane * localK + e;
                    float v  = dist[e];
                    int   id = idx[e];

                    float otherV  = __shfl_xor_sync(FULL_MASK, v,  warpDelta);
                    int   otherId = __shfl_xor_sync(FULL_MASK, id, warpDelta);

                    bool up   = ((g & size) == 0);
                    bool comp = (v > otherV);
                    if (comp == up) {
                        v  = otherV;
                        id = otherId;
                    }

                    dist[e] = v;
                    idx[e]  = id;
                }
            }

            __syncwarp();
        }
    }
}

// Merge the per-warp candidate buffer (in shared memory) with the current
// intermediate result (in registers). The steps are:
//
//   1. Swap buffer (shared) and intermediate (registers), so that the buffer
//      becomes register-resident in regBufDist/regBufIdx and the previous
//      intermediate result is moved to shared memory.
//   2. Bitonic-sort the buffer (regBuf*).
//   3. For each global index i, compute:
//           merged[i] = min( buffer[i], intermediate[k-1-i] ),
//      producing a bitonic sequence in bestDist/bestIdx.
//   4. Bitonic-sort the merged sequence to restore ascending order.
//   5. Update maxDistance from the k-th element.
__device__ __forceinline__
void merge_buffer_with_intermediate(
    float regBufDist[MAX_K_PER_THREAD],
    int   regBufIdx [MAX_K_PER_THREAD],
    float bestDist  [MAX_K_PER_THREAD],
    int   bestIdx   [MAX_K_PER_THREAD],
    float *warpCandDist,
    int   *warpCandIdx,
    int k,
    int localK,
    float &maxDistance)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Step 1: swap buffer (shared) and intermediate result (registers)
    for (int e = 0; e < localK; ++e) {
        int g = lane * localK + e;
        if (g < k) {
            float tmpD = warpCandDist[g];
            int   tmpI = warpCandIdx[g];

            warpCandDist[g] = bestDist[e];
            warpCandIdx[g]  = bestIdx[e];

            regBufDist[e] = tmpD;
            regBufIdx[e]  = tmpI;
        }
    }
    __syncwarp();

    // Step 2: sort the buffer in registers
    bitonic_sort_k(regBufDist, regBufIdx, localK, k);

    // Step 3: merge into a bitonic sequence using minima of (buffer[i], intermediate[k-1-i])
    for (int e = 0; e < localK; ++e) {
        int g = lane * localK + e;
        if (g < k) {
            int j = k - 1 - g;

            float valB = regBufDist[e];
            int   idB  = regBufIdx[e];

            float valI = warpCandDist[j];
            int   idI  = warpCandIdx[j];

            if (valI < valB) {
                valB = valI;
                idB  = idI;
            }

            bestDist[e] = valB;
            bestIdx[e]  = idB;
        }
    }
    __syncwarp();

    // Step 4: sort merged result in registers (ascending)
    bitonic_sort_k(bestDist, bestIdx, localK, k);
    __syncwarp();

    // Step 5: update maxDistance from global position k-1, which is stored at
    // lane = WARP_SIZE-1, local index = localK-1.
    float last = bestDist[localK - 1];
    maxDistance = __shfl_sync(FULL_MASK, last, WARP_SIZE - 1);
    __syncwarp();
}

// Flush the per-warp candidate buffer into the intermediate result.
// The buffer currently contains sCandCount[warpId] entries at the
// beginning; we pad the remainder up to k with +inf before merging.
//
// This function is warp-synchronous and must be called by all threads
// in the warp that owns warpId.
__device__ __forceinline__
void flush_candidate_buffer(
    int warpId,
    float bestDist[MAX_K_PER_THREAD],
    int   bestIdx [MAX_K_PER_THREAD],
    float regBufDist[MAX_K_PER_THREAD],
    int   regBufIdx [MAX_K_PER_THREAD],
    float (*sCandDist)[MAX_K],
    int   (*sCandIdx )[MAX_K],
    int *sCandCount,
    int k,
    int localK,
    float &maxDistance)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    int candCount = sCandCount[warpId];
    if (candCount == 0) {
        return;
    }

    float *warpCandDist = sCandDist[warpId];
    int   *warpCandIdx  = sCandIdx[warpId];

    // Pad unused entries with +inf so that the buffer has exactly k elements.
    if (lane == 0) {
        for (int i = candCount; i < k; ++i) {
            warpCandDist[i] = CUDART_INF_F;
            warpCandIdx[i]  = -1;
        }
    }
    __syncwarp();

    // Merge candidate buffer with current intermediate result.
    merge_buffer_with_intermediate(regBufDist, regBufIdx,
                                   bestDist, bestIdx,
                                   warpCandDist, warpCandIdx,
                                   k, localK, maxDistance);

    if (lane == 0) {
        sCandCount[warpId] = 0;
    }
    __syncwarp();
}

// Kernel: each warp processes one query point.
__global__ void knn_kernel_2d(
    const float2 *__restrict__ query,
    int query_count,
    const float2 *__restrict__ data,
    int data_count,
    std::pair<int, float> *__restrict__ result,
    int k)
{
    // Shared memory layout:
    //   - sData:  tile of DATA_TILE points cached for all warps in the block.
    //   - sCandDist/sCandIdx: WARPS_PER_BLOCK candidate buffers.
    //   - sCandCount: per-warp candidate counts.
    __shared__ float2 sData[DATA_TILE];
    __shared__ float  sCandDist[WARPS_PER_BLOCK][MAX_K];
    __shared__ int    sCandIdx [WARPS_PER_BLOCK][MAX_K];
    __shared__ int    sCandCount[WARPS_PER_BLOCK];

    const int lane         = threadIdx.x & (WARP_SIZE - 1);
    const int warpId       = threadIdx.x / WARP_SIZE;
    const int warpGlobalId = blockIdx.x * WARPS_PER_BLOCK + warpId;
    const bool warpActive  = (warpGlobalId < query_count);
    const unsigned FULL_MASK = 0xffffffffu;

    const int localK = k / WARP_SIZE;

    // Per-thread registers: intermediate result and temporary buffer for merging.
    float bestDist [MAX_K_PER_THREAD];
    int   bestIdx  [MAX_K_PER_THREAD];
    float bufRegDist[MAX_K_PER_THREAD];
    int   bufRegIdx [MAX_K_PER_THREAD];

    float qx = 0.0f;
    float qy = 0.0f;
    float maxDistance = CUDART_INF_F;

    if (warpActive) {
        // Initialize intermediate result with +inf distances.
        for (int i = 0; i < localK; ++i) {
            bestDist[i] = CUDART_INF_F;
            bestIdx[i]  = -1;
        }

        // Initialize candidate count for this warp.
        if (lane == 0) {
            sCandCount[warpId] = 0;
        }

        // Load and broadcast query point.
        if (lane == 0) {
            float2 q = query[warpGlobalId];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);

        maxDistance = CUDART_INF_F;
    }

    __syncthreads();

    // Process data points in tiles cached in shared memory.
    for (int base = 0; base < data_count; base += DATA_TILE) {
        int tileSize = data_count - base;
        if (tileSize > DATA_TILE) tileSize = DATA_TILE;

        // Load tile cooperatively.
        for (int idx = threadIdx.x; idx < tileSize; idx += blockDim.x) {
            sData[idx] = data[base + idx];
        }
        __syncthreads();

        if (warpActive) {
            // Each warp scans all points in the tile.
            for (int i = lane; i < tileSize; i += WARP_SIZE) {
                float2 p = sData[i];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float distSq = dx * dx + dy * dy;

                bool isCandidate = distSq < maxDistance;

                unsigned mask = __ballot_sync(FULL_MASK, isCandidate);
                int warpNew = __popc(mask);
                if (warpNew == 0) {
                    continue;
                }

                int oldCount = sCandCount[warpId];

                // If buffer would overflow, flush it first and recompute candidacy
                // with the updated maxDistance.
                if (oldCount + warpNew > k) {
                    flush_candidate_buffer(warpId,
                                           bestDist, bestIdx,
                                           bufRegDist, bufRegIdx,
                                           sCandDist, sCandIdx,
                                           sCandCount,
                                           k, localK, maxDistance);

                    oldCount = sCandCount[warpId];

                    isCandidate = distSq < maxDistance;
                    mask = __ballot_sync(FULL_MASK, isCandidate);
                    warpNew = __popc(mask);
                    if (warpNew == 0) {
                        continue;
                    }
                }

                // Re-read candidate count after possible flush.
                oldCount = sCandCount[warpId];
                int baseOffset = oldCount;

                // Update candidate count for this warp.
                if (lane == 0) {
                    sCandCount[warpId] = oldCount + warpNew;
                }
                __syncwarp();

                // Each candidate thread reserves a unique slot in the buffer
                // based on the prefix count of candidate lanes before it.
                if (isCandidate) {
                    unsigned laneMask = mask & ((1u << lane) - 1u);
                    int offset = __popc(laneMask);
                    int pos = baseOffset + offset;
                    sCandDist[warpId][pos] = distSq;
                    sCandIdx [warpId][pos] = base + i;  // global data index
                }
                __syncwarp();
            }
        }

        __syncthreads();
    }

    if (warpActive) {
        // Flush remaining candidates after processing all tiles.
        flush_candidate_buffer(warpId,
                               bestDist, bestIdx,
                               bufRegDist, bufRegIdx,
                               sCandDist, sCandIdx,
                               sCandCount,
                               k, localK, maxDistance);

        // Write final k nearest neighbors for this query to global memory.
        const int queryIdx = warpGlobalId;
        for (int e = 0; e < localK; ++e) {
            int g = (lane * localK) + e;  // global neighbor rank in [0, k)
            int outIndex = queryIdx * k + g;
            result[outIndex].first  = bestIdx[e];
            result[outIndex].second = bestDist[e];
        }
    }
}

// Host-side wrapper. Assumes that 'query', 'data', and 'result' are device
// pointers allocated with cudaMalloc, and that k is a power of two between
// 32 and 1024 (inclusive) with data_count >= k.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Simple grid configuration: one warp per query.
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks);
    dim3 block(THREADS_PER_BLOCK);

    knn_kernel_2d<<<grid, block>>>(query, query_count, data, data_count, result, k);

    // Synchronize to ensure completion before returning.
    cudaDeviceSynchronize();
}