#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ----------------------------------------------------------------------------
// Configuration constants tuned for modern data-center NVIDIA GPUs (A100/H100).
// ----------------------------------------------------------------------------
constexpr int WARP_SIZE          = 32;
constexpr int MAX_K              = 1024;               // upper bound from problem statement
constexpr int MAX_K_PER_THREAD   = MAX_K / WARP_SIZE;  // 32
constexpr int BLOCK_THREADS      = 256;                // 8 warps per block
constexpr int WARPS_PER_BLOCK    = BLOCK_THREADS / WARP_SIZE;
constexpr int TILE_SIZE          = 4096;               // number of data points cached per block

// ----------------------------------------------------------------------------
// Helper struct for distance/index pairs used in shared memory.
// ----------------------------------------------------------------------------
struct IdxDist {
    float dist;
    int   idx;
};

// ----------------------------------------------------------------------------
// Bitonic sort of k elements distributed across one warp.
//
// Layout:
//   - Logical array A[0..k-1] is distributed across the warp.
//   - Mapping: global index i -> (lane, local_slot):
//         lane       = i % WARP_SIZE
//         local_slot = i / WARP_SIZE
//   - Each thread stores k_per_thread = k / WARP_SIZE consecutive elements
//     in registers at indices local_slot = 0..k_per_thread-1.
//
// The implementation follows the serial bitonic sort pseudocode, but uses
// warp shuffles for cross-lane exchanges (stride < WARP_SIZE) and in-thread
// swaps for strides >= WARP_SIZE.
// ----------------------------------------------------------------------------
template<int MAX_KPT>
__device__ __forceinline__
void bitonic_sort_warp(float * __restrict__ dist,
                       int   * __restrict__ idx,
                       int k,
                       int k_per_thread,
                       int lane,
                       unsigned warpMask)
{
    // size: current bitonic subsequence length (2,4,...,k)
    for (int size = 2; size <= k; size <<= 1) {
        // stride: distance between elements being compared/swapped
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (stride >= WARP_SIZE) {
                // Comparisons are between elements stored in the same thread
                // (different local indices).
                int strideLocal = stride >> 5; // stride / WARP_SIZE (since WARP_SIZE == 32)
                for (int r = 0; r < k_per_thread; ++r) {
                    int partnerR = r ^ strideLocal;
                    if (partnerR > r && partnerR < k_per_thread) {
                        int i   = (r << 5) | lane;      // global index
                        bool up = ((i & size) == 0);    // sort direction for this index
                        float a  = dist[r];
                        float b  = dist[partnerR];
                        int   ia = idx[r];
                        int   ib = idx[partnerR];
                        bool doSwap = ((a > b) == up);
                        if (doSwap) {
                            dist[r]        = b;
                            dist[partnerR] = a;
                            idx[r]         = ib;
                            idx[partnerR]  = ia;
                        }
                    }
                }
            } else {
                // Comparisons are across threads in the warp (same local index).
                for (int r = 0; r < k_per_thread; ++r) {
                    int i   = (r << 5) | lane;      // global index
                    bool up = ((i & size) == 0);    // sort direction

                    float selfVal = dist[r];
                    int   selfIdx = idx[r];

                    float otherVal = __shfl_xor_sync(warpMask, selfVal, stride);
                    int   otherIdx = __shfl_xor_sync(warpMask, selfIdx, stride);

                    bool doSwap = ((selfVal > otherVal) == up);
                    if (doSwap) {
                        dist[r] = otherVal;
                        idx[r]  = otherIdx;
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Merge candidate buffer (in shared memory) with intermediate result
// (in registers) for one warp.
//
// The candidate buffer has up to 'candidateCount' valid entries at the
// beginning (0..candidateCount-1). The rest of the k buffer slots are
// filled with +INF so they don't affect the result.
//
// Steps (per problem statement):
//   0. Intermediates in interDist/interIdx are sorted ascending (invariant).
//   1. Swap contents so that the candidate buffer becomes resident in
//      registers (bufDist/bufIdx), and the old intermediate result is moved
//      into shared memory.
//   2. Sort the buffer in registers with bitonic sort (ascending).
//   3. Merge buffer and intermediate into interDist/interIdx in registers:
//      merged[i] = min( buffer[i], intermediate[k-1-i] ).
//      This yields a bitonic sequence of length k.
//   4. Sort this merged sequence again with bitonic sort to restore ascending
//      order. The k smallest distances are now in interDist/interIdx.
//
// Returns the updated maxDistance (distance of the k-th nearest neighbor).
// ----------------------------------------------------------------------------
template<int MAX_KPT>
__device__ __forceinline__
float merge_candidate_buffer(float * __restrict__ interDist,
                             int   * __restrict__ interIdx,
                             float * __restrict__ bufDist,
                             int   * __restrict__ bufIdx,
                             IdxDist * __restrict__ warpBuf,
                             int candidateCount,
                             int k,
                             int k_per_thread,
                             int lane,
                             unsigned warpMask)
{
    const float INF = FLT_MAX;

    // Fill unused candidate slots with INF so they don't influence the merge.
    for (int r = 0; r < k_per_thread; ++r) {
        int iGlobal = (r << 5) | lane;  // r * WARP_SIZE + lane
        if (iGlobal >= candidateCount && iGlobal < k) {
            warpBuf[iGlobal].dist = INF;
            warpBuf[iGlobal].idx  = -1;
        }
    }

    // Step 1: swap buffer (shared) with intermediate result (registers).
    for (int r = 0; r < k_per_thread; ++r) {
        int iGlobal = (r << 5) | lane;
        IdxDist tmp = warpBuf[iGlobal];  // candidate or INF
        warpBuf[iGlobal].dist = interDist[r];
        warpBuf[iGlobal].idx  = interIdx[r];
        bufDist[r]            = tmp.dist;
        bufIdx[r]             = tmp.idx;
    }

    // Step 2: sort candidate buffer in registers (ascending).
    bitonic_sort_warp<MAX_KPT>(bufDist, bufIdx, k, k_per_thread, lane, warpMask);

    // Step 3: merge sorted buffer and previous intermediate result into
    // interDist/interIdx registers via mirrored index comparison.
    for (int r = 0; r < k_per_thread; ++r) {
        int iGlobal = (r << 5) | lane;
        int jGlobal = k - 1 - iGlobal;  // mirrored index in the other sequence

        IdxDist bufVal;
        bufVal.dist = bufDist[r];
        bufVal.idx  = bufIdx[r];

        IdxDist intVal = warpBuf[jGlobal];  // previous intermediate result

        if (bufVal.dist <= intVal.dist) {
            interDist[r] = bufVal.dist;
            interIdx[r]  = bufVal.idx;
        } else {
            interDist[r] = intVal.dist;
            interIdx[r]  = intVal.idx;
        }
    }

    // Step 4: sort the merged bitonic sequence (ascending).
    bitonic_sort_warp<MAX_KPT>(interDist, interIdx, k, k_per_thread, lane, warpMask);

    // Compute and broadcast new maxDistance (distance of k-th neighbor).
    float maxDistLane = 0.0f;
    if (lane == WARP_SIZE - 1) {
        maxDistLane = interDist[k_per_thread - 1];
    }
    float maxDist = __shfl_sync(warpMask, maxDistLane, WARP_SIZE - 1);
    return maxDist;
}

// ----------------------------------------------------------------------------
// Main CUDA kernel: each warp processes exactly one query point.
// ----------------------------------------------------------------------------
__global__
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                int k,
                std::pair<int,float> * __restrict__ result)
{
    extern __shared__ unsigned char shared_raw[];

    // Layout of dynamic shared memory:
    // [TILE_SIZE float2] [WARPS_PER_BLOCK * k IdxDist] [WARPS_PER_BLOCK int]
    float2 *tilePoints = reinterpret_cast<float2*>(shared_raw);

    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    size_t offset = TILE_SIZE * sizeof(float2);

    IdxDist *warpBuffers = reinterpret_cast<IdxDist*>(shared_raw + offset);
    offset += static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) * sizeof(IdxDist);

    int *warpCounts = reinterpret_cast<int*>(shared_raw + offset);

    int lane           = threadIdx.x & (WARP_SIZE - 1);
    int warpIdInBlock  = threadIdx.x / WARP_SIZE;
    int globalWarpId   = blockIdx.x * warpsPerBlock + warpIdInBlock;
    bool validWarp     = (globalWarpId < query_count);

    unsigned warpMask  = 0xffffffffu;

    // Initialize candidate count for each warp.
    if (lane == 0) {
        warpCounts[warpIdInBlock] = 0;
    }

    // Per-thread registers for this warp's intermediate result and temp buffer.
    float interDist[MAX_K_PER_THREAD];
    int   interIdx[MAX_K_PER_THREAD];
    float bufDist[MAX_K_PER_THREAD];
    int   bufIdx[MAX_K_PER_THREAD];

    float2 q;
    int    k_per_thread = 0;
    float  maxDist      = FLT_MAX;

    if (validWarp) {
        q = query[globalWarpId];
        k_per_thread = k / WARP_SIZE;

        // Initialize intermediate result to "empty": INF distances and invalid indices.
        for (int r = 0; r < k_per_thread; ++r) {
            interDist[r] = FLT_MAX;
            interIdx[r]  = -1;
        }
        maxDist = FLT_MAX;
    }

    // Process data points in batches cached in shared memory.
    for (int base = 0; base < data_count; base += TILE_SIZE) {
        int remaining = data_count - base;
        int tileSize  = (remaining < TILE_SIZE) ? remaining : TILE_SIZE;

        // All threads in the block cooperatively load the tile into shared memory.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            tilePoints[i] = data[base + i];
        }

        __syncthreads();

        if (validWarp) {
            IdxDist *warpBuf = warpBuffers + warpIdInBlock * k;

            // Each warp walks over the tile, striding by WARP_SIZE.
            for (int idxInTile = lane; idxInTile < tileSize; idxInTile += WARP_SIZE) {
                float2 p = tilePoints[idxInTile];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                float dist = dx * dx + dy * dy;  // squared Euclidean distance

                // Filter by current maxDist to reduce candidate insertions.
                bool isCandidate = (dist < maxDist);

                // Warp-wide ballot to count candidates for this data point.
                unsigned candMask = __ballot_sync(warpMask, isCandidate);
                int numNew = __popc(candMask);

                if (numNew > 0) {
                    int candCountOld;
                    if (lane == 0) {
                        candCountOld = warpCounts[warpIdInBlock];
                    }
                    candCountOld = __shfl_sync(warpMask, candCountOld, 0);

                    // If buffer would overflow, merge current buffer with the
                    // intermediate result first, then re-evaluate this candidate.
                    if (candCountOld + numNew > k) {
                        maxDist = merge_candidate_buffer<MAX_K_PER_THREAD>(
                            interDist, interIdx,
                            bufDist, bufIdx,
                            warpBuf,
                            candCountOld,
                            k,
                            k_per_thread,
                            lane,
                            warpMask);

                        if (lane == 0) {
                            warpCounts[warpIdInBlock] = 0;
                        }
                        candCountOld = 0;

                        // Re-evaluate candidate predicate with updated maxDist.
                        isCandidate = (dist < maxDist);
                        candMask = __ballot_sync(warpMask, isCandidate);
                        numNew   = __popc(candMask);
                    }

                    if (numNew > 0) {
                        // Compute per-thread offset among new candidates
                        // using prefix popcount of the ballot mask.
                        unsigned maskBefore = candMask & ((1u << lane) - 1u);
                        int offsetLocal = __popc(maskBefore);

                        if (isCandidate) {
                            int pos = candCountOld + offsetLocal;
                            warpBuf[pos].dist = dist;
                            warpBuf[pos].idx  = base + idxInTile;
                        }

                        if (lane == 0) {
                            warpCounts[warpIdInBlock] = candCountOld + numNew;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    if (validWarp) {
        IdxDist *warpBuf = warpBuffers + warpIdInBlock * k;

        // Final merge for any remaining candidates in the buffer.
        int candCountFinal;
        if (lane == 0) {
            candCountFinal = warpCounts[warpIdInBlock];
        }
        candCountFinal = __shfl_sync(warpMask, candCountFinal, 0);

        if (candCountFinal > 0) {
            maxDist = merge_candidate_buffer<MAX_K_PER_THREAD>(
                interDist, interIdx,
                bufDist, bufIdx,
                warpBuf,
                candCountFinal,
                k,
                k_per_thread,
                lane,
                warpMask);

            if (lane == 0) {
                warpCounts[warpIdInBlock] = 0;
            }
        }

        // Write the final k nearest neighbors for this query to global memory.
        for (int r = 0; r < k_per_thread; ++r) {
            int iGlobal   = (r << 5) | lane;  // r * WARP_SIZE + lane
            int outIndex  = globalWarpId * k + iGlobal;
            result[outIndex].first  = interIdx[r];
            result[outIndex].second = interDist[r];
        }
    }
}

// ----------------------------------------------------------------------------
// Public host entry point.
// Assumes 'query', 'data', and 'result' are device pointers (cudaMalloc'ed).
// No additional device allocations are performed here.
// ----------------------------------------------------------------------------
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Assumptions from problem statement:
    //   - 32 <= k <= 1024
    //   - k is a power of two, and thus divisible by WARP_SIZE (32).
    // No explicit checks here; behavior is undefined if violated.

    constexpr int blockThreads   = BLOCK_THREADS;
    constexpr int warpsPerBlock  = WARPS_PER_BLOCK;

    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    if (numBlocks == 0) {
        return; // nothing to do
    }

    dim3 grid(numBlocks);
    dim3 block(blockThreads);

    // Dynamic shared memory size per block:
    //   [TILE_SIZE float2] + [warpsPerBlock * k IdxDist] + [warpsPerBlock int]
    size_t sharedBytes =
        TILE_SIZE * sizeof(float2) +
        static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) * sizeof(IdxDist) +
        static_cast<size_t>(warpsPerBlock) * sizeof(int);

    // Allow the kernel to use the required dynamic shared memory size.
    // On modern GPUs (A100/H100), this limit can exceed 48KB.
    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(sharedBytes));

    knn_kernel<<<grid, block, sharedBytes>>>(query, query_count, data, data_count, k, result);
}