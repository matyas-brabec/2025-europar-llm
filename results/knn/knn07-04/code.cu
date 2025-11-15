// k-NN for 2D points on modern NVIDIA GPUs (e.g., A100/H100).
// Each query is processed by a single warp (32 threads).
// The implementation follows the specification in the prompt:
//  - Process data in shared-memory tiles.
//  - For each query (warp), maintain:
//      * A private intermediate result of size k (indices + distances), always sorted.
//      * A shared-memory candidate buffer of size k.
//      * A shared integer tracking the candidate count, updated with atomicAdd.
//  - When the candidate buffer is full, merge it into the intermediate result using:
//      0. Intermediate result is sorted ascending (invariant).
//      1. Sort the buffer with Bitonic Sort (ascending).
//      2. Merge the buffer and intermediate result by taking element-wise minima of
//         R[i] and B[k-1-i], producing a bitonic sequence of length k.
//      3. Bitonic sort this merged sequence to restore ascending order.
//  - After the last batch, merge any remaining candidates (buffer not empty).
//
// Notes on design/optimization:
//  - Each query is handled by exactly one warp.
//  - All warp-cooperative work (sorting, merging, candidate insert) uses warp-level
//    primitives (__ballot_sync, __shfl_sync, __syncwarp) and shared memory.
//  - We use static shared memory sized for the worst case k = 1024 and 4 warps/block.
//    This fits comfortably within the shared memory limits of modern data-center GPUs.
//  - The bitonic sorts and merge operate entirely in shared memory, avoiding global
//    memory traffic except for loading data tiles and writing final results.
//  - k is assumed to be a power of two in [32, 1024], as specified.

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ---- Configuration constants ----

constexpr int WARP_SIZE          = 32;   // CUDA warp size
constexpr int WARPS_PER_BLOCK    = 4;    // Number of queries processed per block
constexpr int THREADS_PER_BLOCK  = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int MAX_K              = 1024; // Maximum allowed k (power of two)
constexpr int TILE_POINTS        = 1024; // Number of data points per shared-memory tile

// ---- Helper struct to mirror std::pair<int, float> in device code ----
//
// We cannot rely on full std::pair support on device in a portable way,
// so we define a trivial POD with the same logical layout and reinterpret
// the std::pair<int,float>* pointer when launching the kernel.
struct KnnPair {
    int   first;   // index of nearest neighbor
    float second;  // squared distance
};

// Optional sanity check for typical compilers. If this fails on a given
// toolchain, adjust KnnPair layout accordingly.
static_assert(sizeof(KnnPair) == sizeof(std::pair<int, float>),
              "KnnPair must match std::pair<int,float> layout");

// ---- Shared memory declarations ----
//
// Layout:
//   sData        : TILE_POINTS float2s (shared tile of data points)
//   sInterIdx    : WARPS_PER_BLOCK x MAX_K intermediate indices
//   sInterDist   : WARPS_PER_BLOCK x MAX_K intermediate distances
//   sCandIdx     : WARPS_PER_BLOCK x MAX_K candidate indices
//   sCandDist    : WARPS_PER_BLOCK x MAX_K candidate distances
//   sMergeIdx    : WARPS_PER_BLOCK x MAX_K temporary indices used in merge
//   sMergeDist   : WARPS_PER_BLOCK x MAX_K temporary distances used in merge
//   sCandCount   : WARPS_PER_BLOCK candidate counts (one per warp)

__shared__ float2 sData[TILE_POINTS];

__shared__ int   sInterIdx [WARPS_PER_BLOCK][MAX_K];
__shared__ float sInterDist[WARPS_PER_BLOCK][MAX_K];

__shared__ int   sCandIdx  [WARPS_PER_BLOCK][MAX_K];
__shared__ float sCandDist [WARPS_PER_BLOCK][MAX_K];

__shared__ int   sMergeIdx [WARPS_PER_BLOCK][MAX_K];
__shared__ float sMergeDist[WARPS_PER_BLOCK][MAX_K];

__shared__ int   sCandCount[WARPS_PER_BLOCK];

// ---- Warp-level bitonic sort ----
//
// Sorts 'dist[0..n-1]' in ascending order and permutes 'idx' accordingly.
// - n is assumed to be a power of two and 1 <= n <= MAX_K.
// - All threads in the warp participate.
// - Data resides in shared memory (idx/dist pointers refer to per-warp slices).
// - The implementation follows the serial bitonic pseudocode provided, but with
//   the innermost loop parallelized over warp lanes.

__device__ __forceinline__
void warpBitonicSort(int lane, int n, int *idx, float *dist)
{
    const unsigned fullMask = 0xFFFFFFFFu;

    // Outer loop: size of the subsequence to be merged (k in pseudocode).
    for (int k = 2; k <= n; k <<= 1) {
        // Middle loop: distance between compared elements (j in pseudocode).
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Inner loop: each thread processes multiple indices i spaced by warp size.
            for (int i = lane; i < n; i += WARP_SIZE) {
                int l = i ^ j;
                if (l > i) {
                    float di  = dist[i];
                    float dl  = dist[l];
                    int   ii  = idx[i];
                    int   il  = idx[l];

                    bool ascending = ((i & k) == 0);
                    // From pseudocode:
                    // if ((i&k)==0 && arr[i]>arr[l]) OR ((i&k)!=0 && arr[i]<arr[l]) swap
                    bool doSwap = (ascending && (di > dl)) || (!ascending && (di < dl));

                    if (doSwap) {
                        dist[i] = dl;
                        dist[l] = di;
                        idx[i]  = il;
                        idx[l]  = ii;
                    }
                }
            }
            // Synchronize warp after each (k,j) stage to ensure compare-exchanges
            // are visible before the next set of stages.
            __syncwarp(fullMask);
        }
    }
}

// ---- Warp-level flush of a FULL candidate buffer ----
//
// Preconditions:
//  - Candidate buffer for this warp contains exactly k valid elements.
//  - sCandIdx/sCandDist hold these k candidates.
//  - Intermediate result (sInterIdx/sInterDist) is sorted ascending.
// Postconditions:
//  - Intermediate result is updated to include the k best elements from
//    union(intermediate, candidates), still sorted ascending.
//  - Candidate buffer is emptied (count reset to 0).
//  - maxDistance is updated to the distance of the k-th nearest neighbor
//    (interDist[k-1]) and broadcast to all lanes in the warp.

__device__ __forceinline__
void warpFlushBufferFull(int lane,
                         int k,
                         int  *candIdx,  float *candDist,
                         int  *mergeIdx, float *mergeDist,
                         int  *interIdx, float *interDist,
                         int  *candCountPtr,
                         float &maxDistance)
{
    const unsigned fullMask = 0xFFFFFFFFu;

    int count;
    if (lane == 0) {
        count = *candCountPtr;
    }
    count = __shfl_sync(fullMask, count, 0);

    // If buffer is empty, nothing to do. This function is typically called
    // only when the buffer is full (count == k), but we guard for safety.
    if (count == 0) {
        return;
    }

    // Step 1: Sort candidate buffer using bitonic sort (ascending).
    // We sort k elements; the caller ensures 'count == k' for main flushes.
    warpBitonicSort(lane, k, candIdx, candDist);
    __syncwarp(fullMask);

    // Step 2: Merge buffer and intermediate result into a bitonic sequence.
    //
    // For i in [0, k):
    //   Let R[i] = intermediate[i], B[i] = buffer[i] (both sorted ascending).
    //   We form merge[i] = min(R[i], B[k - 1 - i]) (distance comparison), which
    //   yields a bitonic sequence containing the smallest k elements from
    //   R and B. The remaining elements can be discarded.
    for (int i = lane; i < k; i += WARP_SIZE) {
        float dR   = interDist[i];
        int   idxR = interIdx[i];

        float dB   = candDist[k - 1 - i];
        int   idxB = candIdx[k - 1 - i];

        if (dR <= dB) {
            mergeDist[i] = dR;
            mergeIdx[i]  = idxR;
        } else {
            mergeDist[i] = dB;
            mergeIdx[i]  = idxB;
        }
    }
    __syncwarp(fullMask);

    // Step 3: Sort the merged bitonic sequence to obtain updated intermediate
    // results in ascending order.
    warpBitonicSort(lane, k, mergeIdx, mergeDist);
    __syncwarp(fullMask);

    // Step 4: Copy merged sequence back into intermediate storage.
    for (int i = lane; i < k; i += WARP_SIZE) {
        interDist[i] = mergeDist[i];
        interIdx[i]  = mergeIdx[i];
    }
    __syncwarp(fullMask);

    // Step 5: Update maxDistance (distance of k-th nearest neighbor) and reset
    // candidate buffer count.
    if (lane == 0) {
        *candCountPtr = 0;
        maxDistance   = interDist[k - 1];
    }
    // Broadcast updated maxDistance to all lanes.
    maxDistance = __shfl_sync(fullMask, maxDistance, 0);
}

// ---- Warp-level flush of a PARTIALLY filled candidate buffer ----
//
// This is used only at the very end, after all data batches are processed.
// The buffer may contain fewer than k elements (count < k).
// We pad the remaining slots with +inf distances and dummy indices, then
// reuse warpFlushBufferFull to merge as if the buffer were full.

__device__ __forceinline__
void warpFlushBufferPartial(int lane,
                            int k,
                            int  *candIdx,  float *candDist,
                            int  *mergeIdx, float *mergeDist,
                            int  *interIdx, float *interDist,
                            int  *candCountPtr,
                            float &maxDistance)
{
    const unsigned fullMask = 0xFFFFFFFFu;

    int count;
    if (lane == 0) {
        count = *candCountPtr;
    }
    count = __shfl_sync(fullMask, count, 0);

    if (count == 0) {
        // No candidates to merge.
        return;
    }

    // Pad remaining slots [count, k) with +infinity, so they are always
    // worse than any real candidate, and will be ignored by the merge.
    const float INF = FLT_MAX;

    for (int i = lane; i < k; i += WARP_SIZE) {
        if (i >= count) {
            candDist[i] = INF;
            candIdx[i]  = -1;
        }
    }
    __syncwarp(fullMask);

    if (lane == 0) {
        *candCountPtr = k;
    }
    __syncwarp(fullMask);

    warpFlushBufferFull(lane, k,
                        candIdx,  candDist,
                        mergeIdx, mergeDist,
                        interIdx, interDist,
                        candCountPtr,
                        maxDistance);
}

// ---- Main k-NN kernel ----
//
// Each warp processes one query point.
// The block cooperatively loads data tiles into shared memory.
// For each query (warp):
//  - Maintain a per-warp intermediate result of size k, sorted.
//  - Maintain a per-warp candidate buffer of size k in shared memory.
//  - Use atomicAdd (on shared-memory counter) to assign positions to new
//    candidates in the buffer.
//  - When the buffer fills, merge it into the intermediate result as described.

__global__
void knn_kernel(const float2 * __restrict__ query,
                int                     query_count,
                const float2 * __restrict__ data,
                int                     data_count,
                KnnPair * __restrict__  result,
                int                     k)
{
    const unsigned fullMask = 0xFFFFFFFFu;

    // Warp and lane identification.
    int tid          = threadIdx.x;
    int lane         = tid & (WARP_SIZE - 1);       // thread index within warp
    int warpInBlock  = tid >> 5;                    // warp index within block
    int warpsPerBlock = blockDim.x >> 5;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpInBlock;

    if (globalWarpId >= query_count) {
        return;
    }

    int queryIdx = globalWarpId;

    // Pointers to per-warp slices in shared memory.
    int   *interIdx   = &sInterIdx [warpInBlock][0];
    float *interDist  = &sInterDist[warpInBlock][0];
    int   *candIdx    = &sCandIdx  [warpInBlock][0];
    float *candDist   = &sCandDist [warpInBlock][0];
    int   *mergeIdx   = &sMergeIdx [warpInBlock][0];
    float *mergeDist  = &sMergeDist[warpInBlock][0];
    int   *candCountPtr = &sCandCount[warpInBlock];

    // Load the query point into a register (same for all lanes in the warp).
    float2 q = query[queryIdx];

    // Initialize intermediate result to "infinite" distances and invalid indices.
    // Each lane initializes multiple entries in [0, k) in a strided fashion.
    const float INF = FLT_MAX;

    for (int i = lane; i < k; i += WARP_SIZE) {
        interDist[i] = INF;
        interIdx[i]  = -1;
    }

    // Initialize candidate buffer count.
    if (lane == 0) {
        *candCountPtr = 0;
    }

    // Initial maxDistance: effectively infinite until we have at least k
    // real neighbors in the intermediate result.
    float maxDistance = INF;

    __syncwarp(fullMask);

    // Process data points in tiles loaded into shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_POINTS) {
        int tileSize = data_count - tileStart;
        if (tileSize > TILE_POINTS) {
            tileSize = TILE_POINTS;
        }

        // Block-wide: load current tile into shared memory.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            sData[i] = data[tileStart + i];
        }

        __syncthreads(); // Ensure tile is fully loaded before any warp uses it.

        // Warp processes all points in the tile for its query point.
        for (int i = lane; i < tileSize; i += WARP_SIZE) {
            float2 p = sData[i];

            float dx   = p.x - q.x;
            float dy   = p.y - q.y;
            float dist = dx * dx + dy * dy;

            // Filter by current maxDistance to prune distant candidates.
            int isCandidate = (dist < maxDistance) ? 1 : 0;

            // Build mask of threads in this warp that have a candidate for this data point.
            unsigned mask = __ballot_sync(fullMask, isCandidate);

            if (mask == 0) {
                // No candidates from this batch of 32 data points.
                continue;
            }

            // Remaining candidates to process for this batch (mask may be reduced across iterations).
            unsigned remaining = mask;

            // Current candidate buffer count shared across lanes.
            int currentCount;
            if (lane == 0) {
                currentCount = *candCountPtr;
            }
            currentCount = __shfl_sync(fullMask, currentCount, 0);

            // Process the set of candidates in chunks such that we never overflow
            // the candidate buffer of size k. If necessary, we flush the buffer
            // (when full) and continue inserting remaining candidates.
            while (remaining) {
                int freeSlots = k - currentCount;

                // If no free slots, flush the full candidate buffer.
                if (freeSlots == 0) {
                    warpFlushBufferFull(lane, k,
                                        candIdx,  candDist,
                                        mergeIdx, mergeDist,
                                        interIdx, interDist,
                                        candCountPtr,
                                        maxDistance);

                    // Reload currentCount after flush.
                    if (lane == 0) {
                        currentCount = *candCountPtr;
                    }
                    currentCount = __shfl_sync(fullMask, currentCount, 0);

                    freeSlots = k - currentCount;
                }

                // Count how many candidates remain to be inserted in this batch.
                int remainingCount = __popc(remaining);
                int chunk = (remainingCount < freeSlots) ? remainingCount : freeSlots;

                // Decide, per lane, whether this lane's candidate will be inserted
                // in this chunk (the first 'chunk' set bits in 'remaining').
                int isPending   = (remaining >> lane) & 1;
                int localPrefix = __popc(remaining & ((1u << lane) - 1));

                int takeThisRound = isPending && (localPrefix < chunk);

                // Build mask of lanes that insert in this chunk.
                unsigned takeMask = __ballot_sync(fullMask, takeThisRound);
                int takenCount    = __popc(takeMask);

                if (takenCount == 0) {
                    // Should not happen, but guard against it.
                    break;
                }

                // Reserve a contiguous block of 'takenCount' slots in the candidate buffer
                // using atomicAdd on the shared counter. The returned 'base' is the
                // starting index, and each lane uses its localPrefix as offset within it.
                int base;
                if (lane == 0) {
                    base = atomicAdd(candCountPtr, takenCount);
                }
                base = __shfl_sync(fullMask, base, 0);

                if (takeThisRound) {
                    int pos = base + localPrefix;
                    // pos is guaranteed to be < k because chunk <= freeSlots = k - currentCount
                    // and base == currentCount.
                    candIdx[pos]  = tileStart + i;
                    candDist[pos] = dist;
                }

                // Update currentCount for subsequent iterations.
                currentCount = base + takenCount;

                // Remove lanes that have just inserted from the 'remaining' mask.
                remaining &= ~takeMask;
            }
        }

        __syncthreads(); // Ensure all warps are done with this tile before loading the next.
    }

    // After all tiles are processed, merge any remaining candidates in the buffer.
    warpFlushBufferPartial(lane, k,
                           candIdx,  candDist,
                           mergeIdx, mergeDist,
                           interIdx, interDist,
                           candCountPtr,
                           maxDistance);

    // At this point, interIdx/interDist hold the k nearest neighbors for this query,
    // sorted ascending by distance.

    // Write results to global memory.
    for (int i = lane; i < k; i += WARP_SIZE) {
        int outIdx = queryIdx * k + i;
        result[outIdx].first  = interIdx[i];
        result[outIdx].second = interDist[i];
    }
}

// ---- Host interface ----
//
// run_knn
//   query       : device pointer to query points (float2 array of size query_count)
//   query_count : number of query points
//   data        : device pointer to data points (float2 array of size data_count)
//   data_count  : number of data points
//   result      : device pointer to std::pair<int,float> array of size query_count * k
//   k           : number of nearest neighbors per query (power of two, 32 <= k <= 1024)
//
// Assumptions:
//  - query, data, and result were allocated with cudaMalloc.
//  - data_count >= k, as specified.

void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Enforce maximum k supported by this implementation.
    if (k > MAX_K) {
        // In a production setting, one might throw, assert, or handle this more gracefully.
        // Here, we simply clamp to MAX_K to avoid undefined behavior.
        k = MAX_K;
    }

    // Configure kernel launch: each block has WARPS_PER_BLOCK warps, each warp
    // processes one query, so we need ceil(query_count / WARPS_PER_BLOCK) blocks.
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    int  numBlocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 gridDim(numBlocks, 1, 1);

    // Reinterpret result pointer as KnnPair* to use inside the kernel.
    KnnPair *deviceResult = reinterpret_cast<KnnPair *>(result);

    knn_kernel<<<gridDim, blockDim>>>(query,
                                      query_count,
                                      data,
                                      data_count,
                                      deviceResult,
                                      k);

    // In performance-critical code, error checking and synchronization may be
    // managed at a higher level. They are omitted here to keep the interface minimal.
}