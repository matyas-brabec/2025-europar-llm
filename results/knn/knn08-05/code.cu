#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// Simple POD type with the same layout as std::pair<int, float>.
struct PairIF {
    int   first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "PairIF must match std::pair<int,float> size");

// Constants for warp- and block-level configuration.
constexpr int WARP_SIZE            = 32;
constexpr int MAX_K                = 1024;
constexpr int MAX_ITEMS_PER_THREAD = MAX_K / WARP_SIZE; // 32
constexpr int THREADS_PER_BLOCK    = 256;                // 8 warps per block
constexpr int WARPS_PER_BLOCK      = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int TILE_POINTS          = 1024;               // data points per tile loaded into shared memory

// Warp-level bitonic sort for a distributed array of length k.
// The array is stored across the warp as:
//   global_index = local_index * WARP_SIZE + lane
// Each thread holds 'itemsPerThread = k / WARP_SIZE' elements in registers
// at positions [0 .. itemsPerThread-1].
__device__ __forceinline__ void warp_bitonic_sort(float *dist, int *idx, int k) {
    const unsigned full_mask     = 0xffffffffu;
    const int      lane          = threadIdx.x & (WARP_SIZE - 1);
    const int      itemsPerThread = k / WARP_SIZE;

    // Standard bitonic sort network on k elements.
    for (int size = 2; size <= k; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {

            __syncwarp(full_mask);

            if (stride >= WARP_SIZE) {
                // Comparisons within the same thread (no cross-lane communication).
                for (int local = 0; local < itemsPerThread; ++local) {
                    int i        = local * WARP_SIZE + lane;
                    int partner  = i ^ stride;
                    int p_local  = partner >> 5;   // divide by 32

                    // Handle each pair only once.
                    if (p_local <= local) continue;

                    bool up = ((i & size) == 0);

                    float a  = dist[local];
                    int   ai = idx[local];
                    float b  = dist[p_local];
                    int   bi = idx[p_local];

                    bool do_swap = ((a > b) == up); // (up && a>b) || (!up && a<b)

                    if (do_swap) {
                        dist[local]  = b;
                        idx[local]   = bi;
                        dist[p_local] = a;
                        idx[p_local]  = ai;
                    }
                }
            } else {
                // Cross-lane comparisons using warp shuffles.
                int xorMask = stride;

                for (int local = 0; local < itemsPerThread; ++local) {
                    int i  = local * WARP_SIZE + lane;
                    bool up = ((i & size) == 0);

                    float val = dist[local];
                    int   id  = idx[local];

                    // Exchange values with the partner lane (same local index).
                    float partnerVal = __shfl_xor_sync(full_mask, val, xorMask, WARP_SIZE);
                    int   partnerId  = __shfl_xor_sync(full_mask, id,  xorMask, WARP_SIZE);

                    int  partner = i ^ stride;
                    bool smaller = (i < partner);

                    float minv = val;
                    int   mini = id;
                    float maxv = partnerVal;
                    int   maxi = partnerId;
                    if (partnerVal < val) {
                        minv = partnerVal; mini = partnerId;
                        maxv = val;        maxi = id;
                    }

                    // For ascending ('up' == true), the smaller index keeps the min;
                    // for descending, the smaller index keeps the max.
                    bool keepMin = (smaller == up);

                    dist[local] = keepMin ? minv : maxv;
                    idx[local]  = keepMin ? mini : maxi;
                }
            }
        }
    }

    __syncwarp(full_mask);
}

// Merge the candidate buffer in shared memory with the current intermediate
// result in registers using the procedure described in the prompt.
//
// Arguments:
//   bestDist / bestIdx   : current intermediate result in registers (sorted ascending)
//   bufDist  / bufIdx    : temporary registers for the candidate buffer
//   sharedCandDist/Idx   : per-warp candidate buffer in shared memory (size >= k)
//   k                    : k for this run (power of two, 32..1024)
//   bufferCount          : number of valid candidate entries currently in sharedCand*
__device__ __forceinline__
void warp_merge_buffer(float       *bestDist,
                       int         *bestIdx,
                       float       *bufDist,
                       int         *bufIdx,
                       float       *sharedCandDist,
                       int         *sharedCandIdx,
                       int          k,
                       int          bufferCount)
{
    const unsigned full_mask      = 0xffffffffu;
    const int      lane           = threadIdx.x & (WARP_SIZE - 1);
    const int      itemsPerThread = k / WARP_SIZE;
    const float    INF            = FLT_MAX;

    if (bufferCount == 0) {
        return;
    }

    // Pad unused positions in the candidate buffer with +INF so that
    // we always work with exactly k candidates.
    for (int local = 0; local < itemsPerThread; ++local) {
        int g = local * WARP_SIZE + lane; // global index in [0, k)
        if (g >= bufferCount && g < k) {
            sharedCandDist[g] = INF;
            sharedCandIdx[g]  = -1;
        }
    }

    __syncwarp(full_mask);

    // Step 1: Swap content between the candidate buffer in shared memory
    //         and the intermediate result in registers so that the buffer
    //         resides in registers and the intermediate result in shared memory.
    for (int local = 0; local < itemsPerThread; ++local) {
        int g   = local * WARP_SIZE + lane;
        float cd = sharedCandDist[g];
        int   ci = sharedCandIdx[g];

        sharedCandDist[g] = bestDist[local];
        sharedCandIdx[g]  = bestIdx[local];

        bufDist[local] = cd;
        bufIdx[local]  = ci;
    }

    __syncwarp(full_mask);

    // Step 2: Sort the buffer in registers in ascending order.
    warp_bitonic_sort(bufDist, bufIdx, k);

    // At this point:
    //   - bufDist/Idx (registers) hold the sorted candidate buffer B[0..k-1]
    //   - sharedCandDist/Idx hold the previous intermediate result A[0..k-1] (sorted)

    // Step 3: Merge buffer and intermediate into a bitonic sequence.
    // The merged sequence C[0..k-1] is:
    //   C[i] = min( B[i], A[k-1-i] )
    for (int local = 0; local < itemsPerThread; ++local) {
        int i = local * WARP_SIZE + lane;   // index into B
        int j = k - 1 - i;                  // corresponding index into reversed A

        float aDist = sharedCandDist[j];
        int   aIdx  = sharedCandIdx[j];
        float bDist = bufDist[local];
        int   bIdx  = bufIdx[local];

        if (bDist < aDist) {
            bestDist[local] = bDist;
            bestIdx[local]  = bIdx;
        } else {
            bestDist[local] = aDist;
            bestIdx[local]  = aIdx;
        }
    }

    __syncwarp(full_mask);

    // Step 4: Sort the merged bitonic sequence in registers in ascending order.
    warp_bitonic_sort(bestDist, bestIdx, k);
}

// Main CUDA kernel: each warp processes one query point.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int                    query_count,
                           const float2 * __restrict__ data,
                           int                    data_count,
                           PairIF * __restrict__  result,
                           int                    k)
{
    extern __shared__ unsigned char smem[];
    unsigned char *ptr = smem;

    // Shared data tile: TILE_POINTS float2 points.
    float2 *tilePoints = reinterpret_cast<float2*>(ptr);
    ptr += TILE_POINTS * sizeof(float2);

    // Per-warp candidate buffers in shared memory:
    //   WARPS_PER_BLOCK * k integers and distances.
    int   *sharedCandIdx  = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(int);

    float *sharedCandDist = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(float);

    // Per-warp candidate counts.
    int   *sharedBufCount = reinterpret_cast<int*>(ptr);
    // ptr += WARPS_PER_BLOCK * sizeof(int); // not needed further

    const int lane            = threadIdx.x & (WARP_SIZE - 1);
    const int warpInBlock     = threadIdx.x >> 5; // / WARP_SIZE
    const int warpGlobal      = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;
    const bool warpActive     = (warpGlobal < query_count);
    const unsigned full_mask  = 0xffffffffu;

    const int itemsPerThread  = k / WARP_SIZE;

    // Each warp's region in the shared candidate buffers.
    float *warpCandDist = sharedCandDist + warpInBlock * k;
    int   *warpCandIdx  = sharedCandIdx  + warpInBlock * k;

    // Initialize per-warp candidate count.
    if (lane == 0) {
        sharedBufCount[warpInBlock] = 0;
    }

    // Registers for intermediate k-NN result (sorted ascending).
    float bestDist[MAX_ITEMS_PER_THREAD];
    int   bestIdx [MAX_ITEMS_PER_THREAD];

    // Temporary registers used when merging candidate buffer.
    float bufDist [MAX_ITEMS_PER_THREAD];
    int   bufIdx  [MAX_ITEMS_PER_THREAD];

    // Initialize intermediate result with +INF distances and invalid indices.
    for (int i = 0; i < MAX_ITEMS_PER_THREAD; ++i) {
        bestDist[i] = FLT_MAX;
        bestIdx[i]  = -1;
    }

    float2 queryPoint;
    if (warpActive) {
        queryPoint = query[warpGlobal];
    }

    float max_distance = FLT_MAX;

    // Process the data set in tiles.
    for (int tileBase = 0; tileBase < data_count; tileBase += TILE_POINTS) {
        int tileCount = data_count - tileBase;
        if (tileCount > TILE_POINTS) tileCount = TILE_POINTS;

        // Load tile into shared memory cooperatively by the whole block.
        for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
            tilePoints[i] = data[tileBase + i];
        }

        __syncthreads();

        if (warpActive) {
            // Iterate over points in the tile in groups of WARP_SIZE.
            for (int tBase = 0; tBase < tileCount; tBase += WARP_SIZE) {
                int idxInTile = tBase + lane;
                bool valid    = (idxInTile < tileCount);

                float dist = 0.0f;
                int   dataIndex = tileBase + idxInTile;

                if (valid) {
                    float2 p = tilePoints[idxInTile];
                    float dx = p.x - queryPoint.x;
                    float dy = p.y - queryPoint.y;
                    dist = dx * dx + dy * dy;
                } else {
                    dataIndex = -1;
                }

                // Filter by current max_distance.
                bool isCandidate = valid && (dist < max_distance);

                unsigned mask = __ballot_sync(full_mask, isCandidate);
                int numCand   = __popc(mask);

                if (numCand == 0) {
                    continue;
                }

                int bufferCount = sharedBufCount[warpInBlock];

                // If adding these candidates would overflow the buffer,
                // merge the existing buffer first.
                if (bufferCount + numCand > k) {
                    if (bufferCount > 0) {
                        warp_merge_buffer(bestDist, bestIdx,
                                          bufDist,  bufIdx,
                                          warpCandDist, warpCandIdx,
                                          k, bufferCount);

                        if (lane == 0) {
                            sharedBufCount[warpInBlock] = 0;
                        }
                        __syncwarp(full_mask);

                        // Update max_distance (distance of k-th nearest neighbor).
                        max_distance = __shfl_sync(full_mask,
                                                   bestDist[itemsPerThread - 1],
                                                   WARP_SIZE - 1);
                    }

                    bufferCount = 0;

                    // Recompute candidate flag with updated max_distance.
                    isCandidate = valid && (dist < max_distance);
                    mask        = __ballot_sync(full_mask, isCandidate);
                    numCand     = __popc(mask);
                    if (numCand == 0) {
                        continue;
                    }
                }

                // Now we know bufferCount + numCand <= k.
                int offset = __popc(mask & ((1u << lane) - 1));

                if (lane == 0) {
                    sharedBufCount[warpInBlock] = bufferCount + numCand;
                }

                if (isCandidate) {
                    int dest = bufferCount + offset;
                    warpCandIdx [dest] = dataIndex;
                    warpCandDist[dest] = dist;
                }

                int newBufferCount = bufferCount + numCand;

                // If the buffer just became full, merge it immediately.
                if (newBufferCount == k) {
                    warp_merge_buffer(bestDist, bestIdx,
                                      bufDist,  bufIdx,
                                      warpCandDist, warpCandIdx,
                                      k, newBufferCount);

                    if (lane == 0) {
                        sharedBufCount[warpInBlock] = 0;
                    }
                    __syncwarp(full_mask);

                    max_distance = __shfl_sync(full_mask,
                                               bestDist[itemsPerThread - 1],
                                               WARP_SIZE - 1);
                }
            }
        }

        __syncthreads();
    }

    if (warpActive) {
        // After processing all tiles, merge any remaining candidates.
        int bufferCount = sharedBufCount[warpInBlock];
        if (bufferCount > 0) {
            warp_merge_buffer(bestDist, bestIdx,
                              bufDist,  bufIdx,
                              warpCandDist, warpCandIdx,
                              k, bufferCount);
        }

        // Write the final k nearest neighbors for this query to global memory.
        for (int local = 0; local < itemsPerThread; ++local) {
            int g = local * WARP_SIZE + lane; // global neighbor index in [0, k)
            PairIF out;
            out.first  = bestIdx[local];
            out.second = bestDist[local];
            result[static_cast<size_t>(warpGlobal) * k + g] = out;
        }
    }
}

// Host interface as specified.
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

    // Reinterpret std::pair<int,float>* as a POD with the same layout.
    PairIF *result_cast = reinterpret_cast<PairIF*>(result);

    // Grid configuration: one warp per query.
    const int warpsPerBlock = WARPS_PER_BLOCK;
    int numWarps  = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    int numBlocks = numWarps;

    // Dynamic shared memory size per block.
    size_t sharedBytes =
        TILE_POINTS * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * k * (sizeof(int) + sizeof(float)) +
        WARPS_PER_BLOCK * sizeof(int);

    knn_kernel<<<numBlocks, THREADS_PER_BLOCK, sharedBytes>>>(
        query, query_count,
        data,  data_count,
        result_cast,
        k
    );

    // The caller may choose to synchronize or check errors outside this function.
}