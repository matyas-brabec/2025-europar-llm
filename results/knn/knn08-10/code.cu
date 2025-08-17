#include <cuda_runtime.h>
#include <stdint.h>

// This implementation assigns one warp (32 threads) to each query.
// Each thread keeps k/32 consecutive nearest neighbors in registers.
// Candidate buffers of size k per warp are stored in shared memory.
// Batches of data points are loaded into shared memory and processed by all warps in the block.
// When the candidate buffer is full (or after the final batch), we merge it into the intermediate
// result using the described bitonic sort and bitonic merge technique.
//
// Notes:
// - k is a power of two in [32, 1024]; therefore k/32 (KK) is a power of two in [1, 32].
// - The code uses warp-synchronous programming with __shfl_sync and __ballot_sync.
// - The intermediate result (top-k) is kept sorted ascending at all times (after merges).

// Tunable parameters and constants
#ifndef KNN_WARP_SIZE
#define KNN_WARP_SIZE 32
#endif

#ifndef KNN_THREADS_PER_BLOCK
#define KNN_THREADS_PER_BLOCK 256  // 8 warps per block
#endif

#ifndef KNN_TILE_POINTS
#define KNN_TILE_POINTS 4096       // Number of data points per shared-memory tile (~32KB)
#endif

#ifndef KNN_MAX_K
#define KNN_MAX_K 1024             // Maximum k we support
#endif

#define FULL_MASK 0xFFFFFFFFu

struct PairOut {
    int first;
    float second;
};

// Compute squared Euclidean distance between two float2 points
__device__ __forceinline__ float distance2(const float2 &a, const float2 &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // Use FMA to improve performance and precision
    return fmaf(dx, dx, dy * dy);
}

// Warp-wide bitonic sort for k elements distributed across threads.
// Each thread holds KK = k / 32 consecutive elements in registers, stored in arrays dist[] and idx[].
// The sort order is ascending by distance.
__device__ __forceinline__ void warp_bitonic_sort(float dist[], int idx[], int k, int KK, int lane) {
    // m = log2(KK). Since KK is a power of two, we can use __ffs(KK)-1 (ffs returns 1-based index).
    const int m = __ffs(KK) - 1;

    // Outer loop over bitonic sequence length
    for (int size = 2; size <= k; size <<= 1) {
        // Inner loop over half-clean distance "stride"
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Inter-thread or intra-thread phase depends on whether stride toggles a bit in the KK part (low m bits)
            if (stride < KK) {
                // Intra-thread compare-exchange: partner is within the same thread registers at index t^stride
                for (int t = 0; t < KK; t++) {
                    // Restrict updates to one side of each pair to avoid double-updates within the thread
                    if ((t & stride) == 0) {
                        int t2 = t ^ stride;

                        int i = lane * KK + t;
                        bool up = ((i & size) == 0);

                        float a_d = dist[t];
                        int   a_i = idx[t];

                        float b_d = dist[t2];
                        int   b_i = idx[t2];

                        // Compare
                        bool swap = (a_d > b_d);
                        // If descending, reverse comparison
                        if (!up) swap = !swap;

                        // Perform compare-exchange
                        float lo_d = swap ? b_d : a_d;
                        int   lo_i = swap ? b_i : a_i;
                        float hi_d = swap ? a_d : b_d;
                        int   hi_i = swap ? a_i : b_i;

                        if (up) {
                            dist[t]  = lo_d; idx[t]  = lo_i;
                            dist[t2] = hi_d; idx[t2] = hi_i;
                        } else {
                            dist[t]  = hi_d; idx[t]  = hi_i;
                            dist[t2] = lo_d; idx[t2] = lo_i;
                        }
                    }
                }
                // No inter-thread data dependency; __syncwarp() not strictly required here
            } else {
                // Inter-thread compare-exchange: partner lane differs; register index t stays the same.
                int partner_delta = stride >> m; // lane xor mask
                unsigned mask = FULL_MASK;

                for (int t = 0; t < KK; t++) {
                    int i = lane * KK + t;
                    bool up = ((i & size) == 0);

                    float a_d = dist[t];
                    int   a_i = idx[t];

                    float b_d = __shfl_xor_sync(mask, a_d, partner_delta);
                    int   b_i = __shfl_xor_sync(mask, a_i, partner_delta);

                    bool swap = (a_d > b_d);
                    if (!up) swap = !swap;

                    float lo_d = swap ? b_d : a_d;
                    int   lo_i = swap ? b_i : a_i;
                    float hi_d = swap ? a_d : b_d;
                    int   hi_i = swap ? a_i : b_i;

                    if (up) {
                        dist[t] = lo_d; idx[t] = lo_i;
                    } else {
                        dist[t] = hi_d; idx[t] = hi_i;
                    }
                }
                __syncwarp(); // Ensure warp-synchronous progress across inter-thread exchanges
            }
        }
    }
}

// Merge the candidate buffer (shared memory) with the intermediate result (registers) as described:
// 0) The intermediate result in registers is sorted ascending (invariant).
// 1) Swap content so that the buffer is in registers.
// 2) Sort the buffer (now in registers) ascending using bitonic sort.
// 3) Merge buffer and intermediate result into registers by taking min of pairs (i, k - i - 1).
// 4) Sort the merged result ascending using bitonic sort.
// After this function, the registers contain the updated intermediate result, sorted ascending,
// and maxDist is updated to the last (k-th) element.
__device__ __forceinline__ void warp_merge_buffer_with_result(
    float bestDist[], int bestIdx[],           // registers: current intermediate top-k
    float tmpDist[], int tmpIdx[],             // registers: temporary storage for buffer during swap/sort
    volatile float *bufDist, volatile int *bufIdx, // shared memory: candidate buffer (size k) per warp
    volatile int *bufCountPtr,                 // shared: candidate count
    int k, int KK, int lane, float &maxDist)
{
    int cc = *bufCountPtr;

    // Pad the buffer with +inf if it's partially filled
    if (cc < k) {
        for (int p = lane; p < (k - cc); p += KNN_WARP_SIZE) {
            int pos = cc + p;
            bufDist[pos] = CUDART_INF_F;
            bufIdx[pos]  = -1;
        }
    }
    __syncwarp();

    // Step 1: Swap content of buffer and intermediate result so that buffer goes to registers.
    // Load buffer into tmpDist/tmpIdx (registers), and write bestDist/bestIdx into shared buffer.
    for (int t = 0; t < KK; t++) {
        int i = lane * KK + t;
        float bd = bufDist[i];
        int   bi = bufIdx[i];
        tmpDist[t] = bd;
        tmpIdx[t]  = bi;

        bufDist[i] = bestDist[t];
        bufIdx[i]  = bestIdx[t];
    }
    __syncwarp();

    // Step 2: Sort buffer (now in tmpDist/tmpIdx registers) ascending
    warp_bitonic_sort(tmpDist, tmpIdx, k, KK, lane);

    __syncwarp();

    // Step 3: Merge buffer and previous result into registers by taking the element-wise minimum
    // of tmpDist[i] and bufDist[k - 1 - i].
    for (int t = 0; t < KK; t++) {
        int i = lane * KK + t;
        int j = k - 1 - i;

        float other_d = bufDist[j];
        int   other_i = bufIdx[j];

        float a_d = tmpDist[t];
        int   a_i = tmpIdx[t];

        bool take_a = (a_d < other_d);
        bestDist[t] = take_a ? a_d : other_d;
        bestIdx[t]  = take_a ? a_i : other_i;
    }
    __syncwarp();

    // Step 4: Sort the merged result ascending to restore invariant
    warp_bitonic_sort(bestDist, bestIdx, k, KK, lane);

    __syncwarp();

    // Update maxDist to the k-th element (last one in ascending order).
    float kth = bestDist[KK - 1];
    // Broadcast from last lane
    maxDist = __shfl_sync(FULL_MASK, kth, KNN_WARP_SIZE - 1);

    // Reset buffer count
    if (lane == 0) {
        *bufCountPtr = 0;
    }
    __syncwarp();
}

// The main kernel. One warp processes one query point.
__global__ void knn_kernel(
    const float2 * __restrict__ query, int query_count,
    const float2 * __restrict__ data,  int data_count,
    PairOut * __restrict__ result,
    int k)
{
    // Compute warp and lane identifiers
    const int lane = threadIdx.x & (KNN_WARP_SIZE - 1);
    const int warpInBlock = threadIdx.x >> 5; // /32
    const int warpsPerBlock = blockDim.x >> 5;

    const int queryIdx = blockIdx.x * warpsPerBlock + warpInBlock;
    if (queryIdx >= query_count) return;

    // Each warp handles a single query
    const float2 q = query[queryIdx];

    // Per-thread register storage for the intermediate top-k list (ascending order).
    // Each thread holds k/32 consecutive entries.
    const int KK = k / KNN_WARP_SIZE; // number of elements per thread
    float bestDist[KNN_MAX_K / KNN_WARP_SIZE];
    int   bestIdx [KNN_MAX_K / KNN_WARP_SIZE];
#pragma unroll
    for (int t = 0; t < KNN_MAX_K / KNN_WARP_SIZE; t++) {
        if (t < KK) {
            bestDist[t] = CUDART_INF_F;
            bestIdx[t]  = -1;
        }
    }

    // Temporary registers used during merge swap (to store buffer contents)
    float tmpDist[KNN_MAX_K / KNN_WARP_SIZE];
    int   tmpIdx [KNN_MAX_K / KNN_WARP_SIZE];

    // Shared memory layout:
    // [0] float2 tile[KNN_TILE_POINTS]
    // [1] per-warp candidate buffers: dist[warps*k], idx[warps*k]
    // [2] per-warp candidate counts: int[warps]
    extern __shared__ unsigned char smem[];
    size_t off = 0;

    // Shared tile for cached data points
    float2 *sTile = reinterpret_cast<float2*>(smem + off);
    off += sizeof(float2) * KNN_TILE_POINTS;

    // Align to 8 bytes for safety
    off = (off + 7) & ~size_t(7);

    // Candidate buffers for all warps in the block
    float *candDistBase = reinterpret_cast<float*>(smem + off);
    off += sizeof(float) * (size_t)warpsPerBlock * (size_t)k;

    int *candIdxBase = reinterpret_cast<int*>(smem + off);
    off += sizeof(int) * (size_t)warpsPerBlock * (size_t)k;

    // Candidate counters per warp
    int *candCountBase = reinterpret_cast<int*>(smem + off);
    // off += sizeof(int) * warpsPerBlock; // Not needed further

    // Pointers for this warp's candidate buffer and count
    float *myCandDist = candDistBase + (size_t)warpInBlock * (size_t)k;
    int   *myCandIdx  = candIdxBase  + (size_t)warpInBlock * (size_t)k;
    volatile int *myCandCount = candCountBase + warpInBlock;

    if (lane == 0) {
        *myCandCount = 0;
    }
    __syncwarp();

    // Current threshold distance: distance of the k-th nearest element in the intermediate result.
    float maxDist = CUDART_INF_F;

    // Process the data in tiles
    for (int tileStart = 0; tileStart < data_count; tileStart += KNN_TILE_POINTS) {
        int tileSize = data_count - tileStart;
        if (tileSize > KNN_TILE_POINTS) tileSize = KNN_TILE_POINTS;

        // Cooperative loading of the tile into shared memory by the whole block
        for (int t = threadIdx.x; t < tileSize; t += blockDim.x) {
            sTile[t] = data[tileStart + t];
        }
        __syncthreads();

        // Each warp processes all points in the tile
        for (int t = lane; t < tileSize; t += KNN_WARP_SIZE) {
            float2 d = sTile[t];
            float dist = distance2(q, d);

            // Filter by current maxDist (k-th neighbor distance)
            int take = (dist < maxDist) ? 1 : 0;

            // Warp ballot to count candidates
            unsigned mask = __ballot_sync(FULL_MASK, take);
            int add = __popc(mask);

            if (add > 0) {
                // If the buffer would overflow, merge it first
                int needMerge = 0;
                if (lane == 0) {
                    int cc = *myCandCount;
                    if (cc == k || (cc + add > k)) {
                        needMerge = 1;
                    }
                }
                needMerge = __shfl_sync(FULL_MASK, needMerge, 0);
                if (needMerge) {
                    // Merge the buffer with the intermediate result
                    warp_merge_buffer_with_result(
                        bestDist, bestIdx,
                        tmpDist, tmpIdx,
                        myCandDist, myCandIdx, myCandCount,
                        k, KK, lane, maxDist
                    );
                }

                // Reserve space in the candidate buffer
                int base = 0;
                if (lane == 0) {
                    int cc = *myCandCount;
                    *myCandCount = cc + add;
                    base = cc;
                }
                base = __shfl_sync(FULL_MASK, base, 0);

                if (take) {
                    // Offset of this lane's element within the newly added block
                    unsigned lower = mask & ((1u << lane) - 1u);
                    int offset = __popc(lower);
                    int pos = base + offset;
                    // Store candidate (we ensured no overflow)
                    myCandDist[pos] = dist;
                    myCandIdx[pos]  = tileStart + t;
                }
            }
        }

        // If the buffer is full after processing this tile, merge it
        int needMergeAfterTile = 0;
        if (lane == 0) {
            int cc = *myCandCount;
            if (cc == k) needMergeAfterTile = 1;
        }
        needMergeAfterTile = __shfl_sync(FULL_MASK, needMergeAfterTile, 0);
        if (needMergeAfterTile) {
            warp_merge_buffer_with_result(
                bestDist, bestIdx,
                tmpDist, tmpIdx,
                myCandDist, myCandIdx, myCandCount,
                k, KK, lane, maxDist
            );
        }

        __syncthreads(); // Ensure all warps finished using sTile before reloading
    }

    // Final merge if the buffer has remaining candidates
    int finalMerge = 0;
    if (lane == 0) {
        int cc = *myCandCount;
        if (cc > 0) finalMerge = 1;
    }
    finalMerge = __shfl_sync(FULL_MASK, finalMerge, 0);
    if (finalMerge) {
        warp_merge_buffer_with_result(
            bestDist, bestIdx,
            tmpDist, tmpIdx,
            myCandDist, myCandIdx, myCandCount,
            k, KK, lane, maxDist
        );
    }

    // Write out the results for this query in row-major order:
    // result[queryIdx * k + j] = (index, distance), j in [0, k)
    size_t outBase = (size_t)queryIdx * (size_t)k;
    for (int t = 0; t < KK; t++) {
        int j = lane * KK + t;
        PairOut out;
        out.first  = bestIdx[t];
        out.second = bestDist[t];
        result[outBase + j] = out;
    }
}

// Host function to launch the kernel.
// query: device pointer to float2 query points
// data: device pointer to float2 data points
// result: device pointer to std::pair<int, float> results (treated as PairOut here)
// k: power of two in [32, 1024]
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    const int threadsPerBlock = KNN_THREADS_PER_BLOCK; // multiple of 32
    const int warpsPerBlock = threadsPerBlock / KNN_WARP_SIZE;
    const int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute dynamic shared memory size:
    // tile: KNN_TILE_POINTS * sizeof(float2)
    // per-warp candidate buffers: warpsPerBlock * k * (sizeof(float) + sizeof(int))
    // per-warp counts: warpsPerBlock * sizeof(int)
    size_t smem_size = 0;
    smem_size += sizeof(float2) * (size_t)KNN_TILE_POINTS;
    // Align to 8
    smem_size = (smem_size + 7) & ~size_t(7);
    smem_size += (size_t)warpsPerBlock * (size_t)k * (sizeof(float) + sizeof(int));
    smem_size += sizeof(int) * (size_t)warpsPerBlock;

    PairOut *out = reinterpret_cast<PairOut*>(result);

    knn_kernel<<<blocks, threadsPerBlock, smem_size>>>(query, query_count, data, data_count, out, k);
}