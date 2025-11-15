#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ----------------------------------------------------------------------------------
// Basic configuration constants
// ----------------------------------------------------------------------------------

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Maximum supported k and per-thread element count
static constexpr int MAX_K  = 1024;
static constexpr int MAX_R  = MAX_K / WARP_SIZE;

// Number of warps per block. Each warp processes one query.
static constexpr int WARPS_PER_BLOCK = 4;

// Number of data points cached per block in shared memory.
static constexpr int TILE_POINTS = 1536;

// ----------------------------------------------------------------------------------
// Device-side helper structures and functions
// ----------------------------------------------------------------------------------

struct Candidate
{
    float dist;
    int   idx;
};

// Initialize the per-thread portion of the intermediate top-k result.
__device__ __forceinline__
void init_topk(float (&bestDist)[MAX_R], int (&bestIdx)[MAX_R], int R)
{
    for (int i = 0; i < R; ++i)
    {
        bestDist[i] = FLT_MAX;
        bestIdx[i]  = -1;
    }
}

// Warp-wide bitonic sort over k elements distributed across the 32 threads.
// Each thread holds R = k / 32 consecutive elements in its registers.
// The arrays bestDist / bestIdx are per-thread and contain R elements.
// The global index of element (thread lane, local position p) is:
//     i = lane * R + p, 0 <= i < k.
// For stride >= R, comparators operate across threads using shuffles,
// and for stride < R, comparators operate within the registers of a single thread.
__device__ __forceinline__
void warp_bitonic_sort(float (&dist)[MAX_R], int (&idx)[MAX_R], int n, int R)
{
    const unsigned FULL_MASK = 0xffffffffu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Bitonic sort network
    for (int size = 2; size <= n; size <<= 1)
    {
        // size: length of subsequences to be sorted in this stage
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            __syncwarp(FULL_MASK);

            if (stride >= R)
            {
                // Cross-thread comparators: same local index in different threads.
                // laneMask is the XOR distance in lane IDs.
                int laneMask = stride / R;

                for (int p = 0; p < R; ++p)
                {
                    int i = lane * R + p;
                    bool up = ((i & size) == 0);

                    float selfVal = dist[p];
                    int   selfIdx = idx[p];

                    float otherVal = __shfl_xor_sync(FULL_MASK, selfVal, laneMask);
                    int   otherIdx = __shfl_xor_sync(FULL_MASK, selfIdx, laneMask);

                    bool cond = (selfVal > otherVal);
                    if ((up && cond) || (!up && !cond))
                    {
                        selfVal = otherVal;
                        selfIdx = otherIdx;
                    }

                    dist[p] = selfVal;
                    idx[p]  = selfIdx;
                }
            }
            else
            {
                // Intra-thread comparators: swap register values inside a single thread.
                int stridePos = stride; // stride < R

                for (int p = 0; p < R; ++p)
                {
                    int partner_pos = p ^ stridePos;
                    if (partner_pos > p)
                    {
                        int i = lane * R + p;
                        bool up = ((i & size) == 0);

                        float valA = dist[p];
                        int   idxA = idx[p];
                        float valB = dist[partner_pos];
                        int   idxB = idx[partner_pos];

                        bool cond = (valA > valB);
                        if ((up && cond) || (!up && !cond))
                        {
                            // Swap values
                            float tmpV = valA; valA = valB; valB = tmpV;
                            int   tmpI = idxA; idxA = idxB; idxB = tmpI;
                        }

                        dist[p]          = valA;
                        idx[p]           = idxA;
                        dist[partner_pos] = valB;
                        idx[partner_pos]  = idxB;
                    }
                }
            }
        }
    }
}

// Merge a full candidate buffer (k elements) into the current intermediate result.
// Preconditions:
//   - bestDist / bestIdx contain the current intermediate result, sorted in ascending order.
//   - candBuf[0..k-1] contains a full buffer of candidate elements (unsorted).
// Steps (per specification):
//   0. Intermediate result in registers is sorted.
//   1. Swap content of registers and buffer so that the buffer is in registers.
//   2. Sort the buffer (now in registers) using bitonic sort.
//   3. Merge the buffer and the intermediate result (now in shared memory) into registers:
//        merged[i] = min( buffer[i], intermediate[k-1-i] ), for i in [0, k-1].
//      The merged sequence is bitonic and contains the best k elements of the union.
//   4. Sort the merged sequence again using bitonic sort to restore ascending order.
//   5. Update maxDist to be the distance of the k-th nearest neighbor.
__device__ __forceinline__
void warp_merge_full_buffer(
    float (&bestDist)[MAX_R],
    int   (&bestIdx)[MAX_R],
    float &maxDist,
    Candidate *candBuf, // per-warp candidate buffer in shared memory
    int k,
    int R)
{
    const unsigned FULL_MASK = 0xffffffffu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Step 1: swap content of buffer and intermediate result.
    for (int p = 0; p < R; ++p)
    {
        int i = lane * R + p;

        float bufDist = candBuf[i].dist;
        int   bufIdx  = candBuf[i].idx;

        candBuf[i].dist = bestDist[p];
        candBuf[i].idx  = bestIdx[p];

        bestDist[p] = bufDist;
        bestIdx[p]  = bufIdx;
    }

    // Step 2: sort the buffer (now in registers).
    warp_bitonic_sort(bestDist, bestIdx, k, R);

    // Step 3: merge buffer (registers, sorted) and intermediate result (shared, sorted).
    // For each global position i, take the minimum of:
    //   - buffer[i]   (registers)
    //   - intermediate[k - 1 - i] (shared memory)
    for (int p = 0; p < R; ++p)
    {
        int i = lane * R + p;

        float regDist = bestDist[p];
        int   regIdx  = bestIdx[p];

        Candidate other = candBuf[k - 1 - i];

        if (other.dist < regDist)
        {
            regDist = other.dist;
            regIdx  = other.idx;
        }

        bestDist[p] = regDist;
        bestIdx[p]  = regIdx;
    }

    // Step 4: sort merged result (still in registers) in ascending order.
    warp_bitonic_sort(bestDist, bestIdx, k, R);

    // Step 5: update maxDist = distance of the k-th nearest neighbor.
    // After sorting ascending, the k-th (largest) element is at index k-1,
    // which is stored in the last register of lane (WARP_SIZE - 1).
    float kth = bestDist[R - 1];
    kth = __shfl_sync(FULL_MASK, kth, WARP_SIZE - 1);
    maxDist = kth;
}

// ----------------------------------------------------------------------------------
// k-NN CUDA kernel
// Each warp processes a single query point.
// Each thread in the warp holds k / 32 consecutive nearest neighbors in registers.
// ----------------------------------------------------------------------------------

__global__
void knn_kernel(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    std::pair<int, float> * __restrict__ result,
    int k)
{
    const unsigned FULL_MASK = 0xffffffffu;

    int lane              = threadIdx.x & (WARP_SIZE - 1);
    int warp_id_in_block  = threadIdx.x >> 5; // threadIdx.x / 32
    int warp_global_id    = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    bool warp_active      = (warp_global_id < query_count);

    // Per-thread number of neighbors kept in registers.
    int R = k / WARP_SIZE;

    // Shared memory layout (dynamic):
    // [0 .. TILE_POINTS-1]        : cached data points for the block
    // [TILE_POINTS .. ]           : WARPS_PER_BLOCK * k Candidate entries
    // [after Candidate array]     : WARPS_PER_BLOCK ints (candidate counts)
    extern __shared__ unsigned char shared_mem[];
    float2  *tilePoints    = reinterpret_cast<float2*>(shared_mem);
    Candidate *allCandBuf  = reinterpret_cast<Candidate*>(tilePoints + TILE_POINTS);
    int     *candCounts    = reinterpret_cast<int*>(allCandBuf + WARPS_PER_BLOCK * k);

    // Per-warp candidate buffer and counter in shared memory.
    Candidate *candBuf = allCandBuf + warp_id_in_block * k;
    int       *candCountPtr = candCounts + warp_id_in_block;

    // Per-thread intermediate top-k result stored in registers.
    float bestDist[MAX_R];
    int   bestIdx[MAX_R];

    float qx = 0.0f;
    float qy = 0.0f;

    if (warp_active)
    {
        // Initialize per-thread top-k.
        init_topk(bestDist, bestIdx, R);

        // Load the query point for this warp.
        float2 q = query[warp_global_id];
        qx = q.x;
        qy = q.y;

        // Initialize candidate count for this warp.
        if (lane == 0)
        {
            *candCountPtr = 0;
        }
    }

    // maxDist: distance of current k-th nearest neighbor (per warp).
    float maxDist = FLT_MAX;

    __syncthreads(); // Ensure shared memory initialization visibility.

    // Process data points in batches cached into shared memory.
    for (int base = 0; base < data_count; base += TILE_POINTS)
    {
        int tileCount = data_count - base;
        if (tileCount > TILE_POINTS)
            tileCount = TILE_POINTS;

        // Load a tile of data points into shared memory cooperatively.
        for (int t = threadIdx.x; t < tileCount; t += blockDim.x)
        {
            tilePoints[t] = data[base + t];
        }

        __syncthreads();

        if (warp_active)
        {
            // Iterate over the points in the tile, one batch of 32 points per iteration.
            for (int t = lane; t < tileCount; t += WARP_SIZE)
            {
                float2 dp = tilePoints[t];
                float dx = dp.x - qx;
                float dy = dp.y - qy;
                float dist = dx * dx + dy * dy;
                int   data_idx = base + t;

                // Determine which threads have a candidate closer than current maxDist.
                bool is_candidate = (dist < maxDist);
                unsigned int mask = __ballot_sync(FULL_MASK, is_candidate);
                int n = __popc(mask);

                if (n == 0)
                {
                    continue;
                }

                // Retrieve the current candidate count from shared memory.
                int candCount = 0;
                if (lane == 0)
                {
                    candCount = *candCountPtr;
                }
                candCount = __shfl_sync(FULL_MASK, candCount, 0);

                int leftover = k - candCount; // remaining capacity in buffer

                // rank: position among new candidates in this iteration.
                int rank = __popc(mask & ((1u << lane) - 1u));

                if (n <= leftover)
                {
                    // All new candidates fit into the current buffer.

                    if (lane == 0)
                    {
                        *candCountPtr = candCount + n;
                    }

                    if (is_candidate)
                    {
                        int dst = candCount + rank;
                        candBuf[dst].dist = dist;
                        candBuf[dst].idx  = data_idx;
                    }

                    // If buffer is exactly full (candCount+n == k), merge it.
                    bool doMerge = false;
                    if (lane == 0)
                    {
                        doMerge = (*candCountPtr == k);
                    }
                    doMerge = __shfl_sync(FULL_MASK, doMerge, 0);

                    if (doMerge)
                    {
                        __syncwarp(FULL_MASK);

                        // Merge full buffer with intermediate result.
                        warp_merge_full_buffer(bestDist, bestIdx, maxDist, candBuf, k, R);

                        if (lane == 0)
                        {
                            *candCountPtr = 0;
                        }
                    }
                }
                else
                {
                    // Not all new candidates fit into the current buffer.
                    // First fill the remaining capacity, then merge, then handle leftovers.

                    // First chunk: candidates with rank < leftover.
                    if (is_candidate && (rank < leftover))
                    {
                        int dst = candCount + rank;
                        candBuf[dst].dist = dist;
                        candBuf[dst].idx  = data_idx;
                    }

                    if (lane == 0)
                    {
                        *candCountPtr = k;
                    }

                    __syncwarp(FULL_MASK);

                    // Merge full buffer with intermediate result.
                    warp_merge_full_buffer(bestDist, bestIdx, maxDist, candBuf, k, R);

                    if (lane == 0)
                    {
                        *candCountPtr = 0;
                    }

                    __syncwarp(FULL_MASK);

                    // Second chunk: remaining candidates (rank >= leftover), re-filtered
                    // against the updated maxDist.
                    bool is_candidate2 = is_candidate && (rank >= leftover) && (dist < maxDist);
                    unsigned int mask2 = __ballot_sync(FULL_MASK, is_candidate2);
                    int n2 = __popc(mask2);

                    if (n2 > 0)
                    {
                        int rank2 = __popc(mask2 & ((1u << lane) - 1u));

                        if (lane == 0)
                        {
                            *candCountPtr = n2;
                        }

                        if (is_candidate2)
                        {
                            int dst = rank2;
                            candBuf[dst].dist = dist;
                            candBuf[dst].idx  = data_idx;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    if (warp_active)
    {
        // After processing all tiles, merge remaining candidates (if any).
        int candCount = 0;
        if (lane == 0)
        {
            candCount = *candCountPtr;
        }
        candCount = __shfl_sync(FULL_MASK, candCount, 0);

        if (candCount > 0)
        {
            // Fill the rest of the buffer with sentinel values so that it has k elements.
            // Sentinel distance is maxDist; they will not improve the top-k set.
            int fillCount = k - candCount;
            for (int offset = lane; offset < fillCount; offset += WARP_SIZE)
            {
                int dst = candCount + offset;
                candBuf[dst].dist = maxDist;
                candBuf[dst].idx  = -1;
            }

            if (lane == 0)
            {
                *candCountPtr = k;
            }

            __syncwarp(FULL_MASK);

            // Merge full buffer with intermediate result.
            warp_merge_full_buffer(bestDist, bestIdx, maxDist, candBuf, k, R);

            if (lane == 0)
            {
                *candCountPtr = 0;
            }
        }

        // Write final top-k for this query to global memory.
        int baseOut = warp_global_id * k;
        for (int p = 0; p < R; ++p)
        {
            int globalPos = (lane * R) + p;
            int outIdx = baseOut + globalPos;

            result[outIdx].first  = bestIdx[p];
            result[outIdx].second = bestDist[p];
        }
    }
}

// ----------------------------------------------------------------------------------
// Host-side interface
// ----------------------------------------------------------------------------------

void run_knn(
    const float2 *query, int query_count,
    const float2 *data,  int data_count,
    std::pair<int, float> *result,
    int k)
{
    // Each warp processes one query.
    int totalWarps = query_count;
    int threadsPerBlock = WARPS_PER_BLOCK * WARP_SIZE;
    int blocks = (totalWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Shared memory size:
    //   TILE_POINTS * sizeof(float2)                         : cached data points
    // + WARPS_PER_BLOCK * k * sizeof(Candidate)              : candidate buffers
    // + WARPS_PER_BLOCK * sizeof(int)                        : candidate counts
    size_t sharedMemBytes =
        TILE_POINTS * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(Candidate) +
        WARPS_PER_BLOCK * sizeof(int);

    knn_kernel<<<blocks, threadsPerBlock, sharedMemBytes>>>(
        query, query_count, data, data_count, result, k);

    // Ensure completion before returning.
    cudaDeviceSynchronize();
}