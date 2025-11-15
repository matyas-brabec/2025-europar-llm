// CUDA implementation of k-NN for 2D points with one warp (32 threads) per query.
// The algorithm uses a per-warp intermediate result stored in registers and
// a per-warp candidate buffer in shared memory. Candidates are accumulated and
// periodically merged into the intermediate result using bitonic sort and a
// bitonic merge operation implemented with warp shuffle intrinsics.

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <utility>

// Constants tuned for modern data-center GPUs (A100/H100-like).
// These can be adjusted if necessary.
static constexpr int WARP_SIZE            = 32;
static constexpr int MAX_K                = 1024;                 // Maximum supported k
static constexpr int MAX_ELEMS_PER_THREAD = MAX_K / WARP_SIZE;    // 32
static constexpr int THREADS_PER_BLOCK    = 256;                  // 8 warps per block
static constexpr int BATCH_SIZE           = 2048;                 // Number of data points per shared-memory batch

// Internal representation for results on device, assumed to be layout-compatible
// with std::pair<int,float> for the given platform (two 32-bit values).
struct ResultPairDevice {
    int   first;
    float second;
};

// Warp-level bitonic sort of k elements distributed across a warp.
//
// - Each thread stores elemsPerThread = k / WARP_SIZE consecutive elements
//   in arrays dist[0..elemsPerThread-1] and idx[0..elemsPerThread-1].
// - The global index of element (lane, e) is i = lane * elemsPerThread + e.
// - k is a power of two between 32 and 1024 (inclusive), so elemsPerThread is
//   a power of two between 1 and 32.
//
// The implementation follows the classical bitonic sort network:
//   for (size = 2; size <= k; size <<= 1)
//     for (stride = size >> 1; stride > 0; stride >>= 1)
//       for each i in parallel:
//         partner = i ^ stride
//         dir = ((i & size) == 0)
//         if ((dist[i] > dist[partner]) == dir) swap(dist[i], dist[partner])
//
// We adapt this to a distributed layout and use warp shuffles for exchanges
// that occur between different lanes. For exchanges within the same lane,
// we swap registers directly. Thanks to the structure of the network,
// exchanges between lanes always involve elements with the same local index
// within each thread.
__device__ __forceinline__
void warpBitonicSort(float (&dist)[MAX_ELEMS_PER_THREAD],
                     int   (&idx)[MAX_ELEMS_PER_THREAD],
                     int k)
{
    const unsigned fullMask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int elemsPerThread = k / WARP_SIZE;

    // Bitonic sort over k elements.
    for (int size = 2; size <= k; size <<= 1) {
        // size is the size of the bitonic subsequences being merged.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // stride controls the distance between compared elements.

            if (stride >= elemsPerThread) {
                // Cross-lane comparisons: partner is in another lane, but with the
                // same local index. Because k and elemsPerThread are powers of two
                // and stride is also a power of two, stride / elemsPerThread is an
                // integer power of two that can be used as the XOR mask for lanes.
                const int laneMask = stride / elemsPerThread;

                #pragma unroll
                for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
                    if (e >= elemsPerThread) break;

                    const int globalIdx = lane * elemsPerThread + e;
                    const bool up = ((globalIdx & size) == 0); // sort direction for this element

                    float selfVal = dist[e];
                    int   selfIdx = idx[e];

                    float otherVal = __shfl_xor_sync(fullMask, selfVal, laneMask, WARP_SIZE);
                    int   otherIdx = __shfl_xor_sync(fullMask, selfIdx, laneMask, WARP_SIZE);

                    const bool swap = (selfVal > otherVal) == up;
                    if (swap) {
                        selfVal = otherVal;
                        selfIdx = otherIdx;
                    }

                    dist[e] = selfVal;
                    idx[e]  = selfIdx;
                }
            } else {
                // Intra-lane comparisons: partner is in the same lane, with a
                // different local index. We swap registers directly.
                const int strideLocal = stride;

                #pragma unroll
                for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
                    if (e >= elemsPerThread) break;

                    const int globalIdx = lane * elemsPerThread + e;
                    const bool up = ((globalIdx & size) == 0);

                    const int partner = e ^ strideLocal; // partner in same lane

                    float selfVal = dist[e];
                    int   selfIdx = idx[e];

                    float otherVal = dist[partner];
                    int   otherIdx = idx[partner];

                    const bool swap = (selfVal > otherVal) == up;
                    if (swap) {
                        dist[e] = otherVal;
                        idx[e]  = otherIdx;
                    }
                }
            }
        }
    }
}

// Merge the candidate buffer in shared memory with the current intermediate
// result stored in registers for a single warp.
//
// Inputs:
//   regDist, regIdx: current intermediate result in registers, sorted ascending.
//   sharedBufDist, sharedBufIdx: shared-memory buffer for this warp, containing
//       candCount candidate entries in positions [0 .. candCount-1] (unsorted).
//   k: total number of neighbors to keep per query (power of two in [32,1024]).
//   candCount: number of valid candidates currently stored in shared buffer.
//   maxDist: reference to current max_distance (distance of k-th neighbor).
//
// Procedure (per specification):
//   0. Intermediate result in registers is sorted ascending (invariant).
//   1. Pad buffer with +INF up to k elements, then swap contents of buffer
//      and intermediate result so that the buffer is in registers.
//   2. Sort the buffer in registers using bitonic sort (ascending).
//   3. Merge the sorted buffer (in registers) and the former intermediate
//      result (now in shared memory) by taking, for each i, the minimum of
//      buffer[i] and intermediate[k-1-i]; this yields a bitonic sequence of
//      length k in registers containing the k smallest elements overall.
//   4. Sort the merged bitonic sequence in registers using bitonic sort.
//   5. Update maxDist to the last element (k-th neighbor) of the sorted result.
__device__ __forceinline__
void warpMergeWithBuffer(float (&regDist)[MAX_ELEMS_PER_THREAD],
                         int   (&regIdx)[MAX_ELEMS_PER_THREAD],
                         float *sharedBufDist,
                         int   *sharedBufIdx,
                         int k,
                         int candCount,
                         float &maxDist)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned fullMask = 0xFFFFFFFFu;
    const int elemsPerThread = k / WARP_SIZE;

    // Step 0: pad the candidate buffer with +INF and dummy indices so that
    // it conceptually contains exactly k elements.
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
        if (e >= elemsPerThread) break;

        int globalIdx = lane * elemsPerThread + e;
        if (globalIdx >= candCount && globalIdx < k) {
            sharedBufDist[globalIdx] = FLT_MAX;
            sharedBufIdx[globalIdx]  = -1;
        }
    }
    __syncwarp();

    // Step 1: swap contents of buffer (shared memory) and intermediate result
    // (registers), so that the buffer is now in registers and the old
    // intermediate result resides in shared memory.
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
        if (e >= elemsPerThread) break;

        int globalIdx = lane * elemsPerThread + e;
        if (globalIdx < k) {
            float tmpDist = sharedBufDist[globalIdx];
            int   tmpIdx  = sharedBufIdx[globalIdx];

            sharedBufDist[globalIdx] = regDist[e];
            sharedBufIdx[globalIdx]  = regIdx[e];

            regDist[e] = tmpDist;
            regIdx[e]  = tmpIdx;
        }
    }
    __syncwarp();

    // Step 2: sort the buffer (now in registers) in ascending order.
    warpBitonicSort(regDist, regIdx, k);

    // Step 3: merge the sorted buffer (in registers) with the previous
    // intermediate result (now in shared memory). For each position i in [0,k),
    // we compare buffer[i] with intermediate[k-1-i] and keep the minimum.
    // The resulting sequence in registers is bitonic and contains the k
    // smallest elements from both sequences.
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
        if (e >= elemsPerThread) break;

        int i = lane * elemsPerThread + e;
        if (i < k) {
            int j = k - 1 - i;

            float aDist = regDist[e];
            int   aIdx  = regIdx[e];

            float bDist = sharedBufDist[j];
            int   bIdx  = sharedBufIdx[j];

            if (bDist < aDist) {
                regDist[e] = bDist;
                regIdx[e]  = bIdx;
            }
        }
    }
    __syncwarp();

    // Step 4: sort the resulting bitonic sequence in registers.
    warpBitonicSort(regDist, regIdx, k);

    // Step 5: update maxDist to the distance of the k-th nearest neighbor,
    // which is the last element in the sorted sequence.
    float kthDist = 0.0f;
    if (lane == WARP_SIZE - 1) {
        // Each thread holds elemsPerThread consecutive elements; the last element
        // in the sequence is in lane (WARP_SIZE-1) at index (elemsPerThread-1).
        kthDist = regDist[elemsPerThread - 1];
    }
    kthDist = __shfl_sync(fullMask, kthDist, WARP_SIZE - 1);
    maxDist = kthDist;
}

// Kernel: one warp processes one query point.
// Each warp maintains:
//   - A register-resident sorted array of k nearest neighbors (indices & distances).
//   - A shared-memory candidate buffer of up to k (index, distance) pairs.
//
// The data points are processed in batches. Each batch is loaded into shared memory
// by the entire block. Then, each warp computes distances from its query point to
// the cached data points, filters them using maxDist, and inserts the qualifying
// candidates into its shared buffer using warp-wide ballot and prefix sums.
// When the buffer is "close" to full (more than k - WARP_SIZE elements), it is
// merged into the intermediate result using the warpMergeWithBuffer() routine.
__global__
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                ResultPairDevice * __restrict__ result,
                int k)
{
    const int threadId      = threadIdx.x;
    const int lane          = threadId & (WARP_SIZE - 1);
    const int warpId        = threadId >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int queryId = blockIdx.x * warpsPerBlock + warpId;
    const bool warpActive = (queryId < query_count);

    const int elemsPerThread = k / WARP_SIZE;

    // Shared memory layout:
    // [0 .. BATCH_SIZE-1]      : float2 sharedData[BATCH_SIZE]
    // [next ..]                : float candDist[warpsPerBlock * k]
    // [next .. end]            : int   candIdx [warpsPerBlock * k]
    extern __shared__ unsigned char sharedRaw[];
    unsigned char *ptr = sharedRaw;

    float2 *sharedData = reinterpret_cast<float2*>(ptr);
    ptr += BATCH_SIZE * sizeof(float2);

    float *candDistAll = reinterpret_cast<float*>(ptr);
    ptr += warpsPerBlock * k * sizeof(float);

    int *candIdxAll = reinterpret_cast<int*>(ptr);
    // ptr += warpsPerBlock * k * sizeof(int); // Not needed further

    // Per-warp pointers into candidate buffers.
    float *warpCandDist = candDistAll + warpId * k;
    int   *warpCandIdx  = candIdxAll  + warpId * k;

    // Load the query point for this warp.
    float2 q;
    if (warpActive) {
        if (lane == 0) {
            q = query[queryId];
        }
        const unsigned fullMask = 0xFFFFFFFFu;
        q.x = __shfl_sync(fullMask, q.x, 0);
        q.y = __shfl_sync(fullMask, q.y, 0);
    }

    // Intermediate k-NN result in registers: each thread holds elemsPerThread
    // consecutive elements. Initially, all distances are +INF and indices are -1.
    float regDist[MAX_ELEMS_PER_THREAD];
    int   regIdx [MAX_ELEMS_PER_THREAD];

    #pragma unroll
    for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
        if (e >= elemsPerThread) break;
        regDist[e] = FLT_MAX;
        regIdx[e]  = -1;
    }

    float maxDist = FLT_MAX;    // Distance of the current k-th nearest neighbor.
    int   candCount = 0;        // Number of candidates stored in this warp's buffer.

    const unsigned fullMask = 0xFFFFFFFFu;

    // Process data in batches.
    for (int base = 0; base < data_count; base += BATCH_SIZE) {
        int batchSize = data_count - base;
        if (batchSize > BATCH_SIZE) batchSize = BATCH_SIZE;

        // Load this batch of data points into shared memory cooperatively.
        for (int i = threadId; i < batchSize; i += blockDim.x) {
            sharedData[i] = data[base + i];
        }

        __syncthreads();

        if (warpActive) {
            // Number of iterations required so that each warp thread processes
            // all elements of the batch. Each iteration handles WARP_SIZE points.
            const int warpIters = (batchSize + WARP_SIZE - 1) / WARP_SIZE;

            for (int it = 0; it < warpIters; ++it) {
                // Ensure there is enough room in the candidate buffer for up to
                // WARP_SIZE new candidates. If not, merge the current buffer into
                // the intermediate result first.
                if (candCount > k - WARP_SIZE && candCount > 0) {
                    warpMergeWithBuffer(regDist, regIdx,
                                        warpCandDist, warpCandIdx,
                                        k, candCount, maxDist);
                    candCount = 0;
                }

                const int dIdx = it * WARP_SIZE + lane;
                const bool valid = (dIdx < batchSize);

                float dist = 0.0f;
                bool  isCandidate = false;
                int   globalDataIndex = -1;

                if (valid) {
                    const float2 p = sharedData[dIdx];
                    const float dx = p.x - q.x;
                    const float dy = p.y - q.y;
                    // Squared Euclidean distance.
                    dist = fmaf(dx, dx, dy * dy);
                    globalDataIndex = base + dIdx;
                    isCandidate = (dist < maxDist);
                }

                // Use warp-wide ballot to determine which threads found candidates.
                unsigned int mask = __ballot_sync(fullMask, isCandidate);
                int newCount = __popc(mask);

                // Append new candidates into this warp's shared buffer.
                int baseOffset = candCount;
                candCount += newCount;

                if (isCandidate) {
                    // Offset of this thread's candidate within the new segment.
                    unsigned int laneMask = (1u << lane) - 1u;
                    int offset = __popc(mask & laneMask);
                    int outPos = baseOffset + offset;

                    warpCandDist[outPos] = dist;
                    warpCandIdx[outPos]  = globalDataIndex;
                }
            }
        }

        __syncthreads();
    }

    // After processing all batches, merge the remaining candidates (if any)
    // into the intermediate result.
    if (warpActive && candCount > 0) {
        warpMergeWithBuffer(regDist, regIdx,
                            warpCandDist, warpCandIdx,
                            k, candCount, maxDist);
        candCount = 0;
    }

    // Write the final k nearest neighbors for this query to global memory.
    if (warpActive) {
        const int outBase = queryId * k;

        #pragma unroll
        for (int e = 0; e < MAX_ELEMS_PER_THREAD; ++e) {
            if (e >= elemsPerThread) break;

            int globalIdx = lane * elemsPerThread + e;
            int outIdx    = outBase + globalIdx;

            result[outIdx].first  = regIdx[e];
            result[outIdx].second = regDist[e];
        }
    }
}

// Host interface as requested.
//
// query:       device pointer to array of query_count float2 points.
// data:        device pointer to array of data_count float2 points.
// result:      device pointer to array of query_count * k std::pair<int,float>.
// k:           power-of-two between 32 and 1024, inclusive.
//
// The function launches the CUDA kernel with one warp per query and uses
// dynamic shared memory for batches and candidate buffers.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0) {
        return;
    }

    // Assumptions: k is a power of two between 32 and 1024, inclusive.
    // (As specified in the problem statement.)
    const int warpsPerBlock = THREADS_PER_BLOCK / WARP_SIZE;
    const int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Dynamic shared memory size:
    //  - BATCH_SIZE float2s for cached data points.
    //  - warpsPerBlock * k floats and ints for candidate buffers.
    size_t sharedMemSize =
        static_cast<size_t>(BATCH_SIZE) * sizeof(float2) +
        static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) *
            (sizeof(float) + sizeof(int));

    // Reinterpret result pointer for device use; ResultPairDevice is layout-compatible
    // with std::pair<int,float> for the purposes of this implementation.
    ResultPairDevice *resultDevice =
        reinterpret_cast<ResultPairDevice*>(result);

    knn_kernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(
        query,
        query_count,
        data,
        data_count,
        resultDevice,
        k
    );

    // Optional synchronization; can be removed if the caller handles it.
    cudaDeviceSynchronize();
}