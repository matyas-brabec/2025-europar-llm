#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

//------------------------------------------------------------------------------
// In this implementation each query (a float2 point) is processed by one warp
// (32 threads). Each warp maintains its own “private” intermediate k‐nearest
// neighbor list in registers (distributed evenly among its 32 lanes) and a candidate
// buffer in shared memory (capacity = k candidates). The candidate buffer is used
// to accumulate newly–found data–points (from batches of data loaded into shared
// memory) that are closer than the current “max_distance” (i.e. the k-th nearest
// neighbor so far). Whenever the candidate buffer fills, it is merged in parallel
// (using warp–synchronous, odd–even transposition sort) with the intermediate result.
// At the end, if any candidates remain in the candidate buffer, a final merge is done.
// The final sorted list is then written to the global result array.
//------------------------------------------------------------------------------

// Candidate structure representing a data point candidate
// (with its index and squared Euclidean distance).
struct Candidate {
    int idx;
    float dist;
};

// A simple structure analogous to std::pair<int,float> for output.
struct KNNResult {
    int first;
    float second;
};

//------------------------------------------------------------------------------
// Kernel hyper-parameters (block size, batch size, and maximum k).
// BLOCK_SIZE: number of threads per block (must be a multiple of 32)
// BATCH_SIZE: number of data points to load per batch into shared memory.
// MAX_K: maximum candidate buffer capacity per query (k is guaranteed <= MAX_K)
// MAX_WARPS: maximum warps per block computed as BLOCK_SIZE/32
//------------------------------------------------------------------------------
#ifndef BLOCK_SIZE
  #define BLOCK_SIZE 128
#endif
#ifndef BATCH_SIZE
  #define BATCH_SIZE 1024
#endif
#ifndef MAX_K
  #define MAX_K 1024
#endif
#define MAX_WARPS (BLOCK_SIZE/32)

//------------------------------------------------------------------------------
// __device__ function: merge_intermediate
// This function is called by a warp (all 32 threads participate) to merge its
// private intermediate result (of k candidates, stored in registers distributed
// among the warp threads) and the candidate buffer (of recently-found candidates,
// stored in shared memory). The two lists are concatenated and then sorted using
// an odd-even transposition sort. Only the k smallest candidates (i.e., the new
// intermediate k-NN) are retained and written back to the private registers.
// Parameters:
//   warp_id      - warp's ID within the block (0 <= warp_id < MAX_WARPS)
//   k            - number of neighbors requested for each query
//   numPerThread - number of candidates held by each thread in the intermediate result (k/32)
//   inter_idx    - pointer to the per-thread private candidate indices (length = numPerThread)
//   inter_dist   - pointer to the per-thread private candidate distances (length = numPerThread)
//   max_distance - reference to the warp's current k-th neighbor distance; updated after merge
//   sCandBuf     - pointer to shared candidate buffer (size = MAX_K per warp)
//   sMergeBuf    - pointer to shared merge buffer (size = 2*MAX_K per warp)
//   candCount    - current number of candidates stored in the candidate buffer
//                  (if merging early due to full buffer, candCount==k; for final merge, it may be less)
// Note: The candidate buffer for this warp is at offset (warp_id*MAX_K)
//       The merge buffer for this warp is at offset (warp_id*(2*MAX_K))
//------------------------------------------------------------------------------
__device__ inline void merge_intermediate(int warp_id, int k, int numPerThread,
                                           int inter_idx[], float inter_dist[],
                                           float &max_distance,
                                           Candidate *sCandBuf,
                                           Candidate *sMergeBuf,
                                           int candCount)
{
    // Each warp uses a dedicated portion of the merge buffer.
    int mergeBase = warp_id * (2 * MAX_K);
    int warpCandOffset = warp_id * MAX_K;

    // Step 1. Copy the private intermediate result (k candidates) from registers
    // into the first half of the merge buffer.
    // The intermediate result is distributed among the warp threads:
    // each thread holds 'numPerThread' sorted candidates.
    #pragma unroll
    for (int i = 0; i < numPerThread; i++) {
        int globPos = i * 32 + (threadIdx.x & 31);
        // Each thread writes its candidate to the merge buffer.
        sMergeBuf[mergeBase + globPos].idx  = inter_idx[i];
        sMergeBuf[mergeBase + globPos].dist = inter_dist[i];
    }

    // Step 2. Copy the candidate buffer from shared memory into the merge buffer.
    // We copy the valid candidates (candCount entries) into the second half.
    // For positions not filled (if candCount < k), fill with "empty" candidate (dist=FLT_MAX).
    for (int i = (threadIdx.x & 31); i < candCount; i += 32) {
        sMergeBuf[mergeBase + k + i] = sCandBuf[warpCandOffset + i];
    }
    for (int i = (threadIdx.x & 31); i < (k - candCount); i += 32) {
        sMergeBuf[mergeBase + k + candCount + i].idx  = -1;
        sMergeBuf[mergeBase + k + candCount + i].dist = FLT_MAX;
    }
    __syncwarp();

    // Total elements to sort = 2*k (the concatenation of intermediate result and candidate buffer).
    int total = 2 * k;

    // Step 3. Odd-even transposition sort over the merge buffer for this warp.
    // We perform "total" phases. In each phase, threads cooperatively compare and swap
    // adjacent pairs.
    for (int phase = 0; phase < total; phase++) {
        // Each thread processes multiple pairs, striding by warp size (32).
        for (int i = (threadIdx.x & 31); i < total - 1; i += 32) {
            // For even (or odd) phase, process pairs where the first index has the same parity as phase.
            if ((i & 1) == (phase & 1)) {
                int idxA = mergeBase + i;
                int idxB = mergeBase + i + 1;
                Candidate a = sMergeBuf[idxA];
                Candidate b = sMergeBuf[idxB];
                if (a.dist > b.dist) {
                    sMergeBuf[idxA] = b;
                    sMergeBuf[idxB] = a;
                }
            }
        }
        __syncwarp();
    }
    // Now sMergeBuf[mergeBase ... mergeBase+total-1] is sorted in ascending order by distance.
    // The new intermediate result (k best candidates) are the first k elements.
    #pragma unroll
    for (int i = 0; i < numPerThread; i++) {
        int globPos = i * 32 + (threadIdx.x & 31);
        Candidate cand = sMergeBuf[mergeBase + globPos];
        inter_idx[i]  = cand.idx;
        inter_dist[i] = cand.dist;
    }
    // Update max_distance (the worst distance among the current k best, i.e. the k-th neighbor).
    if ((threadIdx.x & 31) == 31) {
        max_distance = sMergeBuf[mergeBase + k - 1].dist;
    }
    max_distance = __shfl_sync(0xFFFFFFFF, max_distance, 31);

    // Reset candidate buffer count for this warp (only lane 0 does this).
    // Note: The candidate buffer for this warp will be reused after merge.
    extern __shared__ int dummy[]; // dummy use to allow __syncthreads if needed
    if ((threadIdx.x & 31) == 0)
        ((int *)(&sCandBuf[warp_id * MAX_K]))[0] = 0; // not actually used; we use sCandBuf in global index form below.
    // Instead, we assume that the sCandBuf count is stored in a separate shared memory array (see kernel below).
    // (The actual candidate counter is reset in the kernel after merge.)
    __syncwarp();
}

//------------------------------------------------------------------------------
// The main CUDA kernel that computes the k-nearest neighbors for 2D query points.
// Each warp (32 threads) works on one query point.
//------------------------------------------------------------------------------
__global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                           const float2* __restrict__ data, int data_count,
                           KNNResult* __restrict__ result, int k)
{
    // Each warp processes one query.
    int warpInBlock = (threadIdx.x >> 5);  // threadIdx.x/32
    int lane       = threadIdx.x & 31;       // lane id within the warp
    int warpsPerBlock = blockDim.x >> 5;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpInBlock;
    if (globalWarpId >= query_count) return;

    // Each warp loads its query in a warp-synchronous manner.
    float2 q;
    if (lane == 0)
        q = query[globalWarpId];
    q.x = __shfl_sync(0xFFFFFFFF, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFF, q.y, 0);

    // Number of intermediate candidates per thread (k is power-of-two and divisible by 32).
    int numPerThread = k / 32;
    // Private intermediate result: each thread holds numPerThread candidates (stored in registers / local memory).
    int inter_idx[32];   // maximum possible numPerThread is MAX_K/32, which is <= 32 when k<=1024.
    float inter_dist[32];
#pragma unroll
    for (int i = 0; i < numPerThread; i++) {
        inter_idx[i] = -1;
        inter_dist[i] = FLT_MAX;
    }
    // Initially, max_distance is FLT_MAX.
    float max_distance = FLT_MAX;

    // Shared memory declarations.
    // s_data: shared memory buffer to cache a batch of data points.
    __shared__ float2 s_data[BATCH_SIZE];
    // s_candidateBuffer: per–warp candidate buffers, contiguous memory.
    __shared__ Candidate s_candidateBuffer[MAX_WARPS * MAX_K];
    // s_mergeBuffer: per–warp temporary merge workspace.
    __shared__ Candidate s_mergeBuffer[MAX_WARPS * (2 * MAX_K)];
    // s_candidateCount: per–warp candidate counter.
    __shared__ int s_candidateCount[MAX_WARPS];
    // Initialize candidate counter for the warp (only lane 0 of each warp does it).
    if (lane == 0)
        s_candidateCount[warpInBlock] = 0;
    __syncwarp();

    // Process 'data_count' data points in batches loaded into shared memory.
    for (int base = 0; base < data_count; base += BATCH_SIZE)
    {
        int batch_size = (base + BATCH_SIZE <= data_count) ? BATCH_SIZE : (data_count - base);
        // Each block loads the batch into shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            s_data[i] = data[base + i];
        }
        __syncthreads();

        // Each warp processes the loaded data points.
        for (int i = lane; i < batch_size; i += 32)
        {
            float2 d = s_data[i];
            float dx = q.x - d.x;
            float dy = q.y - d.y;
            float dist = dx * dx + dy * dy;
            if (dist < max_distance) {
                // Prepare candidate with global index.
                Candidate cand;
                cand.idx = base + i;
                cand.dist = dist;
                // Atomically append candidate to the warp's candidate buffer.
                int pos = atomicAdd(&s_candidateCount[warpInBlock], 1);
                int warpCandOffset = warpInBlock * MAX_K;
                if (pos < k) {
                    s_candidateBuffer[warpCandOffset + pos] = cand;
                } else {
                    // Buffer overflow; if this thread's atomicAdd returned exactly k, trigger merge.
                    if (pos == k) {
                        __syncwarp();
                        merge_intermediate(warpInBlock, k, numPerThread, inter_idx, inter_dist,
                                           max_distance, s_candidateBuffer, s_mergeBuffer, k);
                        // Reset candidate counter (only one thread resets it).
                        if (lane == 0)
                            s_candidateCount[warpInBlock] = 0;
                        __syncwarp();
                    }
                    // After merge the candidate buffer is empty; try to insert candidate again.
                    if (cand.dist < max_distance) {
                        pos = atomicAdd(&s_candidateCount[warpInBlock], 1);
                        if (pos < k) {
                            s_candidateBuffer[warpCandOffset + pos] = cand;
                        }
                    }
                }
            }
        }
        __syncthreads();
    } // end for each batch

    // After processing all batches, if any candidates remain, merge them.
    if (s_candidateCount[warpInBlock] > 0) {
        __syncwarp();
        merge_intermediate(warpInBlock, k, numPerThread, inter_idx, inter_dist,
                           max_distance, s_candidateBuffer, s_mergeBuffer, s_candidateCount[warpInBlock]);
        if (lane == 0)
            s_candidateCount[warpInBlock] = 0;
        __syncwarp();
    }

    // At this point, the private intermediate result holds the sorted k nearest neighbors.
    // Each warp writes its results (of total k candidates, distributed over 32 threads)
    // to the global result array.
    // Global result layout: for query i, result[i*k + j] is the j-th nearest neighbor.
    for (int i = 0; i < numPerThread; i++) {
        int globPos = (threadIdx.x & 31) + 32 * i;
        // Use a simple assignment to our KNNResult structure.
        // (A reinterpret_cast to std::pair<int,float> is assumed to be layout–compatible.)
        result[globalWarpId * k + globPos].first  = inter_idx[i];
        result[globalWarpId * k + globPos].second = inter_dist[i];
    }
}

//------------------------------------------------------------------------------
// Host interface: run_knn
// Launches the CUDA kernel to compute the k-nearest neighbors for all queries.
// Inputs:
//   query       - pointer to device array of float2 query points (length = query_count)
//   query_count - number of query points
//   data        - pointer to device array of float2 data points (length = data_count)
//   data_count  - number of data points
//   result      - pointer to device array where results will be stored; layout:
//                 for query[i], the j-th neighbor is stored at result[i*k + j] as a pair (index, distance)
//   k           - number of nearest neighbors to compute (power-of-two between 32 and 1024)
//------------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Set up grid dimensions so that each warp processes one query.
    int warpsPerBlock = BLOCK_SIZE >> 5; // BLOCK_SIZE/32
    /// @FIXED (-1:+0)
    /// int totalWarps   = (query_count + 0 - 1) / 1; // one warp per query.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Launch kernel.
    knn_kernel<<<blocks, BLOCK_SIZE>>>(query, query_count, data, data_count,
                                      reinterpret_cast<KNNResult*>(result), k);
    cudaDeviceSynchronize();
}

// End of source code.
