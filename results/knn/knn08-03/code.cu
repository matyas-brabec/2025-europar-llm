#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <utility>

// This code implements a k-nearest neighbors (k-NN) search for 2D points on the GPU.
// Each query point is processed by one warp (32 threads). The k nearest neighbors are kept in an intermediate result
// that is distributed among the registers – each warp thread holds k/32 consecutive candidates. A candidate buffer
// (of k candidates) is allocated in shared memory per warp to temporarily store better candidates from the current batch.
// When the candidate buffer fills, it is merged with the intermediate result using a two‐step Bitonic Sort procedure.
// In the bitonic sort, we use a shared‐memory based implementation and intra‐warp synchronizations; when possible, warp
// shuffle instructions are used for early-outs in intra‐thread exchanges.
//
// This implementation is tuned for a modern NVIDIA GPU and compiled with the latest CUDA toolkit.


// -------------------------
// Data structure definitions
// -------------------------

// “Candidate” holds a data point index and its squared Euclidean distance from the query.
struct Candidate {
    int idx;
    float dist;
};

// Comparison: returns true if candidate a is “less” than candidate b (i.e. lower distance).
__device__ inline bool candidateLess(const Candidate &a, const Candidate &b) {
    return a.dist < b.dist;
}

// Swap two Candidate objects.
__device__ inline void swapCandidate(Candidate &a, Candidate &b) {
    Candidate tmp = a;
    a = b;
    b = tmp;
}

// -------------------------
// Bitonic Sort in shared memory for a warp’s candidate array.
// This sorts an array "data" of n elements (n is a power-of-two) in ascending order.
// All threads in the warp cooperate: each thread iterates over indices starting from its lane.
// Since the candidate arrays for merge are stored contiguously in shared memory, we can use this routine.
__device__ void bitonicSortShared(Candidate* data, int n)
{
    // n must be a power of two.
    // Use the warp’s lane id for partitioning.
    unsigned mask = 0xFFFFFFFF;
    int lane = threadIdx.x & 31;  // warp lane id

    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes a subset of indices.
            for (int i = lane; i < n; i += 32) {
                int j = i ^ stride;
                if (j > i) {
                    bool ascending = ((i & size) == 0);
                    Candidate ai = data[i];
                    Candidate aj = data[j];
                    // Swap if out-of-order.
                    if ((ascending && ai.dist > aj.dist) || (!ascending && ai.dist < aj.dist)) {
                        data[i] = aj;
                        data[j] = ai;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// -------------------------
// Merge routine for a warp:
// Merges the candidate buffer (in shared memory) with the warp’s intermediate result (in registers).
// The merge is done in two stages using Bitonic Sort as described:
//
//  0. Swap the content of the buffer and the intermediate result so that the candidate buffer is now in registers.
//  1. Sort the swapped-out content (old intermediate) using Bitonic Sort.
//  2. Merge the two sorted arrays by, for each global index i in [0,k), taking the minimum of buffer[i] and intermediate[k-i-1].
//  3. Sort the merged result using Bitonic Sort and store it back into the intermediate result registers.
//
// The distributed intermediate result is stored in registers "warpRes[]" (each thread holds m = k/32 candidates).
// The candidate buffer for the warp is in shared memory starting at warpBuff (of size k).
// A temporary shared memory segment "warpMerge" (of size k) is provided for the merge.
__device__ void mergeWarpCandidates(Candidate warpRes[], Candidate* warpBuff, Candidate* warpMerge, int k)
{
    // k is the number of candidates distributed across the warp.
    // m is the number of candidates per thread.
    int m = k / 32;  // since warp size is 32
    int lane = threadIdx.x & 31;
    
    // Step 0: Swap: copy the candidate buffer (warpBuff) into a temporary registers array newCand[].
    Candidate newCand[32];  // maximum m is 32 because k<=1024 => m<=32.
    // Each thread loads m consecutive candidates from warpBuff.
    for (int r = 0; r < m; r++) {
        int globalPos = lane * m + r;
        newCand[r] = warpBuff[globalPos];
    }
    // Save the old intermediate result from registers into a temporary array "oldRes[]" (in registers).
    Candidate oldRes[32];
    for (int r = 0; r < m; r++) {
        oldRes[r] = warpRes[r];
    }
    
    // Step 1: Sort the new candidate array (which came from candidate buffer) into ascending order.
    // To sort the entire warp’s new candidates (k elements), first store them into the shared merge buffer.
    // Each thread writes its m new candidates into warpMerge.
    for (int r = 0; r < m; r++) {
        int globalPos = lane * m + r;
        warpMerge[globalPos] = newCand[r];
    }
    __syncwarp();
    bitonicSortShared(warpMerge, k);
    __syncwarp();
    // Now, load the sorted new candidates back into registers (into newCand[]).
    for (int r = 0; r < m; r++) {
        int globalPos = lane * m + r;
        newCand[r] = warpMerge[globalPos];
    }
    
    // (oldRes is assumed to be already sorted in ascending order as invariant.)
    // Step 2: Merge: create a merged array "merged" of size k in the shared merge buffer.
    // For each global index i in [0, k), merged[i] = min(newSorted[i], oldRes[k - i - 1])
    // Since the arrays are distributed, we let each thread process m indices.
    for (int r = 0; r < m; r++) {
        int globalPos = lane * m + r;
        int partner = k - globalPos - 1;
        // Retrieve candidate from oldRes distributed registers.
        // Determine which thread holds the candidate for index "partner".
        int partnerLane = partner / m;
        int partnerIdx  = partner % m;
        // Use warp shuffle to get the candidate from the partner lane.
        Candidate partnerCand;
        // If partner is in the same lane, simply read from oldRes.
        if (partnerLane == lane) {
            partnerCand = oldRes[partnerIdx];
        } else {
            partnerCand.idx  = __shfl_sync(0xFFFFFFFF, oldRes[partnerIdx].idx, partnerLane);
            partnerCand.dist = __shfl_sync(0xFFFFFFFF, oldRes[partnerIdx].dist, partnerLane);
        }
        // Choose the minimum of newCand[globalPos] and partnerCand.
        Candidate mergedCand;
        if (newCand[r].dist < partnerCand.dist)
            mergedCand = newCand[r];
        else
            mergedCand = partnerCand;
        warpMerge[globalPos] = mergedCand;
    }
    __syncwarp();
    
    // Step 3: Sort the merged result using Bitonic Sort.
    bitonicSortShared(warpMerge, k);
    __syncwarp();
    
    // Step 4: Write the sorted merged result back into the intermediate result registers.
    for (int r = 0; r < m; r++) {
        int globalPos = lane * m + r;
        warpRes[r] = warpMerge[globalPos];
    }
    __syncwarp();
}

// -------------------------
// The k-NN kernel. 
// Each warp processes one query. Data points are processed in batches that are loaded in shared memory.
#define THREADS_PER_BLOCK 128
#define WARP_SIZE 32
#define BATCH_SIZE 256  // number of data points per batch

__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k)
{
    // Each warp (32 threads) processes one query.
    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    
    if (globalWarpId >= query_count) return;  // More warps than queries.
    
    // Declare the number of candidates per thread (k is assumed divisible by 32).
    const int m = k / WARP_SIZE;
    
    // Load the query point for this warp.
    float2 q = query[globalWarpId];
    
    // Initialize the intermediate result registers.
    // Each thread holds m candidates. Initially, all candidate distances are set to a very high value.
    Candidate warpRes[32];  // max m is 32.
    for (int r = 0; r < m; r++) {
        warpRes[r].idx = -1;
        warpRes[r].dist = CUDART_INF_F;
    }
    
    // Compute shared memory layout.
    // Shared memory is dynamically allocated; layout:
    //   [0, BATCH_SIZE) float2 data tile,
    //   [BATCH_SIZE, BATCH_SIZE + warpsPerBlock*k) Candidate warp candidate buffers,
    //   [BATCH_SIZE + warpsPerBlock*k, BATCH_SIZE + 2*warpsPerBlock*k) Candidate merge buffers,
    //   [BATCH_SIZE + 2*warpsPerBlock*k, ... ) int warp candidate counts.
    extern __shared__ char smem[];
    float2* dataTile = (float2*)smem;
    size_t offset = BATCH_SIZE * sizeof(float2);
    Candidate* warpCandBuffer = (Candidate*)(smem + offset);
    offset += warpsPerBlock * k * sizeof(Candidate);
    Candidate* warpMergeBuffer = (Candidate*)(smem + offset);
    offset += warpsPerBlock * k * sizeof(Candidate);
    int* warpCandCount = (int*)(smem + offset);
    
    // Pointers for this warp's candidate buffer and merge buffer.
    Candidate* myCandBuffer = warpCandBuffer + warpIdInBlock * k;
    Candidate* myMergeBuffer = warpMergeBuffer + warpIdInBlock * k;
    // Candidate count for this warp is stored in shared memory.
    if (lane == 0) {
        warpCandCount[warpIdInBlock] = 0;
    }
    __syncwarp();
    
    // Process data points in batches.
    for (int batch = 0; batch < data_count; batch += BATCH_SIZE) {
        int batchCount = (batch + BATCH_SIZE <= data_count) ? BATCH_SIZE : (data_count - batch);
        // Load batch data into shared memory.
        // Each thread in the block loads one or more data points.
        for (int i = threadIdx.x; i < batchCount; i += blockDim.x) {
            dataTile[i] = data[batch + i];
        }
        __syncthreads();
        
        // Each warp processes the batch loaded in dataTile.
        // Each thread in the warp iterates over some data points (using lane striding).
        for (int i = lane; i < batchCount; i += WARP_SIZE) {
            float2 d = dataTile[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            // Get the current "max" distance from the intermediate result (i.e. kth candidate).
            // The kth candidate (worst so far) is at global index k-1.
            int worstThread = (k - 1) / m;
            int worstIdx = (k - 1) % m;
            float currentMax;
            if (lane == worstThread)
                currentMax = warpRes[worstIdx].dist;
            currentMax = __shfl_sync(0xFFFFFFFF, currentMax, worstThread);
            // If the new distance is smaller than current maximum, consider it.
            if (dist < currentMax) {
                Candidate cand;
                cand.idx = batch + i; // global index of the data point
                cand.dist = dist;
                // Atomically append candidate to the warp's candidate buffer.
                int pos = atomicAdd(&warpCandCount[warpIdInBlock], 1);
                if (pos < k) {
                    myCandBuffer[pos] = cand;
                }
            }
        }
        __syncthreads();
        // If the candidate buffer is full, merge it with the intermediate result.
        if (warpCandCount[warpIdInBlock] >= k) {
            mergeWarpCandidates(warpRes, myCandBuffer, myMergeBuffer, k);
            if (lane == 0) {
                warpCandCount[warpIdInBlock] = 0;
            }
            __syncwarp();
        }
        __syncthreads();
    }
    
    // After processing all batches, if there are any leftover candidates in the candidate buffer, merge them.
    if (warpCandCount[warpIdInBlock] > 0) {
        // Fill the remainder of the candidate buffer with “dummy” candidates.
        int count = warpCandCount[warpIdInBlock];
        for (int pos = lane; pos < k; pos += WARP_SIZE) {
            if (pos >= count) {
                myCandBuffer[pos].idx = -1;
                myCandBuffer[pos].dist = CUDART_INF_F;
            }
        }
        __syncwarp();
        mergeWarpCandidates(warpRes, myCandBuffer, myMergeBuffer, k);
        if (lane == 0) {
            warpCandCount[warpIdInBlock] = 0;
        }
        __syncwarp();
    }
    
    // Write the final sorted intermediate result to global memory.
    // Each warp writes its k candidates (distributed among its 32 threads, each writing m candidates)
    for (int r = 0; r < m; r++) {
        int globalPos = globalWarpId * k + (lane * m + r);
        result[globalPos] = std::make_pair(warpRes[r].idx, warpRes[r].dist);
    }
}

// -------------------------
// Host interface: run_knn
// This function launches the knn_kernel. It assumes that query, data, and result have been allocated on the GPU.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Choose block size and grid size.
    // Each warp processes one query. With THREADS_PER_BLOCK threads per block, each block processes (THREADS_PER_BLOCK / 32) queries.
    int warpsPerBlock = THREADS_PER_BLOCK / WARP_SIZE;
    int gridSize = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    
    // Calculate shared memory size needed per block.
    // Shared memory layout:
    //   data tile: BATCH_SIZE * sizeof(float2)
    //   warp candidate buffers: warpsPerBlock * k * sizeof(Candidate)
    //   warp merge buffers:   warpsPerBlock * k * sizeof(Candidate)
    //   warp candidate counts: warpsPerBlock * sizeof(int)
    size_t shmem_size = BATCH_SIZE * sizeof(float2)
                      + warpsPerBlock * k * sizeof(Candidate)
                      + warpsPerBlock * k * sizeof(Candidate)
                      + warpsPerBlock * sizeof(int);
    
    knn_kernel<<<gridSize, THREADS_PER_BLOCK, shmem_size>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}