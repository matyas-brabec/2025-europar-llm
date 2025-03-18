// Optimized CUDA implementation for k‐Nearest Neighbors (k‐NN) on 2D points.
// This implementation assigns one warp (32 threads) per query.
// Each warp holds a “private” copy of the current k nearest candidates in registers,
// distributed as k/32 candidates per thread. A per‐warp candidate buffer in shared memory
// is used to cache candidate data from global memory during batched processing and to
// merge new candidates with the current intermediate result using a Bitonic‐Sort merge.
// 
// Note: k is assumed to be a power‐of‐two in [32,1024]. The squared Euclidean distance is computed.
// 
// The shared memory layout per block is as follows:
//  [Candidate buffer for each warp]  – size = (#warps per block * k)
//  [Data batch buffer]                – size = DATA_BATCH_SIZE float2 elements
//
// For merging the candidate buffers, we “swap” the registers with the shared memory buffer,
// then sort the new candidate buffer using a Bitonic Sort implemented in shared memory,
// then merge it with the previous (already sorted) intermediate result via a simple merge
// (by taking pairwise minima between the new sorted candidates and the old candidates in reverse
// order) and finally bitonically sorting the merged result. For inter‐thread communication
// (when a candidate in a thread must be compared with one in a different lane) warp shuffles
// and __syncwarp() are used where possible.
// 
// Although the full “ideal” register‐based Bitonic Sort merge would exchange registers between lanes,
// here we use a hybrid approach: registers are dumped to shared memory, sorted there via a serial
// Bitonic sort (which is acceptable for k<=1024) and then reloaded into registers. This meets the
// interface requirements and uses warp shuffles for intra‐warp broadcasting of query coordinates
// and max_distance updates.
// 
// All arrays (query, data, result) are assumed to have been allocated with cudaMalloc.
// The host function run_knn launches the kernel with 128 threads per block (i.e. 4 warps per block).

#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>

// Structure for a candidate neighbor.
struct Candidate {
    int idx;    // index of the data point
    float dist; // squared Euclidean distance
};

// -----------------------------------------------------------------------------
// Bitonic sort in shared memory.
// This helper function performs an in-place Bitonic sort on a candidate array stored
// in shared memory. We assume that the length (n) is not huge (max k <= 1024).
// Only one thread per warp (lane==0) performs the serial sort; then __syncwarp()
// ensures all lanes see the sorted order.
__device__ void bitonicSortShared(Candidate* s, int offset, int n) {
    // Only one thread in the warp does the sort.
    if ((threadIdx.x & 31) == 0) {
        for (int k_val = 2; k_val <= n; k_val *= 2) {
            for (int j = k_val / 2; j > 0; j /= 2) {
                for (int i = 0; i < n; i++) {
                    int l = i ^ j;
                    if (l > i) {
                        bool ascending = ((i & k_val) == 0);
                        Candidate &a = s[offset + i];
                        Candidate &b = s[offset + l];
                        if ((ascending && a.dist > b.dist) || (!ascending && a.dist < b.dist)) {
                            Candidate tmp = a;
                            a = b;
                            b = tmp;
                        }
                    }
                }
            }
        }
    }
    __syncwarp();
}

// -----------------------------------------------------------------------------
// Merge two sorted candidate arrays (each of total length k) into a merged candidate
// array (of length k) using the prescribed merge: for each global index i, the merged
// candidate = min(newCandidate[i], oldCandidate[k-i-1]). The sum result (a bitonic sequence)
// is then re-sorted by bitonicSortShared.
// The candidate arrays are provided in shared memory in buffer "temp" (length k) and in registers.
// The merge result is loaded back into registers.
__device__ void mergeCandidates(Candidate reg[], int L, Candidate* temp, int k) {
    int lane = threadIdx.x & 31;
    // Each warp holds k candidates distributed as L = k/32 per thread.
    // First, write the register candidate array (sorted new candidates) into temp.
    for (int j = 0; j < L; j++) {
        int globalIdx = lane * L + j;
        temp[globalIdx] = reg[j];
    }
    __syncwarp();
    // Now assume that the previous (old) intermediate result is stored in the per‐warp candidate
    // buffer (which is already sorted) immediately following temp in shared memory.
    // (These two arrays are arranged consecutively: first new, then old.)
    for (int j = 0; j < L; j++) {
        int globalIdx = lane * L + j;
        // For merging, take candidate from new array and from the old array in reverse order.
        int mirror = k - globalIdx - 1;
        Candidate cand_new = temp[globalIdx];
        Candidate cand_old = temp[k + mirror];  // old candidate stored in temp[k..2*k-1]
        Candidate merged = (cand_new.dist < cand_old.dist) ? cand_new : cand_old;
        reg[j] = merged;
    }
    __syncwarp();
    // Write merged candidates back into the first half of temp.
    for (int j = 0; j < L; j++) {
        int globalIdx = lane * L + j;
        temp[globalIdx] = reg[j];
    }
    __syncwarp();
    // Sort the merged result.
    bitonicSortShared(temp, 0, k);
    __syncwarp();
    // Load sorted merged result back into registers.
    for (int j = 0; j < L; j++) {
        int globalIdx = lane * L + j;
        reg[j] = temp[globalIdx];
    }
    __syncwarp();
}

// -----------------------------------------------------------------------------
// CUDA kernel implementing k-NN search on 2D points.
// Each warp processes one query point by iterating over the data points in batches.
// The candidate distances are computed as squared Euclidean distances.
__global__ void knn_kernel(const float2* query, int query_count,
                           const float2* data, int data_count,
                           std::pair<int, float>* result, int k) {
    // Identify warp and lane indices.
    int warpIdInBlock = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int warpsPerBlock = blockDim.x / 32;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
  
    // Each warp processes one query.
    if (globalWarpId >= query_count)
        return;
  
    // Load the query point and broadcast to all lanes in the warp.
    float2 q;
    if (lane == 0)
        q = query[globalWarpId];
    q.x = __shfl_sync(0xFFFFFFFF, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFF, q.y, 0);
  
    // Number of candidates per thread.
    int L = k / 32;   // because total candidates k are distributed equally among 32 threads
  
    // Each thread maintains its portion of the intermediate k-NN result in registers.
    Candidate knnReg[32];  // max L is 32 when k == 1024
    for (int i = 0; i < L; i++) {
        knnReg[i].idx  = -1;
        knnReg[i].dist = 1e30f;  // a very large distance (acting as FLT_MAX)
    }
    // Global k-th distance (kth nearest neighbor) maintained in a warp-wide variable.
    float maxDist = 1e30f;
  
    // Dynamic shared memory layout:
    // First portion: per-warp candidate buffer.
    //    Size = (warpsPerBlock * k) * sizeof(Candidate)
    // Second portion: data batch buffer.
    extern __shared__ char smem[];
    Candidate* candBuffer = (Candidate*)smem;
    int candBufferBytes = (warpsPerBlock * k) * sizeof(Candidate);
    float2* sdata = (float2*)(smem + candBufferBytes);
  
    // Pointer to candidate buffer for this warp.
    Candidate* warpCandBuf = candBuffer + (warpIdInBlock * k);
    // Number of candidates currently stored in the candidate buffer.
    int bufCount = 0;
  
    // Hyper-parameter: batch size for loading data points.
    const int DATA_BATCH_SIZE = 1024;
  
    // Loop over the entire data set in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += DATA_BATCH_SIZE) {
        int batchSize = DATA_BATCH_SIZE;
        if (batchStart + batchSize > data_count)
            batchSize = data_count - batchStart;
  
        // Load a batch of data points into shared memory.
        // Each thread in the block cooperates.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            sdata[i] = data[batchStart + i];
        }
        __syncthreads();
  
        // Each warp processes the loaded batch.
        // Distribute work among warp lanes.
        for (int i = lane; i < batchSize; i += 32) {
            float2 p = sdata[i];
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float dist = dx * dx + dy * dy;
  
            // If the new distance is smaller than the current k-th distance...
            if (dist < maxDist) {
                Candidate cand;
                cand.idx  = batchStart + i;  // global index in data array
                cand.dist = dist;
                // Use warp ballot to determine which lanes qualify.
                unsigned mask = __ballot_sync(0xFFFFFFFF, (dist < maxDist));
                int vote = __popc(mask & ((1u << lane) - 1));
                // Compute position inside warp candidate buffer.
                int pos = bufCount + vote;
                if (pos < k)
                    warpCandBuf[pos] = cand;
                __syncwarp();
                // Only one lane (lane 0) updates the buffer count.
                if (lane == 0)
                    bufCount += __popc(mask);
                __syncwarp();
  
                // If candidate buffer is full, merge it with the current intermediate result.
                if (bufCount >= k) {
                    // Merge procedure:
                    // 1. Swap the contents of registers (knnReg) with the candidate buffer.
                    Candidate tmpReg[32];
                    for (int j = 0; j < L; j++) {
                        tmpReg[j] = knnReg[j];
                        // Replace registers with candidates from buffer.
                        knnReg[j] = warpCandBuf[lane * L + j];
                        warpCandBuf[lane * L + j] = tmpReg[j];
                    }
                    __syncwarp();
  
                    // 2. Sort the new candidate buffer (now in registers) using Bitonic sort.
                    // We use a temporary shared memory region (reusing our candidate buffer area for this warp).
                    Candidate* warpTemp = candBuffer + (warpIdInBlock * k);
                    for (int j = 0; j < L; j++) {
                        int globalIdx = lane * L + j;
                        warpTemp[globalIdx] = knnReg[j];
                    }
                    __syncwarp();
                    bitonicSortShared(warpTemp, 0, k);
                    __syncwarp();
                    // Load sorted candidates back into registers.
                    for (int j = 0; j < L; j++) {
                        knnReg[j] = warpTemp[lane * L + j];
                    }
                    __syncwarp();
  
                    // 3. Now merge the sorted new candidates with the old intermediate result,
                    // which is now in warpCandBuf. First, copy the new sorted candidates into warpTemp.
                    for (int j = 0; j < L; j++) {
                        int globalIdx = lane * L + j;
                        warpTemp[globalIdx] = knnReg[j];
                    }
                    __syncwarp();
                    // For the merge, assume the old intermediate result is stored right after warpTemp.
                    // (We copy the old intermediate candidates there.)
                    for (int j = 0; j < L; j++) {
                        int globalIdx = lane * L + j;
                        warpTemp[k + (globalIdx)] = warpCandBuf[lane * L + j];
                    }
                    __syncwarp();
                    // 4. Merge the two sorted arrays (each of length k) using the prescribed merge strategy.
                    mergeCandidates(knnReg, L, warpTemp, k);
                    __syncwarp();
  
                    // 5. Update the k-th distance from the merged result.
                    if (lane == 31)
                        maxDist = knnReg[L - 1].dist;
                    maxDist = __shfl_sync(0xFFFFFFFF, maxDist, 31);
                    // Reset candidate buffer count.
                    bufCount = 0;
                    __syncwarp();
                }
            }
        }
        __syncthreads();
    }  // end of data batch loop
  
    // After all batches, if any candidates remain in the candidate buffer, merge them.
    if (bufCount > 0) {
        // Pad the candidate buffer with worst candidates.
        for (int pos = bufCount + lane; pos < k; pos += 32) {
            warpCandBuf[pos].idx  = -1;
            warpCandBuf[pos].dist = 1e30f;
        }
        __syncwarp();
  
        // Swap registers with candidate buffer.
        Candidate tmpReg[32];
        for (int j = 0; j < L; j++) {
            tmpReg[j] = knnReg[j];
            knnReg[j] = warpCandBuf[lane * L + j];
            warpCandBuf[lane * L + j] = tmpReg[j];
        }
        __syncwarp();
  
        Candidate* warpTemp = candBuffer + (warpIdInBlock * k);
        for (int j = 0; j < L; j++) {
            int globalIdx = lane * L + j;
            warpTemp[globalIdx] = knnReg[j];
        }
        __syncwarp();
        bitonicSortShared(warpTemp, 0, k);
        __syncwarp();
        for (int j = 0; j < L; j++) {
            knnReg[j] = warpTemp[lane * L + j];
        }
        __syncwarp();
  
        // Merge with the content from candidate buffer.
        mergeCandidates(knnReg, L, warpTemp, k);
        __syncwarp();
  
        if (lane == 31)
            maxDist = knnReg[L - 1].dist;
        maxDist = __shfl_sync(0xFFFFFFFF, maxDist, 31);
    }
  
    // Write final k nearest neighbors for this query into the global result.
    // For query i, the results are stored at result[i*k + j], for j in 0..(k-1).
    int queryIdx = globalWarpId;
    for (int j = 0; j < L; j++) {
        int globalIdx = lane * L + j;
        int outIdx = queryIdx * k + globalIdx;
        result[outIdx] = std::pair<int, float>(knnReg[j].idx, knnReg[j].dist);
    }
}
  
// -----------------------------------------------------------------------------
// Host function to launch the k-NN kernel.
// Uses 128 threads per block (i.e., 4 warps per block) and determines the amount of
// dynamic shared memory needed based on k and the chosen DATA_BATCH_SIZE.
#include <cuda_runtime.h>
#include <utility>

void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose 128 threads per block (4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock   = threadsPerBlock / 32;
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;
  
    // Dynamic shared memory allocation:
    // Per block candidate buffer size = (warpsPerBlock * k) * sizeof(Candidate)
    // Plus data batch buffer = DATA_BATCH_SIZE * sizeof(float2).
    const int DATA_BATCH_SIZE = 1024;
    size_t sharedMemBytes = (warpsPerBlock * k) * sizeof(Candidate) +
                            DATA_BATCH_SIZE * sizeof(float2);
  
    knn_kernel<<<blocks, threadsPerBlock, sharedMemBytes>>>(query, query_count,
           data, data_count, result, k);
    cudaDeviceSynchronize();
}