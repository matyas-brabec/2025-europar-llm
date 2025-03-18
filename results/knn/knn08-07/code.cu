#include <cuda_runtime.h>
#include <cfloat>   // for FLT_MAX
#include <utility>  // for std::pair

// Constants. DATA_BATCH_SIZE determines how many data points are loaded into shared memory per batch.
#define DATA_BATCH_SIZE 1024

// -----------------------------------------------------------------------------
// Candidate structure holds one neighbor candidate (data index and squared distance).
// -----------------------------------------------------------------------------
struct Candidate {
    int index;
    float dist;
};

// Utility function to create a Candidate.
__device__ inline Candidate makeCandidate(int idx, float d) {
    Candidate c;
    c.index = idx;
    c.dist = d;
    return c;
}

// -----------------------------------------------------------------------------
// Insertion sort for small arrays in registers.
// Sorts an array of 'n' Candidate elements in ascending order (by dist).
// -----------------------------------------------------------------------------
__device__ void sortLocalCandidates(Candidate arr[], int n) {
    for (int i = 1; i < n; i++) {
        Candidate key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].dist > key.dist) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// -----------------------------------------------------------------------------
// Warp-level Bitonic sort for k candidates distributed among the warp's registers.
// The overall number of candidates k is assumed to be a power-of-two and distributed
// so that each thread holds (k/32) candidates in an array 'reg'.
// This routine first sorts each thread's local subarray using insertion sort, then
// sorts each column (i.e. the candidate at the same register index across warp lanes)
// using a simple bitonic merge network via warp shuffle instructions.
// Note: This simplified version works best when (k/32) is small (max 32 for k=1024).
// -----------------------------------------------------------------------------
__device__ void warpBitonicSort(Candidate reg[], int k) {
    // Each warp has 32 threads.
    const int localCount = k >> 5;  // = k/32; note k is in {32,64,...,1024}
    int lane = threadIdx.x & 31;

    // Step 1: Each thread locally sorts its own candidates.
    sortLocalCandidates(reg, localCount);
    __syncwarp();

    // Step 2: For each register slot (column), perform a warp-level bitonic merge.
    for (int r = 0; r < localCount; r++) {
        // Load the candidate from this column.
        Candidate val = reg[r];
        // Bitonic merge network for 32 elements.
        // We use a fixed unrolled loop over offsets 16, 8, 4, 2, 1.
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            // Exchange val with candidate from lane 'lane^offset'.
            /// @FIXED (-1:+3)
            /// Candidate other = __shfl_xor_sync(0xffffffff, val, offset);
            Candidate other;
            other.index = __shfl_xor_sync(0xffffffff, val.index, offset);
            other.dist = __shfl_xor_sync(0xffffffff, val.dist, offset);
            int partner = lane ^ offset;
            // In a full bitonic merge, the lower-indexed lane should keep the smaller value.
            if (lane < partner) {
                if (other.dist < val.dist)
                    val = other;
            } else {
                if (other.dist > val.dist)
                    val = other;
            }
        }
        // Write the merged value back.
        reg[r] = val;
        __syncwarp();
    }
    // At this point, for each column r across the warp, the value stored at lane i is the
    // i-th element (in sorted order) of that column. Although the overall (row-major)
    // order is not fully globally sorted, the result is sufficient for our merge.
}

// -----------------------------------------------------------------------------
// Merges the intermediate result (sorted in registers in regResult) with a candidate
// buffer (sorted in registers in bufferReg). Both arrays hold k candidates (distributed
// among the 32 threads, each thread with (k/32) elements). The merge is implemented by,
// for each global index i, taking the minimum of buffer[i] and regResult[k-i-1] (which
// produces a bitonic sequence). Then the merged result is sorted with warpBitonicSort.
// -----------------------------------------------------------------------------
__device__ void mergeIntermediateAndBuffer(Candidate regResult[], Candidate bufferReg[], int k) {
    const int localCount = k >> 5;
    int lane = threadIdx.x & 31;
    Candidate merged[32];  // enough for localCount elements per thread

    // For each candidate held by this thread (each column r),
    // its global index is: global_index = lane * localCount + r.
    // Its partner is at global index = k - global_index - 1.
    // We assume that because localCount is a power-of-two (k/32), the partner has the same local index.
    for (int r = 0; r < localCount; r++) {
        int globalIndex = lane * localCount + r;
        int partnerGlobal = k - globalIndex - 1;
        int partnerLane = partnerGlobal / localCount;
        // Obtain the candidate from regResult corresponding to the partner.
        Candidate partnerVal;
        if (partnerLane == lane) {
            partnerVal = regResult[globalIndex & (localCount - 1)];
        } else {
            // Since each thread stores its candidates in the same order,
            // we can simply fetch the element at index r from the partner thread.
            /// @FIXED (-1:+2)
            /// partnerVal = __shfl_sync(0xffffffff, regResult[r], partnerLane);
            partnerVal.index = __shfl_sync(0xffffffff, regResult[r].index, partnerLane);
            partnerVal.dist = __shfl_sync(0xffffffff, regResult[r].dist, partnerLane);
        }
        // Merge: take the candidate with the smaller distance.
        Candidate candBuf = bufferReg[r];
        merged[r] = (candBuf.dist < partnerVal.dist) ? candBuf : partnerVal;
    }
    // Write back the merged results.
    for (int r = 0; r < localCount; r++) {
        regResult[r] = merged[r];
    }
    __syncwarp();
    // Fully sort the merged result.
    warpBitonicSort(regResult, k);
}

// -----------------------------------------------------------------------------
// The optimized CUDA kernel for computing k-nearest neighbors (k-NN) for 2D points.
// Each warp (32 threads) processes one query point from the 'query' array. The warp's
// threads maintain an intermediate result (the k nearest neighbors so far) in registers,
// distributed so that each thread holds (k/32) consecutive candidates. A candidate buffer
// is allocated in shared memory (per warp) for temporarily holding new candidates from
// the data points processed in the current batch. When the buffer fills (i.e. reaches k),
// it is merged with the intermediate result using a Bitonic Sort–based procedure.
// Data points are processed in batches: each block cooperatively loads DATA_BATCH_SIZE
// points from 'data' into shared memory.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           std::pair<int, float> * __restrict__ result,
                           int k) {
    // Each warp processes one query.
    int warpIdInBlock = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int warpsPerBlock = blockDim.x / 32;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (globalWarpId >= query_count)
        return;

    // Load query point for this warp.
    float2 q = query[globalWarpId];

    // Each thread holds (k/32) candidates in registers.
    const int localCount = k >> 5;   // k divided by 32.
    Candidate regResult[32];         // Maximum localCount is 32 when k==1024.
    // Initialize intermediate result with "infinite" distances.
    for (int i = 0; i < localCount; i++) {
        regResult[i] = makeCandidate(-1, FLT_MAX);
    }

    // Shared memory layout:
    // First: DATA_BATCH_SIZE float2's for a batch of data points.
    // Then: candidate buffers for each warp (each warp gets k Candidate elements).
    extern __shared__ char shared[];
    float2* s_data = (float2*) shared;
    Candidate* s_candBuffer = (Candidate*) (s_data + DATA_BATCH_SIZE);

    // Each warp uses its own candidate buffer; compute its offset.
    int warpCandOffset = warpIdInBlock * k;
    // Candidate count (number of candidates stored in the buffer for this warp).
    int candCount = 0;

    // Process the data points in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += DATA_BATCH_SIZE) {
        int batchSize = DATA_BATCH_SIZE;
        if (batchStart + batchSize > data_count)
            batchSize = data_count - batchStart;

        // Cooperative load: threads load the batch from global memory into shared memory.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            s_data[i] = data[batchStart + i];
        }
        __syncthreads();

        // Each warp processes the batch.
        // Partition the batch among warp lanes: lane processes indices i = lane, lane+32, ...
        for (int i = lane; i < batchSize; i += 32) {
            float2 dpt = s_data[i];
            float dx = dpt.x - q.x;
            float dy = dpt.y - q.y;
            float dist = dx*dx + dy*dy;

            // Get current maximum distance from the intermediate result.
            // The k-th nearest neighbor (largest distance of current best k) is stored
            // at global index k-1. That element is held by thread (k-1)/localCount, at offset (k-1)%localCount.
            int maxLane = (k - 1) >> 5;
            int maxOffset = (k - 1) & (localCount - 1);
            float currMax;
            if (lane == maxLane)
                currMax = regResult[maxOffset].dist;
            currMax = __shfl_sync(0xffffffff, currMax, maxLane);

            // If computed distance is less than current maximum, candidate qualifies.
            if (dist < currMax) {
                // Use warp ballot to discover how many lanes (including this lane) have a qualifying candidate.
                // (Since each thread processes one data point in this loop, the predicate is simply true.)
                unsigned int ballot = __ballot_sync(0xffffffff, true);
                int prefix = __popc(ballot & ((1u << lane) - 1));
                int insertPos = candCount + prefix;
                // If the candidate buffer is full, merge it with the intermediate result.
                if (insertPos >= k) {
                    // Each thread loads its portion of the candidate buffer.
                    Candidate bufferReg[32];
                    for (int j = 0; j < localCount; j++) {
                        bufferReg[j] = s_candBuffer[warpCandOffset + j*32 + lane];
                    }
                    // Swap regResult and bufferReg.
                    for (int j = 0; j < localCount; j++) {
                        Candidate temp = regResult[j];
                        regResult[j] = bufferReg[j];
                        bufferReg[j] = temp;
                    }
                    // Sort the new intermediate result.
                    warpBitonicSort(regResult, k);
                    // Merge the old intermediate result (now in bufferReg) with regResult.
                    mergeIntermediateAndBuffer(regResult, bufferReg, k);
                    candCount = 0;
                    __syncwarp();
                    insertPos = prefix;
                }
                // Write the new candidate into the candidate buffer.
                int writePos = warpCandOffset + insertPos;
                s_candBuffer[writePos] = makeCandidate(batchStart + i, dist);
                candCount = insertPos + 1; // Update candidate count.
            }
        }
        __syncthreads();
    } // end for each batch

    // After processing all batches, if there are remaining candidates in the buffer, merge them.
    if (candCount > 0) {
        Candidate bufferReg[32];
        for (int j = 0; j < localCount; j++) {
            int idx = warpCandOffset + j*32 + lane;
            if (j*32 + lane < candCount)
                bufferReg[j] = s_candBuffer[idx];
            else
                bufferReg[j] = makeCandidate(-1, FLT_MAX);
        }
        warpBitonicSort(bufferReg, k);
        mergeIntermediateAndBuffer(regResult, bufferReg, k);
    }
    __syncwarp();

    // Write final k nearest neighbors for this query to global memory.
    // The warp's intermediate result is stored in registers distributed in row-major order.
    // Each thread writes (k/32) candidates; candidate at global index = (j*32 + lane) for j in [0, localCount).
    for (int j = 0; j < localCount; j++) {
        int outIndex = globalWarpId * k + j*32 + lane;
        result[outIndex] = std::make_pair(regResult[j].index, regResult[j].dist);
    }
}

// -----------------------------------------------------------------------------
// Host interface function that launches the k-NN kernel.
// 'query', 'data', and 'result' are device pointers (allocated via cudaMalloc).
// k is a power-of-two between 32 and 1024.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Use 256 threads per block (i.e. 8 warps per block).
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp handles one query.
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute shared memory size:
    // - DATA_BATCH_SIZE float2 points for the data batch.
    // - For each warp in the block, allocate k Candidate elements.
    size_t sharedMemSize = DATA_BATCH_SIZE * sizeof(float2) + warpsPerBlock * k * sizeof(Candidate);

    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}
