#include <cuda_runtime.h>
#include <utility>
#include <float.h>

// -----------------------------------------------------------------------------
// This implementation computes k-nearest neighbors for 2D points in Euclidean space.
// Each query is processed by one warp (32 threads). The warp maintains a private
// candidate list of k best (smallest squared distance) data points. The candidate
// list is distributed evenly among the 32 threads such that each thread holds
// candPerThread = k/32 candidates in registers, kept in sorted (ascending) order
// so that the worst candidate in a thread is at index (candPerThread-1).
//
// For each batch of data points (loaded into shared memory), every warp’s lanes
// compute distances from the query point and accumulate candidate updates in a
// local update buffer (capacity = UPDATE_CAP per lane). Then, in a warp‐synchronous
// loop, the candidate updates from each lane are merged into the global candidate
// list using warp-wide reduction (to find the current global threshold – the
// maximum among the worst candidates held by each lane) and by replacing that
// worst candidate (if the update improves it) via insertion into the sorted array.
//
// Finally, one lane (lane 0) gathers the entire candidate list from the warp,
// sorts it in ascending order and writes out the results.
//
// Hyper‐parameters:
//   THREADS_PER_BLOCK = 256
//   BATCH_SIZE = 256  (number of data points loaded per iteration into shared memory)
//   UPDATE_CAP = 8    (capacity for per-lane local update buffer)
// Note: k is a power of two between 32 and 1024, so that each warp’s candidate list
//       is evenly divisible among 32 threads.
// -----------------------------------------------------------------------------


// Structure to hold a candidate (data index and squared distance)
struct Candidate {
    int idx;
    float dist;
};

//--------------------------------------------------------------------------
// Device helper: Warp reduction to compute the maximum value (and corresponding lane)
// among a given value provided by each warp thread.
//
// "val" is the per-thread value, "lane" is the thread's lane id; after reduction,
// max_val will hold the maximum value found in the warp and max_lane the lane id that provided it.
__device__ void warpReduceMax(float val, int lane, float &max_val, int &max_lane) {
    // Use shuffle down for warp reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        int other_lane = __shfl_down_sync(0xffffffff, lane, offset);
        if (other > val) {
            val = other;
            lane = other_lane;
        }
    }
    max_val = val;
    max_lane = lane;
}

//--------------------------------------------------------------------------
// Device helper: Insert a candidate 'newcand' into a sorted (ascending) array of length 'len'.
// The array 'cand_d' holds distances and 'cand_idx' holds corresponding indices.
// It is assumed that newcand.dist is less than cand_d[len-1] (the worst candidate).
// The function shifts elements appropriately to preserve ascending order.
__device__ void localInsertCandidate(const Candidate newcand, int len, float *cand_d, int *cand_idx) {
    int insertPos = 0;
    // Find insertion position: first index where candidate distance is greater than newcand.dist.
    while(insertPos < len && cand_d[insertPos] <= newcand.dist)
        insertPos++;
    // Shift elements right from position (len-1) down to insertPos+1.
    for (int i = len-1; i > insertPos; i--) {
        cand_d[i] = cand_d[i-1];
        cand_idx[i] = cand_idx[i-1];
    }
    cand_d[insertPos] = newcand.dist;
    cand_idx[insertPos] = newcand.idx;
}

//--------------------------------------------------------------------------
// Device helper: Insert a candidate into a per-thread update buffer (sorted ascending).
// The update buffer has capacity 'cap'. 'upd_d' holds distances, 'upd_idx' holds indices,
// and 'updCount' (passed by reference) is the current number of entries.
__device__ void localInsertUpdate(const Candidate newcand, int cap, float *upd_d, int *upd_idx, int &updCount) {
    if (updCount < cap) {
        // Insert newcand into update buffer in sorted (ascending) order.
        int pos = 0;
        while (pos < updCount && upd_d[pos] <= newcand.dist)
            pos++;
        // Shift to right.
        for (int i = updCount; i > pos; i--) {
            upd_d[i] = upd_d[i-1];
            upd_idx[i] = upd_idx[i-1];
        }
        upd_d[pos] = newcand.dist;
        upd_idx[pos] = newcand.idx;
        updCount++;
    } else {
        // Buffer is full. Check if newcand improves the worst candidate (which is at index cap-1).
        if (newcand.dist < upd_d[cap-1]) {
            int pos = 0;
            while (pos < cap && upd_d[pos] <= newcand.dist)
                pos++;
            // Shift elements to right.
            for (int i = cap-1; i > pos; i--) {
                upd_d[i] = upd_d[i-1];
                upd_idx[i] = upd_idx[i-1];
            }
            upd_d[pos] = newcand.dist;
            upd_idx[pos] = newcand.idx;
        }
    }
}

//--------------------------------------------------------------------------
// Device helper: Warp-level candidate update. This function performs a warp-wide
// reduction to get the current global threshold (the maximum among the worst candidate
// values held by each thread in the warp) and, if newcand improves upon it, updates the
// candidate list stored (in registers) for the thread that holds that worst candidate.
// 'len' is the per-thread candidate list length.
// 'cand_d' and 'cand_idx' are pointers to the calling thread's candidate list arrays.
// 'lane' is the current warp lane id.
__device__ void warpCandidateUpdateDynamic(const Candidate newcand, int len, float *cand_d, int *cand_idx, int lane) {
    // Each thread's worst candidate is at index len-1 (since list is sorted ascending).
    float myWorst = cand_d[len - 1];
    float globalWorst = myWorst;
    int tgtLane = lane;
    // Warp reduction over the worst candidate values.
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, globalWorst, offset);
        int otherLane = __shfl_down_sync(0xffffffff, tgtLane, offset);
        if (other > globalWorst) {
            globalWorst = other;
            tgtLane = otherLane;
        }
    }
    // If new candidate improves the global worst, update the candidate list of the thread that holds it.
    if (newcand.dist < globalWorst) {
        if (lane == tgtLane) {
            localInsertCandidate(newcand, len, cand_d, cand_idx);
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel: Each warp processes one query point.
//  - query     : array of float2 query points
//  - queryCount: number of queries
//  - data      : array of float2 data points
//  - dataCount : number of data points
//  - result    : output array (row-major) where for each query, result[i*k + j]
//                is a std::pair<int,float> (data index and squared distance)
//  - k         : number of nearest neighbors (power of two between 32 and 1024)
// -----------------------------------------------------------------------------
/// @FIXED
/// extern "C" __global__
__global__
void knn_kernel(const float2 *query, int queryCount,
                const float2 *data, int dataCount,
                std::pair<int, float> *result, int k)
{
    // Hyper-parameters.
    const int THREADS_PER_BLOCK = 256;      // Fixed block size.
    const int BATCH_SIZE = 256;             // Number of data points loaded per batch.
    const int UPDATE_CAP = 8;               // Per-thread update buffer capacity.

    // Each warp (32 threads) processes one query.
    int warpId = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    if (warpId >= queryCount)
        return;

    int lane = threadIdx.x & 31; // lane id in the warp.
    // Load the query point for this warp.
    float2 q = query[warpId];

    // Determine per-thread candidate list length.
    // Since k is a power-of-two between 32 and 1024, each warp gets k candidates.
    // Distribute evenly among 32 threads:
    int candPerThread = k / 32;  // This value is one of: 1,2,4,8,16,32.
    // We'll use dynamic loops based on candPerThread; maximum candidate list size per thread is 32.
    // Allocate registers for candidate list.
    float localCand[32];
    int localCandIdx[32];
    // Initialize candidate list to "infinite" distance.
    for (int i = 0; i < candPerThread; i++) {
        localCand[i] = FLT_MAX;
        localCandIdx[i] = -1;
    }

    // Per-thread local update buffer for candidate update (from current batch).
    float updBuf[UPDATE_CAP];
    int updIdxBuf[UPDATE_CAP];
    int updCount = 0;

    // Shared memory for loading a batch of data points.
    extern __shared__ float2 sharedData[];
    // Process data points in batches.
    for (int base = 0; base < dataCount; base += BATCH_SIZE) {
        int batchSize = BATCH_SIZE;
        if (base + batchSize > dataCount)
            batchSize = dataCount - base;

        // Cooperatively load the current batch into shared memory.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            sharedData[i] = data[base + i];
        }
        __syncthreads();

        // Compute current global threshold: each thread’s worst candidate is at index (candPerThread-1).
        float myThresh = localCand[candPerThread - 1];
        float globalThresh = myThresh;
        int tmpLane = lane;
        warpReduceMax(myThresh, tmpLane, globalThresh, tmpLane);
        // globalThresh is now the worst (largest) distance among all candidate lists in this warp.

        // Each lane processes a subset of the batch (stride = warp size).
        for (int j = lane; j < batchSize; j += 32) {
            float2 pt = sharedData[j];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float dist = dx * dx + dy * dy;
            // If the computed distance is less than the current global threshold,
            // add this candidate to the local update buffer.
            if (dist < globalThresh) {
                Candidate cand;
                cand.dist = dist;
                cand.idx = base + j;  // global index in 'data'
                localInsertUpdate(cand, UPDATE_CAP, updBuf, updIdxBuf, updCount);
            }
        }
        __syncwarp();

        // Merge the update buffer into the global candidate list.
        // We process updates in warp-synchronous order: for each lane src (0 .. 31),
        // if that lane has pending update candidates, process them.
        for (int src = 0; src < 32; src++) {
            if (lane == src) {
                for (int r = 0; r < updCount; r++) {
                    Candidate updcand;
                    updcand.dist = updBuf[r];
                    updcand.idx = updIdxBuf[r];
                    warpCandidateUpdateDynamic(updcand, candPerThread, localCand, localCandIdx, lane);
                }
            }
            __syncwarp();
        }
        // Clear local update buffer for this batch.
        updCount = 0;
        __syncthreads();
    } // end for each batch

    // After processing all batches, the warp's candidate list (held in registers across 32 threads)
    // contains the k-nearest candidates (unsorted globally). Now, lane 0 will collect, sort, and write them.
    if (lane == 0) {
        // Maximum k is 1024.
        Candidate finalCandidates[1024];
        int count = 0;
        // Gather candidate lists from each warp lane.
        // Each lane holds candPerThread candidates.
        for (int src = 0; src < 32; src++) {
            for (int r = 0; r < candPerThread; r++) {
                // Use warp shuffle to get candidate data from lane 'src'.
                float candVal = __shfl_sync(0xffffffff, localCand[r], src);
                int candD = __shfl_sync(0xffffffff, localCandIdx[r], src);
                // Only add valid candidates (distance less than FLT_MAX).
                if (candVal < FLT_MAX) {
                    finalCandidates[count].dist = candVal;
                    finalCandidates[count].idx = candD;
                    count++;
                }
            }
        }

        // In the ideal case, count should equal k.
        // Sort finalCandidates in ascending order (by distance) using a simple insertion sort.
        for (int i = 1; i < count; i++) {
            Candidate key = finalCandidates[i];
            int j = i - 1;
            while (j >= 0 && finalCandidates[j].dist > key.dist) {
                finalCandidates[j+1] = finalCandidates[j];
                j--;
            }
            finalCandidates[j+1] = key;
        }
        // Write the sorted k-nearest neighbors to the result array.
        int outIdx = warpId * k;
        for (int i = 0; i < count; i++) {
            result[outIdx + i] = std::pair<int, float>(finalCandidates[i].idx, finalCandidates[i].dist);
        }
    }
}

// -----------------------------------------------------------------------------
// C++ host interface: Launch kernel.
// This function is assumed to be called with arrays allocated via cudaMalloc.
// It chooses grid and block dimensions based on the number of queries.
// -----------------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                          const float2 *data, int data_count,
                          std::pair<int, float> *result, int k)
{
    // Use 256 threads per block.
    const int THREADS_PER_BLOCK = 256;
    // Each warp processes 1 query; hence, queries per block = (THREADS_PER_BLOCK/32).
    int warpsPerBlock = THREADS_PER_BLOCK / 32;
    int numWarps = query_count;
    int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Allocate shared memory size: We use BATCH_SIZE data points per block.
    // BATCH_SIZE is fixed at 256 and each float2 is 8 bytes.
    size_t sharedMemSize = 256 * sizeof(float2);

    knn_kernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}
