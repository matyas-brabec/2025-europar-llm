#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Candidate structure to hold an index and its associated squared distance.
struct Candidate {
    int idx;
    float dist;
};

// ***********************************************************************
// Device function: warp-level maximum reduction for a float value.
// Uses full-warp mask (assumes warp size 32).
// ***********************************************************************
__device__ inline float warpMax(float val) {
    // Use warp shuffle down to compute maximum in the warp.
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) val = other;
    }
    return val;
}

// ***********************************************************************
// Device function: merge a lane’s private result with the candidate 
// buffer portion in shared memory.  Each warp thread holds a sorted 
// array 'priv' of length 'r' (its private candidates, in ascending order)
// and will merge it with the candidates in the candidate buffer that 
// belong to that thread.  After merging, the thread keeps the smallest 'r'
// elements (still sorted in ascending order).
// Parameters:
//    priv         : pointer to the private candidate array (length = r)
//    buffer       : pointer to the candidate buffer for the warp 
//                   (size may be < or == k)
//    bufCount     : total number of candidates stored in the warp’s candidate buffer
//    r            : number of private candidates per thread (= k/32)
//    lane         : the thread’s lane id within its warp (0..31)
// ***********************************************************************
__device__ void mergePrivateAndBuffer(Candidate* priv, const Candidate* buffer, int bufCount, int r, int lane) {
    // Each warp thread extracts its assigned candidates from the shared candidate buffer.
    // The candidates in the warp buffer are stored contiguously; thread 'lane' gets all indices:
    // lane, lane+32, lane+64, ... up to bufCount.
    Candidate localCand[32]; // Maximum possible candidates per lane; k is at most 1024 so (1024/32)==32.
    int localCount = 0;
    for (int i = lane; i < bufCount; i += 32) {
        localCand[localCount++] = buffer[i];
    }
    // Simple insertion sort on the local candidate array (in ascending order by distance).
    for (int i = 1; i < localCount; i++) {
        Candidate key = localCand[i];
        int j = i - 1;
        while (j >= 0 && localCand[j].dist > key.dist) {
            localCand[j+1] = localCand[j];
            j--;
        }
        localCand[j+1] = key;
    }
    // Merge the already sorted private array 'priv' (of length r) with the sorted local candidate array.
    // We want to keep the r smallest elements overall.
    Candidate merged[64]; // Maximum size = r + localCount, with r<=32 and localCount<=32.
    int i = 0, j = 0, m = 0;
    while (m < r && (i < r || j < localCount)) {
        // Take candidate from priv if available and either localCand is exhausted or priv is smaller.
        if (i < r && (j >= localCount || priv[i].dist <= localCand[j].dist)) {
            merged[m++] = priv[i++];
        } else {
            merged[m++] = localCand[j++];
        }
    }
    // Write the merged r best candidates back to the private array.
    for (int k_i = 0; k_i < r; k_i++) {
        priv[k_i] = merged[k_i];
    }
}

// ***********************************************************************
// Kernel: each warp processes one query point to compute its k nearest neighbors.
// The algorithm processes the data in batches. A block–wide shared memory buffer
// caches a batch of data points. Each warp maintains two buffers: a private intermediate
// result (stored in registers, distributed among the warp lanes) and a candidate buffer 
// (in shared memory) for potential neighbor candidates from the current batch.
// When the candidate buffer is full, it is merged into the private result.
// After processing all batches, the private result is merged across the warp to produce 
// a sorted list of k nearest neighbors, which is then written to global memory.
// ***********************************************************************
#define BATCH_SIZE 1024  // Number of data points loaded per batch (tunable)

__global__ void knn_kernel(const float2 *query, int query_count, const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp processes one query.
    int lane      = threadIdx.x % 32;             // Lane id within warp.
    int warpId    = threadIdx.x / 32;             // Warp index within block.
    int warpsPerBlock = blockDim.x / 32;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpId;
    if (globalWarpId >= query_count) return;  // Guard: ensure valid query.

    // Load the query point. (Could use warp shuffle to broadcast but all lanes load same memory.)
    float2 q = query[globalWarpId];

    // Each warp will maintain an intermediate result (k candidates) distributed across its 32 threads.
    // Each thread holds r = k/32 candidates in a private array (kept sorted in ascending order, i.e. smallest first).
    int r = k / 32;  // Guaranteed to be an integer since k is a power of two (>=32)
    Candidate priv[32];  // maximum size is 32 (r <= 32 when k<=1024)
    // Initialize private result with "infinite" distances.
    for (int i = 0; i < r; i++) {
        priv[i].idx = -1;
        priv[i].dist = FLT_MAX;
    }

    // Define shared memory layout.
    // Shared memory is allocated dynamically; see host launch configuration below.
    extern __shared__ char shared_mem[];
    // sData: batch of data points.
    float2* sData = (float2*)shared_mem;
    // warpBufCount: per-warp candidate-buffer counters.
    int* warpBufCount = (int*)(sData + BATCH_SIZE);
    // sCandBuffer: candidate buffers for each warp, one contiguous block of size (k) per warp.
    Candidate* sCandBuffer = (Candidate*)(warpBufCount + warpsPerBlock);

    // Initialize candidate buffer counter for this warp (one thread does it, then broadcast).
    if (lane == 0) {
        warpBufCount[warpId] = 0;
    }
    __syncwarp();

    // Process the data points in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += BATCH_SIZE) {
        // Determine the number of points in this batch.
        int batchLen = BATCH_SIZE;
        if (batchStart + BATCH_SIZE > data_count)
            batchLen = data_count - batchStart;

        // Cooperative loading of data points into shared memory.
        // All threads in the block load points (each thread loads multiple points if needed).
        for (int i = threadIdx.x; i < batchLen; i += blockDim.x) {
            sData[i] = data[batchStart + i];
        }
        __syncthreads();  // Ensure batch is loaded.

        // Before processing the batch, get the current threshold.
        // Each thread has its private worst (largest) candidate: since priv[] is sorted in ascending order,
        // the worst candidate is at index r-1.
        float localWorst = priv[r - 1].dist;
        float currThresh = warpMax(localWorst);  // Global threshold for the warp.

        // Each warp processes the batch from shared memory.
        // The 32 threads in the warp partition the batch points by taking indices starting at lane and stepping by 32.
        for (int i = lane; i < batchLen; i += 32) {
            float2 d = sData[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            // Only consider this candidate if it is closer than the current worst.
            if (dist < currThresh) {
                Candidate cand;
                cand.idx = batchStart + i;  // Global data point index.
                cand.dist = dist;
                // Insert candidate into this warp's candidate buffer in shared memory.
                Candidate* warpCandBuffer = sCandBuffer + warpId * k;
                int pos = atomicAdd(&warpBufCount[warpId], 1);
                if (pos < k) {
                    warpCandBuffer[pos] = cand;
                }
                // If the candidate buffer becomes full, merge it into the private result.
                if (pos + 1 == k) {
                    mergePrivateAndBuffer(priv, warpCandBuffer, k, r, lane);
                    // Reset the candidate buffer counter (only one thread does it).
                    if (lane == 0) {
                        warpBufCount[warpId] = 0;
                    }
                    __syncwarp();
                    // After merging, update the threshold.
                    localWorst = priv[r - 1].dist;
                    currThresh = warpMax(localWorst);
                }
            }
        }
        __syncthreads();  // Make sure all threads have finished processing the batch.
    } // end for each batch

    // After all batches, if there are any remaining candidates in the candidate buffer, merge them.
    int bufCount = warpBufCount[warpId];
    if (bufCount > 0) {
        Candidate* warpCandBuffer = sCandBuffer + warpId * k;
        mergePrivateAndBuffer(priv, warpCandBuffer, bufCount, r, lane);
        if (lane == 0) warpBufCount[warpId] = 0;
        __syncwarp();
    }

    // ---------------------------------------------------------------------------------
    // Final merge step: each warp now has k candidates distributed across its lanes in the
    // private arrays. Gather these into a contiguous shared memory buffer and perform a 
    // simple parallel odd-even sort to order the k candidates in ascending order of distance.
    // ---------------------------------------------------------------------------------
    Candidate* finalBuffer = sCandBuffer + warpId * k;  // Reuse candidate buffer area for final results.
    // Each thread writes its r private candidates into finalBuffer.
    // We use a strided layout so that the finalBuffer becomes contiguous.
    for (int i = 0; i < r; i++) {
        finalBuffer[lane + i * 32] = priv[i];
    }
    __syncwarp();

    // Perform an odd-even transposition sort on finalBuffer.
    // There are k elements; we use the 32 threads in the warp to cooperatively sort them.
    for (int iter = 0; iter < k; iter++) {
        // Even phase: compare and swap pairs (i, i+1) for even i.
        for (int i = lane; i < k - 1; i += 32) {
            if ((i & 1) == 0) { // even index
                if (finalBuffer[i].dist > finalBuffer[i+1].dist) {
                    Candidate tmp = finalBuffer[i];
                    finalBuffer[i] = finalBuffer[i+1];
                    finalBuffer[i+1] = tmp;
                }
            }
        }
        __syncwarp();
        // Odd phase: compare and swap pairs for odd i.
        for (int i = lane; i < k - 1; i += 32) {
            if ((i & 1) == 1) { // odd index
                if (finalBuffer[i].dist > finalBuffer[i+1].dist) {
                    Candidate tmp = finalBuffer[i];
                    finalBuffer[i] = finalBuffer[i+1];
                    finalBuffer[i+1] = tmp;
                }
            }
        }
        __syncwarp();
    }

    // Write final sorted k nearest neighbors to global memory.
    // The output for query 'globalWarpId' is written starting at result[globalWarpId * k].
    for (int i = lane; i < k; i += 32) {
        // Store as std::pair<int,float>: first is index, second is squared distance.
        result[globalWarpId * k + i] = std::pair<int, float>(finalBuffer[i].idx, finalBuffer[i].dist);
    }
}

// ***********************************************************************
// Host function: run_knn
// This function configures and launches the CUDA kernel to perform k-NN on 2D points.
// Parameters:
//    query       : device pointer to query points (float2)
//    query_count : number of queries
//    data        : device pointer to data points (float2)
//    data_count  : number of data points
//    result      : device pointer to output (std::pair<int, float>) array; must have query_count*k elements.
//    k           : number of nearest neighbors to find (power of two, between 32 and 1024)
// ***********************************************************************
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block configuration.
    // We'll use 256 threads per block (i.e., 8 warps per block).
    const int blockSize = 256;
    const int warpsPerBlock = blockSize / 32;
    // Compute number of blocks needed so that each warp processes one query.
    int totalWarps = (query_count + 0 - 1) / 1; // each warp = one query.
    int numBlocks = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;
    
    // Calculate dynamic shared memory size:
    // Shared memory layout: 
    //   sData: BATCH_SIZE float2 elements,
    //   warpBufCount: warpsPerBlock int elements,
    //   sCandBuffer: warpsPerBlock * k Candidate elements.
    size_t shmemSize = BATCH_SIZE * sizeof(float2)
                     + warpsPerBlock * sizeof(int)
                     + warpsPerBlock * k * sizeof(Candidate);
    
    // Launch the kernel.
    knn_kernel<<<numBlocks, blockSize, shmemSize>>>(query, query_count, data, data_count, result, k);
    // (Note: error checking omitted for brevity.)
}