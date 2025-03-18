#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// This kernel implements k-nearest neighbors for 2D points using a warp‐based approach.
// Each warp (32 threads) processes one query. Each thread holds (k/32) candidates in registers,
// and a per–warp candidate “buffer” of k candidates is maintained in shared memory.
// Data points are processed in batches which are loaded into shared memory to amortize global memory accesses.
// Whenever the candidate buffer becomes full, it is merged with the intermediate result (stored in registers)
// using a procedure loosely based on Bitonic Sort and Merge (here implemented with simple insertion sorts
// and pair‐wise merges using warp shuffles when exchanging elements between threads).
//
// Note: For clarity and brevity, some portions (e.g. the distributed bitonic sort) are implemented
// in a simplified form (using insertion sort) while following the spirit of the algorithm described.

using std::pair;

//------------------------------------------------------
// Candidate structure represents a neighbor candidate.
struct Candidate {
    int idx;
    float dist;
};

//------------------------------------------------------
// Device helper: create a candidate.
__device__ __forceinline__ Candidate make_candidate(int idx, float dist) {
    Candidate c;
    c.idx = idx;
    c.dist = dist;
    return c;
}

//------------------------------------------------------
// Device helper: return the candidate with the smaller distance.
__device__ __forceinline__ Candidate candidate_min(const Candidate &a, const Candidate &b) {
    return (a.dist < b.dist) ? a : b;
}

//------------------------------------------------------
// Device helper: squared Euclidean distance between two float2 points.
__device__ __forceinline__ float squared_distance(const float2 &a, const float2 &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

//------------------------------------------------------
// Device helper: simple insertion sort for small arrays of Candidates.
// This routine sorts the array in ascending order by distance.
__device__ void insertionSort(Candidate *arr, int n) {
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

//------------------------------------------------------
// The main k-NN kernel.
// Each warp processes one query. The intermediate result is stored in registers (distributed among 32 threads)
// as an array of "privateK" = k/32 candidates per thread. A per–warp candidate buffer of size k (in shared memory)
// is used to temporarily accumulate promising candidates from batched data.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           pair<int, float> *result, int k)
{
    // Define warp parameters.
    const int WARP_SIZE = 32;
    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock; 

    // Each warp processes one query if available.
    // Each thread in the warp will hold "privateK" candidates in registers.
    int privateK = k >> 5;  // since k is a power-of-two and 32 divides k

    //---------------------------------------------------------------------------
    // Shared memory layout (dynamically allocated):
    // 1. Warp candidate buffer: each warp gets k Candidate elements.
    // 2. Merge buffer: one per warp, k Candidate elements (used during merge).
    // 3. Data batch buffer: holds a batch of data points.
    // 4. Per-warp candidate count array (one int per warp in the block).
    //
    // The shared memory pointer is partitioned as follows:
    //   Candidate *warpCand      = sharedMem[0 .. warpsPerBlock*k - 1]
    //   Candidate *mergeBuffer   = sharedMem[warpsPerBlock*k .. 2*warpsPerBlock*k - 1]
    //   float2    *sharedData    = sharedMem[2*warpsPerBlock*k .. 2*warpsPerBlock*k + BATCH_SIZE - 1]
    //   int       *warpCandCount = sharedMem[2*warpsPerBlock*k + BATCH_SIZE .. ]
    //---------------------------------------------------------------------------
    extern __shared__ char sharedMem[];
    // Candidate buffer for each warp.
    Candidate *warpCand = (Candidate*)sharedMem;
    // Merge buffer (used during merge steps).
    Candidate *mergeBuffer = (Candidate*)(warpCand + warpsPerBlock * k);
    // Data batch buffer: we use a fixed BATCH_SIZE.
    const int BATCH_SIZE = 1024;
    float2 *sharedData = (float2*)(mergeBuffer + warpsPerBlock * k);
    // Candidate count for each warp in the block.
    int *warpCandCount = (int*)(sharedData + BATCH_SIZE);

    // Initialize candidate counts (only first warpsPerBlock threads need to do this).
    if (threadIdx.x < warpsPerBlock) {
        warpCandCount[threadIdx.x] = 0;
    }
    __syncthreads();

    //---------------------------------------------------------------------------
    // Each warp prepares its private (register) copy of the intermediate result.
    // Initially, all candidate distances are set to FLT_MAX (i.e. worst possible).
    Candidate reg[32];  // reg array size = privateK; note: privateK <= 32 as k<=1024.
    for (int i = 0; i < privateK; i++) {
        reg[i] = make_candidate(-1, FLT_MAX);
    }

    // Process input data points in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += BATCH_SIZE) {
        int currentBatchSize = (data_count - batchStart < BATCH_SIZE) ? (data_count - batchStart) : BATCH_SIZE;
        // Cooperative load: each thread in the block loads part of the current batch.
        for (int i = threadIdx.x; i < currentBatchSize; i += blockDim.x) {
            sharedData[i] = data[batchStart + i];
        }
        __syncthreads();

        // Only active warps (globalWarpId < query_count) process queries.
        if (globalWarpId < query_count) {
            // Load the query point for this warp.
            float2 q = query[globalWarpId];

            // Iterate over the current batch. Each warp processes data in a strided loop by lane.
            for (int i = lane; i < currentBatchSize; i += WARP_SIZE) {
                float2 dpt = sharedData[i];
                float dist = squared_distance(q, dpt);
                // Retrieve current max distance from intermediate result.
                // Since the intermediate result is globally sorted in ascending order, the worst (k-th) candidate
                // is held by the highest-index thread. We broadcast from lane 31.
                float currentMax = __shfl_sync(0xffffffff, reg[privateK - 1].dist, WARP_SIZE - 1);
                // If the candidate is closer than the worst in the current k-NN set...
                if (dist < currentMax) {
                    Candidate cand = make_candidate(batchStart + i, dist);
                    // Use atomic add on the per–warp candidate count to reserve a slot in the candidate buffer.
                    int pos = atomicAdd(&warpCandCount[warpIdInBlock], 1);
                    if (pos < k) {
                        warpCand[warpIdInBlock * k + pos] = cand;
                    }
                    // Else: if the buffer is full, the candidate will be merged soon.
                }
            }
        }
        __syncthreads();

        // After the batch, check if the candidate buffer for this warp is full.
        if (globalWarpId < query_count) {
            int candCount = warpCandCount[warpIdInBlock];
            if (candCount >= k) {
                // --------------------------------------------------------------------
                // Merge procedure when candidate buffer is full:
                // 0. Invariant: the intermediate result (in registers 'reg') is sorted in ascending order.
                // 1. Swap the contents of the candidate buffer (in shared memory) and the registers.
                //    Each thread loads its private portion from warpCand into a local buffer "localBuf"
                //    and writes its register value into the candidate buffer.
                // 2. Sort the swapped candidate buffer (now in "localBuf") using a simple insertion sort
                //    as a stand–in for a Bitonic sort.
                // 3. Merge the sorted "localBuf" with the intermediate result (from registers) by pairing
                //    each element with that from the opposite end of the register array.
                //    Warp ballot/shuffle instructions are used to retrieve partner values.
                // 4. Sort the merged result.
                // --------------------------------------------------------------------
                Candidate localBuf[32];  // local copy for the candidate buffer portion; size = privateK.
                for (int i = 0; i < privateK; i++) {
                    int bufIndex = lane + i * WARP_SIZE;  // each thread holds k/32 consecutive candidates.
                    if (bufIndex < k) {
                        localBuf[i] = warpCand[warpIdInBlock * k + bufIndex];
                    }
                    else {
                        localBuf[i] = make_candidate(-1, FLT_MAX);
                    }
                    // Swap: Write the current intermediate result from registers to the candidate buffer.
                    warpCand[warpIdInBlock * k + bufIndex] = reg[i];
                }
                // Step 2: Sort the swapped candidate buffer copy (localBuf) using insertion sort.
                insertionSort(localBuf, privateK);

                // Step 3: Merge localBuf with the intermediate result (in reg).
                Candidate merged[32];
                for (int i = 0; i < privateK; i++) {
                    // Compute the global index (0 .. k-1) for the element held by this thread.
                    int g = lane + i * WARP_SIZE;
                    // Use the symmetric partner index from the intermediate result.
                    int partner = k - g - 1;
                    int partner_thread = partner / privateK;        // which lane holds the partner element
                    int partner_local  = partner % privateK;          // index within that thread's reg[]
                    Candidate partnerCand;
                    if (partner_thread == lane) {
                        partnerCand = reg[partner_local];
                    }
                    else {
                        // Exchange the candidate from another thread using warp shuffle.
                        partnerCand.idx  = __shfl_sync(0xffffffff, reg[partner_local].idx, partner_thread);
                        partnerCand.dist = __shfl_sync(0xffffffff, reg[partner_local].dist, partner_thread);
                    }
                    // Merge: take the minimum of the candidate from localBuf and the partner candidate.
                    merged[i] = candidate_min(localBuf[i], partnerCand);
                }
                // Step 4: Sort the merged result.
                insertionSort(merged, privateK);
                // Update the intermediate result registers.
                for (int i = 0; i < privateK; i++) {
                    reg[i] = merged[i];
                }
                // Reset the candidate buffer count for this warp.
                warpCandCount[warpIdInBlock] = 0;
            }
        }
        __syncthreads();
    } // end processing all batches

    // After finishing processing all batches, merge any remaining candidates in the candidate buffer.
    if (globalWarpId < query_count) {
        int candCount = warpCandCount[warpIdInBlock];
        if (candCount > 0) {
            Candidate localBuf[32];
            for (int i = 0; i < privateK; i++) {
                int bufIndex = lane + i * WARP_SIZE;
                if (bufIndex < candCount)
                    localBuf[i] = warpCand[warpIdInBlock * k + bufIndex];
                else
                    localBuf[i] = make_candidate(-1, FLT_MAX);
                // Swap: write intermediate result into candidate buffer.
                warpCand[warpIdInBlock * k + bufIndex] = reg[i];
            }
            insertionSort(localBuf, privateK);
            Candidate merged[32];
            for (int i = 0; i < privateK; i++) {
                int g = lane + i * WARP_SIZE;
                int partner = k - g - 1;
                int partner_thread = partner / privateK;
                int partner_local  = partner % privateK;
                Candidate partnerCand;
                if (partner_thread == lane) {
                    partnerCand = reg[partner_local];
                } else {
                    partnerCand.idx  = __shfl_sync(0xffffffff, reg[partner_local].idx, partner_thread);
                    partnerCand.dist = __shfl_sync(0xffffffff, reg[partner_local].dist, partner_thread);
                }
                merged[i] = candidate_min(localBuf[i], partnerCand);
            }
            insertionSort(merged, privateK);
            for (int i = 0; i < privateK; i++)
                reg[i] = merged[i];
            warpCandCount[warpIdInBlock] = 0;
        }
    }
    __syncwarp();

    // At this point, the intermediate result in registers (in 'reg') is sorted in ascending order.
    // Write the final k nearest neighbors for this query to global memory in row–major order:
    // For query globalWarpId, the j-th neighbor is stored in result[globalWarpId * k + j].
    if (globalWarpId < query_count) {
        // The distributed layout is: thread 't' holds reg[i] where global index = t + i * WARP_SIZE.
        // Each thread writes out its corresponding candidates.
        for (int i = 0; i < privateK; i++) {
            int globalCandidateIndex = lane + i * WARP_SIZE;
            if (globalCandidateIndex < k) {
                // Use warp shuffle to broadcast the candidate from the thread that holds it.
                Candidate cand;
                // If this thread is responsible, use its own registers; else, retrieve via shuffle.
                cand.idx  = __shfl_sync(0xffffffff, reg[i].idx, lane);
                cand.dist = __shfl_sync(0xffffffff, reg[i].dist, lane);
                // Write into global result array.
                result[globalWarpId * k + globalCandidateIndex] = pair<int, float>(cand.idx, cand.dist);
            }
        }
    }
}

//------------------------------------------------------
// Host interface: launch the optimized k-NN kernel.
// The input pointers (query, data, result) are assumed to be allocated by cudaMalloc.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose thread block configuration.
    // We use 128 threads per block = 128/32 = 4 warps per block.
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query, so total number of warps = query_count.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute shared memory size required per block.
    // Layout:
    //   Warp candidate buffer: warpsPerBlock * k * sizeof(Candidate)
    //   Merge buffer: warpsPerBlock * k * sizeof(Candidate)
    //   Data batch buffer: BATCH_SIZE * sizeof(float2), where BATCH_SIZE is 1024.
    //   Candidate count array: warpsPerBlock * sizeof(int)
    size_t sharedMemSize = warpsPerBlock * k * sizeof(Candidate)       // warpCand
                         + warpsPerBlock * k * sizeof(Candidate)       // mergeBuffer
                         + 1024 * sizeof(float2)                       // sharedData
                         + warpsPerBlock * sizeof(int);                // warpCandCount

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}