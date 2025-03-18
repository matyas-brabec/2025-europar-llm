#include <cuda_runtime.h>
#include <cfloat>    // for FLT_MAX
#include <utility>   // for std::pair

// -----------------------------------------------------------------------------
// In this implementation we assume that each query is processed by one warp (32 threads).
// Each warp maintains a private intermediate result stored in registers
// (each lane holds k/32 candidate entries) representing the current k best (smallest)
// distances. In addition, each warp has a candidate buffer allocated in shared memory
// of size k to accumulate new candidate points from the currently loaded batch of
// "data" points. When the candidate buffer fills up (or at the end of processing),
// the warp merges the candidate buffer with its intermediate result using an
// in-warp odd-even transposition sort on the union of (intermediate result + candidate buffer)
// stored in shared memory. The resulting sorted best-k candidates then replace the previous
// intermediate result. Finally, the k sorted neighbors are written to the global result array.
//
// We process the data points in batches. For each batch, the entire block loads a subset
// of global data points into shared memory. Then, in each warp, the 32 threads cooperate
// to compute the squared Euclidean distance from the query point (loaded from global memory)
// to each data point in the batch. A candidate is added to the per-warp candidate buffer only
// if its distance is smaller than the current worst (largest) distance among the intermediate result.
// When the candidate buffer is full (or after the last batch), the warp performs a merge.
// -----------------------------------------------------------------------------


// Define our candidate structure (index and squared distance).
struct Candidate {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// The merge_candidates function merges the warp's candidate buffer (of size "candCount")
// with its private intermediate result (stored in registers as an array per thread).
// The union (of size = k + candCount, padded to 2*k) is written into the warp's merge buffer
// (a per-warp shared memory region of size 2*k). Then, an odd-even transposition sort runs
// over the entire union (of size 2*k) to sort it by ascending distance. Finally, the first k
// sorted candidates are re-distributed among the warp threads to update their private intermediate
// result storage.
// Parameters:
//    candBuffer   - pointer to the candidate buffer in shared memory for this warp (length k)
//    candCount    - number of candidates accumulated in candBuffer (may be less than k on final merge)
//    localBest    - per-thread register array holding intermediate result candidates (size L = k/32)
//    mergeBuf     - pointer to the per-warp merge buffer in shared memory (length 2*k)
//    k            - the number of nearest neighbors to select
//    lane         - the warp lane id (0..31)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void merge_candidates(
    Candidate* candBuffer, int candCount, Candidate localBest[],
    Candidate* mergeBuf, int k, int lane)
{
    // Each warp has 32 threads.
    // Each thread holds L = k/32 candidates from the intermediate result.
    const int L = k / 32;
    // The union array size is total = k (from intermediate result) + candCount.
    // We'll form a union in mergeBuf of length N = 2*k (pad extra entries with FLT_MAX).
    int unionSize = k + candCount;  // valid entries

    // Step 1: Write the intermediate result from registers into mergeBuf[0:k].
    // The layout: each thread writes its L elements at positions: i*32 + lane, for i=0..L-1.
    for (int i = 0; i < L; i++) {
        mergeBuf[i * 32 + lane] = localBest[i];
    }
    // Step 2: Load candidate buffer (from shared memory) into mergeBuf[k : k + k - 1].
    // Only candCount entries are valid; pad the rest with { -1, FLT_MAX }.
    for (int i = lane; i < k; i += 32) {
        int idx = i;
        Candidate tmp;
        if (idx < candCount) {
            tmp = candBuffer[idx];
        } else {
            tmp.idx = -1;
            tmp.dist = FLT_MAX;
        }
        mergeBuf[k + i] = tmp;
    }
    __syncwarp();  // Ensure all lanes have written their data.

    // Now, mergeBuf holds 2*k candidate entries.
    const int N = 2 * k;

    // Perform odd-even transposition sort on mergeBuf.
    // We run N iterations; in each phase, each thread processes multiple pairs.
    for (int phase = 0; phase < N; phase++) {
        int start = (phase & 1);  // even phases start at index 0; odd phases start at index 1.
        // Each thread processes indices: i = lane + start, then i += 32.
        for (int i = lane + start; i < N - 1; i += 32) {
            Candidate a = mergeBuf[i];
            Candidate b = mergeBuf[i + 1];
            // Swap if out of order (we want ascending order by distance).
            if (a.dist > b.dist) {
                mergeBuf[i] = b;
                mergeBuf[i + 1] = a;
            }
        }
        __syncwarp();  // Synchronize warp lanes.
    }
    // After sorting, the first k elements in mergeBuf are the new best k candidates.
    // Distribute these back to the intermediate result registers.
    for (int i = 0; i < L; i++) {
        localBest[i] = mergeBuf[i * 32 + lane];
    }
    __syncwarp();
}

// -----------------------------------------------------------------------------
// Kernel function implementing k-NN for 2D points.
// Each warp (32 threads) processes one query; warps are mapped such that
// global warp index = query index. The kernel processes "data" points in batches,
// caching each batch in shared memory. For each data point in the batch,
// the warp computes its squared Euclidean distance to the query, and if the
// distance is smaller than the current worst candidate in the intermediate result,
// the candidate is added to the warp's candidate buffer (in shared memory).
// When the candidate buffer becomes full, it is merged with the intermediate result.
// Finally, after processing all batches, a final merge is performed if needed,
// and the k sorted nearest neighbors for the query are written to global memory.
//
// The shared memory usage per block is partitioned as follows:
//    - A batch buffer for data points: BATCH_SIZE * sizeof(float2)
//    - Per-warp candidate buffer: (numWarps_in_block * k * sizeof(Candidate))
//    - Per-warp merge buffer: (numWarps_in_block * 2*k * sizeof(Candidate))
//    - Per-warp candidate count array: (numWarps_in_block * sizeof(int))
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           Candidate *result, int k)
{
    // Define batch size for loading data points into shared memory.
    const int BATCH_SIZE = 256;

    // Each block has (blockDim.x/32) warps.
    int warpsPerBlock = blockDim.x / 32;
    int warpIdInBlock = threadIdx.x / 32;       // warp index within this block
    int lane = threadIdx.x & 31;                  // lane index (0..31)

    // Global warp id corresponds to a unique query.
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (globalWarpId >= query_count)
        return;

    // Each warp processes one query.
    float2 q = query[globalWarpId];

    // Each thread in a warp holds a portion of the intermediate results.
    // Let L = k/32 be the number of candidates stored per thread.
    const int L = k / 32;
    /// @FIXED
    /// Candidate localBest[L];
    Candidate localBest[/*MAX_K*/1024 / 32];  // Each thread holds k/32 candidates.
    #pragma unroll
    for (int i = 0; i < L; i++) {
        localBest[i].idx = -1;
        localBest[i].dist = FLT_MAX;
    }

    // Partition the shared memory (dynamically allocated) among:
    //   shData   : BATCH_SIZE float2 elements (for caching data points)
    //   warpCand : per-block candidate buffers, one per warp, each of length k (Candidate)
    //   mergeBuf : per-block merge buffers, one per warp, each of length 2*k (Candidate)
    //   candCount: per-block candidate count array, one int per warp.
    extern __shared__ char shared_mem[];
    float2* shData = (float2*) shared_mem;
    Candidate* warpCand = (Candidate*)(shData + BATCH_SIZE);
    Candidate* mergeBuf = (Candidate*)(warpCand + warpsPerBlock * k);
    int* candCount = (int*)(mergeBuf + warpsPerBlock * (2 * k));

    // Get pointers for this warp's candidate buffer, merge buffer, and candidate count.
    Candidate* myBuffer = warpCand + warpIdInBlock * k;
    Candidate* myMergeBuf = mergeBuf + warpIdInBlock * (2 * k);
    int* myCount = candCount + warpIdInBlock;
    if (lane == 0)
        *myCount = 0;
    __syncwarp();

    // Process the "data" points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE)
    {
        // Load a batch of data points from global memory into shared memory.
        // All threads in the block cooperate.
        for (int i = threadIdx.x; i < BATCH_SIZE && (batch_start + i) < data_count; i += blockDim.x) {
            shData[i] = data[batch_start + i];
        }
        __syncthreads();

        // Determine the actual number of points in this batch.
        int currentBatchSize = (data_count - batch_start < BATCH_SIZE) ? (data_count - batch_start) : BATCH_SIZE;

        // Each warp processes the batch cooperatively.
        // Each thread in the warp handles a strided portion of the batch.
        for (int i = lane; i < currentBatchSize; i += 32) {
            float2 p = shData[i];
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float dist = dx * dx + dy * dy;

            // Compute current threshold from intermediate results.
            // Each thread computes the maximum distance among its local candidates.
            float localMax = -1.0f;
            #pragma unroll
            for (int j = 0; j < L; j++) {
                float d = localBest[j].dist;
                if (d > localMax)
                    localMax = d;
            }
            // Reduce across warp to find the global worst (maximum) distance.
            for (int offset = 16; offset > 0; offset /= 2) {
                float other = __shfl_down_sync(0xffffffff, localMax, offset);
                if (other > localMax)
                    localMax = other;
            }
            float threshold = localMax;

            // Only add the candidate if its distance is less than the current worst.
            if (dist < threshold) {
                // Atomically add to the candidate buffer.
                int pos = atomicAdd(myCount, 1);
                if (pos < k) {
                    myBuffer[pos].idx = batch_start + i;  // global index of data point
                    myBuffer[pos].dist = dist;
                }
                // When the candidate buffer is full (or overfull), merge it with the intermediate result.
                if (*myCount >= k) {
                    __syncwarp();
                    merge_candidates(myBuffer, *myCount, localBest, myMergeBuf, k, lane);
                    if (lane == 0)
                        *myCount = 0;
                    __syncwarp();
                }
            }
        }
        __syncthreads();  // Ensure the entire block is done with the batch.
    }

    // After processing all batches, if any candidates remain in the buffer, merge them.
    if (*myCount > 0) {
        __syncwarp();
        merge_candidates(myBuffer, *myCount, localBest, myMergeBuf, k, lane);
        if (lane == 0)
            *myCount = 0;
        __syncwarp();
    }

    // At this point, the warp's intermediate result (in localBest registers) holds the final k nearest neighbors,
    // sorted in ascending order (nearest first). Write them to the global result array.
    // The result array is arranged so that for query index q, the neighbors are stored in result[q*k ... q*k + k - 1].
    int base = globalWarpId * k;
    for (int j = 0; j < L; j++) {
        int outIndex = base + lane + j * 32;
        // Write as std::pair<int, float> -> reinterpret the Candidate struct.
        /// @FIXED
        /// result[outIndex].first = localBest[j].idx;
        result[outIndex].idx = localBest[j].idx;
        /// @FIXED
        /// result[outIndex].second = localBest[j].dist;
        result[outIndex].dist = localBest[j].dist;
    }
}

// -----------------------------------------------------------------------------
// Host interface function that launches the k-NN kernel.
// Given query points and data points (as float2 arrays) and pre-allocated result buffer,
// this function computes the k nearest neighbors for each query (using squared Euclidean distance)
// and writes the indices and distances to the result array.
// The parameter k is a power of 2 between 32 and 1024, and data_count is >= k.
// -----------------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                          const float2 *data, int data_count,
                          std::pair<int, float> *result, int k)
{
    // Choose a reasonable block size that contains multiple warps.
    // We choose 128 threads per block (4 warps per block), so that many queries can share the same loaded data batch.
    const int threadsPerBlock = 128;
    const int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query, so total warps needed equals query_count.
    int numWarps = query_count;
    int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Define BATCH_SIZE used inside the kernel.
    const int BATCH_SIZE = 256;

    // Compute dynamic shared memory size:
    // Layout:
    //    shData   : BATCH_SIZE * sizeof(float2)
    //    warpCand : (warpsPerBlock * k) * sizeof(Candidate)
    //    mergeBuf : (warpsPerBlock * 2*k) * sizeof(Candidate)
    //    candCount: (warpsPerBlock) * sizeof(int)
    size_t shmemSize = BATCH_SIZE * sizeof(float2)
                     + warpsPerBlock * k * sizeof(Candidate)
                     + warpsPerBlock * 2 * k * sizeof(Candidate)
                     + warpsPerBlock * sizeof(int);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, shmemSize>>>(query, query_count, data, data_count,
                                                          reinterpret_cast<Candidate*>(result), k);
    // It's assumed that error checking and synchronization will be handled by the caller.
}
