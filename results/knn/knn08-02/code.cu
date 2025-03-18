#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>
#include <utility>

// This code implements an optimized CUDA kernel for k‐nearest neighbors (k‐NN)
// for 2D points using squared Euclidean distance. Each query is processed by a
// single warp (32 threads). The k nearest neighbors (k is a power‐of‐two
// between 32 and 1024) are maintained in “intermediate result” registers
// distributed in “row‐major” order: each thread in the warp holds k/32
// consecutive neighbor candidates (each candidate is a pair of (int index, float distance)).
// A per–warp candidate buffer is allocated in shared memory to accumulate
// promising candidates from the data (loaded in batches). When the buffer
// becomes full, it is merged with the intermediate result using a multi–step
// procedure that swaps the two arrays, bitonically sorts the temporary copy,
// then “merges” the two sorted arrays with a reversal (to pair large and small)
// and finally bitonically sorts the merged result. In several steps, warp–shuffle
// instructions are used to exchange candidate values between threads (when the
// partner candidate is in a different thread) and no extra device memory is allocated.
//
// The overall kernel structure:
//  - Each warp processes one query point.
//  - The data points are processed in batches. Each batch is loaded into shared memory.
//  - Each warp computes distances for its query and, if a computed distance is less than the current
//    kth largest distance (max_distance), the candidate is appended into the candidate buffer (in shared memory)
//    for that warp. A warp–ballot and warp–synchronous prefix–sum are used to determine insertion positions.
//  - When the candidate buffer for the warp becomes full (>= k candidates) or after the final batch,
//    the candidate buffer is merged with the intermediate result (stored in registers) via the merge procedure.
//  - Finally, the sorted intermediate result is written to global memory for each query.
//
// Note: This implementation uses “row–major” layout for the register arrays so that each candidate’s global
// index is given by: global_index = (lane * chunk + j), where chunk = k/32. In the bitonic sort routine,
// when an exchange is required between threads, the invariant holds because when the inner step index
// is >= chunk, it only affects the thread (lane) index leaving the local index j unchanged.
//
// We define a Candidate struct (each candidate holds an index and its squared distance).

// Struct to hold a candidate neighbor (data index and squared distance)
struct Candidate {
    int idx;
    float dist;
};

// Swap helper for two Candidate values (used for intra–thread swaps)
__device__ inline void swapCandidates(Candidate &a, Candidate &b) {
    Candidate tmp = a;
    a = b;
    b = tmp;
}

//---------------------------------------------------------------------
// Bitonic sort on the full candidate array of size k distributed
// among a warp’s registers. Each thread holds (k/32) candidates in a local array "reg[]".
// The global index of a candidate in a warp is defined as:
//    global_index = (lane * chunk + j)  ,  where lane is the warp–lane (0..31)
// and chunk = k/32.
// The bitonic sort algorithm (adapted from the pseudocode below)
// uses warp–shuffle instructions to exchange candidates stored in registers between threads when needed.
// For intra–thread comparisons (when the partner candidate is in the same thread),
// a normal register swap is performed.
//
// Pseudocode reference:
// for (size = 2; size <= n; size *= 2)
//   for (stride = size/2; stride > 0; stride /= 2)
//     for (i = 0; i < n; i++) {
//       l = i XOR stride;
//       if (l > i)
//         if ( ((i & size)==0 && arr[i] > arr[l]) || ((i & size)!=0 && arr[i] < arr[l]) )
//           swap(arr[i], arr[l]);
//     }
//
// In our implementation n = k, and we decompose i = (lane * chunk + j).
// When stride >= chunk, one obtains inter–thread exchanges (via __shfl_xor_sync)
// with offset = stride/chunk (which is a power–of–two < 32).
// When stride < chunk, both candidates are in the same thread.
__device__ void bitonic_sort(int k, Candidate *reg, int chunk)
{
    int lane = threadIdx.x & 31;  // warp lane id (0..31)

    // Loop over the stages of bitonic sort.
    for (int size = 2; size <= k; size *= 2) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes its local candidates (j from 0 to chunk-1)
            for (int j = 0; j < chunk; j++) {
                // Compute global index of candidate held in reg[j]
                int gi = lane * chunk + j;
                int partner = gi ^ stride;
                if (gi < partner) {
                    // Determine sort order for this pair (ascending if (gi & size)==0)
                    bool ascending = ((gi & size) == 0);
                    if (stride < chunk) {
                        // Intra–thread exchange; partner candidate is stored in the same thread
                        int partner_local = j ^ stride;
                        if (ascending) {
                            if (reg[j].dist > reg[partner_local].dist) {
                                swapCandidates(reg[j], reg[partner_local]);
                            }
                        } else {
                            if (reg[j].dist < reg[partner_local].dist) {
                                swapCandidates(reg[j], reg[partner_local]);
                            }
                        }
                    } else {
                        // Inter–thread exchange: For stride multiples of chunk, the partner's local position remains j.
                        int offset = stride / chunk; // offset for __shfl_xor_sync
                        // Exchange candidate reg[j] with partner candidate from lane: lane XOR offset.
                        Candidate partnerCand;
                        partnerCand.idx  = __shfl_xor_sync(0xffffffff, reg[j].idx, offset);
                        partnerCand.dist = __shfl_xor_sync(0xffffffff, reg[j].dist, offset);
                        if (ascending) {
                            if (reg[j].dist > partnerCand.dist) {
                                reg[j] = partnerCand;
                            }
                        } else {
                            if (reg[j].dist < partnerCand.dist) {
                                reg[j] = partnerCand;
                            }
                        }
                    }
                }
            }
            __syncwarp();
        }
    }
    // End of bitonic_sort: the array "reg" (of k candidates distributed over the warp)
    // is now sorted in ascending order (by candidate.dist).
}

//---------------------------------------------------------------------
// Merge the candidate buffer (accumulated in shared memory) with the intermediate
// result registers. When the candidate buffer is full (>= k candidates),
// we perform the following merge steps:
//   1. Swap the contents of the candidate buffer and the intermediate result registers.
//   2. Sort the now–register candidate buffer (which came from shared memory) using bitonic sort.
//   3. Merge the two sorted arrays by, for each candidate index, replacing the register candidate
//      with the minimum of (current register candidate, and a candidate from the swapped array read in reversed order).
//      To pair elements in different threads without extra communication, we reverse the order across warp lanes;
//      each thread obtains the candidate from its “partner” lane using __shfl_sync.
//   4. Sort the merged result using bitonic sort to yield an updated, fully sorted intermediate result.
//
// Parameters:
//    reg           - pointer to the per–thread register array (size = chunk) holding the intermediate result.
//    chunk         - number of candidates per thread (k/32)
//    buffer        - pointer to the candidate buffer for this warp in shared memory (size = k elements; layout row–major: each thread's block is contiguous).
//    bufferCount   - pointer to the candidate buffer count for this warp (shared memory variable).
//    k             - total number of candidates (k)
__device__ void merge_candidates(Candidate *reg, int chunk, Candidate *buffer, int *bufferCount, int k)
{
    int lane = threadIdx.x & 31;  // warp lane

    // Step 1: Swap the contents of reg (intermediate result) with the first k candidates in the buffer.
    // The buffer is organized in row–major order: candidate for global index = (lane * chunk + j)
    for (int j = 0; j < chunk; j++) {
        int idx = lane * chunk + j;
        Candidate temp = reg[j];
        reg[j] = buffer[idx];
        buffer[idx] = temp;
    }
    __syncwarp();

    // Step 2: Sort the candidate buffer now in registers (reg) using bitonic sort.
    bitonic_sort(k, reg, chunk);

    // Step 3: Merge the two sorted arrays.
    // The old intermediate result is now in the shared memory buffer.
    // To merge without extra memory accesses, we use a warp–shuffle to read from the buffer of the partner lane.
    // We assume that reversing the order across warp lanes pairs the large and small candidates appropriately.
    for (int j = 0; j < chunk; j++) {
        // First, load the candidate from the swapped intermediate result (in shared memory buffer)
        // from the partner lane. The partner lane is defined as: reversed order of the current lane.
        int revLane = 31 - lane;
        // Each thread reads its own candidate from buffer at position (lane * chunk + j),
        // then obtains the candidate from the partner lane at the same local index using shuffle.
        Candidate temp = buffer[lane * chunk + j];
        Candidate partnerCand;
        partnerCand.idx  = __shfl_sync(0xffffffff, temp.idx, revLane);
        partnerCand.dist = __shfl_sync(0xffffffff, temp.dist, revLane);
        // Merge: take the candidate with the smaller distance.
        if (reg[j].dist > partnerCand.dist) {
            reg[j] = partnerCand;
        }
    }
    __syncwarp();

    // Step 4: Bitonically sort the merged result to produce the updated intermediate result.
    bitonic_sort(k, reg, chunk);

    // Reset the candidate buffer count for this warp to 0.
    if ((threadIdx.x & 31) == 0) {
        *bufferCount = 0;
    }
    __syncwarp();
}

//---------------------------------------------------------------------
// Kernel to compute k–nearest neighbors (k–NN) for a set of 2D query points.
// Each query point is processed by a single warp (32 threads).
// Data points are processed in batches and loaded into shared memory, and each warp
// computes distances for its query. Candidates with a distance less than the current
// kth neighbor (max_distance) are appended into a per–warp candidate buffer in shared memory.
// When the candidate buffer is full (has at least k candidates) or after processing all data,
// it is merged with the intermediate result registers using the merge procedure.
/// @FIXED
/// extern "C" __global__
__global__
void knn_kernel(const float2 *query, int query_count,
                const float2 *data, int data_count,
                std::pair<int, float> *result, int k)
{
    // Hyper–parameter: Batch size for caching data points in shared memory.
    // Adjust DATA_BATCH_SIZE as needed.
    #define DATA_BATCH_SIZE 1024

    // Compute warp–level indices.
    int warpId   = threadIdx.x / 32;          // warp index within block
    int laneId   = threadIdx.x & 31;            // lane index (0..31)
    int warpsPerBlock = blockDim.x / 32;
    // Each warp processes one query point.
    int queryIdx = blockIdx.x * warpsPerBlock + warpId;
    if (queryIdx >= query_count) return;

    // Partition shared memory:
    //  Shared memory layout:
    //    [0, DATA_BATCH_SIZE) -> float2 sData[]: batch of data points.
    //    [DATA_BATCH_SIZE, DATA_BATCH_SIZE + (warpsPerBlock * k)) -> Candidate warpBuffer[]
    //    [DATA_BATCH_SIZE + (warpsPerBlock * k), ... ) -> int warpBufferCount[]
    extern __shared__ char smem[];
    float2 *sData = (float2*)smem;
    Candidate *warpBuffer = (Candidate*)(sData + DATA_BATCH_SIZE);
    int *warpBufferCount  = (int*)(warpBuffer + warpsPerBlock * k);

    // Initialize per–warp candidate buffer count to 0 (only once per warp).
    if(laneId == 0) {
        warpBufferCount[warpId] = 0;
    }
    __syncwarp();

    // Determine the number of candidates each thread holds in registers.
    int chunk = k / 32; // k is assumed to be divisible by 32

    // Initialize the warp's intermediate result registers.
    // We use a per–thread register array "inter" of size "chunk". Initially, all distances are FLT_MAX.
    Candidate inter[32];
    for (int j = 0; j < chunk; j++) {
        inter[j].dist = FLT_MAX;
        inter[j].idx  = -1;
    }

    // Load the query point for this warp.
    float2 q = query[queryIdx];

    // A warp–local variable to hold candidate buffer count (mirrors the shared memory value).
    int localBufferCount = 0;
    // (We will update warpBufferCount[warpId] and then reload localBufferCount.)
    localBufferCount = warpBufferCount[warpId];

    // Process the data points iteratively in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += DATA_BATCH_SIZE) {
        // Determine current batch size.
        int batchSize = ( (batchStart + DATA_BATCH_SIZE) <= data_count ) ? DATA_BATCH_SIZE : (data_count - batchStart);

        // Cooperative load of current batch of data points into shared memory.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            sData[i] = data[batchStart + i];
        }
        __syncthreads();

        // Each warp processes the batch:
        // Each thread in the warp iterates over data points in the batch with stride = 32.
        for (int i = laneId; i < batchSize; i += 32) {
            float2 pt = sData[i];
            // Compute squared Euclidean distance.
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float dist = dx*dx + dy*dy;
            // The current maximum accepted distance is the kth neighbor in the intermediate result.
            // The kth neighbor (largest distance among current best k) is stored at global index k-1.
            // In our row–major layout, global index k-1 is in thread with lane 31, at local index (chunk-1).
            float maxD = __shfl_sync(0xffffffff, inter[chunk-1].dist, 31);
            if (dist < maxD) {
                // Create a candidate with global data index and distance.
                Candidate cand;
                cand.idx  = batchStart + i;
                cand.dist = dist;
                // Insert the candidate into the per–warp candidate buffer.
                // Use warp–ballot to determine which lanes have a valid candidate.
                unsigned int vote = __ballot_sync(0xffffffff, true); // each lane that enters here is valid
                // Compute prefix sum within warp (using __popc on the mask of lanes with a lower lane id).
                int prefix = __popc(vote & ((1u << laneId) - 1));
                // The insertion position is determined by the current localBufferCount plus the prefix.
                int pos = atomicAdd(&warpBufferCount[warpId], 1);
                // Write the candidate into the warp buffer.
                warpBuffer[warpId * k + pos] = cand;
            }
        }
        __syncwarp();

        // Reload local candidate buffer count.
        localBufferCount = warpBufferCount[warpId];

        // If candidate buffer is full (>= k candidates), merge it with intermediate result.
        if (localBufferCount >= k) {
            merge_candidates(inter, chunk, &warpBuffer[warpId * k], &warpBufferCount[warpId], k);
            // After merge, localBufferCount is reset.
            localBufferCount = 0;
        }
        __syncthreads(); // synchronize threads in block before next batch load.
    }

    __syncwarp();
    // After processing all batches, if there are leftover candidates in the buffer, merge them.
    if (warpBufferCount[warpId] > 0) {
        merge_candidates(inter, chunk, &warpBuffer[warpId * k], &warpBufferCount[warpId], k);
    }
    __syncwarp();

    // Write the final k nearest neighbors for this query from the registers to global memory.
    // The intermediate result "inter" is stored in row–major order: each thread holds chunk candidates.
    // Global index for candidate from this warp: (queryIdx * k) + (laneId * chunk + j).
    for (int j = 0; j < chunk; j++) {
        int out_idx = queryIdx * k + (laneId * chunk + j);
        result[out_idx] = std::make_pair(inter[j].idx, inter[j].dist);
    }
}

//---------------------------------------------------------------------
// C++ host function to launch the k–NN kernel.
// All arrays (query, data, result) are assumed to be allocated on the device via cudaMalloc.
// The number of queries and data points can be large. This function chooses a block size
// of 256 threads (8 warps per block) and enough dynamic shared memory to store both the
// data batch and the per–warp candidate buffers.
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                          const float2 *data, int data_count,
                          std::pair<int, float> *result, int k)
{
    // Launch configuration:
    // Each warp processes one query. Use 256 threads per block -> 8 queries per block.
    const int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Shared memory requirements:
    //  - DATA_BATCH_SIZE float2 numbers for data batch.
    //  - For each warp in the block, k Candidate entries for candidate buffer.
    //  - For each warp in the block, 1 int for candidate buffer count.
    /// @FIXED (-1:+0)
    /// const int DATA_BATCH_SIZE = 1024;
    size_t shmem_size = DATA_BATCH_SIZE * sizeof(float2)
                      + warpsPerBlock * k * sizeof(Candidate)
                      + warpsPerBlock * sizeof(int);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, shmem_size>>>(query, query_count, data, data_count, result, k);

    // Check for kernel errors (error handling omitted for brevity).
    cudaDeviceSynchronize();
}
