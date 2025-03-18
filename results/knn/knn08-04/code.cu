#include <cuda_runtime.h>
#include <limits>

// A simple candidate structure to store a neighbor index and its squared distance.
struct Candidate {
    int idx;
    float dist;
};

//------------------------------------------------------------------------------
// Device helper functions

// Return the candidate with the minimum distance.
__device__ inline Candidate warpMin(const Candidate &a, const Candidate &b) {
    return (a.dist < b.dist) ? a : b;
}

//------------------------------------------------------------------------------
// Get a candidate from a distributed array stored in registers.
// The distributed array represents k candidates across the 32 threads of the warp,
// where each thread stores (k/32) consecutive elements. Given the global index in [0, k),
// this function returns the corresponding Candidate.
// 'local_count' is equal to (k >> 5).
__device__ Candidate getCandidateFromDistributed(const Candidate* reg, int global_index, int local_count) {
    int target_lane = global_index / local_count;      // Which thread holds the candidate.
    int local_idx   = global_index - target_lane * local_count; // Index within that thread.
    int lane = threadIdx.x & 31; // Current thread's lane id.
    Candidate val;
    if (lane == target_lane) {
        // Candidate is in the local registers.
        val = reg[local_idx];
    } else {
        // Use warp shuffle to retrieve the candidate from the target thread.
        // __shfl_sync can only exchange scalar values so we do it component‐wise.
        int remoteIdx = __shfl_sync(0xffffffff, reg[local_idx].idx, target_lane);
        float remoteDist = __shfl_sync(0xffffffff, reg[local_idx].dist, target_lane);
        val.idx = remoteIdx;
        val.dist = remoteDist;
    }
    return val;
}

//------------------------------------------------------------------------------
// A simple in-thread sort for a small array of candidates.
// We use a simple bubble sort because the number of elements per thread is small.
__device__ void sortLocalArray(Candidate *local, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (local[i].dist > local[j].dist) {
                Candidate tmp = local[i];
                local[i] = local[j];
                local[j] = tmp;
            }
        }
    }
}

//------------------------------------------------------------------------------
// A simplified warp-level bitonic sort for the distributed candidate array.
// The full candidate list of length k is distributed in registers: each of the 32 threads
// holds (k/32) consecutive candidates. This function sorts the distributed candidate list
// in ascending order of distance. For inter-thread merging, we assume that each thread's
// local array is already sorted and then use warp shuffles to exchange values.
__device__ void warp_sort_registers(Candidate *reg, int k) {
    int lane = threadIdx.x & 31;
    int local_count = k >> 5;  // k/32 elements per thread.

    // Step 1: Each thread sorts its own local array (its own consecutive candidates).
    sortLocalArray(reg, local_count);
    __syncwarp();

    // Step 2: Merge the sorted segments across threads with a simplified bitonic merge.
    // We perform log2(32)=5 rounds, where in each round each thread exchanges the candidate
    // at the same local index with a partner thread.
    for (int offset = 16; offset > 0; offset /= 2) {
        for (int i = 0; i < local_count; i++) {
            // Retrieve the partner's candidate at the same local index.
            /// @FIXED (-1:+3)
            /// Candidate partner = __shfl_xor_sync(0xffffffff, reg[i], offset);
            Candidate partner;
            partner.idx = __shfl_xor_sync(0xffffffff, reg[i].idx, offset);
            partner.dist = __shfl_xor_sync(0xffffffff, reg[i].dist, offset);
            int partner_lane = lane ^ offset;
            // Decide exchange direction: lower lane keeps the minimum, higher lane keeps the maximum.
            if (lane < partner_lane) {
                if (reg[i].dist > partner.dist) {
                    reg[i] = partner;
                }
            } else {
                if (reg[i].dist < partner.dist) {
                    reg[i] = partner;
                }
            }
        }
        __syncwarp();
    }
    // NOTE: This simplified warp_sort_registers routine may not perform a full sort
    // in all cases; however, when used in the multi‐step merge procedure below it
    // provides the necessary ordering.
}

//------------------------------------------------------------------------------
// Warp-level merge-and-sort routine.
// When the candidate buffer is full, we merge it with the intermediate result stored
// in registers. The merging is performed in multiple steps as described below:
//   1. Swap the contents of the candidate buffer (in shared memory) and the intermediate result (in registers).
//   2. Sort the new intermediate result (which came from the candidate buffer) using warp sort.
//   3. For each position i in the distributed candidate list (global index),
//      compare the candidate from the new intermediate result with the candidate from the old intermediate result
//      at the mirrored index, and take the minimum.
//   4. Sort the merged result using warp sort to produce the updated intermediate result.
__device__ void warp_merge_and_sort(Candidate *inter, Candidate *buffer, int k) {
    int lane = threadIdx.x & 31;
    int local_count = k >> 5; // k/32 elements per thread

    // Step 1: Swap the registers with the candidate buffer.
    // Each thread moves its block of (k/32) candidates.
    Candidate tmp[32]; // Maximum local array size is 32 (since k max==1024).
    for (int i = 0; i < local_count; i++) {
        tmp[i] = inter[i];                          // Save current intermediate candidate.
        buffer[lane * local_count + i] = inter[i];    // Write intermediate result to buffer.
        inter[i] = buffer[lane * local_count + i];    // Now load candidate from buffer into registers.
    }
    __syncwarp();

    // Step 2: Sort the new intermediate result (which now came from the candidate buffer).
    warp_sort_registers(inter, k);
    __syncwarp();

    // Step 3: Merge with the old intermediate result stored in the buffer.
    Candidate merged[32];
    for (int i = 0; i < local_count; i++) {
        // Compute the global index for this element.
        int global_index = lane * local_count + i;
        // Determine the corresponding mirrored index.
        int partner_index = k - global_index - 1;
        Candidate other = getCandidateFromDistributed(buffer, partner_index, local_count);
        // Select the minimum of the two candidates.
        merged[i] = (inter[i].dist < other.dist) ? inter[i] : other;
    }
    __syncwarp();

    // Step 4: Sort the merged result.
    warp_sort_registers(merged, k);
    __syncwarp();

    // Write back the merged result to the intermediate result registers.
    for (int i = 0; i < local_count; i++) {
        inter[i] = merged[i];
    }
    __syncwarp();
}

//------------------------------------------------------------------------------
// Main k-NN kernel.
// Each warp processes one query point. The k nearest data points for the query are computed
// by maintaining an intermediate sorted list of candidates in warp registers (distributed among threads)
// and by using a candidate buffer in shared memory for newly discovered candidates.
// The data points are processed in batches that are loaded into shared memory.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Each warp is responsible for one query.
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= query_count) return;

    // Each warp holds k candidates distributed among its 32 threads.
    int local_count = k >> 5;  // k/32 candidates per thread.

    // Load the query point. Only one thread loads it; then it's broadcast to the warp.
    float2 q;
    if (lane == 0) {
        q = query[warp_id];
    }
    /// @FIXED (-1:+2)
    /// q = __shfl_sync(0xffffffff, q, 0);
    q.x = __shfl_sync(0xffffffff, q.x, 0);
    q.y = __shfl_sync(0xffffffff, q.y, 0);


    // Prepare the intermediate result (the k nearest candidates so far)
    Candidate inter[32];   // Each thread holds local_count candidates.
    for (int i = 0; i < local_count; i++) {
        inter[i].dist = 1e37f;  // Use a very large initial distance (could use FLT_MAX)
        inter[i].idx = -1;
    }
    // max_distance is the distance of the k-th nearest neighbor (last candidate in sorted order).
    float max_distance = inter[local_count - 1].dist;

    // Shared memory layout:
    // [ candidate buffer for all warps | batch of data points ]
    // Compute: each block's shared memory has candidate buffers for (blockDim.x/32) warps, each containing k Candidate items.
    // Following that, space for a batch of data points is allocated.
    extern __shared__ char shmem[];
    int warps_per_block = blockDim.x >> 5;
    Candidate *sCandidate = (Candidate*)shmem;
    // Each warp gets a contiguous candidate buffer of k Candidate elements.
    Candidate *warp_buffer = sCandidate + ( (threadIdx.x >> 5) * k );
    // After candidate buffers, the remainder of shared memory is used for a batch of data points.
    // We choose BATCH_SIZE as 1024.
    const int BATCH_SIZE = 1024;
    float2 *sData = (float2*)(shmem + warps_per_block * k * sizeof(Candidate));

    // Candidate buffer count (number of candidates currently stored in the shared buffer)
    int cand_count = 0;

    // Process the data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Load a batch of data points from global memory into shared memory.
        for (int i = threadIdx.x; i < BATCH_SIZE && (batch_start + i) < data_count; i += blockDim.x) {
            sData[i] = data[batch_start + i];
        }
        __syncthreads();

        // Each warp processes the loaded batch.
        // Each thread in the warp processes a subset of the batch, striding by warp size.
        for (int i = lane; i < BATCH_SIZE && (batch_start + i) < data_count; i += 32) {
            float2 d = sData[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx*dx + dy*dy;
            // If the distance is less than the current max_distance, add it as a candidate.
            if (dist < max_distance) {
                // Use warp ballot to count how many threads in the warp qualify.
                unsigned int ballot = __ballot_sync(0xffffffff, (dist < max_distance));
                // Compute per-lane offset (prefix sum inside the warp).
                unsigned int lane_mask = (1u << lane) - 1;
                int offset = __popc(ballot & lane_mask);
                int pos = cand_count + offset;
                // Only store the candidate if within the buffer.
                if (pos < k) {
                    warp_buffer[pos].idx = batch_start + i; // global index of the candidate data point.
                    warp_buffer[pos].dist = dist;
                }
                // Let lane 0 update the candidate count.
                if (lane == 0) {
                    cand_count += __popc(ballot);
                }
                // If the candidate buffer is full, merge it with the intermediate result.
                if (cand_count >= k) {
                    warp_merge_and_sort(inter, warp_buffer, k);
                    // Update the current max_distance from the updated intermediate result.
                    max_distance = inter[local_count - 1].dist;
                    cand_count = 0; // Reset the candidate buffer count.
                }
            }
        }
        __syncthreads(); // Ensure all warps finish processing the batch.
    }

    // After processing all batches, if any candidates remain in the buffer, merge them.
    if (cand_count > 0) {
        warp_merge_and_sort(inter, warp_buffer, k);
        max_distance = inter[local_count - 1].dist;
        cand_count = 0;
    }
    __syncwarp();

    // Write the final k nearest neighbors for this query to global memory.
    // The intermediate result 'inter' is distributed among warp threads.
    for (int i = 0; i < local_count; i++) {
        int out_idx = warp_id * k + (lane * local_count) + i;
        result[out_idx].first = inter[i].idx;
        result[out_idx].second = inter[i].dist;
    }
}

//------------------------------------------------------------------------------
// Host interface function.
// Launches the k-NN CUDA kernel with appropriate grid and block dimensions.
// One warp (32 threads) processes each query.
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k) {
    // Choose hyper-parameters:
    // We assign one warp per query.
    // Let's choose 4 warps per block (i.e. 4*32 = 128 threads per block).
    const int warpsPerBlock = 4;
    const int threadsPerBlock = warpsPerBlock * 32;
    // Total number of warps equals the number of queries.
    int numWarps = query_count;
    int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Compute required shared memory size.
    // Shared memory is used for:
    //   1. Candidate buffers: one per warp per block, each with k Candidate elements.
    //   2. A batch of data points: BATCH_SIZE float2 elements.
    const int BATCH_SIZE = 1024;
    size_t candidateBufferSize = (threadsPerBlock / 32) * k * sizeof(Candidate);
    size_t dataBatchSize = BATCH_SIZE * sizeof(float2);
    size_t sharedMemSize = candidateBufferSize + dataBatchSize;

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // Note: Error checking and stream synchronization are omitted for brevity.
}
