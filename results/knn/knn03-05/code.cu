// This is an optimized CUDA implementation of k‐nearest neighbors for 2D points.
// Each query is processed by one warp (32 threads per query).
// The kernel processes the input data in batches; each batch of data points is loaded
// into shared memory to reduce global memory accesses.
// Each warp maintains its own private candidate list of k nearest neighbors in shared memory,
// which is maintained in sorted (ascending) order by squared distance.
// To reduce the overhead of updating the candidate list (which is maintained by lane 0),
// each thread buffers a few candidate proposals from the data it processes. Once the buffer
// is full, the warp cooperatively flushes (via warp shuffles) all proposals to lane 0,
// and lane 0 updates the candidate list via an insertion (binary‐like) procedure.
// The candidate list is stored in shared memory and later written to global memory.
// 
// Hyper-parameters chosen:
//   - BATCH_SIZE: 1024 (number of data points loaded per batch).
//   - Threads per block: 256 (8 warps per block).
//   - Buffer size per thread (L): 4.
// These choices are tuned for modern GPUs such as the A100/H100.
//
// Note: k is a power of two between 32 and 1024. Each warp’s candidate list is stored as
// an array of k std::pair<int,float> (index, distance) in shared memory.
// 
// The squared distance between two float2 points is computed as (dx*dx + dy*dy).
//
// The C++ host interface is provided by run_knn().
//
// Compile with the latest CUDA toolkit and a modern host compiler.

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Define the batch size (number of data points loaded into shared memory per iteration).
#define BATCH_SIZE 1024

// __device__ function to update a warp's candidate list (stored in shared memory).
// The candidate list "cand" is maintained in ascending order (best candidate at index 0,
// worst candidate at index k-1). If the new candidate (with index idx and squared distance dist)
// is better than the current worst candidate, it is inserted in sorted order.
__device__ inline void update_candidate(std::pair<int, float>* cand, int k, int idx, float dist) {
    // Only update if the new distance is smaller than the worst candidate.
    if (dist >= cand[k - 1].second)
        return;
    // Perform an insertion sort: shift candidates one step to the right until proper slot is found.
    int pos = k - 1;
    while (pos > 0 && cand[pos - 1].second > dist) {
        cand[pos] = cand[pos - 1];
        pos--;
    }
    cand[pos] = std::pair<int, float>(idx, dist);
}

// The CUDA kernel that computes k-NN for 2D points.
// Each warp (32 threads) handles one query point.
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k)
{
    // Obtain lane id (0...31) and warp id (within the block)
    const int lane = threadIdx.x & 31;
    const int warpIdInBlock = threadIdx.x >> 5;
    // Global warp id corresponds to the query index.
    const int globalWarpId = blockIdx.x * (blockDim.x >> 5) + warpIdInBlock;
    if (globalWarpId >= query_count)
        return;

    // Load the query point associated with this warp.
    float2 q = query[globalWarpId];

    // Shared memory layout:
    //   First part: BATCH_SIZE float2 elements for caching a batch of data points.
    //   Second part: candidate lists for each warp in the block, each of size k std::pair<int,float>.
    extern __shared__ char smem[];
    float2* sdata = (float2*)smem;  // Data cache buffer (size: BATCH_SIZE * sizeof(float2))
    std::pair<int, float>* cand_lists = (std::pair<int, float>*)
                                         (smem + BATCH_SIZE * sizeof(float2));
    // Each warp's candidate list pointer.
    std::pair<int, float>* cand = cand_lists + warpIdInBlock * k;

    // Initialize the candidate list to "empty" (distance = FLT_MAX, index = -1).
    if (lane == 0) {
        for (int i = 0; i < k; i++) {
            cand[i] = std::pair<int, float>(-1, FLT_MAX);
        }
    }
    // Synchronize within warp to ensure candidate list is visible.
    __syncwarp();

    // Each thread uses a small local buffer to postpone candidate list updates.
    // Buffer size L is chosen to amortize the cost of candidate list update.
    const int L = 4;
    int bufCount = 0;
    int localBufIdx[L];
    float localBufDist[L];

    // Process input data in batches.
    for (int batch = 0; batch < data_count; batch += BATCH_SIZE) {
        // Determine number of points in this batch.
        int batchCount = (data_count - batch < BATCH_SIZE) ? (data_count - batch) : BATCH_SIZE;

        // Cooperative load: all threads in block load the current batch into shared memory.
        for (int i = threadIdx.x; i < BATCH_SIZE; i += blockDim.x) {
            if (i < batchCount)
                sdata[i] = data[batch + i];
        }
        __syncthreads(); // Ensure the batch is fully loaded.

        // Each warp processes the batch: iterate over data indices with stride 32.
        for (int j = lane; j < batchCount; j += 32) {
            float2 dpt = sdata[j];
            float dx = dpt.x - q.x;
            float dy = dpt.y - q.y;
            float dist = dx * dx + dy * dy;
            int idx = batch + j;
            // Read the current worst candidate distance from the warp's candidate list.
            float worst = cand[k - 1].second;
            if (dist < worst) {
                // Buffer the candidate proposal.
                localBufIdx[bufCount] = idx;
                localBufDist[bufCount] = dist;
                bufCount++;
                // If the buffer is full, flush all proposals from the warp.
                if (bufCount == L) {
                    // Only lane 0 will flush the proposals from all lanes in the warp.
                    if (lane == 0) {
                        // For each lane in the warp, retrieve its buffered proposals.
                        for (int src = 0; src < 32; src++) {
                            int srcCount = __shfl_sync(0xFFFFFFFF, bufCount, src);
                            for (int pos = 0; pos < L; pos++) {
                                if (pos < srcCount) {
                                    int proposalIdx = __shfl_sync(0xFFFFFFFF, localBufIdx[pos], src);
                                    float proposalDist = __shfl_sync(0xFFFFFFFF, localBufDist[pos], src);
                                    update_candidate(cand, k, proposalIdx, proposalDist);
                                }
                            }
                        }
                    }
                    __syncwarp(); // Ensure all threads see the updated candidate list.
                    bufCount = 0; // Reset each thread's buffer.
                }
            }
        }
        __syncthreads(); // Ensure all warps are done with the current batch before loading next.
    }

    // Flush any remaining proposals in the local buffers.
    if (bufCount > 0) {
        if (lane == 0) {
            for (int src = 0; src < 32; src++) {
                int srcCount = __shfl_sync(0xFFFFFFFF, bufCount, src);
                for (int pos = 0; pos < L; pos++) {
                    if (pos < srcCount) {
                        int proposalIdx = __shfl_sync(0xFFFFFFFF, localBufIdx[pos], src);
                        float proposalDist = __shfl_sync(0xFFFFFFFF, localBufDist[pos], src);
                        update_candidate(cand, k, proposalIdx, proposalDist);
                    }
                }
            }
        }
        __syncwarp();
    }

    // Write the candidate list for this query to the results array in global memory.
    // Each warp writes its k candidates cooperatively.
    int base = globalWarpId * k;
    for (int i = lane; i < k; i += 32) {
        result[base + i] = cand[i];
    }
}

// Host function that sets up and launches the knn_kernel.
// Input arrays (query, data) and the output array (result) are assumed to be allocated
// in device memory (via cudaMalloc).
//   - query: array of float2 query points (query_count elements).
//   - data: array of float2 data points (data_count elements).
//   - result: array of std::pair<int,float> to hold k nearest neighbors per query.
//             For query i, result[i*k + j] is the j-th nearest neighbor.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Choose 256 threads per block (8 warps per block).
    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock >> 5; // 256/32 = 8

    // Each warp processes one query.
    int numWarps = query_count;
    int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate the required shared memory per block:
    //   - Data cache: BATCH_SIZE * sizeof(float2)
    //   - Candidate lists: (warpsPerBlock * k * sizeof(std::pair<int,float>))
    size_t sharedMemBytes = BATCH_SIZE * sizeof(float2)
                          + warpsPerBlock * k * sizeof(std::pair<int, float>);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(query, query_count,
                                                               data, data_count,
                                                               result, k);
    cudaDeviceSynchronize();
}