#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <utility>
// @FIXED
#include <math_constants.h>

// Define the batch size for loading data points from global memory into shared memory.
// This value is chosen to trade off between shared‐memory usage and off‐chip bandwidth.
#define BATCH_SIZE 1024

// -----------------------------------------------------------------------------------
// Struct to hold a candidate neighbor (data index and its squared Euclidean distance).
// We use 32-bit int for the index and 32-bit float for the distance.
struct Candidate {
    int index;
    float distance;
};

// -----------------------------------------------------------------------------------
// Device inline function to swap two Candidate objects.
__device__ inline void swapCandidate(Candidate &a, Candidate &b) {
    Candidate temp = a;
    a = b;
    b = temp;
}

// -----------------------------------------------------------------------------------
// Device function implementing Bitonic Sort in ascending order on an array of Candidates.
// This implementation is designed to be executed cooperatively by a warp (32 threads).
// The sort is performed on 'n' elements stored in shared memory.
// Each thread in the warp processes multiple elements in a strided loop.
__device__ void bitonicSort(Candidate *arr, int n) {
    // Assume warp-synchronous execution: use lane index for iteration.
    int lane = threadIdx.x & 31;  // lane index within warp

    // Outer loop: sequence size doubles each iteration.
    for (int seq = 2; seq <= n; seq *= 2) {
        // Inner loop: step reduces by half each iteration.
        for (int stride = seq >> 1; stride > 0; stride = stride >> 1) {
            // Each thread handles indices i in a strided manner.
            for (int i = lane; i < n; i += 32) {
                int j = i ^ stride; // partner index computed as bitwise XOR.
                if (j > i) {
                    // Determine whether the current sequence is in ascending order.
                    bool ascending = ((i & seq) == 0);
                    if (ascending) {
                        if (arr[i].distance > arr[j].distance) {
                            swapCandidate(arr[i], arr[j]);
                        }
                    } else {
                        if (arr[i].distance < arr[j].distance) {
                            swapCandidate(arr[i], arr[j]);
                        }
                    }
                }
            }
            __syncwarp(); // synchronize lanes in this warp before next stride.
        }
    }
}

// -----------------------------------------------------------------------------------
// Kernel implementing k-NN for 2D points.
// Each query point is processed by one warp (32 threads).
// Each warp maintains a private (register) copy of its current best k candidates
// and uses a shared-memory candidate buffer to accumulate new candidate points
// from successive batches of data points loaded into shared memory.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Each warp (32 threads) processes one query.
    int lane = threadIdx.x & 31;               // lane index in warp (0..31)
    int warpIdInBlock = threadIdx.x >> 5;        // warp index within the block
    int globalWarpId = blockIdx.x * (blockDim.x >> 5) + warpIdInBlock;
    if (globalWarpId >= query_count) return;     // out-of-bound check

    // Load the query point for this warp.
    float2 q = query[globalWarpId];

    // Determine number of candidates stored per thread in this warp's private (register) copy.
    // Since k is a power-of-two between 32 and 1024, k/32 yields an integer.
    int local_k = k >> 5; // equivalent to k / 32

    // Private array for the warp's current k nearest neighbors, partitioned among lanes.
    // Each lane holds local_k Candidate objects.
    Candidate local_int[32];  // Maximum local_k will be 1024/32 = 32.
#pragma unroll
    for (int i = 0; i < local_k; i++) {
        local_int[i].index = -1;
        local_int[i].distance = CUDART_INF_F;
    }
    // The current maximum distance among the k nearest neighbors.
    float warp_max = CUDART_INF_F;

    // --------------------------------------------------------------------------------
    // Shared memory layout (allocated dynamically):
    //  1. shared_data: BATCH_SIZE float2 elements for data points.
    //  2. candidate buffers: one per warp, each with k Candidate elements.
    //  3. candidate counts: one int per warp to count candidates in the buffer.
    //  4. warp_intermediate: temporary buffer (per warp) to copy the private k-results.
    //  5. warp_merge: temporary buffer (per warp) for merging.
    //
    // Total shared memory per block (in bytes):
    //   BATCH_SIZE * sizeof(float2)
    // + (warpsPerBlock * k) * sizeof(Candidate)
    // + (warpsPerBlock) * sizeof(int)
    // + (warpsPerBlock * k) * sizeof(Candidate)
    // + (warpsPerBlock * k) * sizeof(Candidate)
    //
    // Compute warps per block.
    int warpsPerBlock = blockDim.x >> 5; // blockDim.x / 32

    extern __shared__ char smem[]; // Dynamically allocated shared memory.
    // Pointer to shared data batch (float2 array).
    float2 *shared_data = (float2*)smem;
    // Pointer to candidate buffers for all warps.
    Candidate *cand_buf_global = (Candidate*)(shared_data + BATCH_SIZE);
    // Pointer to candidate count array (one int per warp).
    int *cand_count_global = (int*)(cand_buf_global + warpsPerBlock * k);
    // Pointer to temporary buffer to copy the private intermediate results.
    Candidate *warp_int_global = (Candidate*)(cand_count_global + warpsPerBlock);
    // Pointer to temporary merge buffer.
    Candidate *warp_merge_global = (Candidate*)(warp_int_global + warpsPerBlock * k);

    // Get pointers for this warp's candidate buffer and candidate count.
    Candidate *my_cand_buf = cand_buf_global + warpIdInBlock * k;
    int *my_cand_count = cand_count_global + warpIdInBlock;
    if (lane == 0) {
        *my_cand_count = 0;
    }
    __syncwarp();

    // Pointers for this warp's temporary buffers.
    Candidate *my_warp_int = warp_int_global + warpIdInBlock * k;
    Candidate *my_warp_merge = warp_merge_global + warpIdInBlock * k;

    // --------------------------------------------------------------------------------
    // Process the data points in batches.
    for (int batch = 0; batch < data_count; batch += BATCH_SIZE) {
        // Determine the actual number of data points in this batch.
        int batch_size = (batch + BATCH_SIZE <= data_count) ? BATCH_SIZE : (data_count - batch);
        // Load a batch of data points from global memory to shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            shared_data[i] = data[batch + i];
        }
        __syncthreads();  // Ensure entire batch is loaded.

        // Each warp processes the batch in parallel using its 32 lanes.
        for (int i = lane; i < batch_size; i += 32) {
            float2 d = shared_data[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            // Only consider this candidate if its distance is less than the current max.
            if (dist < warp_max) {
                // Atomically add candidate to the candidate buffer.
                int pos = atomicAdd(my_cand_count, 1);
                if (pos < k) {
                    my_cand_buf[pos].index = batch + i; // Global data index.
                    my_cand_buf[pos].distance = dist;
                }
            }
        }
        __syncwarp();

        // If the candidate buffer is full or overfull, merge it with the private intermediate result.
        if (*my_cand_count >= k) {
            // (1) Sort the candidate buffer using Bitonic Sort.
            bitonicSort(my_cand_buf, k);
            __syncwarp();

            // (2) Copy the private intermediate result from registers to the temporary shared buffer.
            for (int i = 0; i < local_k; i++) {
                my_warp_int[lane * local_k + i] = local_int[i];
            }
            __syncwarp();

            // (3) Merge: for each index i in [0, k), compute:
            // merged[i] = min( my_warp_int[i], my_cand_buf[k - 1 - i] )
            for (int i = lane; i < k; i += 32) {
                Candidate a = my_warp_int[i];
                Candidate b = my_cand_buf[k - 1 - i];
                my_warp_merge[i] = (a.distance < b.distance) ? a : b;
            }
            __syncwarp();

            // (4) Sort the merged result to obtain an updated sorted intermediate result.
            bitonicSort(my_warp_merge, k);
            __syncwarp();

            // (5) Copy the merged sorted result back to the private registers.
            for (int i = 0; i < local_k; i++) {
                local_int[i] = my_warp_merge[lane * local_k + i];
            }
            __syncwarp();

            // (6) Update the current maximum distance from the merged result.
            if (lane == 0) {
                warp_max = my_warp_merge[k - 1].distance;
            }
            warp_max = __shfl_sync(0xffffffff, warp_max, 0);

            // (7) Reset the candidate buffer count.
            if (lane == 0) {
                *my_cand_count = 0;
            }
            __syncwarp();
        }
        __syncthreads(); // Ensure all threads finished this batch before next.
    }  // End of batching loop

    // --------------------------------------------------------------------------------
    // After processing all batches, if there remain candidates in the candidate buffer,
    // merge them with the private intermediate result.
    if (*my_cand_count > 0) {
        int cnt = *my_cand_count;
        // Fill remaining entries of the candidate buffer with sentinel values.
        for (int i = cnt + lane; i < k; i += 32) {
            my_cand_buf[i].index = -1;
            my_cand_buf[i].distance = CUDART_INF_F;
        }
        __syncwarp();

        bitonicSort(my_cand_buf, k);
        __syncwarp();

        for (int i = 0; i < local_k; i++) {
            my_warp_int[lane * local_k + i] = local_int[i];
        }
        __syncwarp();

        for (int i = lane; i < k; i += 32) {
            Candidate a = my_warp_int[i];
            Candidate b = my_cand_buf[k - 1 - i];
            my_warp_merge[i] = (a.distance < b.distance) ? a : b;
        }
        __syncwarp();

        bitonicSort(my_warp_merge, k);
        __syncwarp();

        for (int i = 0; i < local_k; i++) {
            local_int[i] = my_warp_merge[lane * local_k + i];
        }
        __syncwarp();

        if (lane == 0) {
            warp_max = my_warp_merge[k - 1].distance;
        }
        warp_max = __shfl_sync(0xffffffff, warp_max, 0);

        if (lane == 0) {
            *my_cand_count = 0;
        }
        __syncwarp();
    }

    // --------------------------------------------------------------------------------
    // Write final sorted k nearest neighbors to global memory.
    // The result for query 'globalWarpId' is stored in row-major order in the output array.
    // Each warp’s private result (stored in registers and partitioned among 32 threads)
    // is written out contiguously.
    for (int i = 0; i < local_k; i++) {
        int out_idx = globalWarpId * k + (lane * local_k + i);
        result[out_idx].first = local_int[i].index;
        result[out_idx].second = local_int[i].distance;
    }
}

// -----------------------------------------------------------------------------------
// Host-side interface function that sets up and launches the CUDA kernel.
// All pointers are assumed to be allocated with cudaMalloc.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose a block size that is a multiple of warp size.
    // Here we choose 128 threads per block (i.e. 4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock >> 5;  // threadsPerBlock / 32

    // Each warp processes one query; thus total warps needed equals query_count.
    int numWarps = query_count;
    // Compute grid dimensions: number of blocks needed.
    int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Compute the required shared memory size per block.
    // Layout: [shared_data (BATCH_SIZE float2)] +
    //         [candidate buffers: warpsPerBlock * k Candidates] +
    //         [candidate counts: warpsPerBlock ints] +
    //         [warp_intermediate: warpsPerBlock * k Candidates] +
    //         [warp_merge: warpsPerBlock * k Candidates]
    size_t sharedMemSize = BATCH_SIZE * sizeof(float2)
                           + warpsPerBlock * k * sizeof(Candidate)
                           + warpsPerBlock * sizeof(int)
                           + warpsPerBlock * k * sizeof(Candidate)
                           + warpsPerBlock * k * sizeof(Candidate);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // Note: In production code one should check for kernel launch errors and synchronize.
}
