#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// -----------------------------------------------------------------------------
// This CUDA code implements a k-nearest neighbors search for 2D points
// using a warp‐level algorithm. Each warp (32 threads) processes one query.
// Each warp keeps a private (register‐based) list of k best neighbors (unsorted)
// and uses a shared‐memory candidate buffer (capacity k) to temporarily store
// new candidates. When the candidate buffer is full, the warp merges it with
// its private result using a warp‐synchronized bitonic sort on a merge buffer,
// and then updates its k-th best (max_distance).
//
// The whole data array is processed in batches; each batch of data points is
// loaded into shared memory by the block and then processed by all warps.
// Finally, each warp does a final sort of its intermediate results and writes
// the k sorted nearest neighbors (index, squared distance pairs) to global memory.
// -----------------------------------------------------------------------------


// Constant batch size for processing data points.
// (If k is large, one may choose a smaller batch size so that overall shared memory
// usage stays below device limits.)
#define BATCH_SIZE 1024

// Structure of a neighbor candidate.
struct Neighbor {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// __device__ bitonic sort routine: sorts an array "buffer" of N elements in
// ascending order (by .dist).  N must be a power-of-two. The 32 warp threads
// cooperatively process the array (each thread works on elements with stride 32).
// __syncwarp() is used for warp-level synchronization.
// -----------------------------------------------------------------------------
/// @FIXED
/// __device__ void bitonicSortWarp(volatile Neighbor *buffer, int N) {
__device__ void bitonicSortWarp(Neighbor *buffer, int N) {
    // Loop over the bitonic sort stages
    for (int size = 2; size <= N; size *= 2) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncwarp(0xffffffff);
            // Each thread processes indices: starting at its lane id and then every 32
            int lane = threadIdx.x % 32;
            for (int i = lane; i < N; i += 32) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    // Determine sorting order for this pair.
                    bool ascending = ((i & size) == 0);
                    Neighbor a = buffer[i];
                    Neighbor b = buffer[ixj];
                    // Swap if out-of-order.
                    if ((a.dist > b.dist) == ascending) {
                        buffer[i] = b;
                        buffer[ixj] = a;
                    }
                }
            }
        }
    }
    __syncwarp(0xffffffff);
}

// -----------------------------------------------------------------------------
// Merge candidate buffer with the warp's private (register) intermediate result.
// The private k best (of size k) is stored across the warp in arrays held in registers:
// each of the 32 threads holds k_per_thread = (k/32) elements in its local array "privateNN".
// The candidate buffer (of variable length candCount, up to k) is in shared memory.
// The two sets are merged into a temporary merge buffer (in shared memory) of size 2*k.
// The merge buffer is padded (if needed) so that its length equals 2*k (a power-of-two).
// A warp-synchronous bitonic sort is applied to the merge buffer and then the first k elements
// (the best candidates) are distributed back to the private result and the new max_distance is updated.
// -----------------------------------------------------------------------------
__device__ void mergeCandidates(Neighbor privateNN[], int k, int lane_id, int kpt,
/// @FIXED
///                                 volatile Neighbor *mergeBuffer, volatile Neighbor *candBuffer,
                                Neighbor *mergeBuffer, Neighbor *candBuffer,
                                int candCount, float *max_dist) {
    // Total elements = k (private intermediate result) + candCount (candidates)
    int total = k + candCount;
    // We pad the merge buffer to 2*k (which is a power-of-two, since k is assumed to be a power-of-two).
    int padTotal = 2 * k;

    // 1) Write the private intermediate results (stored in registers distributed among the warp)
    //    into mergeBuffer indices [0, k).
    // Each thread writes its own kpt elements.
    for (int j = 0; j < kpt; j++) {
        int index = lane_id + j * 32;
        if (index < k) {
            mergeBuffer[index] = privateNN[j];
        }
    }
    // 2) Write the candidate buffer (from shared memory) into mergeBuffer starting at index = k.
    for (int i = lane_id; i < candCount; i += 32) {
        mergeBuffer[k + i] = candBuffer[i];
    }
    // 3) Pad the remaining positions with dummy entries (with FLT_MAX distance).
    for (int i = lane_id; i < (padTotal - total); i += 32) {
        mergeBuffer[total + i] = { -1, FLT_MAX };
    }
    __syncwarp(0xffffffff);

    // 4) Bitonic sort the mergeBuffer array of length padTotal.
    bitonicSortWarp(mergeBuffer, padTotal);
    __syncwarp(0xffffffff);

    // 5) Write back the first k sorted elements into the private result.
    for (int j = 0; j < kpt; j++) {
        int index = lane_id + j * 32;
        if (index < k) {
            privateNN[j] = mergeBuffer[index];
        }
    }
    __syncwarp(0xffffffff);

    // 6) Update the new max_distance.
    // The worst neighbor (k-th nearest) is at mergeBuffer[k-1]. Since k is a multiple of 32,
    // lane 31 holds that value. Broadcast from lane 31.
    float new_max;
    if (lane_id == 31) {
        new_max = mergeBuffer[k - 1].dist;
    }
    new_max = __shfl_sync(0xffffffff, new_max, 31);
    *max_dist = new_max;
    __syncwarp(0xffffffff);
}

// -----------------------------------------------------------------------------
// Final sort of the warp's private intermediate result.
// This function writes the k private results into the merge buffer, sorts them (length = k),
// and then loads them back into the private register arrays. Also update max_dist.
// -----------------------------------------------------------------------------
__device__ void warpSortIntermediate(Neighbor privateNN[], int k, int lane_id, int kpt,
/// @FIXED
///                                      volatile Neighbor *mergeBuffer, float *max_dist) {
                                     Neighbor *mergeBuffer, float *max_dist) {
    int N = k;
    for (int j = 0; j < kpt; j++) {
        int index = lane_id + j * 32;
        if (index < N) {
            mergeBuffer[index] = privateNN[j];
        }
    }
    __syncwarp(0xffffffff);
    bitonicSortWarp(mergeBuffer, N);
    __syncwarp(0xffffffff);
    for (int j = 0; j < kpt; j++) {
        int index = lane_id + j * 32;
        if (index < N) {
            privateNN[j] = mergeBuffer[index];
        }
    }
    __syncwarp(0xffffffff);
    float new_max;
    if (lane_id == 31) {
        new_max = mergeBuffer[N - 1].dist;
    }
    new_max = __shfl_sync(0xffffffff, new_max, 31);
    *max_dist = new_max;
    __syncwarp(0xffffffff);
}

// -----------------------------------------------------------------------------
// The main k-NN kernel.
// Each warp (32 threads) processes one query.
// Shared memory usage per block:
//   - A data cache for a batch of data points (BATCH_SIZE float2 elements)
//   - A candidate buffer for each warp (capacity k, array of Neighbor)
//   - An array of candidate counts (one int per warp)
//   - A merge buffer for each warp (capacity 2*k, array of Neighbor)
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Each warp (32 threads) processes one query.
    int warp_id_in_block = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;

    // Global query index = blockIdx.x * warpsPerBlock + warp_id_in_block.
    int q_idx = blockIdx.x * warpsPerBlock + warp_id_in_block;
    if (q_idx >= query_count)
        return;

    // Load query point.
    float2 q = query[q_idx];

    // Each warp will maintain its private, register-held intermediate result.
    // Each thread holds k/32 candidates.
    int kpt = k / 32;
    Neighbor privateNN[64]; // Maximum space needed if k/32 <= 64.
    for (int i = 0; i < kpt; i++) {
        privateNN[i].idx = -1;
        privateNN[i].dist = FLT_MAX;
    }
    // Initialize max_distance used for candidate filtering.
    float max_distance = FLT_MAX;

    // Partition shared memory.
    // Shared memory layout (per block):
    //   1) Data cache for one batch: BATCH_SIZE float2 elements.
    //   2) Candidate buffers: warpsPerBlock * k Neighbor elements.
    //   3) Candidate counts: warpsPerBlock ints.
    //   4) Merge buffers: warpsPerBlock * (2*k) Neighbor elements.
    extern __shared__ char shared_mem[];
    // Data cache comes first.
    float2 *dataCache = (float2*)shared_mem;
    // Next candidate buffers.
    Neighbor *candBufferBase = (Neighbor*)(dataCache + BATCH_SIZE);
    // Candidate counts.
    int *candCountBase = (int*)(candBufferBase + warpsPerBlock * k);
    // Merge buffers.
    Neighbor *mergeBufferBase = (Neighbor*)(candCountBase + warpsPerBlock);

    // Pointers for this warp.
    Neighbor *myCandBuffer = &candBufferBase[warp_id_in_block * k];
    int *myCandCount = &candCountBase[warp_id_in_block];
    // Each warp gets its own merge buffer (size 2*k).
    /// @FIXED
    /// volatile Neighbor *myMergeBuffer = mergeBufferBase + warp_id_in_block * (2 * k);
    Neighbor *myMergeBuffer = mergeBufferBase + warp_id_in_block * (2 * k);

    // Initialize candidate count to 0 (only one thread in the warp does it).
    if (lane_id == 0)
        *myCandCount = 0;
    __syncwarp(0xffffffff);

    // Process the data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        int current_batch = BATCH_SIZE;
        if (batch_start + current_batch > data_count)
            current_batch = data_count - batch_start;
        // Load this batch from global memory into shared data cache.
        for (int i = threadIdx.x; i < current_batch; i += blockDim.x) {
            dataCache[i] = data[batch_start + i];
        }
        __syncthreads();

        // Each warp processes the batch.
        for (int i = 0; i < current_batch; i++) {
            float2 p = dataCache[i];
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float d = dx * dx + dy * dy;
            // If the computed distance is less than our current worst (max_distance)
            if (d < max_distance) {
                // Use atomicAdd on our candidate count for this warp.
                int pos = atomicAdd(myCandCount, 1);
                // Only store if position is within capacity.
                if (pos < k) {
                    myCandBuffer[pos].idx = batch_start + i;  // Global data index.
                    myCandBuffer[pos].dist = d;
                }
            }
            // Check periodically if the candidate buffer is full.
            if (lane_id == 0 && *myCandCount >= k) {
                // Merge candidates with private intermediate result.
                mergeCandidates(privateNN, k, lane_id, kpt, myMergeBuffer, myCandBuffer, *myCandCount, &max_distance);
                *myCandCount = 0;
            }
            __syncwarp(0xffffffff);
        }
        __syncthreads();
    }

    // After all batches, merge any remaining candidates.
    if (*myCandCount > 0) {
        if (lane_id == 0) {
            mergeCandidates(privateNN, k, lane_id, kpt, myMergeBuffer, myCandBuffer, *myCandCount, &max_distance);
            *myCandCount = 0;
        }
        __syncwarp(0xffffffff);
    }

    // Final sort of the private intermediate result.
    warpSortIntermediate(privateNN, k, lane_id, kpt, myMergeBuffer, &max_distance);

    // Write the k sorted results for this query to global memory.
    // The output array is laid out in row-major order.
    for (int j = 0; j < kpt; j++) {
        int index = lane_id + j * 32;
        if (index < k) {
            result[q_idx * k + index] = std::pair<int, float>(privateNN[j].idx, privateNN[j].dist);
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface function. This function configures kernel launch parameters
// and shared memory size, then launches the kernel.
// Input:
//   query     - pointer to device array of float2 queries (query_count items)
//   data      - pointer to device array of float2 data points (data_count items)
//   result    - pointer to device array of std::pair<int,float> (query_count*k items)
//   k         - number of nearest neighbors to compute (power-of-two between 32 and 1024)
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose number of warps per block.
    // For larger k, we use fewer warps to reduce shared memory usage.
    int warpsPerBlock = (k > 256) ? 4 : 8;
    int threadsPerBlock = warpsPerBlock * 32;
    // Each warp processes one query, so total warps = query_count.
    int totalWarps = query_count;
    int blocks = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Compute shared memory size required per block.
    // Layout:
    //   Data cache: BATCH_SIZE * sizeof(float2)
    //   Candidate buffers: warpsPerBlock * k * sizeof(Neighbor)
    //   Candidate counts: warpsPerBlock * sizeof(int)
    //   Merge buffers: warpsPerBlock * (2 * k) * sizeof(Neighbor)
    size_t shmem = BATCH_SIZE * sizeof(float2)
                 + warpsPerBlock * k * sizeof(Neighbor)
                 + warpsPerBlock * sizeof(int)
                 + warpsPerBlock * (2 * k) * sizeof(Neighbor);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, shmem>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}
