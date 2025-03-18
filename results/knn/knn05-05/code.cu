#include <cuda_runtime.h>
#include <utility>     // for std::pair

// Define simple min/max macros.
#ifndef MIN
  #define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
  #define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// A very large float value used for “infinity.”
#define INF 1e30f

//-----------------------------------------------------------------------------
// Data structure to hold a candidate neighbor.
// It contains the index of the data point and its squared Euclidean distance.
struct Neighbor {
    int idx;
    float dist;
};

//-----------------------------------------------------------------------------
// Device helper function prototypes.
/// @FIXED
/// __device__ void warp_sort(volatile Neighbor *arr, int n, int lane);
__device__ void warp_sort(Neighbor *arr, int n, int lane);
__device__ int2 merge_path(int pos, const Neighbor *A, const Neighbor *B, int k);
/// @FIXED
/// __device__ void merge_buffers(volatile Neighbor *A, volatile Neighbor *B, int k, int lane);
__device__ void merge_buffers(Neighbor *A, Neighbor *B, int k, int lane);

//-----------------------------------------------------------------------------
// warp_sort: In-warp bitonic sort of an array of Neighbor.
// The array "arr" (of length n) is stored in shared memory and distributed among
// the 32 threads of the warp (each thread handles indices with stride 32).
// Assumes n is a power of two.
/// @FIXED
/// __device__ void warp_sort(volatile Neighbor *arr, int n, int lane) {
__device__ void warp_sort(Neighbor *arr, int n, int lane) {
    // Bitonic sort.
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes multiple elements with stride 32.
            for (int i = lane; i < n; i += 32) {
                int partner = i ^ stride;
                if (partner > i && partner < n) {
                    // Determine sort direction from bit of index.
                    bool ascending = ((i & size) == 0);
                    if ((ascending && arr[i].dist > arr[partner].dist) ||
                        (!ascending && arr[i].dist < arr[partner].dist)) {
                        // Swap the two elements.
                        Neighbor temp = arr[i];
                        arr[i] = arr[partner];
                        arr[partner] = temp;
                    }
                }
            }
            __syncwarp();
        }
    }
}

//-----------------------------------------------------------------------------
// merge_path: Given two sorted arrays A and B (each of length k) and an
// output position pos (with 0 <= pos <= k), compute the partition indices for
// merging. Returns an int2 (i,j) where i elements are taken from A and j from B.
// This implements the “merge path” binary search used in parallel merging.
__device__ int2 merge_path(int pos, const Neighbor *A, const Neighbor *B, int k) {
    int low = MAX(0, pos - k);
    int high = MIN(pos, k);
    while (low < high) {
        int mid = (low + high) >> 1;
        float A_val = A[mid].dist;
        float B_val = B[pos - mid - 1].dist;
        if (A_val <= B_val)
            low = mid + 1;
        else
            high = mid;
    }
    int i = low;
    int j = pos - i;
    return make_int2(i, j);
}

//-----------------------------------------------------------------------------
// merge_buffers: Merge two sorted arrays A and B (each of length k) stored in
// shared memory. The union of these arrays (of size 2k) is merged and the
// k smallest elements (i.e. the best candidates) are stored back into A.
// This function is executed cooperatively by the 32 threads of a warp.
// A and B are passed as pointers offset to the warp’s region.
/// @FIXED
/// __device__ void merge_buffers(volatile Neighbor *A, volatile Neighbor *B, int k, int lane) {
__device__ void merge_buffers(Neighbor *A, Neighbor *B, int k, int lane) {
    // Determine how many output elements each thread is in charge of.
    int outPerThread = k >> 5;  // k/32, k is guaranteed a power-of-two (min 32).
    int start = lane * outPerThread;
    int end = (lane + 1) * outPerThread;

    // Each thread will compute its segment of the merged output.
    Neighbor localOut[32];  // Maximum outPerThread is 32 when k==1024.
    int count = 0;
    for (int pos = start; pos < end; pos++) {
        int2 part = merge_path(pos, A, B, k);
        int i_idx = part.x;
        int j_idx = part.y;
        float a_val = (i_idx < k) ? A[i_idx].dist : INF;
        float b_val = (j_idx < k) ? B[j_idx].dist : INF;
        if (i_idx < k && (j_idx >= k || a_val <= b_val)) {
            localOut[count++] = A[i_idx];
        } else {
            localOut[count++] = B[j_idx];
        }
    }
    __syncwarp();

    // Write each thread’s merged segment into a temporary slot.
    for (int pos = start, t = 0; pos < end; pos++, t++) {
        B[pos] = localOut[t];
    }
    __syncwarp();

    // Copy the merged result from the temporary buffer back to A.
    for (int pos = start; pos < end; pos++) {
        A[pos] = B[pos];
    }
    __syncwarp();
}

//-----------------------------------------------------------------------------
// knn_kernel: Each warp (32 threads) processes one query. The warp maintains
// two candidate lists:
//   1) A private (per-warp) intermediate result stored in shared memory and
//      kept sorted (lowest squared distance first).
//   2) A candidate buffer (in shared memory) where new candidate points (from
//      the current batch) are appended.
// Data points are processed in batches: first each block loads a batch of data
// points (of size BATCH_SIZE) from global memory into shared memory. Then, each
// warp computes distances and appends qualified candidates.
// When the candidate buffer fills (reaches capacity k), a merge is done to update
// the intermediate result.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           Neighbor *result, int k) {
    // Each warp handles one query.
    int warp_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (warp_id >= query_count)
        return;  // Extra warps simply exit.

    // Determine the number of warps per block.
    int numWarps = blockDim.x >> 5;

    // Shared memory layout:
    // [0, candidateBufferBytes): candidate buffers for all warps, size = numWarps * k * sizeof(Neighbor)
    // Next: intermediate results for all warps, size = numWarps * k * sizeof(Neighbor)
    // Next: candidate counts for all warps, size = numWarps * sizeof(int)
    // Next: batch data points, size = BATCH_SIZE * sizeof(float2)
    extern __shared__ char sharedMem[];

    size_t candidateBufferBytes = numWarps * k * sizeof(Neighbor);
    Neighbor *candBuffer = (Neighbor *)sharedMem;

    size_t interResBytes = numWarps * k * sizeof(Neighbor);
    Neighbor *interRes = (Neighbor *)((char*)sharedMem + candidateBufferBytes);

    size_t candCountBytes = numWarps * sizeof(int);
    int *candCount = (int *)((char*)sharedMem + candidateBufferBytes + interResBytes);

    const int BATCH_SIZE = 1024;
    float2 *shData = (float2 *)((char*)sharedMem + candidateBufferBytes + interResBytes + candCountBytes);

    // Get warp index within the block.
    int warpInBlock = threadIdx.x >> 5;

    // Initialize candidate count for this warp.
    if (lane == 0)
        candCount[warpInBlock] = 0;
    __syncwarp();

    // Load the query point (a float2) for this warp from global memory.
    float2 qpt = query[warp_id];
    __syncwarp();

    // Initialize the intermediate result with the first k data points.
    // Each thread in the warp loads k/32 candidates.
    int chunk = k >> 5;  // Because k is divisible by 32.
    for (int i = 0; i < chunk; i++) {
        int index = lane + i * 32;  // Distributed indices.
        if (index < k) {
            float2 dpt = data[index];
            float dx = dpt.x - qpt.x;
            float dy = dpt.y - qpt.y;
            float dist = dx * dx + dy * dy;
            interRes[warp_id * k + index].idx = index;
            interRes[warp_id * k + index].dist = dist;
        }
    }
    __syncwarp();

    // Sort the intermediate result for this query using in-warp bitonic sort.
    warp_sort(interRes + warp_id * k, k, lane);
    __syncwarp();

    // Process the remaining data points in batches.
    for (int batchStart = k; batchStart < data_count; batchStart += BATCH_SIZE) {
        int batchSize = ((data_count - batchStart) < BATCH_SIZE) ? (data_count - batchStart) : BATCH_SIZE;
        // Cooperative load: each block loads parts of the batch from global memory.
        for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
            shData[i] = data[batchStart + i];
        }
        __syncthreads();  // Ensure entire batch is loaded.

        // Each warp processes the batch using its 32 threads.
        for (int i = lane; i < batchSize; i += 32) {
            float2 pt = shData[i];
            float dx = pt.x - qpt.x;
            float dy = pt.y - qpt.y;
            float dist = dx * dx + dy * dy;
            // The worst (largest) distance in the sorted intermediate result is at index k-1.
            Neighbor worst = interRes[warp_id * k + k - 1];
            if (dist < worst.dist) {
                Neighbor cand;
                cand.idx = batchStart + i;
                cand.dist = dist;
                // Append the candidate to the candidate buffer (using atomicAdd to obtain a unique slot).
                int pos = atomicAdd(&candCount[warpInBlock], 1);
                if (pos < k)
                    candBuffer[warp_id * k + pos] = cand;
                // If the candidate buffer reaches capacity, merge it with the intermediate result.
                if (pos + 1 == k) {
                    merge_buffers(interRes + warp_id * k, candBuffer + warp_id * k, k, lane);
                    if (lane == 0)
                        candCount[warpInBlock] = 0;
                    __syncwarp();
                }
            }
        }
        __syncthreads();
    }

    // After finishing all batches, if there remain candidates in the candidate buffer, merge them.
    int remaining = candCount[warpInBlock];
    if (remaining > 0) {
        // Pad the candidate buffer with INF values.
        for (int i = lane; i < k; i += 32) {
            if (i >= remaining) {
                candBuffer[warp_id * k + i].idx = -1;
                candBuffer[warp_id * k + i].dist = INF;
            }
        }
        __syncwarp();
        merge_buffers(interRes + warp_id * k, candBuffer + warp_id * k, k, lane);
    }
    __syncwarp();

    // Write the sorted intermediate result (the k nearest neighbors) for this query to global memory.
    for (int i = lane; i < k; i += 32) {
        /// @FIXED
        /// result[warp_id * k + i] = std::pair<int, float>(
        result[warp_id * k + i] = Neighbor(
            interRes[warp_id * k + i].idx,
            interRes[warp_id * k + i].dist
        );
    }
}

//-----------------------------------------------------------------------------
// run_knn: Host interface function to launch the k-NN kernel. It assumes that
// the arrays query, data, and result have been allocated on the device (using cudaMalloc).
// The kernel launches such that each warp (32 threads) processes one query.
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k) {
    // Choose 256 threads per block (i.e. 8 warps per block).
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock >> 5;  // 256/32 = 8.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Determine shared memory size per block.
    // Layout:
    //  candidate buffer: warpsPerBlock * k * sizeof(Neighbor)
    //  intermediate result: warpsPerBlock * k * sizeof(Neighbor)
    //  candidate counts: warpsPerBlock * sizeof(int)
    //  batch data points: 1024 * sizeof(float2)
    const int BATCH_SIZE = 1024;
    size_t sharedMemSize = warpsPerBlock * k * sizeof(Neighbor)
                         + warpsPerBlock * k * sizeof(Neighbor)
                         + warpsPerBlock * sizeof(int)
                         + BATCH_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count,
                                                           data, data_count,
                                                           (Neighbor*)result, k);
}
