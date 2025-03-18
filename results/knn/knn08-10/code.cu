// This complete CUDA source implements a k‐nearest neighbors (k‐NN)
// algorithm for 2D points using a single warp per query. The overall design
// is as follows:
//
// • Each warp (32 threads) is assigned one query point.
// • The current “intermediate” k‐nearest neighbor result is maintained in registers,
//   distributed in “row–major” order: each thread holds k/32 consecutive candidate (index,distance)
//   pairs that together form a sorted (ascending by distance) array of length k.
// • A per–warp candidate buffer is allocated in shared memory (of size k candidates).
//   As the warp examines batches of data points (uploaded cooperatively into shared memory),
//   each thread computes the squared Euclidean distance from its query point and, if the
//   distance is lower than the current max distance (i.e. the k-th neighbor’s distance),
//   the candidate is appended into the shared buffer.
// • When the candidate buffer becomes full (or after processing the last batch),
//   the candidate buffer is merged with the intermediate result by (a) “swapping” the
//   registers and shared memory arrays, (b) sorting the candidate buffer (now in registers)
//   using a warp–level Bitonic Sort, (c) merging the two sorted arrays into a new sorted
//   distributed array of k candidates, and (d) finally sorting the merged result with
//   Bitonic Sort.  (The “merge” is implemented using a simple linear–scan merge for clarity.)
// • Finally, the warp writes its k sorted nearest neighbors to the global result array.
//
// The distributed Bitonic sort works on an array of “k” elements that is distributed
// in row–major order among the 32 threads of a warp, so that each thread holds k/32 consecutive
// elements. In the Bitonic sort kernels, exchanges between registers are performed via warp–shuffle
// instructions and are restricted to elements with the same intra–thread (local) index.
//
// NOTE: This implementation is optimized for modern NVIDIA GPUs (e.g. A100/H100)
// and uses cooperative warp–level programming. The kernel launch configuration must supply
// enough dynamic shared memory as explained below.
//
// Compile with the latest CUDA toolkit and a C++14–compatible host compiler.

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ----------------------------------------------------------------------
// Data structure for a candidate (neighbor index and squared distance)
struct Candidate {
    int idx;
    float dist;
};

// ----------------------------------------------------------------------
// Swap two Candidate objects.
__device__ inline void swap_candidate(Candidate &a, Candidate &b) {
    Candidate tmp = a;
    a = b;
    b = tmp;
}

// ----------------------------------------------------------------------
// A simple insertion sort to sort a small array of Candidates (ascending by distance).
// This is used to sort each thread's local candidate array.
__device__ inline void local_insertion_sort(Candidate arr[], int n) {
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

// ----------------------------------------------------------------------
// Given a distributed candidate array stored in registers in row–major order,
// this helper returns the Candidate at global index "global_idx".
// Each thread holds an array of "local_n" elements; the global array length is (32 * local_n = k).
// We assume all threads in the warp execute this function synchronously.
__device__ inline Candidate getDistributedCandidate(const Candidate reg[], int local_n, int global_idx) {
    int lane = threadIdx.x & 31; // lane within warp
    int owner = global_idx / local_n;
    int pos = global_idx % local_n;
    Candidate val;
    // Use warp shuffle to broadcast the element at index "pos" from thread "owner"
    val.idx  = __shfl_sync(0xFFFFFFFF, reg[pos].idx, owner);
    val.dist = __shfl_sync(0xFFFFFFFF, reg[pos].dist, owner);
    return val;
}

// ----------------------------------------------------------------------
// This function performs a Bitonic Sort on a distributed array of "n" elements,
// where n = 32 * local_n, stored in row–major order among the warp threads.
// Each thread holds a local array "reg" of "local_n" Candidates. The algorithm
// first sorts the local arrays, then performs a global bitonic sort with pair–wise
// exchanges using warp shuffle instructions. (Exchanges are done only on elements
// with the same local index.)
__device__ inline void warp_sort_distributed(Candidate reg[], int n, int local_n) {
    // First, each thread sorts its own local array.
    local_insertion_sort(reg, local_n);
    // Now n = 32 * local_n.
    // The Bitonic sort algorithm (pseudocode provided) is applied in a distributed manner.
    // Each thread iterates over its local elements and participates in exchanges when
    // its global index (computed as: (lane * local_n + local_index)) satisfies the condition.
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i_local = 0; i_local < local_n; i_local++) {
                int global_idx = (threadIdx.x & 31) * local_n + i_local;
                int partner = global_idx ^ j;  // partner index
                if (partner > global_idx) {
                    // Only perform exchange if the two elements reside in the same register slot,
                    // i.e. if (partner % local_n) equals our local index.
                    if ((partner % local_n) == i_local) {
                        Candidate myVal = reg[i_local];
                        int partner_lane = partner / local_n;
                        Candidate otherVal;
                        otherVal.idx  = __shfl_sync(0xFFFFFFFF, reg[i_local].idx, partner_lane);
                        otherVal.dist = __shfl_sync(0xFFFFFFFF, reg[i_local].dist, partner_lane);
                        bool swapFlag = false;
                        // Determine sort direction: if (global_idx & k)==0, ascending sort; else descending.
                        if ((global_idx & k) == 0) {
                            if (myVal.dist > otherVal.dist)
                                swapFlag = true;
                        } else {
                            if (myVal.dist < otherVal.dist)
                                swapFlag = true;
                        }
                        if (swapFlag) {
                            // In a compare–exchange step of Bitonic Sort,
                            // the new value for this position is the minimum (for ascending order)
                            // or maximum (for descending order) of the two.
                            Candidate newVal;
                            if ((global_idx & k) == 0)
                                newVal = (myVal.dist < otherVal.dist) ? myVal : otherVal;
                            else
                                newVal = (myVal.dist > otherVal.dist) ? myVal : otherVal;
                            reg[i_local] = newVal;
                        }
                    }
                }
            }
            // Synchronize lanes in the warp.
            __syncwarp();
        }
    }
}

// ----------------------------------------------------------------------
// A simple wrapper for Bitonic sort on a local array in a single thread.
// (Not needed for global sort; provided for clarity.)
__device__ inline void warp_bitonic_sort(Candidate arr[], int local_n) {
    local_insertion_sort(arr, local_n);
}

// ----------------------------------------------------------------------
// This function merges two sorted distributed arrays (each of length k, where k = 32*local_n)
// into a new sorted distributed array (keeps only the k smallest elements).
// The two sorted arrays are:
//    - "intermediate": the current warp’s intermediate result (in registers)
//    - A second array from a candidate buffer that has been collected in shared memory.
// For clarity (and simplicity), we implement the merge via a linear scan that, for each
// global output index, selects the appropriate candidate from the union.
// The merge is performed cooperatively by the warp using warp–shuffle operations.
__device__ inline void warp_merge_shared(
    Candidate intermediate[],         // distributed array in registers (length = k, sorted)
    Candidate *warpBuffer,            // pointer to the warp's candidate buffer in shared memory
    int candidate_count,              // number of candidates in the warp buffer (may be < k)
    int k,                            // full capacity (k must be a multiple of 32)
    int local_n                       // k/32 (number of elements per thread)
) {
    int lane = threadIdx.x & 31;
    // Each thread copies its share from the candidate buffer (row–major order).
    Candidate buffer_reg[32];  // local copy (max local_n assumed <= 32)
    for (int i = 0; i < local_n; i++) {
        int global_idx = lane * local_n + i;
        if (global_idx < candidate_count) {
            buffer_reg[i] = warpBuffer[global_idx];
        } else {
            buffer_reg[i].dist = FLT_MAX;
            buffer_reg[i].idx  = -1;
        }
    }
    // Sort the candidate buffer copy (distributed array) using warp sort.
    warp_sort_distributed(buffer_reg, k, local_n);
    // Merge the two sorted distributed arrays into a new intermediate result.
    // Each thread will compute its local output (i.e. the elements corresponding to
    // global indices: (lane * local_n) through (lane * local_n + local_n - 1)).
    Candidate merged[32];
    for (int i = 0; i < local_n; i++) {
        int global_idx = lane * local_n + i;
        int count = 0;
        int ptrA = 0, ptrB = 0;
        Candidate cand;
        // Perform a simple linear merge (scan) until "global_idx+1" elements have been chosen.
        while (count <= global_idx) {
            Candidate a = (ptrA < k) ? getDistributedCandidate(buffer_reg, local_n, ptrA)
                                     : Candidate{-1, FLT_MAX};
            Candidate b = (ptrB < k) ? getDistributedCandidate(intermediate, local_n, ptrB)
                                     : Candidate{-1, FLT_MAX};
            if (a.dist <= b.dist) {
                cand = a;
                ptrA++;
            } else {
                cand = b;
                ptrB++;
            }
            count++;
        }
        merged[i] = cand;
    }
    // Write the merged values back into the intermediate distributed array.
    for (int i = 0; i < local_n; i++) {
        intermediate[i] = merged[i];
    }
    // Final sort of the new intermediate array.
    warp_sort_distributed(intermediate, k, local_n);
}

// ----------------------------------------------------------------------
// The main k-NN kernel. Each warp (32 threads) processes one query and computes
// its k nearest neighbors from a (large) data set. Data points are processed in batches,
// with each batch loaded cooperatively into shared memory.
__global__ void knn_kernel(
    const float2 *query, int query_count,
    const float2 *data, int data_count,
    std::pair<int, float> *result,
    int k  // k is a power of two between 32 and 1024 (inclusive)
) {
    // Each warp (32 threads) is assigned one query.
    int warp_id_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x / 32;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    if (global_warp_id >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[global_warp_id];

    // Each thread holds k/32 candidate entries in registers.
    int local_n = k / 32;
    Candidate knn[32];  // local storage; maximum local_n is assumed <= 32.
    // Initialize intermediate result with "infinite" distances.
    for (int i = 0; i < local_n; i++) {
        knn[i].dist = FLT_MAX;
        knn[i].idx  = -1;
    }
    // At all times the distributed array (of total length k) stored in registers
    // is kept sorted in ascending order.

    //--------------------------------------------------------------------
    // Shared memory layout (dynamically allocated):
    // [0, warps_per_block * k * sizeof(Candidate)): per–warp candidate buffers.
    // [warps_per_block * k * sizeof(Candidate), warps_per_block * k * sizeof(Candidate) + warps_per_block * sizeof(int)):
    //    per–warp candidate counts.
    // After that, a data tile buffer of TILE_SIZE elements (float2) is allocated.
    extern __shared__ char shared_mem[];
    int buffer_area = warps_per_block * k;
    Candidate *warpBuffer_all = (Candidate*)shared_mem;
    int *warpCandidateCount = (int*)(shared_mem + buffer_area * sizeof(Candidate));
    // Define tile size (number of data points processed per batch).
    const int TILE_SIZE = 256;
    float2 *dataTile = (float2*)(shared_mem + buffer_area * sizeof(Candidate) + warps_per_block * sizeof(int));

    // Initialize candidate count for this warp (only one thread in warp does it).
    if (lane == 0) {
        warpCandidateCount[warp_id_in_block] = 0;
    }
    __syncwarp();

    // Process the data in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_SIZE) {
        int batch_size = (batch_start + TILE_SIZE <= data_count) ? TILE_SIZE : (data_count - batch_start);
        // Load the current batch (tile) of data points into shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            dataTile[i] = data[batch_start + i];
        }
        __syncthreads();  // ensure whole block has loaded the tile

        // Each warp processes the tile cooperatively.
        // Each thread in the warp examines a subset of the tile with stride 32.
        for (int i = lane; i < batch_size; i += 32) {
            float2 d_point = dataTile[i];
            float dx = d_point.x - q.x;
            float dy = d_point.y - q.y;
            float dist = dx * dx + dy * dy;
            // Determine the current maximum distance from the intermediate result.
            // The maximum is the last element of the sorted distributed array.
            float current_max;
            if (lane == 31) {
                current_max = knn[local_n - 1].dist;
            }
            current_max = __shfl_sync(0xFFFFFFFF, current_max, 31);
            if (dist < current_max) {
                // Prepare candidate.
                Candidate cand;
                cand.idx  = batch_start + i;  // Global index of data point.
                cand.dist = dist;
                // Append the candidate into the warp's candidate buffer in shared memory.
                int pos = atomicAdd(&warpCandidateCount[warp_id_in_block], 1);
                warpBuffer_all[warp_id_in_block * k + pos] = cand;
                // If the buffer becomes full, merge it immediately with the intermediate result.
                if (pos + 1 == k) {
                    warp_merge_shared(knn, warpBuffer_all + warp_id_in_block * k,
                                        k, k, local_n);
                    // Reset the candidate count.
                    if (lane == 0) {
                        warpCandidateCount[warp_id_in_block] = 0;
                    }
                    __syncwarp();
                }
            }
        }
        __syncthreads();  // ensure tile processing is complete for all threads.
    }
    // After the last batch, if there remain candidates in the buffer, merge them.
    int candCount = warpCandidateCount[warp_id_in_block];
    if (candCount > 0) {
        warp_merge_shared(knn, warpBuffer_all + warp_id_in_block * k, candCount, k, local_n);
    }
    __syncwarp();

    // Write the final sorted k nearest neighbors to global memory.
    // The distributed array "knn" (of total length k) is stored in row–major order:
    // thread "lane" contains candidates for global indices [lane*local_n, lane*local_n+local_n-1].
    int query_base = global_warp_id * k;
    for (int i = 0; i < local_n; i++) {
        int out_idx = query_base + (lane * local_n + i);
        result[out_idx] = std::make_pair(knn[i].idx, knn[i].dist);
    }
}

// ----------------------------------------------------------------------
// Host function that launches the k-NN kernel.
// The input arrays "query", "data", and "result" are assumed to be allocated using cudaMalloc().
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Choose an appropriate block size.
    int blockSize = 128;  // number of threads per block (must be a multiple of 32)
    int warpsPerBlock = blockSize / 32;
    // Each warp processes one query. Compute grid size accordingly.
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Define tile size (for data batching) consistent with knn_kernel.
    const int TILE_SIZE = 256;
    // Compute the shared memory size (in bytes):
    //   - Each block has warpsPerBlock candidate buffers, each of size k Candidates.
    //   - Plus warpsPerBlock candidate count integers.
    //   - Plus a data tile buffer of TILE_SIZE float2 elements.
    size_t sharedMemSize = warpsPerBlock * k * sizeof(Candidate)
                         + warpsPerBlock * sizeof(int)
                         + TILE_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<numBlocks, blockSize, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // (Error checking omitted for brevity.)
}