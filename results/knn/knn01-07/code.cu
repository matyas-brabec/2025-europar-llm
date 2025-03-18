#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

//---------------------------------------------------------------------
// This CUDA code implements a brute‐force k-nearest neighbors (k-NN)
// search for 2D points using squared Euclidean distance. The kernel
// is optimized for modern NVIDIA GPUs (e.g. A100/H100) and is designed
// to process many queries concurrently by mapping one query per warp.
// Each warp (32 threads) cooperatively scans all data points, with each
// thread processing a strided subset of the data points. Each thread
// maintains a local candidate list (of size L = k/32) of the closest points
// encountered, and then the warp gathers these candidates and performs a
// final sort (using bitonic sort) to produce a fully sorted list of k
// nearest neighbors for its query.
//---------------------------------------------------------------------

// Structure to hold a candidate neighbor (data point index and its distance).
struct Candidate {
    int idx;
    float dist;
};

//---------------------------------------------------------------------
// Device function: Bitonic sort for an array 'arr' of length 'n'.
// This is a simple in-place bitonic sort implemented in a single thread.
// It sorts the array in ascending order with respect to Candidate.dist.
// 'n' is assumed to be a power of two (as k is a power-of-two between 32 and 1024).
//---------------------------------------------------------------------
__device__ void bitonic_sort(Candidate* arr, int n) {
    // Outer loop: size of the bitonic sequence; doubles each iteration.
    for (int size = 2; size <= n; size <<= 1) {
        // Inner loop: stride controls the distance over which comparisons are made.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Loop over the entire array and compare-swap elements as needed.
            for (int i = 0; i < n; i++) {
                int j = i ^ stride;  // Bitwise XOR to compute the paired index.
                if (j > i) {
                    // Determine the sorting order for this subsequence:
                    // If the i-th bit (corresponding to 'size') is 0, sort ascending;
                    // otherwise sort descending.
                    bool ascending = ((i & size) == 0);
                    // Compare and swap elements if they are out of order.
                    if ((ascending && arr[i].dist > arr[j].dist) ||
                        (!ascending && arr[i].dist < arr[j].dist)) {
                        // Swap arr[i] and arr[j]
                        Candidate temp = arr[i];
                        arr[i] = arr[j];
                        arr[j] = temp;
                    }
                }
            } // end for i
        } // end for stride
    } // end for size
}

//---------------------------------------------------------------------
// Kernel: knn_kernel
// Each warp processes one query point. Each thread in a warp scans
// through the data array (in a coalesced manner) computing the squared
// Euclidean distance from the query point. Each thread maintains a local
// candidate list of size L = k/32 (all threads together produce k candidates).
// After processing all data, lane 0 in each warp gathers the candidate lists
// from all threads via warp-shuffle, sorts the combined list (of k candidates)
// using bitonic sort, and writes the sorted results to global memory.
//---------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp processes one query.
    // Get the warp identifier (within block) and lane index.
    int warpId = threadIdx.x >> 5;      // threadIdx.x / 32
    int lane = threadIdx.x & 31;          // threadIdx.x % 32

    // Calculate the global query index based on warp id.
    int warpsPerBlock = blockDim.x >> 5;  // blockDim.x / 32
    int query_id = blockIdx.x * warpsPerBlock + warpId;
    if (query_id >= query_count)
        return;

    // Load the query point into registers.
    float2 q = query[query_id];

    // Determine the local candidate list size per thread.
    // Since k is assumed to be a power of two and >= 32, and warp size is 32,
    // we have L = k / 32 candidates per thread.
    int L = k >> 5;  // equivalent to k/32

    // Each thread maintains its local candidate list.
    Candidate local_candidates[64]; // Maximum L is 1024/32 = 32; use 64 to be safe.
    #pragma unroll
    for (int i = 0; i < L; i++) {
        local_candidates[i].dist = FLT_MAX;
        local_candidates[i].idx  = -1;
    }

    // Loop over all data points in a strided manner.
    // Each thread in the warp processes indices: lane, lane+32, lane+64, ...
    for (int j = lane; j < data_count; j += 32) {
        // Load data point and compute squared Euclidean distance.
        float2 p = data[j];
        float dx = q.x - p.x;
        float dy = q.y - p.y;
        float dist = dx * dx + dy * dy;

        // Check if this candidate should be included in the local candidate list.
        // Find the current worst (maximum) distance in the local list.
        int max_index = 0;
        float current_max = local_candidates[0].dist;
        #pragma unroll
        for (int i = 1; i < L; i++) {
            float d = local_candidates[i].dist;
            if (d > current_max) {
                current_max = d;
                max_index = i;
            }
        }
        // If the new candidate is closer than the worst in the list, update it.
        if (dist < current_max) {
            local_candidates[max_index].dist = dist;
            local_candidates[max_index].idx  = j;
        }
    }

    // (Optional) Sort each thread's local candidate list in ascending order.
    // Since L is small (<=32), a simple insertion sort is efficient.
    for (int i = 1; i < L; i++) {
        Candidate key = local_candidates[i];
        int j = i - 1;
        while (j >= 0 && local_candidates[j].dist > key.dist) {
            local_candidates[j + 1] = local_candidates[j];
            j--;
        }
        local_candidates[j + 1] = key;
    }

    // Now, each thread has L sorted candidates.
    // The warp collectively holds k = 32 * L candidates.
    // Use warp shuffle to gather these candidates into one contiguous array.
    // We have the guarantee that k is a power-of-two and divisible by 32.
    // Only lane 0 in the warp will perform the final merge and sorting.
    if (lane == 0) {
        // Allocate a local array to hold all k candidates.
        // Maximum k is 1024; we allocate a fixed-size array.
        Candidate knn_candidates[1024];
        // Loop over the candidate list positions L per thread and gather from all 32 lanes.
        for (int l = 0; l < L; l++) {
            // For each thread in the warp, retrieve its l-th candidate.
            for (int r = 0; r < 32; r++) {
                int gather_index = r + l * 32; // index into the final merged array
                // Use warp shuffle to get the candidate from lane 'r'
                // __shfl_sync returns a 32-bit value; we do it for each field.
                int cand_idx = __shfl_sync(0xffffffff, local_candidates[l].idx, r);
                float cand_dist = __shfl_sync(0xffffffff, local_candidates[l].dist, r);
                knn_candidates[gather_index].idx = cand_idx;
                knn_candidates[gather_index].dist = cand_dist;
            }
        }

        // Perform a bitonic sort on the combined candidate array.
        // The array length is 'k', which is a power-of-two.
        bitonic_sort(knn_candidates, k);

        // Write the sorted k-nearest neighbors (ascending by distance) to the output.
        int out_base = query_id * k;
        for (int i = 0; i < k; i++) {
            result[out_base + i] = std::pair<int, float>(knn_candidates[i].idx, knn_candidates[i].dist);
        }
    }
}

//---------------------------------------------------------------------
// Host function: run_knn
// This function sets up the grid and block dimensions and launches
// the CUDA kernel. It expects that 'query', 'data', and 'result' arrays
// have been allocated on the device using cudaMalloc.
//---------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                          const float2 *data, int data_count,
                          std::pair<int, float> *result, int k)
{
    // Choose 1024 threads per block (i.e. 32 warps per block).
    int threadsPerBlock = 1024;
    int warpsPerBlock = threadsPerBlock >> 5; // 1024 / 32 = 32 warps per block.
    // Each warp processes one query.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock>>>(query, query_count, data, data_count, result, k);

    // Synchronize the device (error checking omitted for brevity).
    cudaDeviceSynchronize();
}
