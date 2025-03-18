#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// --------------------------------------------------------------------------------
// This CUDA kernel implements k–nearest neighbors (k‐NN) search for 2D points.
// Each query point is processed by one warp (32 threads) and its “private” 
// candidate list of the best k (lowest squared Euclidean distance) data points 
// is maintained in shared memory. The input data points are processed in batches
// (loaded into shared memory) to amortize global memory latency.
// 
// For each batch, each thread in the warp computes distances for a subset of 
// the data points in the batch (using a strided loop) and sorts its local set 
// of candidate pairs (data index, distance). Then, the warp cooperatively 
// “merges” these local candidates into the warp’s global candidate list – stored 
// in shared memory – via a serialized update performed by lane 0 after 
// broadcasting each candidate from each lane. At the end, lane 0 sorts the warp’s 
// candidate list (of size k) and writes the sorted results (in increasing distance)
// to global memory.
// 
// This implementation assumes k is a power-of-two between 32 and 1024, inclusive.
// The distance is computed as squared Euclidean distance and ties may be resolved arbitrarily.
// --------------------------------------------------------------------------------

#define WARP_SIZE 32
// Define the batch size: number of data points loaded into shared memory per iteration.
#define BATCH_SIZE 1024

// -----------------------------------------------------------------------------
// __device__ function: insertion sort for a local array of candidate pairs in registers.
// This sorts the array in ascending order (by the 'second' field = distance).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void insertion_sort_local(std::pair<int, float>* arr, int n)
{
    for (int i = 1; i < n; i++) {
        std::pair<int, float> key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].second > key.second) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// -----------------------------------------------------------------------------
// __device__ function: simple insertion sort for an array of candidate pairs.
// This is used to sort the global candidate list (size k), which is stored in shared memory.
// -----------------------------------------------------------------------------
__device__ void insertion_sort(std::pair<int, float>* arr, int n)
{
    // A straightforward insertion sort (n is at most 1024).
    for (int i = 1; i < n; i++) {
        std::pair<int, float> key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j].second > key.second) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// -----------------------------------------------------------------------------
// __device__ function: update the unsorted candidate list with a new candidate.
// The candidate list (of size k) is stored in shared memory for the warp.
// We scan the list to find its worst (largest distance) candidate and replace it if needed.
// This function is called by exactly one thread (lane 0) in the warp.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void update_candidate_unsorted(std::pair<int, float>* cand_list, int k, int cand_index, float cand_dist)
{
    // Find the candidate with the maximum (worst) distance.
    float worst = cand_list[0].second;
    int worst_idx = 0;
    for (int i = 1; i < k; i++) {
        float d = cand_list[i].second;
        if (d > worst) {
            worst = d;
            worst_idx = i;
        }
    }
    // If the new candidate is better than the current worst, replace it.
    if (cand_dist < worst) {
        cand_list[worst_idx].first = cand_index;
        cand_list[worst_idx].second = cand_dist;
    }
}

// -----------------------------------------------------------------------------
// The main k-NN kernel. Each warp processes one query point.
// Shared memory layout (dynamically allocated):
//   - The first (BATCH_SIZE) elements (of type float2) hold a batch of data points.
//   - The remaining memory is used to store warp–private candidate lists for queries.
//      Each warp gets k std::pair<int, float> entries.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Determine warp and lane within the warp.
    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    // Each warp processes one query; compute global query index.
    int queryId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (queryId >= query_count)
        return;
    
    // Load the query point for this warp.
    float2 q = query[queryId];
    
    // Partition dynamic shared memory:
    // shData: holds a batch of data points (BATCH_SIZE elements).
    extern __shared__ char smem[];
    float2 *shData = (float2*)smem;
    // shCandidates follows shData.
    std::pair<int, float> *shCandidates = (std::pair<int, float>*)
                                         (smem + BATCH_SIZE * sizeof(float2));
    // Each warp has its own candidate list section.
    int warpCandidateOffset = warpIdInBlock * k;
    std::pair<int, float>* cand_list = &shCandidates[warpCandidateOffset];
    
    // Initialize the candidate list with "dummy" values.
    // We use a simple strided loop over the k entries.
    for (int j = lane; j < k; j += WARP_SIZE) {
        cand_list[j].first = -1;
        cand_list[j].second = FLT_MAX;
    }
    
    // Process data in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE)
    {
        // Determine how many data points in this batch.
        int batch_count = BATCH_SIZE;
        if (batch_start + BATCH_SIZE > data_count) {
            batch_count = data_count - batch_start;
        }
        
        // Load the current batch of data points from global memory into shared memory.
        // Each thread in the block loads one or more points.
        for (int i = threadIdx.x; i < batch_count; i += blockDim.x) {
            shData[i] = data[batch_start + i];
        }
        __syncthreads(); // Ensure the batch is fully loaded.
        
        // Each warp processes the batch.
        // Each thread in the warp will process a subset of the batch points in a strided loop.
        // For BATCH_SIZE points and a warp size of 32, each thread processes about BATCH_SIZE/32 points.
        const int maxLocal = BATCH_SIZE / WARP_SIZE;  // Typically 1024/32 = 32.
        std::pair<int, float> localCandidates[maxLocal];
        int localCount = 0;
        for (int i = lane; i < batch_count; i += WARP_SIZE) {
            // Compute squared Euclidean distance.
            float2 pt = shData[i];
            float dx = pt.x - q.x;
            float dy = pt.y - q.y;
            float dist = dx * dx + dy * dy;
            localCandidates[localCount].first = batch_start + i; // Global index.
            localCandidates[localCount].second = dist;
            localCount++;
        }
        // Sort the local candidate list locally (in registers) in ascending order.
        insertion_sort_local(localCandidates, localCount);
        
        // Now, merge the local candidates from each lane into the warp’s
        // global candidate list (stored in shared memory).
        // We iterate over each candidate index (up to maxLocal) and for each lane r,
        // we broadcast its candidate (if exists) and update the candidate list.
        for (int i = 0; i < maxLocal; i++) {
            // Loop over all lanes in the warp in a fixed order.
            for (int r = 0; r < WARP_SIZE; r++) {
                // Get local count from lane r.
                int rLocalCount = __shfl_sync(0xFFFFFFFF, localCount, r);
                // Broadcast candidate from lane r if it has a candidate at index i.
                std::pair<int, float> cand;
                if (i < rLocalCount) {
                    cand.first  = __shfl_sync(0xFFFFFFFF, localCandidates[i].first, r);
                    cand.second = __shfl_sync(0xFFFFFFFF, localCandidates[i].second, r);
                } else {
                    // If lane r has no candidate at this index, set as inactive.
                    cand.first  = -1;
                    cand.second = FLT_MAX;
                }
                // Let lane 0 perform the update to the global candidate list.
                if (lane == 0) {
                    update_candidate_unsorted(cand_list, k, cand.first, cand.second);
                }
                __syncwarp(0xFFFFFFFF); // Ensure update is visible to all warp lanes.
            }
        }
        __syncthreads();  // Ensure candidate list updates complete before next batch.
    } // end for each batch

    // After processing all batches, the candidate list (of size k) in shCandidates
    // contains the best k candidates for this query; however, they are unsorted.
    // Now, have lane 0 sort the candidate list and write the result to global memory.
    if (lane == 0) {
        // Use a local array (max k=1024 elements) to sort.
        std::pair<int, float> localGlobal[1024];
        for (int i = 0; i < k; i++) {
            localGlobal[i] = cand_list[i];
        }
        insertion_sort(localGlobal, k);
        // Write sorted k–NN pairs to the result array.
        // For query 'queryId', results are stored in result[queryId * k + j].
        for (int i = 0; i < k; i++) {
            result[queryId * k + i] = localGlobal[i];
        }
    }
}

// --------------------------------------------------------------------------------
// Host function: run_knn
// This function launches the k-NN kernel. The grid and block dimensions are chosen
// so that each warp processes one query point.
// 
// Shared memory layout per block:
//   size = BATCH_SIZE * sizeof(float2) + (warpsPerBlock * k * sizeof(std::pair<int, float>))
// --------------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose number of warps per block.
    // We choose 8 warps per block => blockDim.x = 8 * 32 = 256 threads.
    const int warpsPerBlock = 8;
    const int threadsPerBlock = warpsPerBlock * WARP_SIZE;

    // Each warp processes one query. Thus, the number of queries processed per block is warpsPerBlock.
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate shared memory size per block.
    // Space for: BATCH_SIZE data points of type float2 + candidate lists for each warp.
    size_t shmemBytes = BATCH_SIZE * sizeof(float2) + warpsPerBlock * k * sizeof(std::pair<int, float>);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, shmemBytes>>>(query, query_count, data, data_count, result, k);
    
    // It's the caller's responsibility to check for errors and synchronize.
    cudaDeviceSynchronize();
}