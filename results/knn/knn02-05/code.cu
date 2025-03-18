#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// This kernel implements k-NN search for 2D points using a warp‐per‐query strategy.
// Each warp (32 threads) processes one query. Each thread in the warp is responsible for
// a "slice" of the final k neighbors. Since k is a power‐of‑two (32 ≤ k ≤ 1024), we let
// each thread hold k/32 candidate entries (its “local candidate list”) in registers.
// The data points are processed in batches loaded into shared memory to amortize global memory latency.
// After processing all batches, each thread sorts its local candidate list in ascending order,
// then the 32 sorted lists (each of length k/32) are merged into one final sorted list (of length k)
// using shared memory and a simple multi‐way merge (performed by lane 0 of the warp).
// Finally, the merged sorted result is written to global memory.

#define BATCH_SIZE 1024  // Number of data points read per batch (fits in shared memory)

/// @FIXED
/// extern "C" __global__ void knn_kernel(const float2 *query, int query_count,
__global__ void knn_kernel(const float2 *query, int query_count,
                                       const float2 *data, int data_count,
                                       std::pair<int, float> *result, int k) {
    // The dynamic shared memory is partitioned as follows:
    // - First: an array of BATCH_SIZE float2 elements for caching a batch of data points.
    // - Second: a merge buffer for each warp, sized (k) elements per warp.
    extern __shared__ char shared_buf[];
    float2* s_data = (float2*)shared_buf;
    // s_merge will be used for storing per-warp intermediate candidate arrays and final merge result.
    std::pair<int, float>* s_merge = (std::pair<int, float>*)(s_data + BATCH_SIZE);

    // Each warp (32 threads) processes one query.
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int query_index = blockIdx.x * warps_per_block + warp_id;
    if (query_index >= query_count) return;

    // Load the query point for this warp (all threads share the same query).
    float2 q = query[query_index];

    // Each thread holds a private candidate list of size local_n = k/32.
    int local_n = k >> 5;  // k/32; note: k is a power of 2 between 32 and 1024.
    const int MAX_LOCAL_N = 32;  // maximum possible local candidate list size
    float local_dists[MAX_LOCAL_N];
    int   local_indices[MAX_LOCAL_N];

    // Initialize the local candidate list with "infinite" distances.
    for (int i = 0; i < local_n; i++) {
        local_dists[i] = FLT_MAX;
        local_indices[i] = -1;
    }

    // Process all data points in batches cached in shared memory.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Determine current batch size.
        int batch_size = BATCH_SIZE;
        if (batch_start + BATCH_SIZE > data_count)
            batch_size = data_count - batch_start;

        // Cooperatively load a batch of data points from global memory to shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            s_data[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the batch is fully loaded.

        // Each warp processes the cached batch.
        // Each thread in the warp loops over a subset of the batch (stride = 32).
        for (int j = lane; j < batch_size; j += 32) {
            float2 dpt = s_data[j];
            float dx = q.x - dpt.x;
            float dy = q.y - dpt.y;
            float dist = dx * dx + dy * dy;
            int data_index = batch_start + j;  // Global index of the candidate.

            // Update the local candidate list.
            // We scan the local list to find the candidate with maximum distance.
            float max_val = local_dists[0];
            int max_idx = 0;
            for (int t = 1; t < local_n; t++) {
                if (local_dists[t] > max_val) {
                    max_val = local_dists[t];
                    max_idx = t;
                }
            }
            // If the current candidate is closer than the worst in the local list, update it.
            if (dist < max_val) {
                local_dists[max_idx] = dist;
                local_indices[max_idx] = data_index;
            }
        }
        __syncthreads();  // Ensure all threads are done before reusing shared memory.
    } // End processing all data batches

    // Sort each thread's local candidate list in ascending order (using insertion sort).
    for (int i = 1; i < local_n; i++) {
        float key_dist = local_dists[i];
        int   key_idx  = local_indices[i];
        int j = i - 1;
        while (j >= 0 && local_dists[j] > key_dist) {
            local_dists[j + 1] = local_dists[j];
            local_indices[j + 1] = local_indices[j];
            j--;
        }
        local_dists[j + 1] = key_dist;
        local_indices[j + 1] = key_idx;
    }

    // Each warp will merge the 32 sorted local candidate lists (one per lane) into one sorted list of k candidates.
    // We'll use the merge buffer in shared memory (s_merge) allocated per warp.
    int warp_merge_offset = warp_id * k;  // Each warp has a contiguous region of k std::pair<int, float> elements.

    // Step 1: Each thread writes its sorted local candidate list into its designated slot in the merge buffer.
    // The layout for each warp: [lane0: indices 0 .. local_n-1, lane1: local_n .. 2*local_n-1, ..., lane31: ...].
    int slot_base = warp_merge_offset + lane * local_n;
    for (int i = 0; i < local_n; i++) {
        s_merge[slot_base + i] = std::make_pair(local_indices[i], local_dists[i]);
    }
    __syncwarp();  // Ensure all 32 lanes have written their candidates.

    // Step 2: Lane 0 of the warp performs a multiway merge of the 32 sorted subarrays.
    if (lane == 0) {
        // Copy the k candidate entries from the merge buffer into a temporary array.
        std::pair<int, float> temp[1024];  // Maximum k is 1024.
        for (int i = 0; i < k; i++) {
            temp[i] = s_merge[warp_merge_offset + i];
        }
        // Each of the 32 subarrays is of length local_n and already sorted in ascending order.
        int pointers[32];  // Pointer (index) for each subarray.
        for (int i = 0; i < 32; i++) {
            pointers[i] = i * local_n;  // Starting index for each lane's candidates in the temp array.
        }
        std::pair<int, float> merged[1024];
        // Perform a multiway merge by selecting the smallest candidate among the heads of the 32 subarrays.
        for (int m = 0; m < k; m++) {
            float min_val = FLT_MAX;
            int min_sub = -1;
            // For each subarray, if elements remain, check the candidate.
            for (int s = 0; s < 32; s++) {
                int begin = s * local_n;
                int end = begin + local_n;
                if (pointers[s] < end) {
                    float candidate_val = temp[pointers[s]].second;
                    if (candidate_val < min_val) {
                        min_val = candidate_val;
                        min_sub = s;
                    }
                }
            }
            merged[m] = temp[pointers[min_sub]];
            pointers[min_sub]++;
        }
        // Write the merged sorted result back into the merge buffer region for this warp.
        for (int m = 0; m < k; m++) {
            s_merge[warp_merge_offset + m] = merged[m];
        }
    }
    __syncwarp();  // Ensure the merged result is visible to all lanes.

    // Step 3: Each lane cooperatively writes the final sorted k nearest neighbors for the query to global memory.
    // The result for query index 'query_index' is stored contiguously in the output array.
    for (int m = lane; m < k; m += 32) {
        result[query_index * k + m] = s_merge[warp_merge_offset + m];
    }
}

// Host function that configures and launches the CUDA kernel.
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                          const float2 *data, int data_count,
                          std::pair<int, float> *result, int k) {
    // Choose block dimensions: 256 threads per block (i.e., 8 warps per block).
    const int block_size = 256;
    const int warps_per_block = block_size >> 5;  // 256/32 = 8 warps per block.
    // Each warp processes one query, so determine the grid size accordingly.
    int grid_size = (query_count + warps_per_block - 1) / warps_per_block;

    // Calculate the dynamic shared memory size:
    //   - s_data: BATCH_SIZE float2 elements.
    //   - s_merge: warps_per_block * k elements of std::pair<int, float>.
    size_t shared_mem_size = BATCH_SIZE * sizeof(float2) + warps_per_block * k * sizeof(std::pair<int, float>);

    // Launch the kernel.
    knn_kernel<<<grid_size, block_size, shared_mem_size>>>(query, query_count, data, data_count, result, k);
    // Optionally, synchronize and check errors here (omitted for brevity).
}
