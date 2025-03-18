#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <utility>

//------------------------------------------------------------------------------
// This CUDA implementation of k-nearest neighbors (k-NN) is optimized for
// modern NVIDIA GPUs (e.g., A100/H100) and uses warp‐level primitives to
// simultaneously process one query per warp (32 threads). Each warp maintains 
// its own private k‐nearest neighbor list in registers (distributed among lanes)
// and a candidate buffer in shared memory for accumulating new candidate data 
// points from batches loaded from global memory. When the candidate buffer fills,
// it is merged with the private list via a warp‐cooperative bitonic sort.
// Finally, the sorted k nearest neighbors for each query are written to global memory.
//
// The following shared-memory layout is used per block:
//   • A batch of data points (float2) of size BATCH_SIZE.
//   • For each warp (i.e., for each query processed in the block):
//       - A candidate buffer of k elements (struct NN).
//       - A candidate counter (int).
//       - A merge workspace of 2*k elements (struct NN).
//
// The total shared memory per block is therefore:
//   BATCH_SIZE*sizeof(float2) +
//   warps_per_block*(k*sizeof(NN) + sizeof(int) + 2*k*sizeof(NN))
//
// The kernel processes the data points in batches to exploit shared memory
// caching and minimizes global memory traffic.
//------------------------------------------------------------------------------

#define BATCH_SIZE 1024  // Number of data points to load per batch (tunable)

// NN: structure to store candidate neighbor (data index and squared distance).
struct NN {
    int index;
    float dist;
};

//------------------------------------------------------------------------------
// bitonic_sort: A warp-level parallel bitonic sort.
// This function sorts an array "arr" of n elements (of type NN) in ascending order
// based on the 'dist' field. The sort is performed cooperatively by the 32 lanes
// of the warp. Each thread processes elements with indices: lane, lane+32, lane+64, ...
// It is assumed that n is a power-of-two.
//------------------------------------------------------------------------------
__device__ inline void bitonic_sort(NN* arr, int n, int lane)
{
    // Outer loop: increasing subsequence size
    for (int size = 2; size <= n; size <<= 1) {
        // Inner loop: bitonic merge
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread works on multiple indices in steps of 32 (warp size)
            for (int i = lane; i < n; i += 32) {
                int j = i ^ stride;
                if (j > i && j < n) {
                    // Determine the desired order direction.
                    bool ascending = ((i & size) == 0);
                    NN a = arr[i];
                    NN b = arr[j];
                    bool swap = false;
                    if (ascending && (a.dist > b.dist))
                        swap = true;
                    if (!ascending && (a.dist < b.dist))
                        swap = true;
                    if (swap) {
                        arr[i] = b;
                        arr[j] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

//------------------------------------------------------------------------------
// merge_candidates: Merges the candidate buffer (shared memory) with the private 
// k-NN intermediate result stored in registers (distributed among 32 warp lanes).
// The merge is accomplished by copying both sets into a merge workspace in shared 
// memory, padding the unused entries with dummy values (FLT_MAX), and then performing
// a bitonic sort on 2*k elements. The best k candidates (lowest distances) are then
// copied back into the private registers.
// Parameters:
//  candidate_buf   - pointer to warp’s candidate buffer (size k)
//  candidate_count_ptr - pointer to the candidate counter (per warp)
//  k               - number of neighbors to keep (global per query)
//  local_idx       - per-thread private array of candidate indices (size seg_size)
//  local_dist      - per-thread private array of candidate distances (size seg_size)
//  seg_size        - number of candidates stored per thread (k/32)
//  merge_buf       - pointer to warp’s merge workspace (size 2*k)
//  lane            - thread lane ID within the warp (0..31)
//------------------------------------------------------------------------------
__device__ inline void merge_candidates(NN* candidate_buf, int* candidate_count_ptr, int k,
                                         int local_idx[], float local_dist[], int seg_size, NN* merge_buf, int lane)
{
    // Step 1: Copy the current private k-NN (intermediate result) from registers
    // into the lower part of merge_buf. Each warp’s intermediate result is distributed
    // among 32 lanes; lane i is responsible for seg_size contiguous elements.
    int base = lane * seg_size;
    for (int j = 0; j < seg_size; j++) {
        merge_buf[base + j].index = local_idx[j];
        merge_buf[base + j].dist  = local_dist[j];
    }
    __syncwarp();

    // Step 2: Read candidate buffer count and copy candidate buffer into merge_buf after k entries.
    int count = *candidate_count_ptr;
    for (int j = lane; j < count; j += 32) {
        merge_buf[k + j] = candidate_buf[j];
    }
    __syncwarp();

    // Total elements to merge = k + count. Pad the remaining slots up to 2*k with dummy entries.
    int total = k + count;
    for (int j = total + lane; j < 2 * k; j += 32) {
        merge_buf[j].index = -1;
        merge_buf[j].dist  = FLT_MAX;
    }
    __syncwarp();

    // Step 3: Sort the 2*k elements in merge_buf using bitonic sort.
    int sort_total = 2 * k; // This is a power-of-two.
    bitonic_sort(merge_buf, sort_total, lane);
    __syncwarp();

    // Step 4: Copy the best k candidates (first k sorted elements) back to the private registers.
    for (int j = 0; j < seg_size; j++) {
        int idx = lane * seg_size + j;
        local_idx[j]  = merge_buf[idx].index;
        local_dist[j] = merge_buf[idx].dist;
    }
    __syncwarp();

    // Reset candidate count (only one lane needs to do this).
    if (lane == 0)
        *candidate_count_ptr = 0;
    __syncwarp();
}

//------------------------------------------------------------------------------
// knn_kernel: CUDA kernel that computes the k-nearest neighbors for 2D query points.
// Each warp (32 threads) processes one query. The input "data" points are processed in
// batches that are loaded into shared memory to reduce global memory traffic.
//------------------------------------------------------------------------------
__global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                           const float2* __restrict__ data, int data_count,
                           std::pair<int, float>* result, int k)
{
    // Each warp processes one query.
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31; // thread index within warp (0..31)
    const int warps_per_block = blockDim.x / 32;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    if (global_warp_id >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[global_warp_id];

    // Divide the k neighbors among the 32 warp lanes.
    // Since k is a power-of-two between 32 and 1024, seg_size = k / 32.
    const int seg_size = k / 32;
    // Private (register) storage for the intermediate k-NN result distributed among lanes.
    int local_idx[32];     // Maximum seg_size is 32 when k==1024.
    float local_dist[32];
    for (int i = 0; i < seg_size; i++) {
        local_idx[i]  = -1;
        local_dist[i] = FLT_MAX;
    }

    // Get pointers into shared memory.
    // Shared memory layout:
    //   [0, BATCH_SIZE*sizeof(float2))         -> Data batch buffer (float2)
    //   [BATCH_SIZE*sizeof(float2), ... )        -> Warp candidate buffers (NN), one per warp, each of size k.
    //   [after candidate buffers]                -> Warp candidate counters (int), one per warp.
    //   [after candidate counters]               -> Warp merge workspace (NN), one per warp, each of size 2*k.
    extern __shared__ char s_mem[];
    float2* s_data = (float2*)s_mem;  // Data batch buffer

    // Offset pointer for candidate buffers.
    NN* s_candidate_buf = (NN*)(s_mem + BATCH_SIZE * sizeof(float2));
    // Pointer for this warp's candidate buffer.
    NN* candidate_buf = s_candidate_buf + warp_id * k;
    
    // Offset pointer for candidate counters.
    int* s_candidate_count = (int*)(s_mem + BATCH_SIZE * sizeof(float2) + warps_per_block * k * sizeof(NN));
    int* candidate_count_ptr = s_candidate_count + warp_id;
    
    // Offset pointer for merge workspace.
    NN* s_merge_buf = (NN*)(s_mem + BATCH_SIZE * sizeof(float2) + 
                            warps_per_block * k * sizeof(NN) + 
                            warps_per_block * sizeof(int));
    // Pointer for this warp's merge buffer (workspace of size 2*k).
    NN* merge_buf = s_merge_buf + warp_id * (2 * k);
    
    // Initialize candidate counter for this warp.
    if (lane == 0)
        *candidate_count_ptr = 0;
    __syncwarp();

    // Process the data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Determine the number of data points in the current batch.
        int current_batch = (batch_start + BATCH_SIZE <= data_count) ? BATCH_SIZE : (data_count - batch_start);
        // Cooperative loading: All threads in the block load parts of the data batch into shared memory.
        for (int i = threadIdx.x; i < current_batch; i += blockDim.x) {
            s_data[i] = data[batch_start + i];
        }
        __syncthreads();

        // Before processing the batch, compute the current threshold (max distance)
        // from the intermediate result. Each lane computes the maximum of its local segment.
        float local_worst = -1.0f;
        for (int j = 0; j < seg_size; j++) {
            local_worst = fmaxf(local_worst, local_dist[j]);
        }
        // Warp-level reduction to compute the maximum among lanes.
        float warp_threshold = local_worst;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, warp_threshold, offset);
            warp_threshold = fmaxf(warp_threshold, other);
        }

        // Each warp thread processes a subset of the data points in this batch.
        for (int i = lane; i < current_batch; i += 32) {
            float2 pt = s_data[i];
            float dx = pt.x - q.x;
            float dy = pt.y - q.y;
            float dist = dx * dx + dy * dy;  // Squared Euclidean distance

            if (dist < warp_threshold) { // Candidate qualifies
                // Use an atomic update on the candidate counter to reserve a slot in the candidate buffer.
                // If the buffer is full (>= k entries), merge it with the intermediate result.
                while (true) {
                    int cnt = *candidate_count_ptr;
                    if (cnt >= k) {
                        // Buffer full: merge candidate buffer with the private intermediate result.
                        merge_candidates(candidate_buf, candidate_count_ptr, k, local_idx, local_dist, seg_size, merge_buf, lane);
                        // Recompute the new threshold from the updated intermediate result.
                        float new_local_worst = -1.0f;
                        for (int j = 0; j < seg_size; j++) {
                            new_local_worst = fmaxf(new_local_worst, local_dist[j]);
                        }
                        warp_threshold = new_local_worst;
                        for (int offset = 16; offset > 0; offset /= 2) {
                            float other = __shfl_down_sync(0xffffffff, warp_threshold, offset);
                            warp_threshold = fmaxf(warp_threshold, other);
                        }
                        // Retry insertion.
                        continue;
                    }
                    // Attempt to reserve a slot with atomicCAS.
                    int old = atomicCAS(candidate_count_ptr, cnt, cnt + 1);
                    if (old == cnt) {
                        candidate_buf[cnt].index = batch_start + i; // Global index in data array.
                        candidate_buf[cnt].dist  = dist;
                        break;
                    }
                }
            }
        }
        __syncthreads();
    } // End of batch loop.

    // After processing all batches, merge any remaining candidates in the candidate buffer.
    if (*candidate_count_ptr > 0) {
        merge_candidates(candidate_buf, candidate_count_ptr, k, local_idx, local_dist, seg_size, merge_buf, lane);
    }

    // Final step: Merge the private intermediate result across warp lanes.
    // Copy the per-thread private arrays into the merge workspace.
    for (int j = 0; j < seg_size; j++) {
        merge_buf[lane * seg_size + j].index = local_idx[j];
        merge_buf[lane * seg_size + j].dist  = local_dist[j];
    }
    __syncwarp();

    // Now, merge_buf holds k = seg_size*32 elements. Sort these k elements.
    bitonic_sort(merge_buf, k, lane);
    __syncwarp();

    // Write the sorted final k-nearest neighbors to the output global result.
    // Each warp writes its own k neighbors.
    int base_out = global_warp_id * k;
    for (int i = 0; i < seg_size; i++) {
        int out_idx = base_out + lane * seg_size + i;
        result[out_idx].first  = merge_buf[lane * seg_size + i].index;
        result[out_idx].second = merge_buf[lane * seg_size + i].dist;
    }
}

//------------------------------------------------------------------------------
// run_knn: Host interface to launch the k-NN kernel.
//   query       - pointer to device array of query points (float2)
//   query_count - number of queries
//   data        - pointer to device array of data points (float2)
//   data_count  - number of data points (>= k)
//   result      - pointer to device array where each query's k nearest neighbors
//                 are stored in row-major order as (int index, float distance) pairs.
//   k           - number of nearest neighbors, a power-of-two between 32 and 1024.
//------------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block size of 128 threads (i.e., 4 warps per block).
    int block_size = 128;
    int warps_per_block = block_size / 32; // should be 4
    // Each warp processes one query; determine grid size accordingly.
    int grid_size = (query_count + warps_per_block - 1) / warps_per_block;

    // Calculate shared memory required per block:
    //   - Data batch buffer: BATCH_SIZE * sizeof(float2)
    //   - Candidate buffers: warps_per_block * k * sizeof(NN)
    //   - Candidate counters: warps_per_block * sizeof(int)
    //   - Merge workspace: warps_per_block * (2 * k) * sizeof(NN)
    size_t shmem = BATCH_SIZE * sizeof(float2)
                 + warps_per_block * k * sizeof(NN)
                 + warps_per_block * sizeof(int)
                 + warps_per_block * (2 * k) * sizeof(NN);

    // Launch the kernel.
    knn_kernel<<<grid_size, block_size, shmem>>>(query, query_count, data, data_count, result, k);
    // Note: Error checking (e.g., cudaGetLastError) should be performed by the caller if desired.
}