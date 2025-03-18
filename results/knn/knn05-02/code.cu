#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ---------------------------------------------------------------------
// This implementation computes the k-nearest neighbors (using squared
// Euclidean distance) for a set of 2D query points against a large set
// of 2D data points. Each query is processed by a single warp (32 threads)
// which maintains two candidate lists:
//  1. A “private” (per-warp, register‐distributed) intermediate candidate
//     list of best k neighbors collected so far.
//  2. A candidate “buffer” stored in shared memory to accumulate candidates
//     from a batch of data points before merging.
// During processing the data are loaded in batches into shared memory.
// Each warp’s threads collaboratively scan the batch and, if a computed
// distance is smaller than the current threshold (i.e. the worst distance
// among the best k so far), they insert the candidate (its global index and
// the squared distance) into the warp’s candidate buffer via an atomic counter.
// When the candidate buffer becomes full (i.e. holds k candidates) or at the end
// of the processing of all batches, the candidate buffer is merged with the
// intermediate candidate list using a warp‐level parallel bitonic sort.
// The merge routine works on 2*k elements (the k intermediate candidates and
// up to k buffer candidates, padded with dummies if needed) stored in shared
// memory. After sorting in ascending order (by distance), the first k elements
// become the new best candidate set. Finally, each warp writes its output to
// the global "result" array.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Tunable hyper-parameters
// We choose 2 warps per block so that shared memory usage remains moderate.
// Each warp (32 threads) processes one query.
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 2
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define BATCH_SIZE 1024  // number of data points per shared-memory batch

// ---------------------------------------------------------------------
// Candidate structure to hold an index and its associated distance.
struct Candidate {
    int idx;
    float dist;
};

// ---------------------------------------------------------------------
// Device function: merge_candidates
//
// Merges the per-warp intermediate candidate list (stored in registers,
// distributed among the 32 threads, each holding k/32 candidates) with 
// a candidate buffer (with 'buffer_count' elements, stored contiguously
// in shared memory) into a new intermediate candidate list.
// To merge efficiently, the two arrays are copied into a temporary
// workspace (of size 2*k) in shared memory, padded (if needed) with dummy
// candidates (with FLT_MAX distance), and then sorted via a bitonic sort.
// The first k elements (sorted in ascending order) become the new set of 
// best candidates; these are distributed back into each thread's register
// array. The function returns the new threshold (i.e., the worst (largest)
// distance among the best k candidates).
//
// Parameters:
//   local_best     - pointer to the per-thread (register) candidate list.
//                    Each thread holds k/32 Candidate elements.
//   candidate_buf  - pointer to the candidate buffer for this warp (size is k).
//   buffer_count   - number of valid candidates in candidate_buf.
//   k              - total number of candidates per query (power-of-2, 32<=k<=1024).
//   merge_workspace- pointer to workspace memory (size 2*k) for the merge.
//
// Note: All threads in the warp call this function and use __syncwarp()
// to coordinate.
__device__ __forceinline__ float merge_candidates(Candidate local_best[], 
                                                   const Candidate* candidate_buf, 
                                                   int buffer_count, 
                                                   int k, 
                                                   Candidate* merge_workspace)
{
    // Each warp is 32 threads; each thread holds a private portion of the k candidates.
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int local_count = k / WARP_SIZE;  // number of candidates per thread

    // Step 1: Each thread writes its private (register) candidates into the merge workspace.
    // The first k elements (indices 0...k-1) will come from the current intermediate result.
    for (int i = 0; i < local_count; i++) {
        merge_workspace[lane * local_count + i] = local_best[i];
    }
    // Step 2: Copy the candidate buffer (of size 'buffer_count') into merge_workspace starting at index k.
    for (int i = lane; i < buffer_count; i += WARP_SIZE) {
        merge_workspace[k + i] = candidate_buf[i];
    }
    // Step 3: Pad the remainder of merge_workspace (from index k+buffer_count to 2*k-1) with dummy candidates.
    for (int i = lane; i < (2 * k - (k + buffer_count)); i += WARP_SIZE) {
        merge_workspace[k + buffer_count + i].idx = -1;
        merge_workspace[k + buffer_count + i].dist = FLT_MAX;
    }
    __syncwarp();

    // Total number of elements in the workspace (L) is 2*k (which is a power-of-2).
    int L = 2 * k;

    // Perform a bitonic sort on merge_workspace.
    // The bitonic sort is implemented in phases; each thread iterates over multiple indices.
    for (int size = 2; size <= L; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            // Each thread processes indices in steps of WARP_SIZE.
            for (int i = lane; i < L; i += WARP_SIZE) {
                int j = i ^ stride;
                if (j > i) {
                    // Determine the sorting direction.
                    bool ascending = ((i & size) == 0);
                    Candidate a = merge_workspace[i];
                    Candidate b = merge_workspace[j];
                    // Swap if out of order according to the direction.
                    if ((ascending && a.dist > b.dist) || (!ascending && a.dist < b.dist)) {
                        merge_workspace[i] = b;
                        merge_workspace[j] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
    // After sorting, the first k elements are the k best candidates in ascending order.
    // Write these sorted values back into the per-thread local_best arrays.
    for (int i = 0; i < local_count; i++) {
        local_best[i] = merge_workspace[lane * local_count + i];
    }
    __syncwarp();

    // The new threshold is the worst (largest) distance among the best k,
    // which is the last element, at index k-1.
    float new_threshold = merge_workspace[k - 1].dist;
    return new_threshold;
}

// ---------------------------------------------------------------------
// Kernel: knn_kernel
//
// Each warp (32 threads) processes one query point. The kernel iterates over
// the data in batches (loaded into shared memory) and for each batch all warps
// compute the distances from their query point. If the computed squared distance
// is below the current threshold, the candidate (global data index and distance)
// is inserted into the candidate buffer (stored in shared memory) using an atomic
// counter. Once the candidate buffer is full (i.e. holds k candidates), the buffer
// is merged with the intermediate result (which is stored in registers, distributed
// among the warp threads) using the merge_candidates() function. After all data have
// been processed, a final merge is performed (if the candidate buffer is not empty),
// and the final k nearest neighbors for that query are written to global memory.
//
// Shared memory layout (per block):
//   [0, BATCH_SIZE*sizeof(float2))           -> sdata (data batch buffer)
//   [BATCH_SIZE*sizeof(float2), BATCH_SIZE*sizeof(float2) + WARPS_PER_BLOCK*k*sizeof(Candidate))
//                                             -> candidate_buffer (per-warp candidate buffers)
//   [next, next + WARPS_PER_BLOCK*sizeof(int)) -> warpBufferCount (per-warp counters)
//   [next, next + WARPS_PER_BLOCK*(2*k*sizeof(Candidate))]
//                                             -> merge_buffer (per-warp merge workspace)
// ---------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count, 
                           const float2 *data, int data_count, 
                           std::pair<int, float> *result, int k)
{
    // Declare shared memory. The total shared memory is allocated dynamically.
    extern __shared__ char smem[];

    // Partition shared memory:
    // sdata: holds the current batch of data points.
    float2 *sdata = (float2*) smem; // size: BATCH_SIZE * sizeof(float2)
    // candidate_buffer: per-warp candidate buffer (each warp gets k Candidate entries)
    Candidate *candidate_buffer = (Candidate*)(smem + BATCH_SIZE * sizeof(float2));
    // warpBufferCount: one integer per warp in the block.
    int *warpBufferCount = (int*)(smem + BATCH_SIZE * sizeof(float2) + WARPS_PER_BLOCK * k * sizeof(Candidate));
    // merge_buffer: per-warp temporary workspace for merge (size: 2*k Candidates each).
    Candidate *merge_buffer = (Candidate*)(smem + BATCH_SIZE * sizeof(float2) +
                                           WARPS_PER_BLOCK * k * sizeof(Candidate) +
                                           WARPS_PER_BLOCK * sizeof(int));

    // Identify the warp within the block.
    int warp_in_block = threadIdx.x / WARP_SIZE;  // range: 0 .. WARPS_PER_BLOCK-1
    int lane = threadIdx.x & (WARP_SIZE - 1);
    // Global warp (query) index.
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    // Each warp processes one query.
    if (global_warp_id < query_count) {
        // Load the query point; use warp leader and broadcast to the lane.
        float2 q;
        if (lane == 0)
            q = query[global_warp_id];
        q.x = __shfl_sync(0xffffffff, q.x, 0);
        q.y = __shfl_sync(0xffffffff, q.y, 0);

        // Each warp maintains an intermediate candidate list (best k) in registers.
        // This list is distributed among the warp's 32 threads.
        int local_count = k / WARP_SIZE; // Guaranteed to be an integer (k is power-of-2, >=32)
        Candidate local_best[32];  // Maximum local_count is k/32 <= 1024/32 = 32.
        for (int i = 0; i < local_count; i++) {
            local_best[i].idx = -1;
            local_best[i].dist = FLT_MAX;
        }

        // Initialize the candidate buffer counter for this warp.
        if (lane == 0)
            warpBufferCount[warp_in_block] = 0;
        __syncwarp();

        // current_threshold (the worst among the best k so far).
        float current_threshold = FLT_MAX;

        // Iterate over data in batches.
        for (int batch_offset = 0; batch_offset < data_count; batch_offset += BATCH_SIZE) {
            // Compute the actual number of points in this batch.
            int batch_size = (data_count - batch_offset) < BATCH_SIZE ? (data_count - batch_offset) : BATCH_SIZE;
            // Load batch data from global memory to shared memory.
            for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
                sdata[i] = data[batch_offset + i];
            }
            __syncthreads();

            // Each warp processes the loaded batch.
            // Distribute the work among lanes.
            for (int i = lane; i < batch_size; i += WARP_SIZE) {
                float2 d = sdata[i];
                float dx = q.x - d.x;
                float dy = q.y - d.y;
                float dist = dx * dx + dy * dy;
                // Only consider this candidate if its distance is smaller than the current threshold.
                if (dist < current_threshold) {
                    int candidate_idx = batch_offset + i;
                    // Insert candidate into the warp's candidate buffer using an atomic within the warp.
                    int pos = atomicAdd(&warpBufferCount[warp_in_block], 1);
                    if (pos < k) {
                        candidate_buffer[warp_in_block * k + pos].idx = candidate_idx;
                        candidate_buffer[warp_in_block * k + pos].dist = dist;
                    }
                    // If the candidate buffer is now full, merge it with the intermediate candidate list.
                    if (pos + 1 == k) {
                        // Merge the k candidate buffer entries with the current intermediate result (local_best).
                        float new_thresh = merge_candidates(local_best, candidate_buffer + warp_in_block * k, k, k,
                                                              merge_buffer + warp_in_block * (2 * k));
                        current_threshold = new_thresh;
                        // Reset the candidate buffer counter.
                        if (lane == 0)
                            warpBufferCount[warp_in_block] = 0;
                        __syncwarp();
                    }
                }
            }
            __syncthreads();  // Ensure all threads are done before loading the next batch.
        }

        // After all batches, merge any remaining candidates in the buffer.
        int buf_count = warpBufferCount[warp_in_block];
        if (buf_count > 0) {
            float new_thresh = merge_candidates(local_best, candidate_buffer + warp_in_block * k, buf_count, k,
                                                merge_buffer + warp_in_block * (2 * k));
            current_threshold = new_thresh;
            if (lane == 0)
                warpBufferCount[warp_in_block] = 0;
            __syncwarp();
        }

        // Write the final k best candidates to the result.
        // The intermediate candidate list is distributed among the warp's lanes.
        // Store in row-major order: for query i, result[i*k + j] is the j-th neighbor.
        int base = global_warp_id * k;
        for (int i = 0; i < local_count; i++) {
            result[base + lane * local_count + i] = std::make_pair(local_best[i].idx, local_best[i].dist);
        }
    }
}

// ---------------------------------------------------------------------
// Host interface: run_knn
//
// This function is responsible for launching the CUDA kernel. It sets up
// grid/block dimensions (each warp processes a query) and computes the total
// shared memory required. The query, data, and result arrays are assumed to be
// allocated via cudaMalloc. No extra device memory is allocated in the kernel.
// ---------------------------------------------------------------------
void run_knn(const float2 *query, int query_count, 
             const float2 *data, int data_count, 
             std::pair<int, float> *result, int k)
{
    // Each warp processes one query; hence, the total number of warps = query_count.
    // With WARPS_PER_BLOCK warps per block, compute the number of blocks.
    int num_warps = query_count;
    int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(num_blocks);

    // Calculate shared memory size (in bytes):
    //   sdata:       BATCH_SIZE * sizeof(float2)
    //   candidate_buffer: WARPS_PER_BLOCK * k * sizeof(Candidate)
    //   warpBufferCount:  WARPS_PER_BLOCK * sizeof(int)
    //   merge_buffer:    WARPS_PER_BLOCK * (2*k) * sizeof(Candidate)
    size_t shared_mem_size = BATCH_SIZE * sizeof(float2)
                           + WARPS_PER_BLOCK * k * sizeof(Candidate)
                           + WARPS_PER_BLOCK * sizeof(int)
                           + WARPS_PER_BLOCK * (2 * k) * sizeof(Candidate);

    // Launch the kernel.
    knn_kernel<<<gridDim, blockDim, shared_mem_size>>>(query, query_count, data, data_count, result, k);
}