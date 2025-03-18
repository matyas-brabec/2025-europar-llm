// This CUDA code implements a k‐nearest neighbors search for 2D points.
// Each query (float2) is processed by one warp (32 threads). Each warp maintains
// an “intermediate result” of k best candidates (nearest neighbors) privately,
// distributed across its 32 threads (each keeping k/32 items in registers, sorted
// in increasing order by squared Euclidean distance). In addition, each warp has
// a candidate “buffer” (of capacity k) stored in shared memory. As the kernel
// iterates over data points loaded in batches into shared memory, promising candidates
// (points with distance less than the current worst candidate) are inserted into the
// candidate buffer. When the buffer gets full (or at the end) the warp “merges” it with
// its intermediate result – using a warp‐synchronous sorting routine that cooperates
// among the 32 threads. Finally, the sorted k candidates are written to global memory.
// 
// This implementation is tuned for modern NVIDIA GPUs (e.g. A100/H100) compiled with
// the latest CUDA toolkit. All shared memory buffers are allocated without extra device
// memory allocations. Note that some parts (especially the merge routine) use a simple
// parallel selection via linear scans; while not optimal for all parameters, it follows
// the requested design and uses warp‐level primitives (e.g. __shfl_down_sync, __syncwarp).
//
// Compile with: nvcc -arch=sm_80 -O3 knn.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cstdio>

// Define a very large float value for "infinity".
#define INF (1e30f)

// Structure to hold a candidate neighbor: its data index and squared distance.
struct Candidate {
    int idx;
    float dist;
};

// Warp-level reduction (maximum) using warp shuffle.
__inline__ __device__ float warpReduceMax(float val) {
    // Use full warp mask.
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

//---------------------------------------------------------------------
// merge_candidates:
// Merge the candidate buffer (of m candidates) with the intermediate result (of k candidates)
// to yield the new best k candidates (sorted in increasing order).
// The intermediate result is stored in registers (each warp holds k items distributed across 32 threads,
// with each thread holding r = k/32 candidates in an array "inter").
// The candidate buffer for this warp (stored in shared memory in s_cand_buffer) holds m candidates.
// The merge is performed into a scratch (merge) buffer of size 2*k; then, using a simple parallel selection
// (each thread picks the candidate of a given global rank by scanning the union array).
//
// All 32 threads of the warp call this function cooperatively.
__device__ void merge_candidates(Candidate inter[], int r, int k, int warp_id, int lane,
                                 Candidate* s_cand_buffer, int* s_cand_count, Candidate* s_merge_buffer)
{
    // Each warp’s candidate buffer is stored contiguously:
    // s_cand_buffer[warp_id*k ... warp_id*k + (k-1)]
    Candidate* warpCand = s_cand_buffer + warp_id * k;
    // Each warp gets its own merge scratch area of size 2*k in s_merge_buffer.
    Candidate* mergeBuf = s_merge_buffer + warp_id * (2 * k);

    //-----------------------------------------------------------------
    // Step 1. Write the current intermediate result (k candidates) from registers into mergeBuf[0...k-1].
    // Each warp holds k items distributed among 32 threads (r = k/32 each).
    for (int i = 0; i < r; i++) {
        int pos = lane + i * 32;
        if (pos < k) {
            mergeBuf[pos] = inter[i];
        }
    }
    __syncwarp();

    //-----------------------------------------------------------------
    // Step 2. Get the candidate buffer count (m). Let only lane 0 do the copy.
    int m = 0;
    if (lane == 0)
        m = s_cand_count[warp_id];
    m = __shfl_sync(0xffffffff, m, 0);
    
    // Copy candidate buffer (m candidates) into mergeBuf[k ... k+m-1].
    if (lane == 0) {
        for (int j = 0; j < m; j++) {
            mergeBuf[k + j] = warpCand[j];
        }
        // Fill remaining slots from (k+m) to (2*k-1) with dummy candidate { -1, INF }.
        for (int j = m; j < k; j++) {
            mergeBuf[k + j].dist = INF;
            mergeBuf[k + j].idx = -1;
        }
        // Reset candidate buffer count.
        s_cand_count[warp_id] = 0;
    }
    __syncwarp();

    // Now, the union array of candidates has total size U = 2*k.
    const int U = 2 * k;
    // We want to select the k smallest candidates (in ascending order) from mergeBuf.
    // We'll use a simple parallel selection algorithm: each thread is responsible for output positions
    // p = lane + 32*i, for i=0 .. (r-1) (since r = k/32, total k positions).
    Candidate newLocal[32]; // r is at most 32 if k <= 1024.
    for (int i = 0; i < r; i++) {
        int globalRank = lane + i * 32;  // This is the rank (0-indexed) in the sorted order we want.
        Candidate best;
        best.dist = INF;
        best.idx = -1;
        // Linear scan over union array to find candidate with rank equal to 'globalRank'.
        // For each candidate cand in mergeBuf, compute its rank by counting how many elements are strictly smaller.
        for (int j = 0; j < U; j++) {
            Candidate cand = mergeBuf[j];
            int count = 0;
            for (int l = 0; l < U; l++) {
                Candidate other = mergeBuf[l];
                // Compare by distance; use index as a tie-breaker.
                if (other.dist < cand.dist || (other.dist == cand.dist && other.idx < cand.idx))
                    count++;
            }
            if (count == globalRank) {
                best = cand;
                break;
            }
        }
        newLocal[i] = best;
    }
    __syncwarp();
    // Step 3. Write the sorted k results back to the per-thread registers.
    for (int i = 0; i < r; i++) {
        inter[i] = newLocal[i];
    }
    __syncwarp();
}

//---------------------------------------------------------------------
// Kernel knn_kernel: Each warp processes one query.
__global__ void knn_kernel(const float2 *query, int query_count, const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Determine warp identity within the thread block.
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane   = tid & 31;  // equivalent to tid % 32
    int warpsPerBlock = blockDim.x / 32;
    
    // Global query index (each warp handles one query).
    int query_idx = blockIdx.x * warpsPerBlock + warp_id;
    if (query_idx >= query_count)
        return;

    // Load the query point. Only one thread (lane 0) loads from global mem,
    // then broadcasts using __shfl_sync.
    float2 q;
    if (lane == 0) {
        q = query[query_idx];
    }
    q.x = __shfl_sync(0xffffffff, q.x, 0);
    q.y = __shfl_sync(0xffffffff, q.y, 0);

    // Each warp maintains an intermediate result of best candidates (k in total),
    // distributed among 32 threads. Let r = k / 32 (k is ensured to be a power of two ≥32).
    const int r = k / 32;
    Candidate inter[32];  // Each thread holds r items; maximum r is 1024/32 = 32.
    for (int i = 0; i < r; i++) {
        inter[i].dist = INF;
        inter[i].idx = -1;
    }

    //---------------------------------------------------------------------
    // Shared memory layout (dynamically allocated):
    // [0, BS)                 : float2 data batch array (shared copy of a batch of data points)
    // [BS, BS + (warpsPerBlock*k)]:
    //       Candidate candidate buffer for all warps (each warp gets k entries)
    // Next, warpsPerBlock ints: candidate count for each warp.
    // Next, warpsPerBlock*(2*k): merge scratch buffer for each warp.
    //
    // We'll define BS as the block batch size.
    const int BS = 256;  // number of data points per batch.
    extern __shared__ char shmem[];
    float2* s_data = (float2*) shmem;
    Candidate* s_cand_buffer = (Candidate*)(shmem + BS * sizeof(float2));
    int* s_cand_count = (int*)(shmem + BS * sizeof(float2) + warpsPerBlock * k * sizeof(Candidate));
    Candidate* s_merge_buffer = (Candidate*)(shmem + BS * sizeof(float2) +
                                             warpsPerBlock * k * sizeof(Candidate) +
                                             warpsPerBlock * sizeof(int));

    // Initialize candidate count for this warp to 0 (one thread per warp does it).
    if (lane == 0)
        s_cand_count[warp_id] = 0;
    __syncwarp();

    // Loop over data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BS)
    {
        // Load a batch of data points into shared memory.
        // Use block threads to load; threads with index < BS load one element each.
        int load_idx = threadIdx.x;
        if (load_idx < BS && (batch_start + load_idx) < data_count)
            s_data[load_idx] = data[batch_start + load_idx];
        __syncthreads();

        // Determine the actual number of data points in this batch.
        int batch_size = (data_count - batch_start < BS) ? (data_count - batch_start) : BS;

        // Each warp processes the batch with a stride of 32 threads.
        for (int j = lane; j < batch_size; j += 32)
        {
            float2 dpt = s_data[j];
            float dx = q.x - dpt.x;
            float dy = q.y - dpt.y;
            float dist = dx * dx + dy * dy;

            // Compute current global threshold from the intermediate result.
            // Each thread’s worst candidate is its last element (largest in its local sorted array).
            float local_thresh = inter[r - 1].dist;
            float global_thresh = warpReduceMax(local_thresh);
            if (dist < global_thresh)
            {
                // Try to insert candidate into the candidate buffer.
                Candidate* warpCand = s_cand_buffer + warp_id * k;
                int prevCount = atomicAdd(&s_cand_count[warp_id], 1);
                if (prevCount < k)
                {
                    warpCand[prevCount].dist = dist;
                    warpCand[prevCount].idx = batch_start + j;  // global data index
                }
                else
                {
                    // Buffer is (or has become) full.
                    // To keep things simple, let only one thread (e.g. lane 0) trigger the merge.
                    // All threads then re-check the candidate.
                    if (prevCount == k)
                    {
                        // Merge candidate buffer with the intermediate result.
                        merge_candidates(inter, r, k, warp_id, lane, s_cand_buffer, s_cand_count, s_merge_buffer);
                    }
                    // After merging, re-read the global threshold.
                    global_thresh = warpReduceMax(inter[r - 1].dist);
                    if (dist < global_thresh)
                    {
                        int pos = atomicAdd(&s_cand_count[warp_id], 1);
                        if (pos < k)
                        {
                            warpCand[pos].dist = dist;
                            warpCand[pos].idx = batch_start + j;
                        }
                    }
                }
            } // end if candidate is promising
        } // end for each data point in batch

        __syncthreads();  // ensure all threads are done with this batch before next batch load
    } // end for each batch

    // After processing all batches, merge any remaining candidates in candidate buffer.
    if (s_cand_count[warp_id] > 0)
        merge_candidates(inter, r, k, warp_id, lane, s_cand_buffer, s_cand_count, s_merge_buffer);

    // At this point, the warp’s intermediate result "inter" holds the final k nearest neighbors,
    // distributed in sorted order (ascending by distance) per our merge routine.
    // Write the result to global memory. Each thread writes its r elements.
    int out_base = query_idx * k;
    for (int i = 0; i < r; i++) {
        int pos = lane + i * 32;
        if (pos < k)
            result[out_base + pos] = std::pair<int, float>(inter[i].idx, inter[i].dist);
    }
}

//---------------------------------------------------------------------
// Host interface: run_knn. Launches the kernel with one warp per query.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block dimensions.
    // We assign one warp (32 threads) per query.
    // For occupancy, we choose a block size that is a multiple of 32.
    // For example, let blockDim.x = 128 (i.e. 4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Number of blocks needed so that each warp processes one query.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate dynamic shared memory size per block.
    // Shared memory layout per block:
    //   s_data: BS * sizeof(float2)
    //   candidate buffer for all warps: warpsPerBlock * k * sizeof(Candidate)
    //   candidate count for all warps: warpsPerBlock * sizeof(int)
    //   merge scratch buffer for all warps: warpsPerBlock * (2*k) * sizeof(Candidate)
    const int BS = 256;
    size_t shmem_size = BS * sizeof(float2)
                        + warpsPerBlock * k * sizeof(Candidate)
                        + warpsPerBlock * sizeof(int)
                        + warpsPerBlock * (2 * k) * sizeof(Candidate);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, shmem_size>>>(query, query_count, data, data_count, result, k);

    // Optionally, check for errors (omitted here for brevity).
    cudaDeviceSynchronize();
}