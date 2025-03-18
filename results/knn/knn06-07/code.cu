#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <cfloat>

// This kernel implements a k-NN search for 2D points using a warp‐per‐query strategy.
// For each query point, a warp (32 threads) cooperates to compute squared Euclidean distances
// between the query and data points (loaded in batches into shared memory). If a candidate
// distance is lower than the current worst neighbor (“max_distance”), the candidate is added
// to a per‐warp candidate buffer (in shared memory) using atomic operations. When the candidate
// buffer becomes full, a merge is performed that combines the candidate buffer with the
// “best” result (private copy stored in shared memory) and updates the worst accepted distance.
// At the end, the final k nearest neighbors are written to global memory.
//
// The code assumes the following shared memory layout per block:
//   [0, BATCH_SIZE)                        : float2 shared_data (data batch from global memory)
//   [BATCH_SIZE, BATCH_SIZE + warps*k*sizeof(Candidate)) : candidate buffer (per-warp)
//   [ ... next, warps*sizeof(int))          : candidate count array (per-warp)
//   [ ... next, warps*MAX_K*sizeof(Candidate)) : best buffer (per-warp; MAX_K is fixed to 1024)
// where k is the requested number of nearest neighbors (k <= MAX_K).
//
// We define BATCH_SIZE as 256 and MAX_K as 1024 (the maximum possible value of k).

#define BATCH_SIZE 256
#define MAX_K 1024

// Struct to hold a candidate neighbor: index and squared distance.
struct Candidate {
    int idx;
    float dist;
};

// __device__ function to merge a warp's candidate buffer with its best result.
// This function is called by all threads of a warp; only lane 0 does the heavy merge work,
// after which the worst distance (max_distance) is updated and broadcast to the warp.
__device__ __forceinline__ void merge_candidates(int warp_id_in_block, int k,
                                                  volatile Candidate* best_buf,      // best buffer for all warps; each warp has MAX_K entries.
                                                  volatile Candidate* cand_buf,      // candidate buffer for all warps; each warp has k entries.
                                                  volatile int* cand_counts,         // candidate count per warp.
                                                  float &current_max,                // local register variable for max distance (will be updated).
                                                  int lane)
{
    // Read candidate count for this warp.
    int m = cand_counts[warp_id_in_block];  // m may be k (when full) or less (at final merge).
    int total = k + m;  // Total number of candidates to merge with current best.

    // Temporary local storage for merging. Since k <= MAX_K (1024), total <= 2048.
    // We allocate a local array of Candidates; only lane 0 will use it.
    Candidate temp[2048];

    if (lane == 0)
    {
        int best_base = warp_id_in_block * MAX_K; // best_buf region for this warp uses MAX_K entries.
        int cand_base = warp_id_in_block * k;      // candidate buffer region for this warp uses k entries.

        // Copy current best candidate list (of size k) from best_buf.
        for (int i = 0; i < k; i++) {
            temp[i].idx = best_buf[best_base + i].idx;
            temp[i].dist = best_buf[best_base + i].dist;
        }
        // Copy candidate buffer (m candidates) from cand_buf.
        for (int i = 0; i < m; i++) {
            temp[k + i].idx = cand_buf[cand_base + i].idx;
            temp[k + i].dist = cand_buf[cand_base + i].dist;
        }
        // Simple selection sort to select the k nearest (i.e. smallest) candidates.
        // For i in [0, k), find the minimum among positions [i, total) and swap.
        for (int i = 0; i < k; i++) {
            int minIndex = i;
            for (int j = i + 1; j < total; j++) {
                if (temp[j].dist < temp[minIndex].dist)
                    minIndex = j;
            }
            // Swap temp[i] and temp[minIndex]
            Candidate tmp = temp[i];
            temp[i] = temp[minIndex];
            temp[minIndex] = tmp;
        }
        // Write the merged k best candidates back to best_buf.
        for (int i = 0; i < k; i++) {
            best_buf[best_base + i].idx = temp[i].idx;
            best_buf[best_base + i].dist = temp[i].dist;
        }
        // Update max distance using the kth candidate (last in best list).
        current_max = temp[k - 1].dist;
        // Reset the candidate count for this warp.
        cand_counts[warp_id_in_block] = 0;
    }
    // Synchronize all lanes of the warp.
    __syncwarp(0xFFFFFFFF);
    // Broadcast updated current_max from lane 0 to all lanes.
    current_max = __shfl_sync(0xFFFFFFFF, current_max, 0);
}

// The main k-NN kernel. Each warp processes one query.
// Shared memory is used for:
//   1. BATCH_SIZE float2 elements (data batch)
//   2. A candidate buffer for each warp (k Candidate elements per warp)
//   3. A candidate count for each warp (1 int per warp)
//   4. A best buffer for each warp (MAX_K Candidate elements per warp; only first k used)
__global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                           const float2* __restrict__ data, int data_count,
                           std::pair<int, float>* result, int k)
{
    // Determine lane and warp indices.
    int lane      = threadIdx.x & 31;            // lane index in warp [0,31]
    int warp_id_in_block = threadIdx.x >> 5;       // warp index within block
    int warps_per_block = blockDim.x >> 5;         // number of warps per block
    int global_warp_id  = blockIdx.x * warps_per_block + warp_id_in_block;  // global warp id

    // Each warp processes one query.
    int query_idx = global_warp_id;
    if (query_idx >= query_count)
        return;

    // Load the query point for this warp. Use lane 0 to load,
    // then broadcast to the other lanes within the warp.
    float2 q;
    if (lane == 0)
        q = query[query_idx];
    q.x = __shfl_sync(0xFFFFFFFF, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFF, q.y, 0);

    // Partition the dynamic shared memory.
    // Shared memory layout:
    // [0, BATCH_SIZE):           float2 shared_data[BATCH_SIZE]
    // [BATCH_SIZE, ...):         candidate buffer for all warps: (warps_per_block * k) Candidates.
    // Next region:                candidate counts for each warp: warps_per_block ints.
    // Next region:                best buffer for all warps: (warps_per_block * MAX_K) Candidates.
    extern __shared__ char smem[];
    float2* shared_data = (float2*)smem; // Batch data storage.

    // Offset for candidate buffer.
    Candidate* cand_buf = (Candidate*)(shared_data + BATCH_SIZE);
    // Offset for candidate counts.
    int* cand_counts = (int*)((char*)cand_buf + (warps_per_block * k * sizeof(Candidate)));
    // Offset for best buffer.
    Candidate* best_buf = (Candidate*)((char*)cand_counts + (warps_per_block * sizeof(int)));

    // Initialize candidate count and best buffer for this warp.
    if (lane == 0) {
        cand_counts[warp_id_in_block] = 0;
        int best_base = warp_id_in_block * MAX_K;
        for (int i = 0; i < k; i++) {
            best_buf[best_base + i].idx = -1;
            best_buf[best_base + i].dist = FLT_MAX;
        }
    }
    // Initialize a local variable for the current worst accepted distance.
    float current_max = FLT_MAX;

    // Process the data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE)
    {
        // Determine number of points in this batch.
        int batch_count = (batch_start + BATCH_SIZE < data_count) ? BATCH_SIZE : (data_count - batch_start);

        // Load batch of data points into shared memory.
        for (int i = threadIdx.x; i < batch_count; i += blockDim.x)
        {
            shared_data[i] = data[batch_start + i];
        }
        __syncthreads();  // Ensure the entire batch is loaded.

        // Each warp processes the batch: each of its 32 lanes iterates over shared_data with stride 32.
        for (int i = lane; i < batch_count; i += 32)
        {
            float2 pt = shared_data[i];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float d = dx * dx + dy * dy;

            // If the computed distance is better than (less than) the current worst distance,
            // attempt to add it to the candidate buffer.
            if (d < current_max)
            {
                bool inserted = false;
                // Loop until the candidate is successfully inserted.
                while (!inserted)
                {
                    int count = cand_counts[warp_id_in_block];  // read current candidate count.
                    if (count < k)
                    {
                        // Try to reserve a slot in candidate buffer.
                        int pos = atomicAdd(&cand_counts[warp_id_in_block], 1);
                        if (pos < k)
                        {
                            int buf_index = warp_id_in_block * k + pos;
                            cand_buf[buf_index].idx = batch_start + i;  // Global data index.
                            cand_buf[buf_index].dist = d;
                            inserted = true;
                            // If we just filled the candidate buffer, trigger a merge.
                            if (pos == k - 1)
                            {
                                merge_candidates(warp_id_in_block, k, best_buf, cand_buf, cand_counts, current_max, lane);
                            }
                        }
                    }
                    else
                    {
                        // If the candidate buffer is full, merge its content with the best result.
                        if (lane == 0) {
                            merge_candidates(warp_id_in_block, k, best_buf, cand_buf, cand_counts, current_max, lane);
                        }
                        __syncwarp(0xFFFFFFFF);
                        // After merge, try to insert again.
                    }
                } // end while
            }
        } // end for each point in batch

        // All warp lanes synchronize before loading the next batch.
        __syncwarp(0xFFFFFFFF);
        __syncthreads();
    } // end for each batch

    // After processing all batches, if any candidates remain in the candidate buffer, do a final merge.
    if (cand_counts[warp_id_in_block] > 0)
    {
        if (lane == 0)
            merge_candidates(warp_id_in_block, k, best_buf, cand_buf, cand_counts, current_max, lane);
        __syncwarp(0xFFFFFFFF);
    }

    // Write out the k nearest neighbors for this query.
    // Only lane 0 writes the final result.
    if (lane == 0)
    {
        int best_base = warp_id_in_block * MAX_K;
        int out_base = query_idx * k;
        // Write each candidate as a std::pair<int,float> (index, squared distance).
        for (int i = 0; i < k; i++) {
            result[out_base + i] = std::pair<int, float>( best_buf[best_base + i].idx, best_buf[best_base + i].dist );
        }
    }
}

// Host function that launches the k-NN kernel.
// Parameters:
//   query         : pointer to an array of float2 query points (allocated by cudaMalloc)
//   query_count   : number of query points
//   data          : pointer to an array of float2 data points (allocated by cudaMalloc)
//   data_count    : number of data points
//   result        : pointer to an array of std::pair<int,float> to store the results (allocated by cudaMalloc)
//   k             : number of nearest neighbors (power-of-two between 32 and 1024)
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k)
{
    // Choose the block configuration.
    // We assign one warp (32 threads) per query.
    // For efficiency, we use 4 warps per block (i.e. 128 threads per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each block processes warpsPerBlock queries.
    int totalWarps = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    int numBlocks = totalWarps;

    // Compute required shared memory size per block:
    // 1. Batch data: BATCH_SIZE * sizeof(float2)
    // 2. Candidate buffer: (warpsPerBlock * k * sizeof(Candidate))
    // 3. Candidate counts: (warpsPerBlock * sizeof(int))
    // 4. Best buffer: (warpsPerBlock * MAX_K * sizeof(Candidate))
    size_t shmemSize = BATCH_SIZE * sizeof(float2)
                     + warpsPerBlock * k * sizeof(Candidate)
                     + warpsPerBlock * sizeof(int)
                     + warpsPerBlock * MAX_K * sizeof(Candidate);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, shmemSize>>>(query, query_count, data, data_count, result, k);
    // Error check and synchronization can be added as needed.
}
