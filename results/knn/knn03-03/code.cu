#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// This optimized CUDA kernel implements k‑nearest neighbors (k‑NN) in 2D Euclidean space.
// Each warp (32 threads) processes one query point. The warp maintains an intermediate
// candidate list of k nearest neighbors (squared distances and indices), distributed evenly
// across its 32 threads (each thread holds k/32 candidates in registers, sorted in ascending order).
// Data points (float2) are processed in batches that are first loaded into shared memory.
// Each warp computes distances from its query point to the batch of data points and then updates
// its intermediate candidate list in a warp‐synchronized “merge‐insertion” procedure using warp
// shuffles. Finally, the candidates maintained in registers are merged (via a bitonic sort in shared memory)
// and the sorted k–NN result is written globally.
// This code targets modern GPUs (e.g. A100/H100) and is compiled with the latest CUDA toolkit.

#define WARP_SIZE 32
// Choose an appropriate batch size that fits in shared memory. 1024 points take 8KB of shared memory.
#define BATCH_SIZE 1024

// The kernel expects that the shared memory allocation is at least:
//   (BATCH_SIZE * sizeof(float2)) + (k * sizeof(int2))
// The first part is used for caching the data points, the second for final merge of candidates.

// Note: We use int2 to temporarily store a pair (index, distance) for the final merge,
// with the distance stored in the int bit‐representation via __float_as_int.

__global__ void knn_kernel(const float2 *query, int query_count, const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp processes one query.
    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    const int globalWarpId = blockIdx.x * warpsPerBlock + (threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x % WARP_SIZE;

    if (globalWarpId >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[globalWarpId];

    // Each warp maintains an intermediate candidate list of k elements distributed across its 32 threads.
    // Each thread holds a sorted (ascending by distance) list of r = k/WARP_SIZE candidates in registers.
    const int r = k / WARP_SIZE;  // k is assumed to be a power-of-two between 32 and 1024.
    float cand_d[32];  // maximum r is 1024/32 = 32 candidates per thread.
    int   cand_idx[32];

    // Initialize each thread's candidate list with FLT_MAX distances and invalid indices (-1).
    #pragma unroll
    for (int i = 0; i < r; i++) {
        cand_d[i] = FLT_MAX;
        cand_idx[i] = -1;
    }

    // Shared memory pointer: the first part is for the current batch of data points.
    extern __shared__ float shared_mem[];
    // Data cache for the batch: BATCH_SIZE float2 elements.
    /// @FIXED
    /// float2 *dataBatch = shared_mem;
    float2 *dataBatch = (float2*)shared_mem;
    // The final merge buffer (for each warp) will be allocated after the first BATCH_SIZE float2.
    // We'll reinterpret that portion as int2 later.

    // Process the full data set in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        // Load batch of data points into shared memory.
        // Each thread in the block loads points with stride blockDim.x.
        for (int i = threadIdx.x; i < BATCH_SIZE && (batch_start + i) < data_count; i += blockDim.x) {
            dataBatch[i] = data[batch_start + i];
        }
        __syncthreads(); // ensure the batch is loaded

        // Each warp processes the cached batch.  We let each thread in the warp loop over
        // the batch indices starting from its lane ID, stepping by 32.
        // Each thread temporarily stores its computed candidate distances (pending candidates)
        // from the batch in a local buffer.
        const int maxPending = (BATCH_SIZE + WARP_SIZE - 1) / WARP_SIZE; // maximum number of candidates per lane in the batch
        float pending[32];
        int   pending_idx[32];
        int pendingCount = 0;

        for (int i = lane; i < BATCH_SIZE && (batch_start + i) < data_count; i += WARP_SIZE) {
            float2 pt = dataBatch[i];
            float dx = pt.x - q.x;
            float dy = pt.y - q.y;
            float dist = dx * dx + dy * dy;
            pending[pendingCount] = dist;
            pending_idx[pendingCount] = batch_start + i; // global index in data
            pendingCount++;
        }
        // Process the pending candidates from this batch.
        // First, get the maximum pending count across the warp.
        int maxPendingWarp = pendingCount;
        // Warp reduction for maximum (using shuffle). Assume full mask.
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            int other = __shfl_down_sync(0xFFFFFFFF, maxPendingWarp, offset);
            if (other > maxPendingWarp)
                maxPendingWarp = other;
        }

        // For each "round" from 0 to maxPendingWarp-1, process candidates from all lanes in warp.
        for (int round = 0; round < maxPendingWarp; round++) {
            // For each lane in the warp (serialized by a loop over proc from 0 to 31),
            // process that lane's pending candidate for the current round.
            for (int proc = 0; proc < WARP_SIZE; proc++) {
                // Each thread extracts the candidate from lane 'proc' if available; else a dummy (FLT_MAX).
                float candCandidate = (round < pendingCount && lane == proc) ? pending[round] : FLT_MAX;
                int   candCandidateIdx = (round < pendingCount && lane == proc) ? pending_idx[round] : -1;
                // Broadcast the candidate from lane 'proc' to all lanes.
                candCandidate = __shfl_sync(0xFFFFFFFF, candCandidate, proc);
                candCandidateIdx = __shfl_sync(0xFFFFFFFF, candCandidateIdx, proc);

                // Compute the current global worst candidate in the warp.
                // Each thread's worst candidate is the last element in its sorted list.
                float myWorst = cand_d[r - 1];
                float globalWorst = myWorst;
                int globalWinner = lane; // candidate index holder (within the warp) for the worst candidate.
                // Use warp reduction to determine the maximum value among cand_d[r-1] and get the lane id.
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                    float other = __shfl_down_sync(0xFFFFFFFF, globalWorst, offset);
                    int otherLane = __shfl_down_sync(0xFFFFFFFF, globalWinner, offset);
                    if (other > globalWorst) {
                        globalWorst = other;
                        globalWinner = otherLane;
                    }
                }
                // If the new candidate is better than the current global worst, update the candidate list.
                if (candCandidate < globalWorst) {
                    // Only the thread holding the worst candidate (globalWinner) updates its local candidate list.
                    if (lane == globalWinner) {
                        // Insert candCandidate into the sorted array cand_d[0..r-1] (ascending order).
                        int pos = r - 1;
                        while (pos > 0 && candCandidate < cand_d[pos - 1]) {
                            cand_d[pos] = cand_d[pos - 1];
                            cand_idx[pos] = cand_idx[pos - 1];
                            pos--;
                        }
                        cand_d[pos] = candCandidate;
                        cand_idx[pos] = candCandidateIdx;
                    }
                }
                __syncwarp(); // synchronize warp after each candidate update.
            }
        }
        __syncthreads();
    } // end batch loop

    // At this point, each warp has an intermediate candidate list distributed across its threads:
    // each of the 32 lanes holds r candidates (total k = r*32 elements) in sorted (ascending) order locally.
    // Next, we must merge these 32 sorted lists into one sorted list of k elements.
    // We copy the per-thread candidate arrays (each of length r) into a contiguous buffer in shared memory,
    // then perform an in-warp bitonic sort over the k elements.

    // The final merge buffer is allocated after the data batch shared memory.
    // Reinterpret the shared memory pointer and offset by BATCH_SIZE float2.
    int offsetBytes = BATCH_SIZE * sizeof(float2);
    // We assume that shared memory was allocated with total size at least: offsetBytes + k*sizeof(int2).
    int2 *finalBuf = (int2*)(((char*)shared_mem) + offsetBytes);
    // Each thread writes its r candidates into the final buffer.
    // We store each candidate as an int2 where .x = candidate index, and .y holds the float bits of distance.
    #pragma unroll
    for (int i = 0; i < r; i++) {
        // We store the candidate in column-major order: index = lane + i * WARP_SIZE.
        finalBuf[lane + i * WARP_SIZE].x = cand_idx[i];
        finalBuf[lane + i * WARP_SIZE].y = __float_as_int(cand_d[i]);
    }
    __syncwarp();

    // Now perform an in-warp bitonic sort on the k elements stored in finalBuf.
    // The total number of elements is k, which is a power-of-two.
    const int total = k;
    // Bitonic sort: iterate over log2(total) stages.
    for (int size = 2; size <= total; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            // Each thread processes multiple indices in the final buffer.
            for (int i = lane; i < total; i += WARP_SIZE) {
                int ixj = i ^ stride;
                if (ixj > i && ixj < total) {
                    // Determine sorting order. We want overall ascending order.
                    // The direction is determined by whether (i & size) == 0.
                    bool ascending = ((i & size) == 0);
                    int2 elem_i = finalBuf[i];
                    int2 elem_ixj = finalBuf[ixj];
                    float d_i = __int_as_float(elem_i.y);
                    float d_ixj = __int_as_float(elem_ixj.y);
                    // Compare and swap if they are out-of-order relative to ascending.
                    if ((ascending && d_i > d_ixj) || (!ascending && d_i < d_ixj)) {
                        finalBuf[i] = elem_ixj;
                        finalBuf[ixj] = elem_i;
                    }
                }
            }
            __syncwarp();
        }
    }
    __syncwarp();

    // Finally, write the sorted k nearest neighbors into the result array.
    // For query globalWarpId, the output segment is result[globalWarpId * k] ... result[globalWarpId * k + k - 1].
    for (int i = lane; i < total; i += WARP_SIZE) {
        int2 cand = finalBuf[i];
        // The result type is std::pair<int, float>.
        result[globalWarpId * k + i].first = cand.x;
        result[globalWarpId * k + i].second = __int_as_float(cand.y);
    }
}


// Host interface function. It launches the k-NN kernel.
// The caller is responsible for allocating device memory for 'query', 'data', and 'result' (via cudaMalloc).
// 'query' : pointer to float2 array of query points (query_count elements)
// 'data'  : pointer to float2 array of data points (data_count elements)
// 'result': pointer to std::pair<int, float> array (query_count * k elements)
// 'k'     : number of neighbors to return (power-of-two between 32 and 1024)
// Other parameters are assumed valid.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Set the number of threads per block.
    // We choose 256 threads per block (8 warps per block).
    const int threadsPerBlock = 256;
    // Each warp processes one query, so warpsPerBlock = 256/32 = 8.
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    // Total number of warps needed equals query_count.
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate shared memory size:
    // – BATCH_SIZE data points (float2) for the batch cache.
    // – k candidates (int2) for the final merge buffer.
    size_t sharedMemSize = BATCH_SIZE * sizeof(float2) + k * sizeof(int2);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // It is assumed that error checking and synchronization are handled externally.
}
