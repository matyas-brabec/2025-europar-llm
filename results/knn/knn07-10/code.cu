#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// Simple struct to hold neighbor data (distance and index) in shared memory.
// The layout is (float, int), but this is independent of std::pair<int,float>.
struct Neighbor {
    float dist;   // squared distance
    int   index;  // index of the data point
};

static constexpr int WARP_SIZE         = 32;
static constexpr int WARPS_PER_BLOCK   = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int BATCH_SIZE        = 2048;

// Warp-wide Bitonic Sort for Neighbor arrays in shared memory.
// - arr: pointer to the first element of the array to sort.
// - n:   number of elements, must be a power of two, n <= 1024.
// Sorting key is 'dist' in ascending order.
__device__ __forceinline__
void bitonic_sort_warp(Neighbor *arr, int n)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);

    // Implementation based on the standard bitonic sorting network.
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = laneId; i < n; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    Neighbor a = arr[i];
                    Neighbor b = arr[ixj];
                    bool up = ((i & k) == 0);
                    bool comp = (a.dist > b.dist);
                    if ((up && comp) || (!up && !comp)) {
                        arr[i]  = b;
                        arr[ixj] = a;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

// Merge the candidate buffer with the current intermediate k-NN result for one warp.
// Preconditions:
//  - 'knn' holds the current intermediate result, sorted ascending by distance.
//  - 'cand[0..cand_count-1]' holds newly found candidates; cand_count <= k.
//  - k is a power of two between 32 and 1024.
// Steps performed:
//  0. Pad the candidate buffer up to k elements with +INF distances.
//  1. Sort the candidate buffer using Bitonic Sort (ascending).
//  2. Merge 'cand' and 'knn' by taking for each i the minimum of knn[i] and cand[k-1-i],
//     storing the result back into knn[i]. This yields a bitonic sequence in 'knn'.
//  3. Sort 'knn' using Bitonic Sort (ascending) to restore the invariant.
//  4. Update max_distance and reset candidate count for this warp.
__device__ __forceinline__
void merge_buffer(
    Neighbor *knn,
    Neighbor *cand,
    int k,
    int cand_count,
    float *sh_max_dists,
    int *sh_cand_counts,
    int warpId)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const int laneId = threadIdx.x & (WARP_SIZE - 1);

    // Pad remaining candidate slots with infinite distance so that
    // we can always sort 'k' elements.
    for (int i = laneId + cand_count; i < k; i += WARP_SIZE) {
        cand[i].dist  = CUDART_INF_F;
        cand[i].index = -1;
    }
    __syncwarp(FULL_MASK);

    // (1) Sort the candidate buffer.
    bitonic_sort_warp(cand, k);

    // (2) Merge 'cand' and 'knn' into a bitonic sequence stored in 'knn'.
    for (int i = laneId; i < k; i += WARP_SIZE) {
        Neighbor a = knn[i];
        Neighbor b = cand[k - 1 - i];
        if (b.dist < a.dist) {
            knn[i] = b;
        }
    }
    __syncwarp(FULL_MASK);

    // (3) Sort the merged result to restore ascending order.
    bitonic_sort_warp(knn, k);

    // (4) Update max_distance (distance of the k-th nearest neighbor)
    //     and reset candidate count for this warp.
    if (laneId == 0) {
        sh_max_dists[warpId]   = knn[k - 1].dist;
        sh_cand_counts[warpId] = 0;
    }
    __syncwarp(FULL_MASK);
}

// CUDA kernel: each warp processes one query point and computes its k nearest neighbors.
__global__ void knn_kernel(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    std::pair<int, float> * __restrict__ result,
    int k)
{
    const int warpId         = threadIdx.x / WARP_SIZE;         // Warp index in block
    const int laneId         = threadIdx.x & (WARP_SIZE - 1);   // Lane index within warp
    const int warpsPerBlock  = blockDim.x / WARP_SIZE;
    const int globalWarpId   = blockIdx.x * warpsPerBlock + warpId;
    const bool active        = (globalWarpId < query_count);
    const unsigned FULL_MASK = 0xffffffffu;

    // Dynamic shared memory layout:
    // [0]  float2 sh_points[BATCH_SIZE]
    // [1]  int    sh_cand_counts[warpsPerBlock]
    // [2]  float  sh_max_dists[warpsPerBlock]
    // [3]  Neighbor sh_knn_all[warpsPerBlock * k]
    // [4]  Neighbor sh_cand_all[warpsPerBlock * k]
    extern __shared__ unsigned char shared_mem[];
    size_t offset = 0;

    // Shared buffer of data points for the current batch (shared by all warps in the block).
    float2 *sh_points = reinterpret_cast<float2*>(shared_mem + offset);
    offset += sizeof(float2) * BATCH_SIZE;

    // Per-warp candidate counts.
    int *sh_cand_counts = reinterpret_cast<int*>(shared_mem + offset);
    offset += sizeof(int) * warpsPerBlock;

    // Per-warp max_distance (distance of current k-th nearest neighbor).
    float *sh_max_dists = reinterpret_cast<float*>(shared_mem + offset);
    offset += sizeof(float) * warpsPerBlock;

    // Per-warp intermediate k-NN results.
    Neighbor *sh_knn_all = reinterpret_cast<Neighbor*>(shared_mem + offset);
    offset += sizeof(Neighbor) * warpsPerBlock * k;

    // Per-warp candidate buffers.
    Neighbor *sh_cand_all = reinterpret_cast<Neighbor*>(shared_mem + offset);
    // offset += sizeof(Neighbor) * warpsPerBlock * k; // Not needed further

    Neighbor *warp_knn  = sh_knn_all  + warpId * k;
    Neighbor *warp_cand = sh_cand_all + warpId * k;

    // Initialize per-warp shared state.
    if (laneId == 0) {
        if (warpId < warpsPerBlock) {
            sh_cand_counts[warpId] = 0;
            sh_max_dists[warpId]   = CUDART_INF_F;
        }
    }

    // Initialize the intermediate k-NN result for active warps.
    if (active) {
        for (int i = laneId; i < k; i += WARP_SIZE) {
            warp_knn[i].dist  = CUDART_INF_F;
            warp_knn[i].index = -1;
        }
    }
    __syncthreads();

    // Load the query point for this warp and broadcast to all lanes.
    float2 q;
    q.x = 0.0f;
    q.y = 0.0f;
    if (active && laneId == 0) {
        q = query[globalWarpId];
    }
    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    // Process all data points in batches cached in shared memory.
    for (int dataStart = 0; dataStart < data_count; dataStart += BATCH_SIZE) {
        int batchSize = data_count - dataStart;
        if (batchSize > BATCH_SIZE) batchSize = BATCH_SIZE;

        // Load the batch of data points into shared memory using the whole block.
        for (int idx = threadIdx.x; idx < batchSize; idx += blockDim.x) {
            sh_points[idx] = data[dataStart + idx];
        }
        __syncthreads();

        if (!active) {
            __syncthreads();
            continue;
        }

        // Each warp processes all points in the current batch for its query.
        for (int base = 0; base < batchSize; base += WARP_SIZE) {
            int idx_in_batch = base + laneId;
            bool valid = (idx_in_batch < batchSize);

            float dist        = 0.0f;
            int   point_index = -1;
            bool  is_candidate = false;

            float max_dist = sh_max_dists[warpId];

            if (valid) {
                float2 p = sh_points[idx_in_batch];
                float dx = q.x - p.x;
                float dy = q.y - p.y;
                dist = dx * dx + dy * dy;

                // Filter by current max_distance threshold.
                if (dist < max_dist) {
                    is_candidate = true;
                    point_index  = dataStart + idx_in_batch;
                }
            }

            // Identify candidate lanes within the warp.
            unsigned int cand_mask = __ballot_sync(FULL_MASK, is_candidate);

            // Process candidates one by one; lane 0 manages the shared buffer.
            while (cand_mask) {
                int srcLane = __ffs(cand_mask) - 1;

                float cdist  = __shfl_sync(FULL_MASK, dist, srcLane);
                int   cindex = __shfl_sync(FULL_MASK, point_index, srcLane);

                bool trigger_merge = false;
                int cand_count_after = 0;

                if (laneId == 0) {
                    int pos = atomicAdd(&sh_cand_counts[warpId], 1);
                    warp_cand[pos].dist  = cdist;
                    warp_cand[pos].index = cindex;
                    cand_count_after = pos + 1;
                    trigger_merge = (cand_count_after >= k);
                }

                // Broadcast the decision to merge and the candidate count after insertion.
                trigger_merge    = __shfl_sync(FULL_MASK, trigger_merge, 0);
                cand_count_after = __shfl_sync(FULL_MASK, cand_count_after, 0);

                if (trigger_merge) {
                    merge_buffer(
                        warp_knn,
                        warp_cand,
                        k,
                        cand_count_after,
                        sh_max_dists,
                        sh_cand_counts,
                        warpId);
                }

                // Remove the processed lane from the mask.
                cand_mask &= cand_mask - 1;
                __syncwarp(FULL_MASK);
            }
        }

        __syncthreads();
    }

    // After processing all batches, merge any remaining candidates for active warps.
    if (active) {
        int final_count = 0;
        if (laneId == 0) {
            final_count = sh_cand_counts[warpId];
        }
        final_count = __shfl_sync(FULL_MASK, final_count, 0);

        if (final_count > 0) {
            merge_buffer(
                warp_knn,
                warp_cand,
                k,
                final_count,
                sh_max_dists,
                sh_cand_counts,
                warpId);
        }

        // Write the final k nearest neighbors for this query to the global result array.
        for (int i = laneId; i < k; i += WARP_SIZE) {
            int out_idx = globalWarpId * k + i;
            result[out_idx].first  = warp_knn[i].index;
            result[out_idx].second = warp_knn[i].dist;
        }
    }
}

// Host function that launches the CUDA kernel.
// - query:  device pointer to query points (float2[query_count]).
// - data:   device pointer to data points  (float2[data_count]).
// - result: device pointer to result pairs (pair<int,float>[query_count * k]).
void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k)
{
    // One warp per query.
    const int warpsPerBlock   = WARPS_PER_BLOCK;
    const int threadsPerBlock = THREADS_PER_BLOCK;

    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Dynamic shared memory size:
    //  - BATCH_SIZE float2 elements for cached data points.
    //  - For each warp:
    //      * 1 int    candidate count.
    //      * 1 float  max_distance.
    //      * k Neighbor elements for intermediate results.
    //      * k Neighbor elements for candidate buffer.
    size_t sharedBytes =
        sizeof(float2) * BATCH_SIZE +
        warpsPerBlock * (sizeof(int) + sizeof(float) + 2 * k * sizeof(Neighbor));

    dim3 grid(numBlocks);
    dim3 block(threadsPerBlock);

    knn_kernel<<<grid, block, sharedBytes>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k);
}