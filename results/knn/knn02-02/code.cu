// Optimized CUDA k-NN for 2D points with one warp (32 threads) per query.
// Target: modern NVIDIA data-center GPUs (e.g., A100, H100).
//
// Key design choices (also see comments inline):
//  - Each query is handled by a single warp (32 threads).
//  - Data points are processed in shared-memory tiles to reduce global memory traffic.
//  - For each query (warp), we maintain a private max-heap of size k in shared memory,
//    storing the current k nearest neighbors (indices + squared distances).
//  - The heap is updated by a single lane (lane 0) using warp-wide shuffles to
//    sequentially consider distances computed in parallel by all 32 threads.
//  - At the end, the heap is heapsorted in-place (by lane 0) to obtain neighbors
//    sorted by increasing distance, and written to the result array.
//

#include <cuda_runtime.h>
#include <cfloat>      // FLT_MAX
#include <utility>     // std::pair

// Tuning parameters for kernel configuration.
namespace {
    // Fixed warp size on NVIDIA GPUs.
    constexpr int KNN_WARP_SIZE = 32;

    // Number of warps (queries) processed per block.
    // 4 warps/block = 128 threads/block. This balances per-block shared memory
    // usage (for heaps) and occupancy on A100/H100 when k can be up to 1024.
    constexpr int KNN_WARPS_PER_BLOCK = 4;
    constexpr int KNN_THREADS_PER_BLOCK = KNN_WARP_SIZE * KNN_WARPS_PER_BLOCK;

    // Maximum k supported (per problem statement).
    constexpr int KNN_MAX_K = 1024;

    // Number of data points per shared-memory tile.
    // Each point is float2 (8 bytes), so 2048 points = 16 KB.
    // Per-block shared memory usage:
    //   - data tile:              2048 * 8 bytes  = 16 KB
    //   - heaps (dist+idx): 4 warps * 1024 * 8   = 32 KB
    //   Total per block: ~48 KB (allows 2 blocks/SM on GPUs with >=96 KB/SM).
    constexpr int KNN_TILE_POINTS = 2048;
}

// Device-side helper: sift-down operation for a max-heap of size heapSize.
//
// The heap is stored in parallel arrays heapDist[] and heapIdx[].
// This function assumes that all elements except possibly the node at "start"
// satisfy the max-heap property. It restores the max-heap property by sifting
// down from "start".
__device__ __forceinline__
void knn_heapify_down(float *heapDist, int *heapIdx, int heapSize, int start)
{
    int i = start;
    while (true) {
        int left  = (i << 1) + 1;
        if (left >= heapSize) break;
        int right = left + 1;

        // Find child with largest distance.
        int largest = left;
        if (right < heapSize && heapDist[right] > heapDist[left]) {
            largest = right;
        }

        // If parent is already larger or equal, heap property holds.
        if (heapDist[i] >= heapDist[largest]) {
            break;
        }

        // Swap parent with larger child.
        float tmpD = heapDist[i];
        heapDist[i] = heapDist[largest];
        heapDist[largest] = tmpD;

        int tmpI = heapIdx[i];
        heapIdx[i] = heapIdx[largest];
        heapIdx[largest] = tmpI;

        i = largest;
    }
}

// Device-side helper: replace the root of a max-heap with a new candidate
// if and only if the candidate distance is smaller than the current maximum.
//
// heapDist[0] is the current worst (largest) distance in the heap.
// If candDist >= heapDist[0], the candidate is not among the k nearest and
// is discarded. Otherwise, the root is replaced and the heap is re-heapified.
__device__ __forceinline__
void knn_heap_replace_root_if_better(float candDist, int candIdx,
                                     float *heapDist, int *heapIdx,
                                     int k)
{
    // If the heap's current worst is better than or equal to the candidate,
    // no update is needed.
    if (candDist >= heapDist[0]) {
        return;
    }

    // Replace root with new candidate and restore max-heap property.
    heapDist[0] = candDist;
    heapIdx[0]  = candIdx;
    knn_heapify_down(heapDist, heapIdx, k, 0);
}

// Device kernel: one warp (32 threads) processes one query point.
//
// - query:  array of query points (float2), length query_count
// - data:   array of data points (float2), length data_count
// - result: array of std::pair<int,float>, length query_count * k
// - k:      number of nearest neighbors to find (power of two, 32..1024)
__global__ void knn_kernel_2d_warp_per_query(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    std::pair<int, float> * __restrict__ result,
    int k)
{
    // Shared-memory tile for data points.
    __shared__ float2 sData[KNN_TILE_POINTS];

    // Per-warp heaps: distances and indices.
    // Each warp keeps its own private heap of size k (k <= KNN_MAX_K).
    __shared__ float sHeapDist[KNN_WARPS_PER_BLOCK][KNN_MAX_K];
    __shared__ int   sHeapIdx[KNN_WARPS_PER_BLOCK][KNN_MAX_K];

    const int tid    = threadIdx.x;
    const int warpId = tid / KNN_WARP_SIZE;       // warp index within block
    const int laneId = tid % KNN_WARP_SIZE;       // lane index within warp

    const int blockFirstWarp = blockIdx.x * KNN_WARPS_PER_BLOCK;
    const int totalWarps     = (query_count + KNN_WARP_SIZE - 1) / KNN_WARP_SIZE; // unused but illustrative
    (void)totalWarps; // silence unused warning

    // Number of active warps (queries) in this block.
    int activeWarpsInBlock = query_count - blockFirstWarp;
    if (activeWarpsInBlock > KNN_WARPS_PER_BLOCK)
        activeWarpsInBlock = KNN_WARPS_PER_BLOCK;
    if (activeWarpsInBlock < 0)
        activeWarpsInBlock = 0;

    const int globalWarpIdx = blockFirstWarp + warpId;
    const bool warpIsActive = (warpId < activeWarpsInBlock);

    // Load query point for this warp (if active).
    float2 q;
    if (warpIsActive) {
        q = query[globalWarpIdx];
    }

    // Initialize heaps for all active warps in this block.
    // Use all threads in the block cooperatively.
    //
    // For each active warp w and heap position pos < k:
    //    sHeapDist[w][pos] = FLT_MAX
    //    sHeapIdx[w][pos]  = -1
    //
    // This means we start with heaps filled with "infinite" distances; as we
    // scan data points, we keep replacing the worst neighbors.
    for (int i = tid; i < activeWarpsInBlock * k; i += blockDim.x) {
        int w   = i / k;
        int pos = i % k;
        sHeapDist[w][pos] = FLT_MAX;
        sHeapIdx[w][pos]  = -1;
    }

    __syncthreads();

    // Process data points in tiles cached in shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += KNN_TILE_POINTS) {
        int tileSize = data_count - tileStart;
        if (tileSize > KNN_TILE_POINTS) tileSize = KNN_TILE_POINTS;

        // Load data tile into shared memory cooperatively by the whole block.
        for (int i = tid; i < tileSize; i += blockDim.x) {
            sData[i] = data[tileStart + i];
        }

        // Ensure the tile is fully loaded before any warp uses it.
        __syncthreads();

        if (warpIsActive) {
            // Pointers to this warp's private heap.
            float *heapDist = sHeapDist[warpId];
            int   *heapIdx  = sHeapIdx[warpId];

            // Process the tile in groups of 32 points so that each lane
            // computes the distance to one point at a time.
            for (int base = 0; base < tileSize; base += KNN_WARP_SIZE) {
                const int idxInTile = base + laneId;

                float pointDist    = FLT_MAX;
                int   globalDataId = -1;
                bool  inRange      = (idxInTile < tileSize);

                if (inRange) {
                    float2 p = sData[idxInTile];
                    float dx = p.x - q.x;
                    float dy = p.y - q.y;
                    // Squared Euclidean distance in 2D.
                    pointDist    = dx * dx + dy * dy;
                    globalDataId = tileStart + idxInTile;
                }

                // Determine which lanes have valid candidates in this group.
                unsigned int activeMask = __ballot_sync(0xffffffffu, inRange);

                // Sequentially (per candidate) present each lane's candidate
                // to lane 0 for a possible heap update.
                //
                // Each iteration of the while-loop processes exactly one
                // candidate (the one from srcLane), using warp shuffles to
                // broadcast its distance and index to lane 0.
                while (activeMask) {
                    // Index of the next lane with a valid candidate.
                    int srcLane = __ffs(activeMask) - 1;

                    // Broadcast candidate distance and index from srcLane.
                    float candDist = __shfl_sync(0xffffffffu, pointDist,    srcLane);
                    int   candIdx  = __shfl_sync(0xffffffffu, globalDataId, srcLane);

                    // Only lane 0 performs heap updates for this warp.
                    if (laneId == 0) {
                        knn_heap_replace_root_if_better(candDist, candIdx,
                                                        heapDist, heapIdx, k);
                    }

                    // Clear the processed candidate bit.
                    activeMask &= (activeMask - 1);
                }
            }
        }

        // Ensure all warps are done using this tile before it is overwritten.
        __syncthreads();
    }

    // After all tiles are processed, each active warp has its k nearest
    // neighbors in a max-heap stored in sHeapDist[warpId][0..k-1] and
    // sHeapIdx[warpId][0..k-1].
    //
    // Now we need to sort them by increasing distance. We can perform an
    // in-place heapsort using the same max-heap data structure:
    //
    //   - Repeatedly swap the root (largest) with the last element of the heap,
    //     shrink the heap size, and sift down.
    //   - After this process, the array will be sorted in ascending order.
    if (warpIsActive && laneId == 0) {
        float *heapDist = sHeapDist[warpId];
        int   *heapIdx  = sHeapIdx[warpId];

        // Perform heapsort on [0, k).
        for (int end = k - 1; end > 0; --end) {
            // Move current maximum (root) to position end.
            float tmpD = heapDist[0];
            heapDist[0] = heapDist[end];
            heapDist[end] = tmpD;

            int tmpI = heapIdx[0];
            heapIdx[0] = heapIdx[end];
            heapIdx[end] = tmpI;

            // Restore heap property on [0, end).
            knn_heapify_down(heapDist, heapIdx, end, 0);
        }

        // Write sorted results (ascending distances) to global memory.
        // For query "globalWarpIdx", its j-th nearest neighbor (0-based) is
        // stored at result[globalWarpIdx * k + j].
        int baseOut = globalWarpIdx * k;
        for (int j = 0; j < k; ++j) {
            result[baseOut + j].first  = heapIdx[j];
            result[baseOut + j].second = heapDist[j];
        }
    }
}

// Host-side interface as specified.
//
// All pointers (query, data, result) are assumed to point to device memory
// allocated with cudaMalloc. This function launches the CUDA kernel and
// waits for completion (synchronous). For high performance in an application
// pipeline, the synchronization can be relaxed or removed as needed.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Guard against invalid k (should not happen per problem statement).
    if (k <= 0 || k > KNN_MAX_K) {
        return;
    }

    // Each query is processed by one warp.
    const int warpsPerBlock   = KNN_WARPS_PER_BLOCK;
    const int threadsPerBlock = KNN_THREADS_PER_BLOCK;

    const int totalWarps  = query_count; // one warp per query
    const int numBlocks   = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(numBlocks);

    knn_kernel_2d_warp_per_query<<<gridDim, blockDim>>>(
        query, query_count,
        data,  data_count,
        result, k);

    // Synchronize for completeness; may be removed for asynchronous usage.
    cudaDeviceSynchronize();
}