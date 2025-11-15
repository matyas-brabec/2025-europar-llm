#include <cuda_runtime.h>
#include <utility>

/*
 * This file implements a CUDA-based k-NN (k-nearest neighbors) search for 2D points.
 *
 * Key design points:
 *  - Each query point is processed by exactly one warp (32 threads).
 *  - The dataset is processed in tiles; each tile of data points is loaded once into
 *    shared memory by the whole block, and then reused by all warps in the block.
 *  - For each query (warp), we maintain an intermediate top-k result in shared memory
 *    as a max-heap (by squared distance). This heap is private to the warp.
 *  - Distances are computed collaboratively by all 32 threads in the warp.
 *  - Candidate distances are broadcast via warp shuffles to lane 0, which updates
 *    the top-k heap (one heap per warp). This keeps the heap logic simple and
 *    branch-free for other lanes.
 *  - At the end, the heap is turned into a sorted list (ascending distance) using
 *    heap sort, and all lanes in the warp cooperatively write results to global memory.
 *
 * Assumptions:
 *  - k is a power of two between 32 and 1024 inclusive.
 *  - data_count >= k.
 *  - query, data, and result point to device memory allocated via cudaMalloc.
 *  - Each point is a float2 (x, y).
 *  - Distances are squared Euclidean distances (no square root).
 */

#ifndef KNN_WARP_SIZE
#define KNN_WARP_SIZE 32
#endif

// Tunable hyper-parameters for this implementation.
static constexpr int KNN_MAX_K             = 1024;         // maximum supported k
static constexpr int KNN_WARPS_PER_BLOCK   = 8;            // warps per block (8 * 32 = 256 threads)
static constexpr int KNN_THREADS_PER_BLOCK = KNN_WARPS_PER_BLOCK * KNN_WARP_SIZE;
static constexpr int KNN_TILE_POINTS       = 1024;         // data points per shared-memory tile

// ----------------------------------------------------------------------------
// Device-side heap helpers (max-heap on squared distances).
// ----------------------------------------------------------------------------

/*
 * Insert a (dist, index) pair into a max-heap of capacity k.
 *
 *  - heapDist: array of distances (length >= k)
 *  - heapIndex: array of indices (length >= k)
 *  - heapSize: current number of elements in the heap (modified in-place)
 *  - dist: candidate squared distance
 *  - index: candidate data-point index
 *  - k: heap capacity (= desired k)
 *
 * The heap invariant:
 *  - For heapSize > 0, heapDist[0] is the largest distance in the heap (worst neighbor).
 *  - Children of node i are at 2*i + 1 and 2*i + 2.
 *
 * Behavior:
 *  - If heapSize < k, insert always.
 *  - Else, insert only if dist < heapDist[0] (better than the current worst).
 *
 * This function is intended to be called only by lane 0 of a warp.
 */
__device__ __forceinline__
void knn_heap_insert(float *heapDist,
                     int   *heapIndex,
                     int   &heapSize,
                     float  dist,
                     int    index,
                     int    k)
{
    // If there is still room, just insert and sift up.
    if (heapSize < k) {
        int i = heapSize;
        heapSize++;

        heapDist[i]  = dist;
        heapIndex[i] = index;

        // Sift up to restore max-heap property.
        while (i > 0) {
            int parent = (i - 1) >> 1;
            float parentDist = heapDist[parent];
            float currentDist = heapDist[i];
            if (parentDist >= currentDist) {
                break;
            }
            // Swap parent and current.
            heapDist[i]  = parentDist;
            heapIndex[i] = heapIndex[parent];

            heapDist[parent]  = currentDist;
            heapIndex[parent] = index;

            i = parent;
        }
    } else {
        // Heap full: only insert if this candidate is better (smaller distance)
        // than the worst currently in the heap (root).
        float rootDist = heapDist[0];
        if (dist >= rootDist) {
            return; // candidate is not among top-k
        }

        // Replace root with new candidate and sift down.
        heapDist[0]  = dist;
        heapIndex[0] = index;

        int i = 0;
        const int size = heapSize; // == k

        while (true) {
            int left  = (i << 1) + 1;
            int right = left + 1;

            if (left >= size) {
                break; // no children
            }

            // Find child with larger distance.
            int largest = left;
            float largestDist = heapDist[left];

            if (right < size) {
                float rightDist = heapDist[right];
                if (rightDist > largestDist) {
                    largest = right;
                    largestDist = rightDist;
                }
            }

            float currentDist = heapDist[i];
            if (currentDist >= largestDist) {
                break; // heap property satisfied
            }

            // Swap current with largest child.
            heapDist[i]  = largestDist;
            heapIndex[i] = heapIndex[largest];

            heapDist[largest]  = currentDist;
            heapIndex[largest] = heapIndex[i];

            i = largest;
        }
    }
}

/*
 * Convert a max-heap (heapDist, heapIndex) of length n into an array sorted
 * ascending by distance using heap sort in-place.
 *
 * After this function:
 *  - heapDist[0] .. heapDist[n-1] are in ascending order.
 *  - heapIndex[] reordered correspondingly.
 *
 * This function is intended to be called only by lane 0 of a warp.
 */
__device__ __forceinline__
void knn_heap_sort_ascending(float *heapDist,
                             int   *heapIndex,
                             int    n)
{
    // Standard heapsort on a max-heap:
    // repeatedly move the root (largest) to the end and restore heap on prefix.
    for (int end = n - 1; end > 0; --end) {
        // Swap root with end.
        float rootDist = heapDist[0];
        int   rootIdx  = heapIndex[0];

        heapDist[0]    = heapDist[end];
        heapIndex[0]   = heapIndex[end];

        heapDist[end]  = rootDist;
        heapIndex[end] = rootIdx;

        // Sift down heapDist[0] .. heapDist[end-1].
        int i = 0;
        while (true) {
            int left  = (i << 1) + 1;
            int right = left + 1;

            if (left >= end) {
                break; // no children within heap
            }

            int largest = left;
            float largestDist = heapDist[left];

            if (right < end) {
                float rightDist = heapDist[right];
                if (rightDist > largestDist) {
                    largest = right;
                    largestDist = rightDist;
                }
            }

            float currentDist = heapDist[i];
            if (currentDist >= largestDist) {
                break;
            }

            // Swap current with largest child.
            heapDist[i]  = largestDist;
            heapIndex[i] = heapIndex[largest];

            heapDist[largest]  = currentDist;
            heapIndex[largest] = heapIndex[i];

            i = largest;
        }
    }

    // After heapsort on max-heap, the array is in ascending order.
}

// ----------------------------------------------------------------------------
// CUDA kernel
// ----------------------------------------------------------------------------

/*
 * Kernel mapping:
 *  - Each warp in the grid processes exactly one query point.
 *  - Block contains KNN_WARPS_PER_BLOCK warps; thus each block processes
 *    KNN_WARPS_PER_BLOCK queries.
 *  - Shared memory layout per block:
 *      [0, KNN_TILE_POINTS)                : float2 s_data[] (tile of data points)
 *      [KNN_TILE_POINTS, ...)              : float s_heapDist[WARPS_PER_BLOCK][KNN_MAX_K]
 *      [.. after previous .., ..)          : int   s_heapIndex[WARPS_PER_BLOCK][KNN_MAX_K]
 */
__global__
void knn_kernel_2d_2nn(const float2 * __restrict__ query,
                       int                      query_count,
                       const float2 * __restrict__ data,
                       int                      data_count,
                       std::pair<int, float> * __restrict__ result,
                       int                      k)
{
    // Thread, warp, and block indexing.
    const int tid            = threadIdx.x;
    const int warpIdInBlock  = tid / KNN_WARP_SIZE;
    const int laneId         = tid & (KNN_WARP_SIZE - 1);
    const int warpGlobalId   = blockIdx.x * KNN_WARPS_PER_BLOCK + warpIdInBlock;

    // Whether this warp corresponds to a valid query.
    const bool activeWarp    = (warpGlobalId < query_count);

    // Shared memory layout.
    extern __shared__ unsigned char shared_raw[];
    // Data tile.
    float2 *s_data = reinterpret_cast<float2*>(shared_raw);

    // Offsets for heap storage.
    const size_t dataTileBytes   = static_cast<size_t>(KNN_TILE_POINTS) * sizeof(float2);
    float *s_heapDist = reinterpret_cast<float*>(shared_raw + dataTileBytes);
    const size_t heapDistBytes   = static_cast<size_t>(KNN_WARPS_PER_BLOCK) *
                                   static_cast<size_t>(KNN_MAX_K) * sizeof(float);
    int   *s_heapIndex = reinterpret_cast<int*>(shared_raw + dataTileBytes + heapDistBytes);

    // Per-warp heap pointers in shared memory.
    float *heapDist  = s_heapDist  + static_cast<size_t>(warpIdInBlock) * KNN_MAX_K;
    int   *heapIndex = s_heapIndex + static_cast<size_t>(warpIdInBlock) * KNN_MAX_K;

    // Load query point for this warp and broadcast to all lanes.
    float qx = 0.0f;
    float qy = 0.0f;
    if (activeWarp && laneId == 0) {
        float2 q = query[warpGlobalId];
        qx = q.x;
        qy = q.y;
    }
    if (activeWarp) {
        // All threads in this warp participate, so full mask is fine.
        unsigned mask = 0xffffffffu;
        qx = __shfl_sync(mask, qx, 0);
        qy = __shfl_sync(mask, qy, 0);
    }

    // Private heap size per warp, maintained by lane 0 only.
    int heapSize = 0;

    // Process the dataset in tiles.
    for (int tileStart = 0; tileStart < data_count; tileStart += KNN_TILE_POINTS) {
        int remaining = data_count - tileStart;
        int tileSize  = (remaining < KNN_TILE_POINTS) ? remaining : KNN_TILE_POINTS;

        // Block-wide cooperative load of the tile into shared memory.
        for (int i = tid; i < tileSize; i += blockDim.x) {
            s_data[i] = data[tileStart + i];
        }

        // Ensure the entire tile is loaded before proceeding.
        __syncthreads();

        if (activeWarp) {
            // Iterate over the tile in chunks of 32, so that each lane processes
            // at most one point per chunk.
            for (int base = 0; base < tileSize; base += KNN_WARP_SIZE) {
                int jLocal = base + laneId;
                bool valid = (jLocal < tileSize);

                // Compute candidate distance for this lane's data point.
                float candDist = 0.0f;
                int   candIdx  = 0;
                if (valid) {
                    float2 p = s_data[jLocal];
                    float dx = qx - p.x;
                    float dy = qy - p.y;
                    candDist = dx * dx + dy * dy;
                    candIdx  = tileStart + jLocal;
                }

                // Now, in a warp-synchronous manner, broadcast each lane's candidate
                // to lane 0, which updates the heap.
                // This ensures exactly one heap update per data point.
                unsigned mask = 0xffffffffu;
                for (int srcLane = 0; srcLane < KNN_WARP_SIZE; ++srcLane) {
                    bool  bValid = __shfl_sync(mask, valid,     srcLane);
                    float bDist  = __shfl_sync(mask, candDist,  srcLane);
                    int   bIdx   = __shfl_sync(mask, candIdx,   srcLane);

                    if (laneId == 0 && bValid) {
                        knn_heap_insert(heapDist, heapIndex, heapSize, bDist, bIdx, k);
                    }
                }

                // Warp-synchronous section ends here. No explicit __syncwarp() is required
                // because control flow is warp-uniform, but it is harmless if added.
            }
        }

        // Ensure all warps in the block are done with the tile before overwriting it.
        __syncthreads();
    }

    // After scanning the entire dataset, each active warp's heap contains its k
    // nearest neighbors (unsorted, in a max-heap).
    if (activeWarp) {
        // Convert heap to ascending order and write results.
        if (laneId == 0) {
            // data_count >= k, so heapSize should be k. As a safety guard, clamp.
            if (heapSize > k) {
                heapSize = k;
            }

            // Ensure we have at least one neighbor; given data_count >= k >= 1, heapSize > 0.
            if (heapSize > 1) {
                knn_heap_sort_ascending(heapDist, heapIndex, heapSize);
            }
        }

        // Make sure lane 0 has finished sorting before other lanes read heap data.
        __syncwarp();

        // Write out results cooperatively: each lane writes a subset of the k neighbors.
        const int outBase = warpGlobalId * k;
        for (int j = laneId; j < k; j += KNN_WARP_SIZE) {
            std::pair<int, float> outPair;
            outPair.first  = heapIndex[j];
            outPair.second = heapDist[j];
            result[outBase + j] = outPair;
        }
    }
}

// ----------------------------------------------------------------------------
// Host interface
// ----------------------------------------------------------------------------

/*
 * Host entry point:
 *
 *  - query: pointer to device array of float2 (query_count elements)
 *  - query_count: number of query points
 *  - data: pointer to device array of float2 (data_count elements)
 *  - data_count: number of data points
 *  - result: pointer to device array of std::pair<int, float>,
 *            size query_count * k
 *  - k: number of nearest neighbors to find (power of 2, 32 <= k <= 1024)
 *
 * This function launches the CUDA kernel with an appropriate configuration.
 * It does not perform any device memory allocation.
 */
void run_knn(const float2 *query,
             int            query_count,
             const float2 *data,
             int            data_count,
             std::pair<int, float> *result,
             int            k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Sanity clamp for k; the problem guarantees 32 <= k <= 1024 and power of two.
    if (k > KNN_MAX_K) {
        k = KNN_MAX_K;
    }

    // Grid configuration: one warp per query.
    const int threadsPerBlock = KNN_THREADS_PER_BLOCK;       // 256 threads (8 warps)
    const int warpsPerBlock   = KNN_WARPS_PER_BLOCK;
    const int numWarps        = query_count;
    const int numBlocks       = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Shared memory size per block:
    //  - Tile of KNN_TILE_POINTS float2.
    //  - Per-warp heaps: KNN_WARPS_PER_BLOCK * KNN_MAX_K (float + int).
    const size_t sharedMemBytes =
        static_cast<size_t>(KNN_TILE_POINTS) * sizeof(float2) +
        static_cast<size_t>(KNN_WARPS_PER_BLOCK) * static_cast<size_t>(KNN_MAX_K) * sizeof(float) +
        static_cast<size_t>(KNN_WARPS_PER_BLOCK) * static_cast<size_t>(KNN_MAX_K) * sizeof(int);

    // Launch kernel.
    knn_kernel_2d_2nn<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(
        query, query_count, data, data_count, result, k
    );

    // Optional: synchronize or check for errors here if desired.
    // cudaDeviceSynchronize();
    // cudaGetLastError();
}