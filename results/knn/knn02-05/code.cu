#include <cuda_runtime.h>
#include <utility>

// -----------------------------------------------------------------------------
// CUDA k-NN implementation for 2D points (squared Euclidean distance).
//
// - One warp (32 threads) processes one query point.
// - All warps in a block cooperatively cache tiles of the data points in shared
//   memory to minimize global memory traffic.
// - Each warp maintains its own private max-heap (in shared memory) of size k,
//   containing the currently best (smallest-distance) neighbors.
//   * The heap root is the worst (largest) distance in the current top-k.
//   * New candidates replace the heap root only if they are closer.
// - At the end of processing, each warp heap-sorts its heap to produce the k
//   nearest neighbors in ascending-distance order, and writes them to global
//   memory.
//
// This implementation is tuned for modern data-center GPUs (e.g., A100/H100).
// -----------------------------------------------------------------------------

// Number of warps per block. Each warp handles one query.
constexpr int WARPS_PER_BLOCK   = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

// Number of data points cached per tile in shared memory.
// 1024 * sizeof(float2) = 8 KiB per block for data cache.
constexpr int TILE_SIZE         = 1024;

// Helper: push a new element into a max-heap (heapDist/heapIdx, size = heapSize).
// The parent node always has a distance >= its children.
__device__ __forceinline__
void heap_push_max(float *heapDist, int *heapIdx, int &heapSize,
                   float dist, int idx)
{
    int pos = heapSize++;
    // Sift up
    while (pos > 0) {
        int parent      = (pos - 1) >> 1;
        float parentVal = heapDist[parent];
        if (parentVal >= dist) {
            break;
        }
        // Move parent down
        heapDist[pos] = parentVal;
        heapIdx[pos]  = heapIdx[parent];
        pos           = parent;
    }
    heapDist[pos] = dist;
    heapIdx[pos]  = idx;
}

// Helper: replace the root of a max-heap with a new smaller value and sift down.
// Assumes dist < heapDist[0] and heapSize > 0.
__device__ __forceinline__
void heap_replace_root_max(float *heapDist, int *heapIdx, int heapSize,
                           float dist, int idx)
{
    int   pos       = 0;
    float valueDist = dist;
    int   valueIdx  = idx;

    // Sift down
    while (true) {
        int left = (pos << 1) + 1;
        if (left >= heapSize) {
            break;
        }
        int right       = left + 1;
        int largestPos  = left;
        float largest   = heapDist[left];

        if (right < heapSize) {
            float rightVal = heapDist[right];
            if (rightVal > largest) {
                largestPos = right;
                largest    = rightVal;
            }
        }

        if (largest <= valueDist) {
            break;
        }

        // Move child up
        heapDist[pos] = largest;
        heapIdx[pos]  = heapIdx[largestPos];
        pos           = largestPos;
    }

    heapDist[pos] = valueDist;
    heapIdx[pos]  = valueIdx;
}

// Helper: in-place heap sort of a max-heap where heapDist/heapIdx has 'size'
// elements. After this, heapDist/heapIdx are sorted in ascending order
// (smallest distance at index 0).
__device__ __forceinline__
void heap_sort_max_to_ascending(float *heapDist, int *heapIdx, int size)
{
    // Standard heapsort on a max-heap.
    for (int end = size - 1; end > 0; --end) {
        // Swap root with last element in [0, end]
        float rootDist = heapDist[0];
        float lastDist = heapDist[end];
        heapDist[0]    = lastDist;
        heapDist[end]  = rootDist;

        int rootIdx = heapIdx[0];
        int lastIdx = heapIdx[end];
        heapIdx[0]  = lastIdx;
        heapIdx[end]= rootIdx;

        // Re-heapify the reduced heap [0, end)
        int   pos       = 0;
        float valueDist = heapDist[0];
        int   valueIdx  = heapIdx[0];

        while (true) {
            int left = (pos << 1) + 1;
            if (left >= end) {
                break;
            }
            int right      = left + 1;
            int largestPos = left;
            float largest  = heapDist[left];

            if (right < end) {
                float rightVal = heapDist[right];
                if (rightVal > largest) {
                    largestPos = right;
                    largest    = rightVal;
                }
            }

            if (largest <= valueDist) {
                break;
            }

            heapDist[pos] = largest;
            heapIdx[pos]  = heapIdx[largestPos];
            pos           = largestPos;
        }

        heapDist[pos] = valueDist;
        heapIdx[pos]  = valueIdx;
    }
}

// Kernel: each warp computes k-NN for one query point.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           int k,
                           std::pair<int, float> * __restrict__ result)
{
    extern __shared__ unsigned char shared_raw[];

    // Layout of shared memory:
    // [0 .. TILE_SIZE-1]              : float2 data tile
    // [TILE_SIZE .. + WARPS*k floats] : per-warp heap distances
    // [floats .. + WARPS*k ints]      : per-warp heap indices
    float2 *sData = reinterpret_cast<float2*>(shared_raw);
    float  *sHeapDists = reinterpret_cast<float*>(sData + TILE_SIZE);
    int    *sHeapIdx   = reinterpret_cast<int*>(sHeapDists + WARPS_PER_BLOCK * k);

    const unsigned FULL_MASK = 0xffffffffu;

    int laneId        = threadIdx.x & 31;     // Thread index within warp
    int warpIdInBlock = threadIdx.x >> 5;     // Warp index within block
    int warpsPerBlock = blockDim.x >> 5;      // Should equal WARPS_PER_BLOCK
    int globalWarpId  = blockIdx.x * warpsPerBlock + warpIdInBlock;

    bool warpActive = (globalWarpId < query_count);

    // Each warp has its own heap region in shared memory.
    float *heapDist = sHeapDists + warpIdInBlock * k;
    int   *heapIdx  = sHeapIdx   + warpIdInBlock * k;

    // Load query point into registers for this warp and broadcast.
    float2 q;
    if (warpActive) {
        if (laneId == 0) {
            q = query[globalWarpId];
        }
        q.x = __shfl_sync(FULL_MASK, q.x, 0);
        q.y = __shfl_sync(FULL_MASK, q.y, 0);
    }

    // Heap size is tracked only in lane 0's register;
    // other lanes never use it directly, only through lane 0.
    int heapSize = 0; // Only valid in lane 0 when warpActive.

    // Iterate over data points in tiles cached in shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {

        int tileCount = data_count - tileStart;
        if (tileCount > TILE_SIZE) {
            tileCount = TILE_SIZE;
        }

        // All threads in the block cooperatively load the tile.
        for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
            sData[i] = data[tileStart + i];
        }
        __syncthreads();

        // Each active warp processes the tile for its query.
        if (warpActive) {
            // Process the tile in groups of warpSize points.
            for (int base = 0; base < tileCount; base += warpSize) {
                int idxInTile = base + laneId;

                float dist = 0.0f;
                int   idx  = -1;

                if (idxInTile < tileCount) {
                    float2 p = sData[idxInTile];
                    float dx = p.x - q.x;
                    float dy = p.y - q.y;
                    // Squared Euclidean distance (no sqrt needed).
                    dist = dx * dx + dy * dy;
                    idx  = tileStart + idxInTile;
                }

                // Number of valid candidates in this group (<= warpSize).
                int groupSize = tileCount - base;
                if (groupSize > warpSize) {
                    groupSize = warpSize;
                }

                // Sequentially feed this group's candidates to lane 0's heap.
                // All threads participate using warp shuffles.
                for (int i = 0; i < groupSize; ++i) {
                    float candDist = __shfl_sync(FULL_MASK, dist, i);
                    int   candIdx  = __shfl_sync(FULL_MASK, idx,  i);

                    if (laneId == 0) {
                        if (heapSize < k) {
                            // Heap not full yet: simply push.
                            heap_push_max(heapDist, heapIdx, heapSize, candDist, candIdx);
                        } else if (candDist < heapDist[0]) {
                            // Candidate is better (closer) than current worst in top-k.
                            heap_replace_root_max(heapDist, heapIdx, heapSize, candDist, candIdx);
                        }
                    }
                    // Ensure heap updates are visible to the warp before the
                    // next candidate is processed.
                    __syncwarp(FULL_MASK);
                }
            }
        }

        __syncthreads();
    }

    // Finalize results for this warp's query.
    if (warpActive) {
        // Heap currently contains k smallest distances in a max-heap.
        // Convert heap into ascending order.
        if (laneId == 0) {
            heap_sort_max_to_ascending(heapDist, heapIdx, heapSize);
        }
        __syncwarp(FULL_MASK);

        // Write sorted k nearest neighbors (index, distance) to global memory.
        if (laneId == 0) {
            int outBase = globalWarpId * k;
            for (int i = 0; i < k; ++i) {
                result[outBase + i].first  = heapIdx[i];
                result[outBase + i].second = heapDist[i];
            }
        }
    }
}

// Host-side interface as requested.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // One warp per query.
    const int warpsPerBlock   = WARPS_PER_BLOCK;
    const int threadsPerBlock = THREADS_PER_BLOCK;

    int numWarps  = query_count;
    int numBlocks = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Shared memory size:
    // - TILE_SIZE float2 for the data tile
    // - WARPS_PER_BLOCK * k floats for heap distances
    // - WARPS_PER_BLOCK * k ints   for heap indices
    size_t sharedBytes =
        TILE_SIZE * sizeof(float2) +
        WARPS_PER_BLOCK * k * (sizeof(float) + sizeof(int));

    knn_kernel<<<numBlocks, threadsPerBlock, sharedBytes>>>(
        query, query_count, data, data_count, k, result);
}