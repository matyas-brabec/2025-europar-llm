#include <cuda_runtime.h>
#include <utility>
#include <cmath>

// Simple POD struct with the same layout as std::pair<int, float>
// used on the device side to avoid relying on the C++ standard library there.
struct PairIF {
    int   first;
    float second;
};

// Ensure that host-side std::pair<int, float> has the same size/layout.
static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "PairIF and std::pair<int,float> must have the same size");

// Tunable constants for the kernel.
constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_BLOCK  = 4;     // 4 warps per block -> 128 threads per block
constexpr int MAX_K            = 1024;  // Given: k is <= 1024
constexpr int TILE_SIZE        = 2048;  // Number of data points cached per block in shared memory

// Shared memory:
//  - shData      : data points tile, reused by all warps in the block.
//  - shTopDist   : per-query (per-warp) heap distances (max-heap of size K).
//  - shTopIdx    : per-query (per-warp) heap indices.
__shared__ float2 shData[TILE_SIZE];
__shared__ float  shTopDist[WARPS_PER_BLOCK * MAX_K];
__shared__ int    shTopIdx [WARPS_PER_BLOCK * MAX_K];

// Device helper: insert/update a max-heap of size up to k stored in shared memory.
// Only lane 0 of each warp calls this function, so no synchronization is needed inside it.
__device__ __forceinline__
void heap_insert_or_update(float* heapDist,
                           int*   heapIdx,
                           int&   heapSize,
                           const int k,
                           const float dist,
                           const int idx)
{
    // If heap not full, insert and bubble up.
    if (heapSize < k) {
        int pos = heapSize++;
        int parent = (pos - 1) >> 1;
        float curDist = dist;
        int   curIdx  = idx;

        // Bubble up to maintain max-heap property (largest distance at root).
        while (pos > 0 && curDist > heapDist[parent]) {
            heapDist[pos] = heapDist[parent];
            heapIdx[pos]  = heapIdx[parent];
            pos    = parent;
            parent = (pos - 1) >> 1;
        }
        heapDist[pos] = curDist;
        heapIdx[pos]  = curIdx;
        return;
    }

    // If heap is full and new distance is not better than current worst, skip.
    if (dist >= heapDist[0]) {
        return;
    }

    // Replace root with new candidate and bubble down.
    float curDist = dist;
    int   curIdx  = idx;
    int   pos     = 0;
    const int n   = heapSize;

    while (true) {
        int left  = (pos << 1) + 1;
        int right = left + 1;
        if (left >= n) {
            break; // no children
        }

        // Select the child with the larger distance.
        int   largerChild = left;
        float largerDist  = heapDist[left];
        if (right < n && heapDist[right] > largerDist) {
            largerChild = right;
            largerDist  = heapDist[right];
        }

        // If the larger child is not greater than current, we're done.
        if (largerDist <= curDist) {
            break;
        }

        heapDist[pos] = largerDist;
        heapIdx[pos]  = heapIdx[largerChild];
        pos = largerChild;
    }

    heapDist[pos] = curDist;
    heapIdx[pos]  = curIdx;
}

// Device helper: convert the max-heap (heapDist/heapIdx, size heapSize=k) into
// a sorted array of k nearest neighbors in ascending order of distance and store
// into the output array. Only lane 0 calls this.
__device__ __forceinline__
void heap_to_sorted_results(float*   heapDist,
                            int*     heapIdx,
                            int      heapSize,
                            PairIF*  out,
                            int      outOffset)
{
    // Classic heap sort using max-heap:
    // Pop max element repeatedly and place it at position [heapSize-1 .. 0].
    // This yields ascending order in out[0..heapSize-1].
    for (int i = heapSize - 1; i >= 0; --i) {
        // The root of the heap is the current largest distance.
        out[outOffset + i].first  = heapIdx[0];
        out[outOffset + i].second = heapDist[0];

        // Move the last element to the root and reduce heap size.
        float lastDist = heapDist[heapSize - 1];
        int   lastIdx  = heapIdx[heapSize - 1];
        --heapSize;

        int pos = 0;
        while (true) {
            int left  = (pos << 1) + 1;
            int right = left + 1;
            if (left >= heapSize) {
                break; // no children
            }

            // Select larger child.
            int   largerChild = left;
            float largerDist  = heapDist[left];
            if (right < heapSize && heapDist[right] > largerDist) {
                largerChild = right;
                largerDist  = heapDist[right];
            }

            // If child is not greater than the last element, we're done.
            if (largerDist <= lastDist) {
                break;
            }

            heapDist[pos] = largerDist;
            heapIdx[pos]  = heapIdx[largerChild];
            pos = largerChild;
        }

        if (heapSize > 0) {
            heapDist[pos] = lastDist;
            heapIdx[pos]  = lastIdx;
        }
    }
}

// Kernel: Each warp processes a single query point.
//  - All warps in a block share a tile of data points cached in shared memory.
//  - Within each warp, all 32 threads compute distances in parallel.
//  - Lane 0 maintains a max-heap of size k in shared memory with the k best candidates.
__global__
void knn_kernel(const float2* __restrict__ query,
                int                        query_count,
                const float2* __restrict__ data,
                int                        data_count,
                PairIF*      __restrict__  result,
                int                        k)
{
    const int tid           = threadIdx.x;
    const int laneId        = tid & (WARP_SIZE - 1);
    const int warpIdInBlock = tid >> 5;  // threadIdx.x / WARP_SIZE
    const int globalWarpId  = blockIdx.x * WARPS_PER_BLOCK + warpIdInBlock;

    if (globalWarpId >= query_count) {
        return;
    }

    // Each warp handles a distinct query.
    const float2 q = query[globalWarpId];

    // Per-warp heap base pointers in shared memory.
    float* heapDist = &shTopDist[warpIdInBlock * MAX_K];
    int*   heapIdx  = &shTopIdx [warpIdInBlock * MAX_K];

    // Lane 0 will maintain the heapSize for this warp's query.
    int heapSize = 0;

    // Process the data points in tiles cached into shared memory.
    for (int base = 0; base < data_count; base += TILE_SIZE) {
        const int chunkSize = min(TILE_SIZE, data_count - base);

        // Block-wide cooperative load of the current tile into shared memory.
        for (int i = tid; i < chunkSize; i += blockDim.x) {
            shData[i] = data[base + i];
        }

        // Ensure the tile is fully loaded before any warp uses it.
        __syncthreads();

        // Number of iterations required so that each data point in the tile is
        // processed once by exactly one thread in the warp.
        const int numWarpIters = (chunkSize + WARP_SIZE - 1) / WARP_SIZE;

        // Warp-level processing: each iteration covers up to 32 points.
        for (int iter = 0; iter < numWarpIters; ++iter) {
            const int tIdx = iter * WARP_SIZE + laneId;

            float dist = 0.0f;
            int   idx  = -1;

            // Compute squared Euclidean distance for owned point (if within range).
            if (tIdx < chunkSize) {
                const float2 p = shData[tIdx];
                const float dx = p.x - q.x;
                const float dy = p.y - q.y;
                dist = dx * dx + dy * dy;
                idx  = base + tIdx;
            }

            // Now each lane has at most one candidate (idx, dist).
            // Broadcast each lane's candidate to lane 0 in turn so that lane 0
            // can update the heap sequentially. This avoids any need for atomic
            // operations or fine-grained synchronization.
#pragma unroll
            for (int srcLane = 0; srcLane < WARP_SIZE; ++srcLane) {
                const float candDist = __shfl_sync(0xffffffff, dist, srcLane);
                const int   candIdx  = __shfl_sync(0xffffffff, idx,  srcLane);

                if (laneId == 0 && candIdx >= 0) {
                    heap_insert_or_update(heapDist, heapIdx, heapSize, k, candDist, candIdx);
                }
            }
        }

        // Ensure all warps are done with the current tile before loading the next.
        __syncthreads();
    }

    // After processing all data points, lane 0 converts its heap into a sorted
    // list of k nearest neighbors for this query and writes the result to global memory.
    if (laneId == 0) {
        PairIF* out = result;
        const int outOffset = globalWarpId * k;
        heap_to_sorted_results(heapDist, heapIdx, heapSize, out, outOffset);
    }
}

// Host interface:
//  - query, data, result are device pointers (allocated with cudaMalloc).
//  - query_count: number of query points.
//  - data_count : number of data points.
//  - result     : output array of size query_count * k, with layout:
//        result[i * k + j].first  = index of j-th nearest neighbor of query i
//        result[i * k + j].second = squared distance to that neighbor
//  - k          : number of nearest neighbors to find (power of two in [32, 1024]).
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Clamp k to MAX_K for safety (inputs are assumed valid per specification).
    if (k > MAX_K) {
        k = MAX_K;
    }

    // Configure launch: one warp per query, WARPS_PER_BLOCK warps per block.
    dim3 blockDim(WARPS_PER_BLOCK * WARP_SIZE);
    dim3 gridDim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Reinterpret result pointer to device-side POD type PairIF.
    PairIF *d_result = reinterpret_cast<PairIF*>(result);

    // Launch kernel. This call is asynchronous; synchronization is left to the caller.
    knn_kernel<<<gridDim, blockDim>>>(query, query_count, data, data_count, d_result, k);
}