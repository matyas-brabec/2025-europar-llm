#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Simple POD type compatible in layout with std::pair<int,float>.
// We use this on device instead of std::pair, which is not guaranteed
// to be __device__-compatible.
struct PairIF {
    int   first;
    float second;
};

// Tunable parameters for the kernel. These values are suitable for
// modern data-center GPUs (e.g., A100, H100) and typical problem sizes.
constexpr int MAX_K             = 1024;           // maximum k supported
constexpr int WARPS_PER_BLOCK   = 8;              // 8 warps * 32 threads = 256 threads per block
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int TILE_SIZE         = 1024;           // number of data points cached per tile (multiple of 32)

// Device helpers implementing a max-heap over distances.
//
// The heap stores the current k best (smallest) distances; as a max-heap,
// the root is the largest (worst among the best). This allows maintaining
// the k nearest neighbors in O(log k) time per candidate that improves
// the current worst.

// Insert a new (dist, idx) into a max-heap of current size heapSize.
__device__ __forceinline__
void heap_push_max(float *dists, int *idxs, int &heapSize,
                   const float dist, const int idx)
{
    int pos = heapSize++;
    float d = dist;
    int i = idx;
    // Sift up.
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        float parentDist = dists[parent];
        if (d <= parentDist) break;
        dists[pos] = parentDist;
        idxs[pos]  = idxs[parent];
        pos = parent;
    }
    dists[pos] = d;
    idxs[pos]  = i;
}

// Restore max-heap property starting at index 'start' in a heap of length heapSize.
__device__ __forceinline__
void heap_sift_down_max(float *dists, int *idxs, int start, int heapSize)
{
    int pos = start;
    while (true) {
        int left = (pos << 1) + 1;
        if (left >= heapSize) break;
        int right = left + 1;
        int largest = pos;
        float largestDist = dists[pos];
        float leftDist = dists[left];
        if (leftDist > largestDist) {
            largest = left;
            largestDist = leftDist;
        }
        if (right < heapSize) {
            float rightDist = dists[right];
            if (rightDist > largestDist) {
                largest = right;
                largestDist = rightDist;
            }
        }
        if (largest == pos) break;
        float tmpDist = dists[pos];
        int   tmpIdx  = idxs[pos];
        dists[pos]    = dists[largest];
        idxs[pos]     = idxs[largest];
        dists[largest]= tmpDist;
        idxs[largest] = tmpIdx;
        pos = largest;
    }
}

// Kernel: each warp (32 threads) processes one query point.
//
// For each query:
//   - Iterate over data in shared-memory tiles.
//   - Within each tile, each thread computes distances to a subset of points.
//   - Candidates are broadcast via warp shuffles and inserted into a per-warp
//     max-heap of size k (stored in shared memory).
//   - After all data are processed, the heap is heap-sorted into ascending order
//     and written to the result array.
__global__
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                PairIF * __restrict__ result,
                int k)
{
    // Shared memory:
    // - s_data_tile: a tile of data points used by all warps in the block.
    // - s_best_dists / s_best_indices: per-warp heaps of current k nearest neighbors.
    __shared__ float2 s_data_tile[TILE_SIZE];
    __shared__ float  s_best_dists[WARPS_PER_BLOCK][MAX_K];
    __shared__ int    s_best_indices[WARPS_PER_BLOCK][MAX_K];

    const int tid       = threadIdx.x;
    const int lane      = tid & 31;               // thread index within warp [0,31]
    const int warp      = tid >> 5;               // warp index within block [0,WARPS_PER_BLOCK-1]
    const int warpGlobal= blockIdx.x * WARPS_PER_BLOCK + warp;

    const bool warpActive = (warpGlobal < query_count);
    const unsigned fullMask = 0xffffffffu;

    float qx = 0.0f;
    float qy = 0.0f;

    // Load query point once per warp and broadcast to all 32 threads.
    if (warpActive) {
        if (lane == 0) {
            float2 q = query[warpGlobal];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(fullMask, qx, 0);
        qy = __shfl_sync(fullMask, qy, 0);

        // Initialize per-warp best distance/index buffers (only first k entries matter).
        float *bestDists = s_best_dists[warp];
        int   *bestIdx   = s_best_indices[warp];
        for (int i = lane; i < k; i += 32) {
            bestDists[i] = FLT_MAX;
            bestIdx[i]   = -1;
        }
    }

    // Heap size is kept in a register; only lane 0 of active warps uses it.
    int heapSize = 0;

    // Iterate over data in tiles cached in shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {
        int tileSize = TILE_SIZE;
        if (tileStart + tileSize > data_count)
            tileSize = data_count - tileStart;

        // Cooperative load of current tile into shared memory by all threads in the block.
        for (int i = tid; i < tileSize; i += blockDim.x) {
            s_data_tile[i] = data[tileStart + i];
        }
        __syncthreads();

        if (warpActive) {
            float *bestDists = s_best_dists[warp];
            int   *bestIdx   = s_best_indices[warp];

            // Process the tile in chunks of 32 points: each lane handles one point per chunk.
            for (int base = 0; base < tileSize; base += 32) {
                int idxInTile     = base + lane;
                int globalDataIdx = tileStart + idxInTile;

                // Each lane computes distance from its query point to one data point (if within bounds).
                float dist = FLT_MAX;
                int   idx  = -1;
                if (idxInTile < tileSize) {
                    float2 p = s_data_tile[idxInTile];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    dist = dx * dx + dy * dy;   // squared Euclidean distance
                    idx  = globalDataIdx;
                }

                // Ensure all threads in the warp have computed their distances.
                __syncwarp(fullMask);

                // Lane 0 sequentially feeds all 32 candidates into the per-warp max-heap.
                // The heap maintains the k smallest distances seen so far.
                for (int srcLane = 0; srcLane < 32; ++srcLane) {
                    float candDist = __shfl_sync(fullMask, dist, srcLane);
                    int   candIdx  = __shfl_sync(fullMask, idx,  srcLane);

                    if (lane == 0 && candIdx >= 0) {
                        if (heapSize < k) {
                            // Heap not full yet: insert the candidate.
                            heap_push_max(bestDists, bestIdx, heapSize, candDist, candIdx);
                        } else if (candDist < bestDists[0]) {
                            // Candidate is better than the current worst in the heap:
                            // replace root and restore heap property.
                            bestDists[0] = candDist;
                            bestIdx[0]   = candIdx;
                            heap_sift_down_max(bestDists, bestIdx, 0, heapSize);
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // After processing all tiles, each active warp has a max-heap of size k that
    // contains the k nearest neighbors (unsorted). Convert the heap into a sorted
    // array (ascending by distance) via in-place heap sort, then write results.
    if (warpActive && lane == 0) {
        float *bestDists = s_best_dists[warp];
        int   *bestIdx   = s_best_indices[warp];

        if (heapSize > k) heapSize = k;

        // In-place heap sort on the max-heap -> ascending order.
        // Complexity: O(k log k), negligible compared to distance computations.
        for (int i = heapSize - 1; i > 0; --i) {
            // Move current maximum to the end.
            float tmpDist = bestDists[0];
            bestDists[0]  = bestDists[i];
            bestDists[i]  = tmpDist;
            int tmpIdx    = bestIdx[0];
            bestIdx[0]    = bestIdx[i];
            bestIdx[i]    = tmpIdx;

            // Restore heap property on the reduced heap [0, i).
            heap_sift_down_max(bestDists, bestIdx, 0, i);
        }

        // Now bestDists[0 .. heapSize-1] are sorted in ascending order.

        PairIF *out = result + warpGlobal * k;
        int outCount = (heapSize < k) ? heapSize : k;

        // Write found neighbors.
        for (int i = 0; i < outCount; ++i) {
            out[i].first  = bestIdx[i];
            out[i].second = bestDists[i];
        }
        // If data_count < k (not expected per problem statement), fill remaining slots.
        for (int i = outCount; i < k; ++i) {
            out[i].first  = -1;
            out[i].second = FLT_MAX;
        }
    }
}

// Host-side entry point.
//
// - query, data, and result are assumed to be device pointers allocated with cudaMalloc.
// - Each query[i] is a float2 representing a 2D point.
// - Each data[j] is a float2 representing a 2D point.
// - result is an array of std::pair<int,float> with length query_count * k in device memory.
//   For query i, result[i * k + j] will contain the index (first) and squared distance (second)
//   of its j-th nearest neighbor among data[0..data_count-1].
// - k is a power of two between 32 and 1024 inclusive.
// - data_count >= k.
//
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Reinterpret the result pointer as a POD type with the same layout as std::pair<int,float>.
    PairIF *result_cast = reinterpret_cast<PairIF*>(result);

    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    int totalWarps = query_count;
    int gridX = (totalWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 gridDim(gridX, 1, 1);

    knn_kernel<<<gridDim, blockDim>>>(query, query_count, data, data_count, result_cast, k);
}