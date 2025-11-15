#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <utility>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Helper: warp-synchronous merge of candidate buffer into a max-heap of current neighbors.
// The heap stores the current k nearest neighbors for one query, as a max-heap keyed by distance.
// Only lane 0 performs the heap operations, but the updated heap_size and max_distance are
// broadcast to all lanes via warp shuffles so that subsequent computations use consistent state.
__device__ __forceinline__
void warp_merge_buffer_heap(int lane,
                            int k,
                            int &heap_size,
                            float &max_distance,
                            int *bestIndex,
                            float *bestDist,
                            const int *candIndex,
                            const float *candDist,
                            int candCount)
{
    if (candCount <= 0) {
        return;
    }

    if (lane == 0) {
        // Sequentially insert/replace candidates into the max-heap.
        for (int i = 0; i < candCount; ++i) {
            float dist = candDist[i];
            int   idx  = candIndex[i];

            if (heap_size < k) {
                // Heap not full yet: standard heap insertion.
                int pos = heap_size++;
                // Sift up.
                while (pos > 0) {
                    int parent = (pos - 1) >> 1;
                    float parentVal = bestDist[parent];
                    if (dist <= parentVal) {
                        break;
                    }
                    // Move parent down.
                    bestDist[pos]  = parentVal;
                    bestIndex[pos] = bestIndex[parent];
                    pos = parent;
                }
                bestDist[pos]  = dist;
                bestIndex[pos] = idx;
            } else if (dist < bestDist[0]) {
                // Candidate is better than current worst (heap root).
                // Replace root and sift down.
                int pos = 0;
                while (true) {
                    int left  = (pos << 1) + 1;
                    int right = left + 1;
                    if (left >= k) {
                        break;
                    }
                    int child = left;
                    if (right < k && bestDist[right] > bestDist[left]) {
                        child = right;
                    }
                    if (dist >= bestDist[child]) {
                        break;
                    }
                    bestDist[pos]  = bestDist[child];
                    bestIndex[pos] = bestIndex[child];
                    pos = child;
                }
                bestDist[pos]  = dist;
                bestIndex[pos] = idx;
            }
        }

        // After processing the buffer, update max_distance to the distance of the k-th neighbor.
        max_distance = (heap_size < k) ? CUDART_INF_F : bestDist[0];
    }

    // Broadcast updated heap_size and max_distance to the whole warp.
    const unsigned FULL_MASK = 0xffffffffu;
    heap_size    = __shfl_sync(FULL_MASK, heap_size, 0);
    max_distance = __shfl_sync(FULL_MASK, max_distance, 0);
}

// Helper: after all data has been processed and the heap contains the final k neighbors,
// convert the max-heap into an array sorted in ascending order of distance and write
// it to the global result array.
__device__ __forceinline__
void warp_heap_sort_and_write(int lane,
                              int k,
                              int heap_size,
                              int *bestIndex,
                              float *bestDist,
                              int queryIdx,
                              std::pair<int, float> *result)
{
    if (heap_size <= 0) {
        return;
    }

    // In-place heap sort performed by lane 0; all other lanes wait.
    if (lane == 0) {
        int n = heap_size;
        // Standard heapsort on a max-heap: repeatedly move the largest element
        // to the end of the array and restore the heap property for the prefix.
        while (n > 1) {
            // Swap root (largest distance) with the last element in the heap.
            float distTmp = bestDist[0];
            int   idxTmp  = bestIndex[0];
            bestDist[0]      = bestDist[n - 1];
            bestIndex[0]     = bestIndex[n - 1];
            bestDist[n - 1]  = distTmp;
            bestIndex[n - 1] = idxTmp;

            // Sift down the new root in the heap of size n-1.
            int pos = 0;
            float dist = bestDist[0];
            idxTmp = bestIndex[0];
            int heapLimit = n - 1;
            while (true) {
                int left  = (pos << 1) + 1;
                int right = left + 1;
                if (left >= heapLimit) {
                    break;
                }
                int child = left;
                if (right < heapLimit && bestDist[right] > bestDist[left]) {
                    child = right;
                }
                if (dist >= bestDist[child]) {
                    break;
                }
                bestDist[pos]  = bestDist[child];
                bestIndex[pos] = bestIndex[child];
                pos = child;
            }
            bestDist[pos]  = dist;
            bestIndex[pos] = idxTmp;

            --n;
        }
        // After heapsort, bestDist[0..heap_size-1] is sorted in ascending order.
    }

    __syncwarp();

    // Each lane writes out a subset of the results for this query.
    for (int i = lane; i < heap_size; i += WARP_SIZE) {
        int outPos = queryIdx * k + i;
        result[outPos].first  = bestIndex[i];
        result[outPos].second = bestDist[i];
    }
}

// Main CUDA kernel implementing k-NN for 2D points.
// Each warp is responsible for one query point and computes its k nearest neighbors.
template <int WARPS_PER_BLOCK, int TILE_SIZE>
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           std::pair<int, float> * __restrict__ result,
                           int k)
{
    extern __shared__ unsigned char smem[];
    unsigned char *ptr = smem;

    // Shared memory layout:
    // [ TILE_SIZE * float2 ] data tile cached for the whole block
    // [ WARPS_PER_BLOCK * k * int   ] bestIndex for all warps in the block
    // [ WARPS_PER_BLOCK * k * float ] bestDist  for all warps in the block
    // [ WARPS_PER_BLOCK * k * int   ] candIndex for all warps in the block
    // [ WARPS_PER_BLOCK * k * float ] candDist  for all warps in the block
    // [ WARPS_PER_BLOCK * int       ] candCount per warp

    // Tile with data points.
    float2 *tile_data = reinterpret_cast<float2*>(ptr);
    ptr += TILE_SIZE * sizeof(float2);

    // Align to 4 bytes for integer arrays.
    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t align = alignof(int);
    uptr = (uptr + align - 1) & ~(align - 1);
    ptr = reinterpret_cast<unsigned char*>(uptr);

    int *bestIndex_all = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(int);

    float *bestDist_all = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(float);

    int *candIndex_all = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(int);

    float *candDist_all = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * static_cast<size_t>(k) * sizeof(float);

    int *candCount_all = reinterpret_cast<int*>(ptr);
    // No further shared memory after this.

    const unsigned FULL_MASK = 0xffffffffu;

    int threadId    = threadIdx.x;
    int warpInBlock = threadId >> 5;                   // warp index within block
    int lane        = threadId & (WARP_SIZE - 1);      // lane index within warp
    int warpGlobal  = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;

    if (warpGlobal >= query_count) {
        return;
    }

    // Offsets into shared memory for this warp's private data structures.
    int warpOffset = warpInBlock * k;

    int   *myBestIndex = bestIndex_all + warpOffset;
    float *myBestDist  = bestDist_all  + warpOffset;
    int   *myCandIndex = candIndex_all + warpOffset;
    float *myCandDist  = candDist_all  + warpOffset;
    int   *myCandCount = candCount_all + warpInBlock;

    // Initialize the intermediate result (heap) with empty entries.
    // All distances are set to +inf and indices to -1.
    for (int i = lane; i < k; i += WARP_SIZE) {
        myBestIndex[i] = -1;
        myBestDist[i]  = CUDART_INF_F;
    }
    if (lane == 0) {
        *myCandCount = 0;
    }
    __syncwarp();

    // Load the query point for this warp and broadcast it to all lanes.
    float2 q;
    if (lane == 0) {
        q = query[warpGlobal];
    }
    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    int   heap_size    = 0;               // number of neighbors currently in the heap
    float max_distance = CUDART_INF_F;    // distance of the current k-th nearest neighbor

    // Iterate over the data points in batches that fit into shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {
        int tileCount = TILE_SIZE;
        if (tileStart + tileCount > data_count) {
            tileCount = data_count - tileStart;
        }

        // Load this tile of data points into shared memory cooperatively.
        for (int i = threadId; i < tileCount; i += blockDim.x) {
            tile_data[i] = data[tileStart + i];
        }
        __syncthreads();

        // Each warp processes the entire tile for its own query.
        for (int i = lane; i < tileCount; i += WARP_SIZE) {
            float2 p = tile_data[i];

            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float dist = dx * dx + dy * dy;

            // Decide whether this point is a candidate based on the current max_distance.
            bool is_candidate = (heap_size < k) || (dist < max_distance);

            unsigned mask  = __ballot_sync(FULL_MASK, is_candidate);
            int total_cand = __popc(mask);

            if (total_cand > 0) {
                // Number of candidates in this warp before the current lane.
                int lane_rank = __popc(mask & ((1u << lane) - 1));

                int base = 0;
                int flushCount = 0;

                if (lane == 0) {
                    int oldCount = *myCandCount;
                    if (oldCount + total_cand > k) {
                        // Buffer would overflow: flush existing candidates.
                        flushCount   = oldCount;
                        *myCandCount = 0;
                    }
                }

                // Broadcast flushCount from lane 0 to all lanes.
                flushCount = __shfl_sync(FULL_MASK, flushCount, 0);

                // Merge existing buffer with intermediate result, if requested.
                if (flushCount > 0) {
                    warp_merge_buffer_heap(lane, k, heap_size, max_distance,
                                           myBestIndex, myBestDist,
                                           myCandIndex, myCandDist,
                                           flushCount);
                }

                if (lane == 0) {
                    // Reserve positions in the (now empty or partially filled) buffer.
                    base = atomicAdd(myCandCount, total_cand);
                }
                base = __shfl_sync(FULL_MASK, base, 0);

                if (is_candidate) {
                    int pos = base + lane_rank;
                    // pos is guaranteed to be < k after the flush above.
                    myCandIndex[pos] = tileStart + i; // global index of the data point
                    myCandDist[pos]  = dist;
                }
            }
        }

        __syncthreads();
    }

    // After all tiles are processed, flush any remaining candidates in the buffer.
    int finalCount = 0;
    if (lane == 0) {
        finalCount   = *myCandCount;
        *myCandCount = 0;
    }
    finalCount = __shfl_sync(FULL_MASK, finalCount, 0);

    if (finalCount > 0) {
        warp_merge_buffer_heap(lane, k, heap_size, max_distance,
                               myBestIndex, myBestDist,
                               myCandIndex, myCandDist,
                               finalCount);
    }

    // At this point, the heap contains the final k nearest neighbors for this query.
    // Convert the heap to an array sorted in ascending order by distance and
    // write the results to global memory.
    warp_heap_sort_and_write(lane, k, heap_size,
                             myBestIndex, myBestDist,
                             warpGlobal, result);
}

// Host-side entry point.
// The pointers query, data, and result are expected to be device pointers obtained via cudaMalloc.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Tunable kernel parameters: number of warps per block and tile size.
    constexpr int WARPS_PER_BLOCK   = 4;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TILE_SIZE         = 1024;

    dim3 blockDim(THREADS_PER_BLOCK);
    int numWarps  = query_count;
    int numBlocks = (numWarps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 gridDim(numBlocks);

    // Compute required dynamic shared memory size.
    // See layout description in the kernel above.
    size_t sharedMemSize = 0;
    sharedMemSize += static_cast<size_t>(TILE_SIZE) * sizeof(float2);
    // Account for alignment before integer arrays.
    {
        size_t align = alignof(int);
        sharedMemSize = (sharedMemSize + align - 1) & ~(align - 1);
    }
    sharedMemSize += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(int);   // bestIndex
    sharedMemSize += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(float); // bestDist
    sharedMemSize += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(int);   // candIndex
    sharedMemSize += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(float); // candDist
    sharedMemSize += static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(int);                            // candCount

    // Launch the kernel. The caller is responsible for checking for errors if desired.
    knn_kernel<WARPS_PER_BLOCK, TILE_SIZE>
        <<<gridDim, blockDim, sharedMemSize>>>(query, query_count, data, data_count, result, k);
}