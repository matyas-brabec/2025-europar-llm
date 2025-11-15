#include <cuda_runtime.h>
#include <utility>

/*
 * Optimized k-NN CUDA implementation for 2D points.
 *
 * Key design choices:
 * - Each CUDA thread processes one query point.
 * - Data points are processed in tiles stored in shared memory to maximize
 *   reuse across threads in a block and minimize global memory traffic.
 * - Each thread maintains its own top-k structure using a max-heap
 *   (represented by two parallel arrays: distances and indices).
 * - The heap size k is a compile-time template parameter, and run_knn()
 *   dispatches to the appropriate kernel instantiation for each supported k.
 * - The final heap is converted to a sorted array (ascending distance)
 *   using heapsort, and written to the output array.
 *
 * Constraints honored:
 * - No additional device memory allocations (no cudaMalloc in device code).
 * - k is a power of two in [32, 1024], inclusive.
 * - data_count >= k is guaranteed.
 */

/********************* Device-side helper utilities ************************/

// Tile size for loading data points into shared memory.
// 1024 points * sizeof(float2) = 8 KB of shared memory per block.
constexpr int DATA_TILE_SIZE = 1024;

/*
 * Sift-down operation for a max-heap stored in parallel arrays dist / idx.
 * Template parameter K is the maximum heap capacity; heapSize is the current
 * active size (can be <= K, e.g., during heapsort).
 *
 * This function assumes:
 * - dist[0..heapSize-1], idx[0..heapSize-1] contain a valid heap except
 *   possibly at position 'start'.
 * - After return, the max-heap property is restored for the range [0, heapSize).
 */
template <int K>
__device__ __forceinline__ void siftDown(float *dist, int *idx, int start, int heapSize)
{
    int root = start;
    float rootDist = dist[root];
    int rootIdx = idx[root];

    int child = (root << 1) + 1;  // left child index

    while (child < heapSize)
    {
        int right = child + 1;
        int swapIdx = child;
        float swapDist = dist[child];

        if (right < heapSize)
        {
            float rightDist = dist[right];
            if (rightDist > swapDist)
            {
                swapIdx = right;
                swapDist = rightDist;
            }
        }

        if (rootDist >= swapDist)
        {
            // Root is larger than both children; heap property satisfied.
            break;
        }

        // Move larger child up.
        dist[root] = swapDist;
        idx[root] = idx[swapIdx];

        root = swapIdx;
        child = (root << 1) + 1;
    }

    dist[root] = rootDist;
    idx[root] = rootIdx;
}

/*
 * Build a max-heap in-place from an unsorted array of size K.
 * Uses bottom-up heap construction (O(K)).
 */
template <int K>
__device__ __forceinline__ void buildMaxHeap(float *dist, int *idx)
{
    // Start from the last parent node and sift down to restore heap property.
    for (int i = (K >> 1) - 1; i >= 0; --i)
    {
        siftDown<K>(dist, idx, i, K);
    }
}

/*
 * Convert a max-heap to a sorted array in ascending order using heapsort.
 * Precondition: dist / idx represent a valid max-heap of size K.
 * Postcondition: dist / idx are sorted so that dist[0] <= dist[1] <= ... <= dist[K-1].
 */
template <int K>
__device__ __forceinline__ void heapSortAscending(float *dist, int *idx)
{
    for (int end = K - 1; end > 0; --end)
    {
        // Move current maximum (root) to the end of the active range.
        float tmpDist = dist[0];
        dist[0] = dist[end];
        dist[end] = tmpDist;

        int tmpIdx = idx[0];
        idx[0] = idx[end];
        idx[end] = tmpIdx;

        // Restore heap property in the reduced range [0, end).
        siftDown<K>(dist, idx, 0, end);
    }
}

/***************************** CUDA kernel **********************************/

/*
 * Templated k-NN kernel for 2D points.
 *
 * Template parameter K is the number of nearest neighbors to find.
 *
 * Each thread:
 * - Loads one query point.
 * - Iterates over all data points in tiles loaded into shared memory.
 * - Maintains a per-thread max-heap of size K with the best candidates.
 * - After processing all data, sorts the heap and writes the k nearest
 *   neighbors (index + distance) to the output array.
 *
 * Grid/block layout:
 * - blockDim.x = number of threads per block (chosen in run_knn()).
 * - gridDim.x  = ceil(query_count / blockDim.x).
 */
template <int K>
__global__ void knn_kernel_2d(const float2 *__restrict__ query,
                              int query_count,
                              const float2 *__restrict__ data,
                              int data_count,
                              std::pair<int, float> *__restrict__ result)
{
    __shared__ float2 dataTile[DATA_TILE_SIZE];

    const int tid = threadIdx.x;
    const int globalQueryIdx = blockIdx.x * blockDim.x + tid;

    const bool active = (globalQueryIdx < query_count);

    // Per-thread k-NN heap storage: distances and indices.
    // K is a compile-time constant, and data_count >= K (as per problem spec).
    float heapDist[K];
    int   heapIdx[K];

    int count = 0;  // Number of data points processed so far by this thread.

    // Load query point into register for active threads.
    float2 q;
    if (active)
    {
        q = query[globalQueryIdx];
    }

    // Loop over data in tiles.
    for (int base = 0; base < data_count; base += DATA_TILE_SIZE)
    {
        int tileSize = DATA_TILE_SIZE;
        if (base + tileSize > data_count)
        {
            tileSize = data_count - base;
        }

        // Load tile of data points into shared memory (all threads cooperate).
        for (int t = tid; t < tileSize; t += blockDim.x)
        {
            dataTile[t] = data[base + t];
        }

        __syncthreads();

        if (active)
        {
            // Process this tile for the current query point.
            for (int j = 0; j < tileSize; ++j)
            {
                const float2 p = dataTile[j];

                const float dx = p.x - q.x;
                const float dy = p.y - q.y;

                // Squared Euclidean distance (no sqrt needed).
                const float dist = fmaf(dx, dx, dy * dy);
                const int   idx  = base + j;

                if (count < K)
                {
                    // Fill initial buffer until we have K elements.
                    heapDist[count] = dist;
                    heapIdx[count]  = idx;
                    ++count;

                    // Once we reach K elements, build the initial max-heap.
                    if (count == K)
                    {
                        buildMaxHeap<K>(heapDist, heapIdx);
                    }
                }
                else
                {
                    // Heap is full: keep only if better than current worst.
                    if (dist < heapDist[0])
                    {
                        heapDist[0] = dist;
                        heapIdx[0]  = idx;
                        siftDown<K>(heapDist, heapIdx, 0, K);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (active)
    {
        // At this point, data_count >= K ensures count >= K.
        // heapDist / heapIdx represent a max-heap of size K.

        // Sort heap in ascending order of distance.
        heapSortAscending<K>(heapDist, heapIdx);

        // Write results for this query: result[query * K + j] = j-th nearest neighbor.
        const int outBase = globalQueryIdx * K;
        for (int j = 0; j < K; ++j)
        {
            result[outBase + j].first  = heapIdx[j];
            result[outBase + j].second = heapDist[j];
        }
    }
}

/***************************** Host interface *******************************/

/*
 * Host function to launch the appropriate k-NN kernel specialization.
 *
 * Parameters:
 * - query:  device pointer to array of float2 (queries), size query_count.
 * - query_count: number of query points.
 * - data:   device pointer to array of float2 (data points), size data_count.
 * - data_count: number of data points.
 * - result: device pointer to array of std::pair<int,float>, size query_count * k.
 * - k:      number of nearest neighbors to find; power of two in [32, 1024].
 */
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Reasonable default block size for modern data center GPUs.
    // 128 threads per block balances occupancy and per-block workload.
    constexpr int BLOCK_SIZE = 128;

    dim3 block(BLOCK_SIZE);
    dim3 grid((query_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Dispatch to a kernel instantiation with compile-time K.
    switch (k)
    {
        case 32:
            knn_kernel_2d<32><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_kernel_2d<64><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_kernel_2d<128><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_kernel_2d<256><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_kernel_2d<512><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_kernel_2d<1024><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem guarantees k is one of {32, 64, 128, 256, 512, 1024},
            // so this path should not be hit. For robustness, we fall back to
            // the largest supported K (1024) and only use the first k results.
            knn_kernel_2d<1024><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
    }

    // Synchronize to ensure completion before returning to the host.
    cudaDeviceSynchronize();
}