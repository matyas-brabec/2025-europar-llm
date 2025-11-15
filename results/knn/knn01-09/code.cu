#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

/*
 * Optimized k-NN (k-nearest neighbors) for 2D points on modern NVIDIA GPUs.
 *
 * Design overview:
 * ----------------
 * - Each CUDA block processes exactly one query point.
 * - Threads in the block cooperatively compute squared Euclidean distances from
 *   that query to all data points.
 * - Distances are processed in tiles and stored in shared memory.
 * - A single thread (thread 0) in each block maintains a max-heap of size k
 *   (stored in shared memory) containing the current k nearest neighbors.
 *   This guarantees O(N log k) work per query, where N = data_count.
 * - After all data points are processed, the heap is converted into a sorted
 *   list (ascending by distance) and written to the result array.
 *
 * Parallelism:
 * ------------
 * - Distance computation is fully parallel within each tile: all threads in the
 *   block compute distances for different data points.
 * - Heap maintenance is done by a single thread per query (thread 0), but since
 *   there are many queries (thousands) and each query is assigned to a
 *   different block, the GPU executes many such heaps in parallel.
 *
 * Memory usage:
 * -------------
 * Shared memory per block:
 *   - TILE_SIZE distances (float) + indices (int)
 *   - K_MAX best distances (float) + indices (int)
 *   Total = TILE_SIZE * 8 bytes + K_MAX * 8 bytes
 *         = 4096 * 8 + 1024 * 8 = 40,960 bytes (< 48 KB default per-block limit)
 *
 * Constraints:
 * ------------
 * - k is a power of two in [32, 1024].
 * - data_count >= k.
 * - No additional device memory allocations (cudaMalloc) are performed.
 */

#ifndef KNN_K_MAX
#define KNN_K_MAX 1024
#endif

#ifndef KNN_TILE_SIZE
#define KNN_TILE_SIZE 4096
#endif

#ifndef KNN_THREADS_PER_BLOCK
#define KNN_THREADS_PER_BLOCK 256
#endif

// Device function to insert a candidate into a max-heap of size up to k.
// The heap is stored in two parallel arrays: best_dist and best_idx.
// Heap invariant: best_dist[0] is the largest distance in the heap.
// Only thread 0 of each block calls this function, so no synchronization is needed here.
__device__ __forceinline__
void heap_insert_max(float *best_dist, int *best_idx,
                     int &heap_size, const int k,
                     float dist, int idx)
{
    // If heap is not full yet, insert new element and heapify up.
    if (heap_size < k)
    {
        int i = heap_size;
        heap_size++;

        // Heapify up to maintain max-heap property.
        while (i > 0)
        {
            int parent = (i - 1) >> 1;
            if (best_dist[parent] >= dist)
                break;

            best_dist[i] = best_dist[parent];
            best_idx[i] = best_idx[parent];
            i = parent;
        }
        best_dist[i] = dist;
        best_idx[i] = idx;
    }
    // If heap is full and this candidate is better than the current worst,
    // replace the root and heapify down.
    else if (dist < best_dist[0])
    {
        int i = 0;

        // We use "dist" and "idx" as the new value to sink down the heap.
        while (true)
        {
            int left  = (i << 1) + 1;
            if (left >= heap_size)
                break;

            int right = left + 1;
            int largest = left;

            if (right < heap_size && best_dist[right] > best_dist[left])
                largest = right;

            if (best_dist[largest] <= dist)
                break;

            best_dist[i] = best_dist[largest];
            best_idx[i] = best_idx[largest];
            i = largest;
        }

        best_dist[i] = dist;
        best_idx[i] = idx;
    }
}

// Kernel implementing k-NN for 2D points using squared Euclidean distance.
__global__
void knn2d_kernel(const float2 * __restrict__ query,
                  int query_count,
                  const float2 * __restrict__ data,
                  int data_count,
                  std::pair<int, float> * __restrict__ result,
                  int k)
{
    // Shared memory buffers:
    // - s_tile_dist / s_tile_idx store a tile of distances and corresponding indices.
    // - s_best_dist / s_best_idx store the current max-heap of k best neighbors.
    __shared__ float s_tile_dist[KNN_TILE_SIZE];
    __shared__ int   s_tile_idx[KNN_TILE_SIZE];
    __shared__ float s_best_dist[KNN_K_MAX];
    __shared__ int   s_best_idx[KNN_K_MAX];

    const int qid = blockIdx.x;
    if (qid >= query_count)
        return;

    // Load the query point for this block.
    const float2 q = query[qid];

    // Local heap size for this block's query.
    // Only thread 0 uses and updates this variable.
    int heap_size = 0;

    // Process the data in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += KNN_TILE_SIZE)
    {
        int remaining = data_count - tile_start;
        int tile_elems = (remaining > KNN_TILE_SIZE) ? KNN_TILE_SIZE : remaining;

        // Each thread computes distances for a subset of points in this tile.
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x)
        {
            int idx = tile_start + i;
            float2 p = data[idx];

            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dist = dx * dx + dy * dy;  // squared Euclidean distance

            s_tile_dist[i] = dist;
            s_tile_idx[i] = idx;
        }

        // Ensure all distances for this tile are computed before using them.
        __syncthreads();

        // Thread 0 updates the heap with all candidates from this tile.
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < tile_elems; ++i)
            {
                float dist = s_tile_dist[i];
                int idx = s_tile_idx[i];
                heap_insert_max(s_best_dist, s_best_idx, heap_size, k, dist, idx);
            }
        }

        // Ensure thread 0 is done with this tile before reusing shared memory.
        __syncthreads();
    }

    // After processing all data points, the heap in s_best_* contains the k nearest
    // neighbors for this query, but arranged as a max-heap (largest distance at root).
    // We now extract them into the result array in ascending order of distance.
    if (threadIdx.x == 0)
    {
        // heap_size should be exactly k since data_count >= k.
        int hsize = heap_size;
        std::pair<int, float> *row = result + qid * k;

        // Perform an in-place heap sort on the heap arrays, writing into 'row'.
        // We repeatedly remove the maximum (root) and place it at position i;
        // by filling from the end to the beginning, we obtain ascending order.
        for (int i = hsize - 1; i >= 0; --i)
        {
            // Extract root (current maximum distance).
            int   best_index = s_best_idx[0];
            float best_dist  = s_best_dist[0];

            row[i].first  = best_index;
            row[i].second = best_dist;

            // Move the last element to the root and reduce heap size.
            float last_dist = s_best_dist[hsize - 1];
            int   last_idx  = s_best_idx[hsize - 1];
            hsize--;

            int current = 0;

            // Heapify down.
            while (true)
            {
                int left  = (current << 1) + 1;
                if (left >= hsize)
                    break;

                int right = left + 1;
                int largest = left;

                if (right < hsize && s_best_dist[right] > s_best_dist[left])
                    largest = right;

                if (s_best_dist[largest] <= last_dist)
                    break;

                s_best_dist[current] = s_best_dist[largest];
                s_best_idx[current]  = s_best_idx[largest];
                current = largest;
            }

            if (hsize > 0)
            {
                s_best_dist[current] = last_dist;
                s_best_idx[current]  = last_idx;
            }
        }
    }
}

// Host function interface as requested.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Assumptions from the problem statement:
    // - query, data, and result are already allocated on the device using cudaMalloc.
    // - data_count >= k.
    // - k is a power of two between 32 and 1024 inclusive.

    // Guard against k exceeding compile-time maximum.
    if (k > KNN_K_MAX || k <= 0 || query_count <= 0 || data_count <= 0)
        return;

    dim3 block_dim(KNN_THREADS_PER_BLOCK);
    dim3 grid_dim(query_count);

    knn2d_kernel<<<grid_dim, block_dim>>>(query,
                                          query_count,
                                          data,
                                          data_count,
                                          result,
                                          k);

    // No explicit cudaDeviceSynchronize() here to allow asynchronous execution.
    // The caller can synchronize or perform a blocking cudaMemcpy from 'result'
    // to the host to ensure completion.
}