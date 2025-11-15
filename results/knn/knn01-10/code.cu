#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

/*
    Optimized brute-force k-NN for 2D points on modern NVIDIA GPUs (e.g., A100/H100).

    Design overview (see kernel and helper functions below):

    - Each GPU thread is responsible for computing the k nearest neighbors for a single query point.
    - A thread block handles multiple queries concurrently (one query per thread).
    - The data set is traversed in tiles. Each tile of data points is loaded once into shared memory
      and reused by all threads in the block to reduce global memory bandwidth.
    - For each query, the thread maintains a max-heap of size k (stored in thread-local memory) containing
      the current k nearest neighbors (by squared Euclidean distance).
        * The heap is a standard binary max-heap, keyed by distance.
        * For each new data point, if the heap is not yet full, it is appended.
        * Once full, the heap root (current worst neighbor) is replaced when a closer point is found,
          followed by a heapify-down operation (O(log k)).
    - After all data points are processed, a heap sort is applied to each thread's max-heap to obtain
      the final neighbors sorted in ascending order of distance.
    - Result layout: for query i, its neighbors are written into result[i * k + j], where .first is the
      data index and .second is the squared distance.

    Constraints/assumptions (from the problem statement):
      * k is a power of two between 32 and 1024 inclusive.
      * data_count >= k.
      * query, data, and result are device pointers allocated with cudaMalloc.
      * Distances are squared Euclidean distances in 2D: (dx^2 + dy^2).
*/

namespace {

constexpr int MAX_K = 1024;              // Maximum supported k
constexpr int THREADS_PER_BLOCK = 128;   // Threads per block (each handles one query)
                                         // Must be a multiple of 32 for warp efficiency

// Max-heap helper: sift-down operation starting from 'root' index.
// The heap is stored in parallel arrays: dists (keys) and idxs (payload).
// heap_size is the number of elements in the heap.
//
// This maintains the max-heap property: parent distance >= children distances.
__device__ __forceinline__
void heapify_down(float *dists, int *idxs, int heap_size, int root)
{
    int current = root;
    while (true) {
        int left  = (current << 1) + 1;  // left child
        if (left >= heap_size) {
            break; // no children
        }
        int right = left + 1;           // right child
        int largest = current;

        if (dists[left] > dists[largest]) {
            largest = left;
        }
        if (right < heap_size && dists[right] > dists[largest]) {
            largest = right;
        }
        if (largest == current) {
            break; // heap property satisfied
        }

        // Swap current and largest child
        float tmpd = dists[current];
        dists[current] = dists[largest];
        dists[largest] = tmpd;

        int tmpi = idxs[current];
        idxs[current] = idxs[largest];
        idxs[largest] = tmpi;

        current = largest;
    }
}

// Build a max-heap in-place from an unsorted array of length heap_size.
// Complexity: O(heap_size)
__device__ __forceinline__
void build_max_heap(float *dists, int *idxs, int heap_size)
{
    // Start from last internal node and heapify down to the root
    for (int start = (heap_size - 2) >> 1; start >= 0; --start) {
        heapify_down(dists, idxs, heap_size, start);
    }
}

// In-place heap sort on a max-heap to produce ascending order by distance.
// After calling this, dists[0..heap_size-1] and idxs[0..heap_size-1] are sorted
// in ascending order of distance.
__device__ __forceinline__
void heap_sort_ascending(float *dists, int *idxs, int heap_size)
{
    // Standard heapsort from a max-heap:
    // Repeatedly move the max element to the end, reduce heap size, and heapify.
    for (int end = heap_size - 1; end > 0; --end) {
        // Swap root (max) with dists[end]
        float tmpd = dists[0];
        dists[0] = dists[end];
        dists[end] = tmpd;

        int tmpi = idxs[0];
        idxs[0] = idxs[end];
        idxs[end] = tmpi;

        // Restore heap property on the reduced heap [0, end)
        heapify_down(dists, idxs, end, 0);
    }
}

// CUDA kernel performing k-NN search for 2D points.
// Each thread processes one query point and computes its k nearest neighbors.
__global__
void knn_kernel(const float2 *__restrict__ query,
                int query_count,
                const float2 *__restrict__ data,
                int data_count,
                std::pair<int, float> *__restrict__ result,
                int k)
{
    // Shared-memory tile for data points.
    // All threads in the block cooperatively load a tile of data points into sdata,
    // then each thread scans that tile for its own query.
    __shared__ float2 sdata[THREADS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int global_qid = blockIdx.x * blockDim.x + tid;

    // Early exit for threads with no corresponding query
    if (global_qid >= query_count) {
        return;
    }

    // Load this thread's query point
    const float2 q = query[global_qid];

    // Per-thread max-heap for the k nearest neighbors of this query.
    // Stored in thread-local memory; size is fixed at MAX_K, but only the first 'k' entries are used.
    float heap_dists[MAX_K];
    int   heap_idxs[MAX_K];

    int   heap_size = 0;      // number of elements currently in the heap
    float current_max = 0.0f; // distance at heap root (largest among k best)

    // Iterate over data in tiles of size THREADS_PER_BLOCK
    for (int base = 0; base < data_count; base += THREADS_PER_BLOCK) {
        const int data_idx = base + tid;

        // Load one data point per thread into shared memory (if within bounds)
        if (data_idx < data_count) {
            sdata[tid] = data[data_idx];
        }
        __syncthreads();

        // Number of valid elements in this tile
        const int tile_count = min(THREADS_PER_BLOCK, data_count - base);

        // Scan all data points in this tile for this query
        // Use a simple loop; compiler can unroll this for small tile sizes.
        for (int i = 0; i < tile_count; ++i) {
            const float2 p = sdata[i];

            const float dx = p.x - q.x;
            const float dy = p.y - q.y;
            const float dist = dx * dx + dy * dy;
            const int   idx  = base + i;

            if (heap_size < k) {
                // Heap not full yet: just append
                heap_dists[heap_size] = dist;
                heap_idxs[heap_size]  = idx;
                ++heap_size;

                // Once we have exactly k elements, build a max-heap
                if (heap_size == k) {
                    build_max_heap(heap_dists, heap_idxs, heap_size);
                    current_max = heap_dists[0]; // root of max-heap
                }
            } else {
                // Heap is full: keep only if this point is closer than current worst
                if (dist < current_max) {
                    heap_dists[0] = dist;
                    heap_idxs[0]  = idx;
                    heapify_down(heap_dists, heap_idxs, k, 0);
                    current_max = heap_dists[0];
                }
            }
        }

        __syncthreads();
    }

    // At this point, heap_dists/heap_idxs contain up to k best neighbors in a max-heap.
    // data_count >= k by problem statement, so heap_size should be exactly k here.
    const int out_count = heap_size; // should equal k, but use heap_size for safety

    // Convert max-heap into ascending order (nearest first)
    if (out_count > 1) {
        heap_sort_ascending(heap_dists, heap_idxs, out_count);
    }

    // Write results to output array in row-major order:
    // For query global_qid, neighbors occupy result[global_qid * k + 0..k-1].
    const int out_base = global_qid * k;

    // Write the found neighbors
    for (int i = 0; i < out_count; ++i) {
        result[out_base + i].first  = heap_idxs[i];
        result[out_base + i].second = heap_dists[i];
    }

    // In the unlikely case where data_count < k (though problem says data_count >= k),
    // fill remaining entries with sentinel values.
    for (int i = out_count; i < k; ++i) {
        result[out_base + i].first  = -1;
        result[out_base + i].second = FLT_MAX;
    }
}

} // anonymous namespace

// Host-side interface as specified in the problem statement.
//
// query:       device pointer to array of query_count float2 points.
// query_count: number of query points.
// data:        device pointer to array of data_count float2 points.
// data_count:  number of data points.
// result:      device pointer to array of size query_count * k of std::pair<int,float>.
// k:           number of nearest neighbors to return for each query.
//
// This function launches the CUDA kernel; it assumes that 'query', 'data', and 'result'
// have been allocated with cudaMalloc and contain/point to device memory.
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0 || k > MAX_K) {
        return; // invalid parameters; no work
    }

    // Choose grid size so that each thread handles one query.
    const dim3 blockDim(THREADS_PER_BLOCK);
    const dim3 gridDim((query_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Launch the kernel.
    knn_kernel<<<gridDim, blockDim>>>(query, query_count, data, data_count, result, k);

    // Optionally, synchronize or check for errors here if desired by the caller.
    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // (error handling omitted as per problem statement)
}