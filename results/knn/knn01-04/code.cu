#include <cuda_runtime.h>
#include <utility>

// ============================================================================
// Optimized k-NN (k-nearest neighbors) for 2D points using CUDA
// Target hardware: modern NVIDIA data-center GPUs (e.g., A100, H100)
// ============================================================================
//
// High-level algorithm overview (per query):
// -----------------------------------------
// 1. Each query is processed by a single thread block.
// 2. The block is split into several warps; each warp processes a disjoint
//    subset of the data points in a strided manner.
// 3. For each warp, lane threads compute distances to their assigned data
//    points. The distances and indices from all lanes in a warp are collected
//    into per-warp shared memory buffers.
// 4. Lane 0 of each warp maintains a max-heap (of size k) in shared memory
//    containing the k best candidates (smallest distances) seen by that warp.
//    The heap insertion is O(log k).
// 5. After all data points are processed, thread 0 of the block merges all
//    per-warp heaps into a final block-level max-heap of size k.
// 6. The final heap is heap-sorted in-place to obtain the k nearest neighbors
//    in ascending order of distance.
// 7. The results (indices and distances) are written to the output array.
//
// Data layout assumptions:
// ------------------------
// - query:  query_count elements of float2 (x, y)
// - data:   data_count elements of float2 (x, y)
// - result: query_count * k elements of std::pair<int, float>
//   For query i, result[i * k + j] holds the j-th nearest neighbor
//   data index and squared distance.
//
// Performance considerations:
// ---------------------------
// - One block per query ensures good parallelism for thousands of queries.
// - Each block uses WARPS_PER_BLOCK warps; each warp scans a ~1/W subset
//   of the data.
// - Heap sizes per warp and for the final heap are exactly k, where
//   k ∈ [32, 1024] and is a power of two.
// - Shared memory is used for per-warp heaps and staging buffers; no
//   additional global memory allocations are performed.
// - The number of threads per block and the amount of shared memory are
//   chosen so that the kernel runs within the default dynamic shared memory
//   limit on A100/H100 without requiring cudaFuncSetAttribute.
//
// ============================================================================

constexpr int KNN_WARP_SIZE        = 32;
constexpr int KNN_WARPS_PER_BLOCK  = 4;   // 4 warps * 32 threads = 128 threads per block

// ============================================================================
// Device-side heap utilities (max-heap) for distances
// ----------------------------------------------------------------------------
// We maintain a max-heap of size up to 'capacity'. The root (index 0)
// holds the largest distance among the elements in the heap. This allows
// efficient maintenance of the k smallest distances: new candidates are
// inserted only if they are better (smaller) than the current worst
// candidate at the root.
// ============================================================================

__device__ __forceinline__ void max_heap_sift_up(float* heap_dist, int* heap_idx, int index)
{
    // Standard max-heap "bubble up" operation to restore heap property
    while (index > 0)
    {
        int parent = (index - 1) >> 1;
        if (heap_dist[parent] >= heap_dist[index])
            break;

        // Swap parent and current
        float tmp_dist = heap_dist[parent];
        heap_dist[parent] = heap_dist[index];
        heap_dist[index] = tmp_dist;

        int tmp_idx = heap_idx[parent];
        heap_idx[parent] = heap_idx[index];
        heap_idx[index] = tmp_idx;

        index = parent;
    }
}

__device__ __forceinline__ void max_heap_sift_down(float* heap_dist, int* heap_idx, int index, int size)
{
    // Standard max-heap "sift down" operation to restore heap property
    while (true)
    {
        int left  = (index << 1) + 1;
        if (left >= size)
            break;  // No children

        int right = left + 1;
        int largest = left;

        if (right < size && heap_dist[right] > heap_dist[left])
            largest = right;

        if (heap_dist[index] >= heap_dist[largest])
            break;

        // Swap current with largest child
        float tmp_dist = heap_dist[index];
        heap_dist[index] = heap_dist[largest];
        heap_dist[largest] = tmp_dist;

        int tmp_idx = heap_idx[index];
        heap_idx[index] = heap_idx[largest];
        heap_idx[largest] = tmp_idx;

        index = largest;
    }
}

__device__ __forceinline__ void max_heap_insert(
    float* __restrict__ heap_dist,
    int*   __restrict__ heap_idx,
    int&   size,
    int    capacity,
    float  new_dist,
    int    new_idx)
{
    // Insert a new (distance, index) into a fixed-capacity max-heap.
    // Only distances smaller than the current maximum (heap root) are kept
    // once the heap is full.

    if (capacity <= 0)
        return;

    if (size < capacity)
    {
        // Heap not full: append new element and sift up
        int pos = size;
        heap_dist[pos] = new_dist;
        heap_idx[pos]  = new_idx;
        ++size;
        max_heap_sift_up(heap_dist, heap_idx, pos);
    }
    else
    {
        // Heap full: only insert if better (smaller distance) than current max
        if (new_dist >= heap_dist[0])
            return;

        heap_dist[0] = new_dist;
        heap_idx[0]  = new_idx;
        max_heap_sift_down(heap_dist, heap_idx, 0, size);
    }
}

__device__ __forceinline__ void max_heap_sort(float* heap_dist, int* heap_idx, int size)
{
    // In-place heap sort for a max-heap:
    // After this, heap_dist[0..size-1] are in ascending order, and heap_idx
    // is permuted correspondingly.

    for (int i = size - 1; i > 0; --i)
    {
        // Swap root (largest) with the last element in current heap range
        float tmp_dist = heap_dist[0];
        heap_dist[0] = heap_dist[i];
        heap_dist[i] = tmp_dist;

        int tmp_idx = heap_idx[0];
        heap_idx[0] = heap_idx[i];
        heap_idx[i] = tmp_idx;

        // Restore heap property on the reduced heap [0..i-1]
        max_heap_sift_down(heap_dist, heap_idx, 0, i);
    }
}

// ============================================================================
// CUDA kernel: k-NN for 2D points (squared Euclidean distance)
// ============================================================================

__global__ void knn_kernel(
    const float2* __restrict__ query,
    int                         query_count,
    const float2* __restrict__ data,
    int                         data_count,
    std::pair<int, float>* __restrict__ result,
    int                         k)
{
    const int query_idx = blockIdx.x;
    if (query_idx >= query_count)
        return;

    // Load the query point into registers (shared across all threads of the block)
    const float2 q = query[query_idx];
    const float  qx = q.x;
    const float  qy = q.y;

    // ------------------------------------------------------------------------
    // Shared memory layout (dynamic shared memory):
    //
    // - warp_heap_dist  : [KNN_WARPS_PER_BLOCK][k]    (float)
    // - warp_heap_idx   : [KNN_WARPS_PER_BLOCK][k]    (int)
    // - final_heap_dist : [k]                         (float)
    // - final_heap_idx  : [k]                         (int)
    // - warp_input_dist : [KNN_WARPS_PER_BLOCK][32]   (float)
    // - warp_input_idx  : [KNN_WARPS_PER_BLOCK][32]   (int)
    // - warp_heap_sizes : [KNN_WARPS_PER_BLOCK]       (int)
    //
    // All arrays reside in a single contiguous dynamic shared memory block.
    // ------------------------------------------------------------------------

    extern __shared__ unsigned char shared_raw[];
    unsigned char* ptr = shared_raw;

    // Per-warp heaps (distances and indices)
    float* warp_heap_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * k * KNN_WARPS_PER_BLOCK;

    int* warp_heap_idx = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * k * KNN_WARPS_PER_BLOCK;

    // Final block-level heap
    float* final_heap_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * k;

    int* final_heap_idx = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * k;

    // Per-warp input staging buffers (one distance/index per lane)
    float* warp_input_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * KNN_WARP_SIZE * KNN_WARPS_PER_BLOCK;

    int* warp_input_idx = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * KNN_WARP_SIZE * KNN_WARPS_PER_BLOCK;

    // Per-warp heap sizes (number of valid elements in each warp heap)
    int* warp_heap_sizes = reinterpret_cast<int*>(ptr);
    // ptr += sizeof(int) * KNN_WARPS_PER_BLOCK; // Not needed further

    const int tid     = threadIdx.x;
    const int warp_id = tid / KNN_WARP_SIZE;  // warp index within block
    const int lane_id = tid % KNN_WARP_SIZE;  // lane index within warp

    const int warps_per_block = KNN_WARPS_PER_BLOCK;
    const int warp_stride     = warps_per_block * KNN_WARP_SIZE; // stride over data for each warp

    // Per-warp views into shared memory
    float* heap_dist_w = warp_heap_dist + warp_id * k;
    int*   heap_idx_w  = warp_heap_idx  + warp_id * k;

    float* input_dist_w = warp_input_dist + warp_id * KNN_WARP_SIZE;
    int*   input_idx_w  = warp_input_idx  + warp_id * KNN_WARP_SIZE;

    int local_heap_size = 0; // Only meaningful for lane 0 of each warp

    // ------------------------------------------------------------------------
    // Main distance computation loop:
    // Each warp processes a disjoint subset of data points:
    //   For base = warp_id * 32; base < data_count; base += warp_stride
    //     lane i processes index (base + i) if within range.
    // Distances and indices are stored in per-warp shared buffers;
    // lane 0 of each warp then feeds them into a per-warp heap.
    // ------------------------------------------------------------------------

    for (int base = warp_id * KNN_WARP_SIZE; base < data_count; base += warp_stride)
    {
        int data_idx = base + lane_id;

        float dist = 0.0f;
        int   idx  = -1;  // -1 marks "no valid point" in this lane for this iteration

        if (data_idx < data_count)
        {
            float2 p = data[data_idx];
            float dx = p.x - qx;
            float dy = p.y - qy;
            dist = dx * dx + dy * dy;
            idx  = data_idx;
        }

        // Store to per-warp input buffers
        input_dist_w[lane_id] = dist;
        input_idx_w[lane_id]  = idx;

        // Synchronize lanes within the warp to ensure buffers are ready
        __syncwarp();

        if (lane_id == 0)
        {
            // Lane 0 consumes all 32 buffered candidates for this warp
            #pragma unroll
            for (int i = 0; i < KNN_WARP_SIZE; ++i)
            {
                int cand_idx = input_idx_w[i];
                if (cand_idx < 0)
                    continue;  // invalid entry (out-of-range data index)

                float cand_dist = input_dist_w[i];
                max_heap_insert(heap_dist_w, heap_idx_w, local_heap_size, k, cand_dist, cand_idx);
            }
        }

        // Ensure lane 0 has finished using the buffers before they are overwritten
        __syncwarp();
    }

    // Store per-warp heap sizes (only lane 0 of each warp has valid local_heap_size)
    if (lane_id == 0)
    {
        warp_heap_sizes[warp_id] = local_heap_size;
    }

    // Synchronize all threads in the block before merging heaps
    __syncthreads();

    // ------------------------------------------------------------------------
    // Merge per-warp heaps into a single block-level heap and output results.
    // Only thread 0 performs the merge and final sorting.
    // ------------------------------------------------------------------------
    if (tid == 0)
    {
        int final_heap_size = 0;

        // Merge all warp heaps into final heap
        for (int w = 0; w < warps_per_block; ++w)
        {
            int w_size = warp_heap_sizes[w];
            float* hd = warp_heap_dist + w * k;
            int*   hi = warp_heap_idx  + w * k;

            for (int i = 0; i < w_size; ++i)
            {
                max_heap_insert(final_heap_dist, final_heap_idx, final_heap_size, k, hd[i], hi[i]);
            }
        }

        // At this point, final_heap_dist/final_heap_idx form a max-heap containing
        // the k nearest neighbors (or fewer if data_count < k). Given the problem
        // constraints (data_count >= k), final_heap_size will be exactly k.

        max_heap_sort(final_heap_dist, final_heap_idx, final_heap_size);

        // Write sorted results for this query to global memory
        const int out_base = query_idx * k;
        for (int i = 0; i < k; ++i)
        {
            result[out_base + i].first  = final_heap_idx[i];
            result[out_base + i].second = final_heap_dist[i];
        }
    }
}

// ============================================================================
// Host-side wrapper: run_knn
// ============================================================================

void run_knn(
    const float2*              query,
    int                        query_count,
    const float2*              data,
    int                        data_count,
    std::pair<int, float>*     result,
    int                        k)
{
    // Basic sanity check (parameters are assumed valid as per problem statement)
    if (query_count <= 0 || data_count <= 0 || k <= 0)
        return;

    dim3 block_dim(KNN_WARP_SIZE * KNN_WARPS_PER_BLOCK);  // 128 threads per block
    dim3 grid_dim(query_count);                           // one block per query

    // Compute required dynamic shared memory size
    size_t shared_bytes = 0;
    shared_bytes += sizeof(float) * k * KNN_WARPS_PER_BLOCK;            // warp_heap_dist
    shared_bytes += sizeof(int)   * k * KNN_WARPS_PER_BLOCK;            // warp_heap_idx
    shared_bytes += sizeof(float) * k;                                  // final_heap_dist
    shared_bytes += sizeof(int)   * k;                                  // final_heap_idx
    shared_bytes += sizeof(float) * KNN_WARP_SIZE * KNN_WARPS_PER_BLOCK; // warp_input_dist
    shared_bytes += sizeof(int)   * KNN_WARP_SIZE * KNN_WARPS_PER_BLOCK; // warp_input_idx
    shared_bytes += sizeof(int)   * KNN_WARPS_PER_BLOCK;                // warp_heap_sizes

    // Launch kernel (asynchronous)
    knn_kernel<<<grid_dim, block_dim, shared_bytes>>>(query,
                                                      query_count,
                                                      data,
                                                      data_count,
                                                      result,
                                                      k);
}