// Optimized CUDA implementation of k-nearest neighbors (k-NN) for 2D points.
// Target: modern NVIDIA data-center GPUs (e.g., A100/H100).
//
// Design overview (see also inline comments below):
// - One warp (32 threads) processes one query point.
// - The block comprises multiple warps; a shared-memory tile of the data set
//   is loaded once per block and reused by all warps in the block.
// - Each warp maintains its own private top-k structure (a max-heap) in
//   shared memory; only lane 0 manipulates the heap, while all lanes
//   compute distances in parallel.
// - Data points are processed in tiles from global memory, cached in shared
//   memory, then further in warp-sized batches. Each batch of 32 distances
//   is written to a per-warp buffer in shared memory, then consumed by lane 0
//   to update the heap.
// - At the end, the heap is converted into a sorted (ascending) array via
//   heapsort, and the results are written to the output array.

#include <cuda_runtime.h>
#include <utility>   // std::pair
#include <cfloat>    // FLT_MAX

// Full-warp mask for modern GPUs (32-wide warps).
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Device-side heap utilities (max-heap of size up to K).
// The heap stores distances (float) and corresponding indices (int).
// Heap property: the largest distance is at index 0.
// ---------------------------------------------------------------------------

template <int K>
__device__ __forceinline__
void heap_insert(float dist, int idx, float *heap_dist, int *heap_idx, int &heap_size)
{
    // Insert a new element into a max-heap (0-based indexing).
    // Precondition: heap_size < K.
    int i = heap_size;
    ++heap_size;

    // Bubble up new element until heap property holds.
    while (i > 0)
    {
        int parent = (i - 1) >> 1;
        float parentDist = heap_dist[parent];
        if (parentDist >= dist)
            break;

        heap_dist[i] = parentDist;
        heap_idx[i]  = heap_idx[parent];
        i = parent;
    }

    heap_dist[i] = dist;
    heap_idx[i]  = idx;
}

__device__ __forceinline__
void heap_replace_root(float dist, int idx, float *heap_dist, int *heap_idx, int heap_size)
{
    // Replace the root of a max-heap with a new element whose distance is
    // strictly smaller than the old root's distance, then restore the heap.
    // Heap uses 0-based indexing, elements in [0, heap_size).
    int i = 0;
    int left = 1;

    // Sift-down procedure.
    while (left < heap_size)
    {
        int right   = left + 1;
        int largest = left;

        if (right < heap_size && heap_dist[right] > heap_dist[left])
            largest = right;

        if (heap_dist[largest] <= dist)
            break;

        heap_dist[i] = heap_dist[largest];
        heap_idx[i]  = heap_idx[largest];

        i    = largest;
        left = (i << 1) + 1;
    }

    heap_dist[i] = dist;
    heap_idx[i]  = idx;
}

template <int K>
__device__ __forceinline__
void heap_sort_ascending(float *heap_dist, int *heap_idx, int heap_size)
{
    // In-place heapsort on a max-heap represented in [0, heap_size).
    // Result: heap_dist and heap_idx sorted ascending by distance.
    for (int end = heap_size - 1; end > 0; --end)
    {
        // Swap root (largest) with the element at 'end'.
        float droot = heap_dist[0];
        heap_dist[0] = heap_dist[end];
        heap_dist[end] = droot;

        int iroot = heap_idx[0];
        heap_idx[0] = heap_idx[end];
        heap_idx[end] = iroot;

        // Restore max-heap in the reduced range [0, end).
        int parent = 0;
        while (true)
        {
            int left = (parent << 1) + 1;
            if (left >= end)
                break;

            int right   = left + 1;
            int largest = left;
            if (right < end && heap_dist[right] > heap_dist[left])
                largest = right;

            if (heap_dist[parent] >= heap_dist[largest])
                break;

            float td = heap_dist[parent];
            heap_dist[parent] = heap_dist[largest];
            heap_dist[largest] = td;

            int ti = heap_idx[parent];
            heap_idx[parent] = heap_idx[largest];
            heap_idx[largest] = ti;

            parent = largest;
        }
    }
    // After this loop, heap_dist[0..heap_size-1] is in ascending order.
}

// ---------------------------------------------------------------------------
// k-NN kernel
//
// Template parameters:
// - K               : number of nearest neighbors to find (power of two 32..1024)
// - WARPS_PER_BLOCK : number of warps per block (blockDim.x = 32 * WARPS_PER_BLOCK)
// - TILE_POINTS     : number of data points cached per block in shared memory
// ---------------------------------------------------------------------------

template <int K, int WARPS_PER_BLOCK, int TILE_POINTS>
__global__
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                std::pair<int, float> * __restrict__ result)
{
    constexpr int WARP_SIZE = 32;

    // Thread/wrap indices within the block.
    int warp_id_in_block = threadIdx.x / WARP_SIZE;   // warp index [0, WARPS_PER_BLOCK)
    int lane_id          = threadIdx.x & (WARP_SIZE - 1); // lane index [0, 31]

    // Global warp index corresponds to the query index this warp will process.
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    bool is_active_warp = (global_warp_id < query_count);

    // -----------------------------------------------------------------------
    // Shared memory layout (single allocation, partitioned logically):
    //
    // [0]   : data tile, size TILE_POINTS * sizeof(float2)
    // [1]   : top-k index arrays for all warps, size WARPS_PER_BLOCK * K * sizeof(int)
    // [2]   : top-k distance arrays for all warps, size WARPS_PER_BLOCK * K * sizeof(float)
    // [3]   : per-warp candidate index buffer, size WARPS_PER_BLOCK * WARP_SIZE * sizeof(int)
    // [4]   : per-warp candidate distance buffer, size WARPS_PER_BLOCK * WARP_SIZE * sizeof(float)
    // -----------------------------------------------------------------------
    extern __shared__ unsigned char shared_raw[];

    unsigned char *ptr = shared_raw;

    // Data tile
    float2 *tile_data = reinterpret_cast<float2 *>(ptr);
    ptr += TILE_POINTS * sizeof(float2);

    // Top-k buffers
    int *topk_index_base = reinterpret_cast<int *>(ptr);
    ptr += WARPS_PER_BLOCK * K * sizeof(int);

    float *topk_dist_base = reinterpret_cast<float *>(ptr);
    ptr += WARPS_PER_BLOCK * K * sizeof(float);

    // Per-warp candidate buffers (warpSize candidates at a time)
    int *cand_index_base = reinterpret_cast<int *>(ptr);
    ptr += WARPS_PER_BLOCK * WARP_SIZE * sizeof(int);

    float *cand_dist_base = reinterpret_cast<float *>(ptr);
    // ptr += WARPS_PER_BLOCK * WARP_SIZE * sizeof(float); // not needed further

    // Pointers to this warp's private top-k heap and candidate buffer
    int   *heap_idx  = topk_index_base + warp_id_in_block * K;
    float *heap_dist = topk_dist_base  + warp_id_in_block * K;

    int   *cand_idx  = cand_index_base + warp_id_in_block * WARP_SIZE;
    float *cand_dist = cand_dist_base  + warp_id_in_block * WARP_SIZE;

    // -----------------------------------------------------------------------
    // Load the query point for this warp and broadcast to all lanes.
    // -----------------------------------------------------------------------
    float2 q = make_float2(0.0f, 0.0f);
    if (lane_id == 0 && is_active_warp)
        q = query[global_warp_id];

    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    // -----------------------------------------------------------------------
    // Initialize heap (only meaningful for active warps; maintained by lane 0).
    // heap_size is kept in a register, private to lane 0 in each warp.
    // -----------------------------------------------------------------------
    int heap_size = 0;

    // -----------------------------------------------------------------------
    // Process the entire data set in tiles cached in shared memory.
    // Each tile is cooperatively loaded by all threads in the block.
    // -----------------------------------------------------------------------
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS)
    {
        int tile_size = TILE_POINTS;
        if (tile_start + tile_size > data_count)
            tile_size = data_count - tile_start;

        // Load the tile from global memory into shared memory.
        for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x)
        {
            tile_data[idx] = data[tile_start + idx];
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // Within this tile, each warp processes all tile points in batches
        // of 32. Each lane computes one distance per batch, stores it into
        // the per-warp candidate buffer in shared memory.
        // Lane 0 then merges these 32 candidates into the warp-local heap.
        // -------------------------------------------------------------------
        for (int base = 0; base < tile_size; base += WARP_SIZE)
        {
            int idx_in_tile = base + lane_id;

            // Compute squared Euclidean distance for valid points.
            int   global_data_idx = -1;
            float dist_val        = 0.0f;

            if (idx_in_tile < tile_size)
            {
                float2 p = tile_data[idx_in_tile];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                dist_val = dx * dx + dy * dy;
                global_data_idx = tile_start + idx_in_tile;
            }

            // Store candidates into per-warp shared buffers.
            cand_idx[lane_id]  = global_data_idx;
            cand_dist[lane_id] = dist_val;

            // Ensure all lanes have written their candidate before lane 0 reads.
            __syncwarp();

            // Lane 0 updates the heap with all valid candidates from this batch.
            if (is_active_warp && lane_id == 0)
            {
#pragma unroll
                for (int i = 0; i < WARP_SIZE; ++i)
                {
                    int idx_cand = cand_idx[i];
                    if (idx_cand < 0)
                        continue;  // invalid candidate (out of tile range)

                    float d = cand_dist[i];

                    if (heap_size < K)
                    {
                        // Heap not full yet: always insert.
                        heap_insert<K>(d, idx_cand, heap_dist, heap_idx, heap_size);
                    }
                    else
                    {
                        // Heap full: only insert if candidate is closer than current worst.
                        if (d < heap_dist[0])
                        {
                            heap_replace_root(d, idx_cand, heap_dist, heap_idx, heap_size);
                        }
                    }
                }
            }

            // Synchronize warp before reusing candidate buffer.
            __syncwarp();
        }

        // Synchronize block before overwriting tile_data in next iteration.
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // After processing all tiles, each active warp has its top-K heap.
    // Convert the heap to a sorted array (ascending) and write out results.
    // -----------------------------------------------------------------------
    if (is_active_warp && lane_id == 0)
    {
        // data_count >= K by problem statement, so heap_size should be K.
        // As a safety net, limit size to K.
        if (heap_size > K)
            heap_size = K;

        // Sort the heap in ascending order of distance.
        heap_sort_ascending<K>(heap_dist, heap_idx, heap_size);

        // Write out the results for this query.
        int out_base = global_warp_id * K;
        for (int i = 0; i < K; ++i)
        {
            // For completeness, if heap_size < K (should not occur here),
            // fill remaining slots with invalid indices and max distance.
            int   idx_out = (i < heap_size) ? heap_idx[i]  : -1;
            float d_out   = (i < heap_size) ? heap_dist[i] : FLT_MAX;

            result[out_base + i].first  = idx_out;
            result[out_base + i].second = d_out;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher: run_knn
//
// C++ interface:
//   void run_knn(const float2 *query, int query_count,
//                const float2 *data,  int data_count,
//                std::pair<int, float> *result, int k);
//
// Assumptions:
// - query, data, and result are already allocated in device memory.
// - data_count >= k, and k is a power of two between 32 and 1024 inclusive.
// - query_count is large enough to benefit from GPU parallelism.
// ---------------------------------------------------------------------------

template <int K>
inline void launch_knn_kernel(const float2 *query, int query_count,
                              const float2 *data,  int data_count,
                              std::pair<int, float> *result)
{
    constexpr int WARP_SIZE        = 32;
    constexpr int WARPS_PER_BLOCK  = 8;     // 8 warps per block = 256 threads
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TILE_POINTS      = 4096;  // shared-memory tile size for data

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Compute required dynamic shared memory size:
    // tile_data:              TILE_POINTS * sizeof(float2)
    // topk_index_base:        WARPS_PER_BLOCK * K * sizeof(int)
    // topk_dist_base:         WARPS_PER_BLOCK * K * sizeof(float)
    // cand_index_base:        WARPS_PER_BLOCK * WARP_SIZE * sizeof(int)
    // cand_dist_base:         WARPS_PER_BLOCK * WARP_SIZE * sizeof(float)
    size_t shared_bytes =
        TILE_POINTS * sizeof(float2) +
        WARPS_PER_BLOCK * (K * (sizeof(int) + sizeof(float)) +
                           WARP_SIZE * (sizeof(int) + sizeof(float)));

    knn_kernel<K, WARPS_PER_BLOCK, TILE_POINTS>
        <<<grid_dim, block_dim, shared_bytes>>>(query, query_count,
                                                data, data_count,
                                                result);
}

void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    if (query == nullptr || data == nullptr || result == nullptr)
        return;
    if (query_count <= 0 || data_count <= 0 || k <= 0)
        return;

    // Dispatch to an instantiation specialized for the requested k.
    // The problem guarantees k is a power of two in [32, 1024].
    switch (k)
    {
        case 32:
            launch_knn_kernel<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            launch_knn_kernel<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            launch_knn_kernel<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            launch_knn_kernel<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            launch_knn_kernel<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            launch_knn_kernel<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // Unsupported k (should not happen given problem constraints).
            // For safety, do nothing.
            break;
    }

    // The caller can perform cudaDeviceSynchronize() and error checking.
}