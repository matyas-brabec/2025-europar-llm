#include <cuda_runtime.h>
#include <utility>

// -----------------------------------------------------------------------------
// Optimized k-NN (k-nearest neighbors) for 2D points using CUDA
// - One warp (32 threads) processes one query point.
// - All warps in a block share a tile of data points in shared memory.
// - Each warp maintains a private max-heap of size K (top-k neighbors) in shared
//   memory. Only lane 0 of the warp mutates the heap to avoid races.
// - Distances are squared Euclidean (L2^2) in 2D.
// -----------------------------------------------------------------------------

// Configuration for the target GPUs (A100/H100).
//  - 8 warps per block (256 threads).
//  - Data tile of 1024 points in shared memory.
constexpr int WARP_SIZE          = 32;
constexpr int WARPS_PER_BLOCK    = 8;
constexpr int THREADS_PER_BLOCK  = WARP_SIZE * WARPS_PER_BLOCK; // 256
constexpr int TILE_SIZE          = 1024;
constexpr int LOADS_PER_THREAD   = TILE_SIZE / THREADS_PER_BLOCK; // 4

static_assert(TILE_SIZE % THREADS_PER_BLOCK == 0,
              "TILE_SIZE must be a multiple of THREADS_PER_BLOCK");

// -----------------------------------------------------------------------------
// Device helper: max-heap operations for top-K selection
// -----------------------------------------------------------------------------

template<int K>
__device__ __forceinline__ void heap_insert_initial(float* heap_dists,
                                                    int*   heap_indices,
                                                    int&   heap_size,
                                                    float  dist,
                                                    int    idx)
{
    // Insert at the end and sift up to maintain max-heap property.
    int pos = heap_size;
    heap_size++;

    heap_dists[pos]   = dist;
    heap_indices[pos] = idx;

    // Sift up
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        if (heap_dists[parent] >= heap_dists[pos]) break;

        // Swap parent and current
        float td            = heap_dists[parent];
        heap_dists[parent]  = heap_dists[pos];
        heap_dists[pos]     = td;

        int ti              = heap_indices[parent];
        heap_indices[parent]= heap_indices[pos];
        heap_indices[pos]   = ti;

        pos = parent;
    }
}

template<int K>
__device__ __forceinline__ void heap_maybe_insert(float* heap_dists,
                                                  int*   heap_indices,
                                                  int&   heap_size,
                                                  float  dist,
                                                  int    idx)
{
    // If the heap is not yet full, perform initial inserts.
    if (heap_size < K) {
        heap_insert_initial<K>(heap_dists, heap_indices, heap_size, dist, idx);
        return;
    }

    // Heap is full. Root contains the current worst (maximum) distance.
    if (dist >= heap_dists[0]) {
        // Candidate is not better than current worst neighbor; discard.
        return;
    }

    // Replace root with the new candidate and sift down to maintain max-heap.
    heap_dists[0]   = dist;
    heap_indices[0] = idx;

    int pos = 0;
    while (true) {
        int left  = (pos << 1) + 1;
        int right = left + 1;
        if (left >= K) break;

        // Find child with larger distance
        int largest = left;
        if (right < K && heap_dists[right] > heap_dists[left]) {
            largest = right;
        }

        if (heap_dists[pos] >= heap_dists[largest]) break;

        // Swap current with child
        float td              = heap_dists[pos];
        heap_dists[pos]       = heap_dists[largest];
        heap_dists[largest]   = td;

        int ti                = heap_indices[pos];
        heap_indices[pos]     = heap_indices[largest];
        heap_indices[largest] = ti;

        pos = largest;
    }
}

// Pop the current worst (max) element from the heap and return it.
// This is used after all distances are processed to output neighbors
// in ascending order by performing a standard heap sort.
template<int K>
__device__ __forceinline__ void heap_pop_max(float* heap_dists,
                                             int*   heap_indices,
                                             int&   heap_size,
                                             float& out_dist,
                                             int&   out_idx)
{
    // Root of the max-heap is the current worst (maximum) distance.
    out_dist = heap_dists[0];
    out_idx  = heap_indices[0];

    // Move last element to root and decrease size.
    heap_size--;
    if (heap_size > 0) {
        heap_dists[0]   = heap_dists[heap_size];
        heap_indices[0] = heap_indices[heap_size];

        // Sift down from root to restore max-heap.
        int pos = 0;
        while (true) {
            int left  = (pos << 1) + 1;
            int right = left + 1;
            if (left >= heap_size) break;

            int largest = left;
            if (right < heap_size && heap_dists[right] > heap_dists[left]) {
                largest = right;
            }

            if (heap_dists[pos] >= heap_dists[largest]) break;

            float td              = heap_dists[pos];
            heap_dists[pos]       = heap_dists[largest];
            heap_dists[largest]   = td;

            int ti                = heap_indices[pos];
            heap_indices[pos]     = heap_indices[largest];
            heap_indices[largest] = ti;

            pos = largest;
        }
    }
}

// -----------------------------------------------------------------------------
// Core kernel: one warp per query, k supplied as template parameter K.
//
// For each block:
//   - Shared memory holds a TILE_SIZE chunk of data points.
//   - Each warp in the block processes a different query point.
//   - Each warp maintains a max-heap of size K in shared memory as its
//     intermediate top-K result.
// -----------------------------------------------------------------------------

template<int K>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int                    query_count,
                           const float2* __restrict__ data,
                           int                    data_count,
                           std::pair<int, float>* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0,
                  "K must be a power of two between 32 and 1024");

    // Shared memory:
    //  - sh_data: cache of TILE_SIZE data points.
    //  - sh_heap_dists/sh_heap_indices: per-warp max-heap for top-K.
    __shared__ float2 sh_data[TILE_SIZE];
    __shared__ float  sh_heap_dists[WARPS_PER_BLOCK][K];
    __shared__ int    sh_heap_indices[WARPS_PER_BLOCK][K];

    const unsigned full_mask = 0xffffffffu;

    // Warp and lane identifiers
    const int thread_id       = threadIdx.x;
    const int warp_id_in_block= thread_id / WARP_SIZE;
    const int lane_id         = thread_id % WARP_SIZE;

    // Global warp index: one warp corresponds to one query
    const int global_warp_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    const int query_idx       = global_warp_id;

    const bool warp_active    = (query_idx < query_count);

    // Load query point into registers and broadcast within the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active && lane_id == 0) {
        float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    // Broadcast from lane 0 to all lanes in the warp.
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);

    // Pointers to this warp's heap arrays in shared memory.
    float* heap_dists   = &sh_heap_dists[warp_id_in_block][0];
    int*   heap_indices = &sh_heap_indices[warp_id_in_block][0];

    // Per-warp heap size (number of neighbors currently stored).
    int heap_size = 0;

    // Process data in tiles of size TILE_SIZE.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        const int tile_size = min(TILE_SIZE, data_count - tile_start);

        // ---------------------------------------------------------------------
        // Load this tile of data points into shared memory.
        // Each thread loads LOADS_PER_THREAD contiguous points (if in range).
        // ---------------------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; ++i) {
            int dst_index   = thread_id + i * blockDim.x;   // 0 .. TILE_SIZE-1
            int global_idx  = tile_start + dst_index;
            if (dst_index < tile_size) {
                sh_data[dst_index] = data[global_idx];
            }
        }

        // Synchronize so that all data in the tile is visible to all warps.
        __syncthreads();

        // ---------------------------------------------------------------------
        // Each active warp processes the shared tile for its own query:
        //   - Compute squared distances to all points in the tile.
        //   - For each distance, attempt to insert into the warp's heap.
        //
        // We process the tile in chunks of WARP_SIZE data points:
        //   - In each chunk, each lane computes the distance to one point.
        //   - Lane 0 then sequentially processes the 32 candidate distances and
        //     updates the warp's heap using heap_maybe_insert().
        //
        // This design:
        //   - Uses all 32 lanes for distance computation,
        //   - Uses only lane 0 for heap mutation (avoids races),
        //   - Still allows many warps to run concurrently across the GPU.
        // ---------------------------------------------------------------------
        if (warp_active) {
            for (int base = 0; base < tile_size; base += WARP_SIZE) {
                int j = base + lane_id;

                // Compute distance for this lane's candidate (if within tile).
                float dist = 0.0f;
                int   idx  = -1;

                if (j < tile_size) {
                    float2 p = sh_data[j];
                    float dx = qx - p.x;
                    float dy = qy - p.y;
                    dist     = dx * dx + dy * dy;
                    idx      = tile_start + j;  // Global index into data[]
                }

                // Now, cooperatively process all 32 candidates in this chunk.
                // Each candidate is broadcast via warp shuffle and inserted
                // into the shared heap by lane 0 if it is among the current
                // top-K neighbors.
                #pragma unroll
                for (int src = 0; src < WARP_SIZE; ++src) {
                    float cand_dist = __shfl_sync(full_mask, dist, src);
                    int   cand_idx  = __shfl_sync(full_mask, idx,  src);

                    if (cand_idx < 0) {
                        // This lane did not contribute a valid candidate
                        // (out-of-range index at end of tile).
                        continue;
                    }

                    if (lane_id == 0) {
                        heap_maybe_insert<K>(heap_dists,
                                             heap_indices,
                                             heap_size,
                                             cand_dist,
                                             cand_idx);
                    }
                }

                // Ensure all lanes in the warp stay in lockstep before
                // proceeding to the next chunk of candidates.
                __syncwarp(full_mask);
            }
        }

        // Synchronize before loading the next tile of data points.
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // After all data points are processed, each active warp has a max-heap of
    // size K (heap_dists/heap_indices). The heap stores the K nearest
    // neighbors, but in arbitrary (heap) order.
    //
    // We convert the heap into a sorted list (ascending distance) by repeatedly
    // popping the maximum (worst) element. This is equivalent to performing a
    // heap sort. The result is written to global memory.
    // -------------------------------------------------------------------------
    if (warp_active && lane_id == 0) {
        const int base_out = query_idx * K;

        // heap_size should be >= K here; but if data_count == K, it will be
        // exactly K. We only pop K elements.
        int local_heap_size = heap_size;

        // Output in ascending order: we pop the maximum and fill the result
        // array from the end to the beginning.
        for (int out_pos = K - 1; out_pos >= 0; --out_pos) {
            float dist;
            int   idx;
            heap_pop_max<K>(heap_dists, heap_indices, local_heap_size, dist, idx);

            result[base_out + out_pos].first  = idx;
            result[base_out + out_pos].second = dist;
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface: run_knn
//
// query       : pointer to device array of float2 (size = query_count)
// query_count : number of query points
// data        : pointer to device array of float2 (size = data_count)
// data_count  : number of data points
// result      : pointer to device array of std::pair<int,float>
//               (size = query_count * k)
// k           : number of neighbors to retrieve (power of 2 between 32 and 1024)
// -----------------------------------------------------------------------------

void run_knn(const float2 *query,
             int            query_count,
             const float2 *data,
             int            data_count,
             std::pair<int, float> *result,
             int            k)
{
    if (query_count <= 0 || data_count <= 0) {
        return;
    }

    // One warp per query.
    const int warps_per_block = WARPS_PER_BLOCK;
    const dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    const int total_warps = query_count;
    const int grid_x = (total_warps + warps_per_block - 1) / warps_per_block;
    const dim3 grid_dim(grid_x, 1, 1);

    // Dispatch specialized kernel based on k (power of two in [32, 1024]).
    switch (k) {
        case 32:
            knn_kernel<32><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result);
            break;
        case 64:
            knn_kernel<64><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result);
            break;
        case 128:
            knn_kernel<128><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result);
            break;
        case 256:
            knn_kernel<256><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result);
            break;
        case 512:
            knn_kernel<512><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_kernel<1024><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result);
            break;
        default:
            // k is documented to always be a power of two between 32 and 1024.
            // If an unsupported k is provided, do nothing.
            break;
    }
}