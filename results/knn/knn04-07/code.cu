#include <cuda_runtime.h>
#include <utility>

// Simple POD replacement for std::pair<int,float> on device.
// Layout matches std::pair<int,float> for typical ABI (two 4-byte fields).
struct PairIF {
    int   first;
    float second;
};

constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 4;         // 4 warps -> 128 threads per block
constexpr int BLOCK_THREADS     = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int TILE_SIZE_POINTS  = 2048;      // 2048 points * 8 bytes = 16 KB
                                             // Shared heap per block (max k=1024): 4 warps * 1024 * 8 B = 32 KB
                                             // Total shared memory per block ~ 48 KB

// ========================= Heap utilities (max-heap) =========================

// Max-heap insert for top-k maintenance.
// Only lane 0 in each warp calls these functions; they operate on shared memory.
template <int K>
__device__ __forceinline__
void heapify_up(float* __restrict__ heap_dist,
                int*   __restrict__ heap_idx,
                int pos)
{
    // Bubble up to maintain max-heap property.
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        float d_pos    = heap_dist[pos];
        float d_parent = heap_dist[parent];
        if (d_parent >= d_pos) {
            break;
        }
        // Swap parent and current.
        heap_dist[pos] = d_parent;
        heap_idx[pos]  = heap_idx[parent];
        heap_dist[parent] = d_pos;
        heap_idx[parent]  = heap_idx[pos];
        pos = parent;
    }
}

template <int K>
__device__ __forceinline__
void heapify_down(float* __restrict__ heap_dist,
                  int*   __restrict__ heap_idx,
                  int size,
                  int pos)
{
    // Bubble down to maintain max-heap property.
    while (true) {
        int left  = (pos << 1) + 1;
        int right = left + 1;
        int largest = pos;

        if (left < size && heap_dist[left] > heap_dist[largest]) {
            largest = left;
        }
        if (right < size && heap_dist[right] > heap_dist[largest]) {
            largest = right;
        }
        if (largest == pos) {
            break;
        }

        float d_pos     = heap_dist[pos];
        float d_largest = heap_dist[largest];
        int   i_pos     = heap_idx[pos];
        int   i_largest = heap_idx[largest];

        heap_dist[pos]    = d_largest;
        heap_idx[pos]     = i_largest;
        heap_dist[largest] = d_pos;
        heap_idx[largest]  = i_pos;

        pos = largest;
    }
}

// Insert candidate (dist, idx) into max-heap of size at most K.
// Keeps the smallest K distances seen so far.
// heap_size is updated accordingly (<= K).
template <int K>
__device__ __forceinline__
void heap_insert_max(float* __restrict__ heap_dist,
                     int*   __restrict__ heap_idx,
                     int&   heap_size,
                     float  dist,
                     int    idx)
{
    if (heap_size < K) {
        int pos = heap_size++;
        heap_dist[pos] = dist;
        heap_idx[pos]  = idx;
        heapify_up<K>(heap_dist, heap_idx, pos);
    } else {
        // Heap is full; only insert if this distance is better (smaller) than current worst.
        if (dist >= heap_dist[0]) {
            return;
        }
        heap_dist[0] = dist;
        heap_idx[0]  = idx;
        heapify_down<K>(heap_dist, heap_idx, heap_size, 0);
    }
}

// In-place heapsort on a max-heap of size heap_size.
// After sorting, heap_dist[0..heap_size-1] are in ascending order of distance.
template <int K>
__device__ __forceinline__
void heap_sort_max_heap(float* __restrict__ heap_dist,
                        int*   __restrict__ heap_idx,
                        int    heap_size)
{
    // Standard heapsort: repeatedly move max element (root) to the end,
    // shrink heap, and heapify down.
    for (int end = heap_size - 1; end > 0; --end) {
        // Swap root with end.
        float d_root = heap_dist[0];
        float d_end  = heap_dist[end];
        int   i_root = heap_idx[0];
        int   i_end  = heap_idx[end];

        heap_dist[0]  = d_end;
        heap_idx[0]   = i_end;
        heap_dist[end] = d_root;
        heap_idx[end]  = i_root;

        // Restore heap property on the reduced heap [0, end).
        heapify_down<K>(heap_dist, heap_idx, end, 0);
    }
}

// ========================= k-NN kernel (warp-per-query) =========================

template <int K>
__global__
void knn_kernel(const float2* __restrict__ query,
                int                        query_count,
                const float2* __restrict__ data,
                int                        data_count,
                PairIF*       __restrict__ result)
{
    extern __shared__ unsigned char shared_mem[];

    // Layout shared memory as:
    // [0 .. TILE_SIZE_POINTS-1]: tile of float2 data points
    // [next]: WARPS_PER_BLOCK * K floats for distances (heap)
    // [next]: WARPS_PER_BLOCK * K ints   for indices  (heap)
    float2* s_data = reinterpret_cast<float2*>(shared_mem);
    float*  s_heap_dist = reinterpret_cast<float*>(s_data + TILE_SIZE_POINTS);
    int*    s_heap_idx  = reinterpret_cast<int*>(s_heap_dist + WARPS_PER_BLOCK * K);

    const int tid               = threadIdx.x;
    const int lane              = tid & (WARP_SIZE - 1);
    const int warp_id_in_block  = tid >> 5; // tid / WARP_SIZE
    const int warp_global_id    = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    const bool warp_has_query   = (warp_global_id < query_count);
    const unsigned full_mask    = 0xffffffffu;

    // Each warp uses a private segment of the heap arrays in shared memory.
    float* heap_dist = s_heap_dist + warp_id_in_block * K;
    int*   heap_idx  = s_heap_idx  + warp_id_in_block * K;

    float qx = 0.0f;
    float qy = 0.0f;

    int heap_size = 0;

    // Load the query point for this warp and broadcast within warp.
    if (warp_has_query) {
        if (lane == 0) {
            float2 q = query[warp_global_id];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(full_mask, qx, 0);
        qy = __shfl_sync(full_mask, qy, 0);
    }

    // Process data in tiles cached in shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE_POINTS) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_SIZE_POINTS) {
            tile_size = TILE_SIZE_POINTS;
        }

        // Load current tile into shared memory cooperatively by the entire block.
        for (int idx = tid; idx < tile_size; idx += BLOCK_THREADS) {
            s_data[idx] = data[tile_start + idx];
        }
        __syncthreads();

        if (warp_has_query) {
            // Each warp now computes distances from its query point to all points in the tile.
            // We process the tile in chunks of 32 so that each lane contributes one candidate per iteration.
            for (int base = 0; base < tile_size; base += WARP_SIZE) {
                int idx = base + lane;
                bool in_range = (idx < tile_size);

                float cand_dist = 0.0f;
                int   cand_idx  = -1;

                if (in_range) {
                    float2 p = s_data[idx];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    // Squared Euclidean distance in 2D.
                    cand_dist = dx * dx + dy * dy;
                    cand_idx  = tile_start + idx;
                }

                // Collect active lanes (those that have a valid candidate).
                unsigned active_mask = __ballot_sync(full_mask, in_range);
                if (active_mask == 0) {
                    continue;
                }

                // Update the heap with all active candidates in this iteration.
                // Lane 0 performs heap updates; values are broadcast via shuffles.
                for (int src_lane = 0; src_lane < WARP_SIZE; ++src_lane) {
                    if (active_mask & (1u << src_lane)) {
                        float d_src = __shfl_sync(active_mask, cand_dist, src_lane);
                        int   i_src = __shfl_sync(active_mask, cand_idx,  src_lane);
                        if (lane == 0) {
                            heap_insert_max<K>(heap_dist, heap_idx, heap_size, d_src, i_src);
                        }
                    }
                }
                // Optional warp-wide sync to make sure all lanes observe consistent heap_size
                // before next iteration (lane 0 is the only writer).
                __syncwarp(full_mask);
            }
        }

        // Ensure all warps have finished processing this tile before reusing shared memory.
        __syncthreads();
    }

    // After all data tiles have been processed, each active warp holds a max-heap
    // of size K containing the k nearest neighbors for its query (unsorted).
    if (warp_has_query) {
        if (lane == 0) {
            // data_count >= K by problem statement, so heap_size >= K.
            if (heap_size > 0) {
                heap_sort_max_heap<K>(heap_dist, heap_idx, heap_size);
            }

            // Write out sorted results (ascending distance) for this query.
            PairIF* out = result + warp_global_id * K;
            const int out_size = (heap_size < K) ? heap_size : K;
            for (int i = 0; i < out_size; ++i) {
                out[i].first  = heap_idx[i];
                out[i].second = heap_dist[i];
            }
        }
    }
}

// ========================= Host launcher =========================

// Helper to launch the templated kernel for a specific K.
template <int K>
void launch_knn_kernel(const float2* d_query,
                       int            query_count,
                       const float2* d_data,
                       int            data_count,
                       PairIF*        d_result)
{
    if (query_count <= 0 || data_count <= 0) {
        return;
    }

    const int warps_needed = query_count; // 1 warp per query
    const int blocks = (warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 grid(blocks);
    dim3 block(BLOCK_THREADS);

    // Shared memory size: tile of data + heaps for all warps in block.
    const size_t shared_bytes =
        TILE_SIZE_POINTS * sizeof(float2) +
        WARPS_PER_BLOCK * K * (sizeof(float) + sizeof(int));

    knn_kernel<K><<<grid, block, shared_bytes>>>(d_query,
                                                 query_count,
                                                 d_data,
                                                 data_count,
                                                 d_result);
}

// Public C++ interface.
// query, data, and result live in device memory (cudaMalloc'ed).
void run_knn(const float2* query,
             int           query_count,
             const float2* data,
             int           data_count,
             std::pair<int, float>* result,
             int           k)
{
    if (query == nullptr || data == nullptr || result == nullptr) {
        return;
    }
    if (query_count <= 0 || data_count <= 0) {
        return;
    }

    // Reinterpret result pointer as our device-compatible PairIF.
    PairIF* d_result = reinterpret_cast<PairIF*>(result);

    // k is guaranteed to be a power of two between 32 and 1024 inclusive.
    switch (k) {
        case 32:
            launch_knn_kernel<32>(query, query_count, data, data_count, d_result);
            break;
        case 64:
            launch_knn_kernel<64>(query, query_count, data, data_count, d_result);
            break;
        case 128:
            launch_knn_kernel<128>(query, query_count, data, data_count, d_result);
            break;
        case 256:
            launch_knn_kernel<256>(query, query_count, data, data_count, d_result);
            break;
        case 512:
            launch_knn_kernel<512>(query, query_count, data, data_count, d_result);
            break;
        case 1024:
            launch_knn_kernel<1024>(query, query_count, data, data_count, d_result);
            break;
        default:
            // For safety, you could handle invalid k here.
            // Problem statement guarantees k is valid, so we simply return.
            return;
    }
}