#include <cuda_runtime.h>
#include <utility>

/*
 * Optimized CUDA k-NN (k-nearest neighbors) for 2D points and squared Euclidean distance.
 *
 * Design overview (see individual comments for details):
 *  - One warp (32 threads) processes one query point.
 *  - Multiple warps per block (WARPS_PER_BLOCK) improve occupancy.
 *  - All 32 lanes in a warp cooperatively load and compute distances to data points in a coalesced fashion.
 *  - Lane 0 in each warp maintains a max-heap of size K (K = k) in shared memory to track the K nearest neighbors.
 *  - New candidate distances from all 32 lanes are sequentially fed to lane 0 via warp shuffles and inserted into the heap.
 *  - After processing all data points, lane 0 heap-sorts the K neighbors and writes them to the result array in ascending order.
 *
 *  This approach:
 *    - Achieves coalesced global memory access for data points.
 *    - Avoids per-thread large local arrays by using shared memory per warp.
 *    - Uses the GPU mainly for massive parallel distance computation; the inherently sequential top-k maintenance is localized to one lane per warp.
 */

static constexpr int WARP_SIZE        = 32;
static constexpr int WARPS_PER_BLOCK  = 4;                      // 4 warps per block -> 128 threads per block
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

/* ========================= Heap utilities (max-heap, K smallest elements) ========================= */

template <int K>
__device__ __forceinline__
void heap_sift_up(float *dist, int *idx, int pos)
{
    // Standard max-heap sift-up operation
    int parent = (pos - 1) >> 1;
    while (pos > 0 && dist[parent] < dist[pos]) {
        float td = dist[parent];
        dist[parent] = dist[pos];
        dist[pos] = td;

        int ti = idx[parent];
        idx[parent] = idx[pos];
        idx[pos] = ti;

        pos = parent;
        parent = (pos - 1) >> 1;
    }
}

template <int K>
__device__ __forceinline__
void heap_sift_down(float *dist, int *idx, int heap_size, int pos)
{
    // Standard max-heap sift-down operation
    while (true) {
        int left   = (pos << 1) + 1;
        int right  = left + 1;
        int largest = pos;

        if (left < heap_size && dist[left] > dist[largest]) {
            largest = left;
        }
        if (right < heap_size && dist[right] > dist[largest]) {
            largest = right;
        }
        if (largest == pos) {
            break;
        }

        float td = dist[pos];
        dist[pos] = dist[largest];
        dist[largest] = td;

        int ti = idx[pos];
        idx[pos] = idx[largest];
        idx[largest] = ti;

        pos = largest;
    }
}

template <int K>
__device__ __forceinline__
void heap_insert(float *dist, int *idx, int &heap_size, float d, int i)
{
    // Maintain a max-heap of up to K elements containing the K smallest distances seen so far.
    if (heap_size < K) {
        // Heap not full yet: insert new element and sift up.
        int pos = heap_size++;
        dist[pos] = d;
        idx[pos]  = i;
        heap_sift_up<K>(dist, idx, pos);
    } else if (d < dist[0]) {
        // New distance is better (smaller) than the current worst (root of max-heap).
        dist[0] = d;
        idx[0]  = i;
        heap_sift_down<K>(dist, idx, heap_size, 0);
    }
}

template <int K>
__device__
void heap_sort(float *dist, int *idx, int heap_size)
{
    // Heapsort to produce ascending order of distances.
    // The heap is a max-heap in dist[0 .. heap_size-1].
    for (int i = heap_size - 1; i > 0; --i) {
        // Move current largest to the end.
        float td = dist[0];
        dist[0] = dist[i];
        dist[i] = td;

        int ti = idx[0];
        idx[0] = idx[i];
        idx[i] = ti;

        // Restore heap property in reduced heap [0 .. i-1].
        heap_sift_down<K>(dist, idx, i, 0);
    }
}

/* ========================= k-NN kernel (templated by K = k) ========================= */

template <int K>
__global__
void knn_kernel_2d(const float2 * __restrict__ query,
                   int query_count,
                   const float2 * __restrict__ data,
                   int data_count,
                   std::pair<int, float> * __restrict__ result)
{
    // Shared memory heaps: one heap (size K) per warp in the block.
    __shared__ float s_heap_dist[WARPS_PER_BLOCK][K];
    __shared__ int   s_heap_idx [WARPS_PER_BLOCK][K];

    const int thread_id       = threadIdx.x;
    const int warp_id_in_block = thread_id / WARP_SIZE;              // warp index within this block [0, WARPS_PER_BLOCK)
    const int lane_id         = thread_id & (WARP_SIZE - 1);         // lane index within warp [0, 31]
    const int warp_global_id  = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    // Each warp processes one query; extra warps (if any) exit.
    if (warp_global_id >= query_count) {
        return;
    }

    // Load query point. Only lane 0 does the global memory load; then broadcast to all lanes.
    float qx = 0.0f, qy = 0.0f;
    if (lane_id == 0) {
        float2 q = query[warp_global_id];
        qx = q.x;
        qy = q.y;
    }
    const unsigned full_mask = 0xFFFFFFFFu;
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);

    // Pointers to this warp's heap in shared memory.
    float *heap_dist = s_heap_dist[warp_id_in_block];
    int   *heap_idx  = s_heap_idx[warp_id_in_block];

    // Only lane 0 will manipulate the heap, but all lanes share the underlying shared memory.
    int heap_size = 0;

    // Loop over data points in tiles of WARP_SIZE.
    // Each lane loads one data point per iteration; together the warp processes 32 points in parallel.
    for (int base = 0; base < data_count; base += WARP_SIZE) {
        const int idx_global = base + lane_id;

        // Compute squared distance for this lane's data point (if in range).
        float d = 0.0f;
        if (idx_global < data_count) {
            float2 p = data[idx_global];
            float dx = p.x - qx;
            float dy = p.y - qy;
            d = dx * dx + dy * dy;
        }

        // Feed all 32 candidates from the warp to lane 0 sequentially via warp shuffles.
        // This preserves correctness while still doing the distance computations in parallel.
        for (int s = 0; s < WARP_SIZE; ++s) {
            float cand_d   = __shfl_sync(full_mask, d, s);
            int   cand_idx = base + s;
            if (lane_id == 0 && cand_idx < data_count) {
                heap_insert<K>(heap_dist, heap_idx, heap_size, cand_d, cand_idx);
            }
        }
    }

    // After processing all data points, lane 0 sorts the heap and writes out the K nearest neighbors.
    if (lane_id == 0) {
        // Given the problem guarantee data_count >= K, heap_size should be K.
        // For robustness, use heap_size in heapsort; it will be min(K, data_count).
        heap_sort<K>(heap_dist, heap_idx, heap_size);

        const int out_base = warp_global_id * K;
        for (int i = 0; i < K; ++i) {
            result[out_base + i].first  = heap_idx[i];
            result[out_base + i].second = heap_dist[i];
        }
    }
}

/* ========================= Host-side dispatch function ========================= */

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0) {
        return;
    }

    dim3 block_dim(THREADS_PER_BLOCK);
    // Each block contains WARPS_PER_BLOCK warps; each warp processes one query.
    dim3 grid_dim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Dispatch to a kernel instance specialized for the given power-of-two k.
    switch (k) {
        case 32:
            knn_kernel_2d<32><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_kernel_2d<64><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_kernel_2d<128><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_kernel_2d<256><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_kernel_2d<512><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_kernel_2d<1024><<<grid_dim, block_dim>>>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem guarantees k is a power of two between 32 and 1024 inclusive.
            // If an unsupported k is provided, do nothing.
            return;
    }

    // Synchronize to ensure completion before the host continues.
    cudaDeviceSynchronize();
}