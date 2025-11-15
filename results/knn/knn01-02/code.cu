#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one GPU thread to each query point.
// Each thread scans all data points and maintains a per-thread fixed-size
// max-heap of size K containing the K nearest neighbors found so far.
//
// Rationale:
//   - The number of queries is on the order of thousands, which provides
//     enough parallelism (thousands of threads) to keep modern GPUs busy.
//   - Each thread independently processes its query, avoiding inter-thread
//     synchronization and shared-memory complexity.
//   - A max-heap gives O(log K) insertion per candidate, which is important
//     for large K (up to 1024) and large data_count (millions).
//   - We compute squared Euclidean distance (no sqrt) to minimize ALU cost.
//
// Notes:
//   - K is known at compile time via template parameter, enabling the
//     compiler to optimize loops around heap operations.
//   - For K=1024, the per-thread heap arrays reside in local memory (backed
//     by global memory). This is unavoidable because K is large enough that
//     storing all values in registers or shared memory per thread would
//     severely limit occupancy. However, the dominating cost remains the
//     O(query_count * data_count) distance computations and global loads,
//     so this is an acceptable tradeoff.
//   - The final result for each query is sorted by increasing distance
//     (result[i * K + 0] is the nearest neighbor).

namespace {

// Max-heap helper: sift element at index `idx` upward to restore heap property.
template <int K>
__device__ __forceinline__
void heap_sift_up(float* __restrict__ heap_dist,
                  int*   __restrict__ heap_idx,
                  int idx)
{
    // Max-heap: parent index for current node.
    while (idx > 0) {
        int parent = (idx - 1) >> 1;
        if (heap_dist[parent] >= heap_dist[idx])
            break;
        // Swap parent and current.
        float tmpd = heap_dist[parent];
        heap_dist[parent] = heap_dist[idx];
        heap_dist[idx] = tmpd;

        int tmpi = heap_idx[parent];
        heap_idx[parent] = heap_idx[idx];
        heap_idx[idx] = tmpi;

        idx = parent;
    }
}

// Max-heap helper: sift element at index `idx` downward to restore heap property.
template <int K>
__device__ __forceinline__
void heap_sift_down(float* __restrict__ heap_dist,
                    int*   __restrict__ heap_idx,
                    int    heap_size,
                    int    idx)
{
    while (true) {
        int left  = (idx << 1) + 1;
        int right = left + 1;
        int largest = idx;

        if (left < heap_size && heap_dist[left] > heap_dist[largest]) {
            largest = left;
        }
        if (right < heap_size && heap_dist[right] > heap_dist[largest]) {
            largest = right;
        }
        if (largest == idx) {
            break;
        }

        // Swap current and largest child.
        float tmpd = heap_dist[largest];
        heap_dist[largest] = heap_dist[idx];
        heap_dist[idx] = tmpd;

        int tmpi = heap_idx[largest];
        heap_idx[largest] = heap_idx[idx];
        heap_idx[idx] = tmpi;

        idx = largest;
    }
}

// Max-heap helper: insert a new (distance, index) into the heap.
// If heap is not full, grows the heap. If full, replaces the root if the
// new distance is smaller than the current maximum.
template <int K>
__device__ __forceinline__
void heap_insert(float d,
                 int   idx_val,
                 float* __restrict__ heap_dist,
                 int*   __restrict__ heap_idx,
                 int&   heap_size)
{
    if (heap_size < K) {
        // Insert at end and sift up.
        int pos = heap_size;
        heap_dist[pos] = d;
        heap_idx[pos]  = idx_val;
        heap_sift_up<K>(heap_dist, heap_idx, pos);
        ++heap_size;
    } else {
        // Heap is full: only insert if new distance is smaller than current max.
        if (d < heap_dist[0]) {
            heap_dist[0] = d;
            heap_idx[0]  = idx_val;
            heap_sift_down<K>(heap_dist, heap_idx, heap_size, 0);
        }
    }
}

// Pop the maximum element from the heap (root) and shrink heap_size by 1.
// Returns (distance, index) through references.
template <int K>
__device__ __forceinline__
void heap_pop_max(float* __restrict__ heap_dist,
                  int*   __restrict__ heap_idx,
                  int&   heap_size,
                  float& out_dist,
                  int&   out_idx)
{
    out_dist = heap_dist[0];
    out_idx  = heap_idx[0];

    int last = heap_size - 1;
    heap_dist[0] = heap_dist[last];
    heap_idx[0]  = heap_idx[last];
    --heap_size;

    if (heap_size > 0) {
        heap_sift_down<K>(heap_dist, heap_idx, heap_size, 0);
    }
}

// Kernel: one thread per query. Each thread builds a max-heap of size K
// containing the K nearest neighbors for its query.
template <int K>
__global__
void knn_kernel(const float2* __restrict__ query,
                int                     query_count,
                const float2* __restrict__ data,
                int                     data_count,
                std::pair<int, float>* __restrict__ result)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_count) {
        return;
    }

    // Load query point into registers.
    float2 q = query[tid];
    float qx = q.x;
    float qy = q.y;

    // Per-thread max-heap for top-K nearest neighbors.
    // For large K these arrays reside in local memory (backed by global),
    // which is acceptable given that distance computations dominate runtime.
    float heap_dist[K];
    int   heap_idx[K];
    int   heap_size = 0;

    // Scan all data points.
    for (int j = 0; j < data_count; ++j) {
        float2 p = data[j];
        float dx = qx - p.x;
        float dy = qy - p.y;
        float d2 = dx * dx + dy * dy;  // squared Euclidean distance

        heap_insert<K>(d2, j, heap_dist, heap_idx, heap_size);
    }

    // At this point, heap_size == K (because data_count >= K).
    // We now extract elements from the max-heap into the result array
    // in ascending order of distance:
    //   - Popping from a max-heap gives elements in descending order.
    //   - We fill the output from the end to the beginning so that
    //     result[0] is the nearest (smallest distance).
    int base = tid * K;
    for (int out_pos = heap_size - 1; out_pos >= 0; --out_pos) {
        float d;
        int   idx_val;
        heap_pop_max<K>(heap_dist, heap_idx, heap_size, d, idx_val);

        std::pair<int, float>& r = result[base + out_pos];
        r.first  = idx_val;
        r.second = d;
    }
}

} // anonymous namespace

// Host-side launcher.
// query   : device pointer to array of query_count float2 points.
// data    : device pointer to array of data_count float2 points.
// result  : device pointer to array of query_count * k std::pair<int,float>.
// k       : power of two between 32 and 1024 inclusive.
//
// The function launches a CUDA kernel specialized for the given k.
// No additional device memory is allocated here; all buffers are
// expected to be allocated by the caller using cudaMalloc.
void run_knn(const float2* query,
             int           query_count,
             const float2* data,
             int           data_count,
             std::pair<int, float>* result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Tune threads per block for modern NVIDIA GPUs.
    // 256 threads per block is a reasonable default balancing occupancy
    // and per-thread resource usage.
    const int block_size = 256;
    int grid_size = (query_count + block_size - 1) / block_size;

    switch (k) {
        case 32:
            knn_kernel<32><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_kernel<64><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_kernel<128><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_kernel<256><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_kernel<512><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_kernel<1024><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
            break;
        default:
            // According to the problem statement, k is always a power of two
            // between 32 and 1024 inclusive, so we should never get here.
            // If we do, fall back to the nearest larger supported K by
            // clamping to [32, 1024] and rounding up to power of two.
            {
                int kk = k;
                if (kk < 32) kk = 32;
                if (kk > 1024) kk = 1024;

                // Round up to next power of two within [32, 1024].
                int pow2 = 32;
                while (pow2 < kk && pow2 < 1024) {
                    pow2 <<= 1;
                }

                switch (pow2) {
                    case 32:
                        knn_kernel<32><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
                        break;
                    case 64:
                        knn_kernel<64><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
                        break;
                    case 128:
                        knn_kernel<128><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
                        break;
                    case 256:
                        knn_kernel<256><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
                        break;
                    case 512:
                        knn_kernel<512><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
                        break;
                    case 1024:
                    default:
                        knn_kernel<1024><<<grid_size, block_size>>>(query, query_count, data, data_count, result);
                        break;
                }
            }
            break;
    }

    // The caller can choose whether to synchronize and check for errors.
    // cudaDeviceSynchronize() and error handling are intentionally omitted
    // to keep this function as a pure launcher.
}