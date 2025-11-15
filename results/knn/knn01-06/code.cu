#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// -------------------------------------------------------------------------------------------------
// Optimized brute-force k-NN for 2D points on modern NVIDIA GPUs (A100 / H100).
//
// - Each CUDA thread processes one query point.
// - All threads in a block cooperatively load tiles of the data points into shared memory.
// - Each thread scans all data points, computing distances to its query.
// - For each query, a per-thread max-heap of size K (K = k) is maintained in local memory,
//   holding the current k nearest neighbors (smallest distances).
// - To insert a candidate distance d:
//     * If we haven't filled the heap yet, we append it.
//     * Once we have K elements, we build an initial max-heap.
//     * Afterwards, for each new distance d, if d < heap[0] (the current worst among the k best),
//       we replace the root and sift-down to restore max-heap property.
// - After scanning all data points, we perform an in-place heap sort on the per-thread heap,
//   which converts the max-heap into an array sorted by ascending distance.
// - The sorted (index, distance) pairs are written to the result array as std::pair<int,float>.
//
// Design notes:
//   * Complexity per query: O(N log K) distance candidates with extremely cheap log K (~10) for K<=1024.
//   * k is a power of two between 32 and 1024 inclusive; we use template specialization on K
//     for best performance and to allow fixed-size arrays.
//   * We only allocate shared memory for a tile of data points; no extra global memory is allocated.
//   * Local thread heaps are stored in thread-local memory (which may spill to global and be cached).
//   * Block size and tile size are chosen to provide good occupancy and memory reuse.
// -------------------------------------------------------------------------------------------------

// Tune these for the target GPU if needed.
#ifndef KNN_BLOCK_DIM
#define KNN_BLOCK_DIM 128    // threads per block (queries processed per block)
#endif

#ifndef KNN_TILE_DATA
#define KNN_TILE_DATA 256    // number of data points loaded per tile into shared memory
#endif

// -------------------------------------------------------------------------------------------------
// Device helper: sift-down operation for a max-heap stored in flat arrays.
//
// heap_dist, heap_idx : arrays of length >= heap_size
// idx                 : index to start sifting down from (usually 0)
// heap_size           : number of elements in the heap [0, heap_size)
// -------------------------------------------------------------------------------------------------
template<int K>
__device__ __forceinline__
void heap_sift_down(float (&heap_dist)[K], int (&heap_idx)[K], int idx, int heap_size)
{
    while (true) {
        int left  = (idx << 1) + 1;  // left child
        if (left >= heap_size) {
            break; // no children
        }
        int right = left + 1;        // right child

        int   largest    = idx;
        float largestVal = heap_dist[idx];

        float leftVal = heap_dist[left];
        if (leftVal > largestVal) {
            largest    = left;
            largestVal = leftVal;
        }

        if (right < heap_size) {
            float rightVal = heap_dist[right];
            if (rightVal > largestVal) {
                largest    = right;
                largestVal = rightVal;
            }
        }

        if (largest == idx) {
            break; // heap property satisfied
        }

        // Swap current node with the largest child
        float tmpd          = heap_dist[idx];
        heap_dist[idx]      = heap_dist[largest];
        heap_dist[largest]  = tmpd;

        int tmpi            = heap_idx[idx];
        heap_idx[idx]       = heap_idx[largest];
        heap_idx[largest]   = tmpi;

        idx = largest;
    }
}

// -------------------------------------------------------------------------------------------------
// Device kernel: brute-force k-NN for 2D points with per-thread max-heap.
// Template parameters:
//   K    : number of neighbors to keep (k)
//   TILE : tile size for shared-memory loads of data points
//
// Each thread processes a single query point.
// -------------------------------------------------------------------------------------------------
template<int K, int TILE>
__global__
void knn_kernel_2d(const float2* __restrict__ query,
                   int                 query_count,
                   const float2* __restrict__ data,
                   int                 data_count,
                   std::pair<int, float>* __restrict__ result)
{
    // Shared memory tile for data points
    __shared__ float2 sh_data[TILE];

    const int tid       = threadIdx.x;
    const int q_global  = blockIdx.x * blockDim.x + tid;
    const bool valid    = (q_global < query_count);

    // Load query point into a register for faster access
    float2 q;
    if (valid) {
        q = query[q_global];
    }

    // Per-thread heap storage: distances and corresponding data indices.
    // We maintain a max-heap of size K.
    float heap_dist[K];
    int   heap_idx[K];
    int   filled = 0;          // number of elements currently in heap (up to K)
    bool  heap_built = false;  // whether we've built the initial max-heap

    // Loop over data points in tiles
    for (int base = 0; base < data_count; base += TILE) {
        int tile_size = data_count - base;
        if (tile_size > TILE) tile_size = TILE;

        // Cooperative load of data tile into shared memory.
        for (int i = tid; i < tile_size; i += blockDim.x) {
            sh_data[i] = data[base + i];
        }

        __syncthreads();

        if (valid) {
            // Process this tile for the current query
            for (int i = 0; i < tile_size; ++i) {
                int global_idx = base + i;

                float dx = q.x - sh_data[i].x;
                float dy = q.y - sh_data[i].y;
                float dist = dx * dx + dy * dy;  // squared Euclidean distance

                if (filled < K) {
                    // Still filling initial buffer; no heap property needed yet.
                    heap_dist[filled] = dist;
                    heap_idx[filled]  = global_idx;
                    ++filled;

                    // Once we have K elements, build the initial max-heap.
                    if (filled == K) {
                        for (int j = (K / 2) - 1; j >= 0; --j) {
                            heap_sift_down<K>(heap_dist, heap_idx, j, K);
                        }
                        heap_built = true;
                    }
                } else {
                    // Heap already built and full: potential replacement of max element.
                    // For a max-heap, root (index 0) holds the current worst (largest distance)
                    // among the K best; only replace it if this candidate is better (smaller).
                    if (dist < heap_dist[0]) {
                        heap_dist[0] = dist;
                        heap_idx[0]  = global_idx;
                        heap_sift_down<K>(heap_dist, heap_idx, 0, K);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (!valid) {
        return;
    }

    // At this point, we must have at least K elements (data_count >= K by assumption).
    // If for some reason heap_built is still false (e.g., data_count == K),
    // we build the heap now.
    if (!heap_built) {
        // filled should equal K here, but we guard against pathological cases.
        int heap_size = (filled < K) ? filled : K;
        for (int j = (heap_size / 2) - 1; j >= 0; --j) {
            heap_sift_down<K>(heap_dist, heap_idx, j, heap_size);
        }
        heap_built = true;
    }

    // In-place heap sort of the max-heap.
    // After this, heap_dist/heap_idx will be sorted in ascending order of distance.
    for (int end = K - 1; end > 0; --end) {
        // Swap root (largest) with last element in current heap
        float tmpd         = heap_dist[0];
        heap_dist[0]       = heap_dist[end];
        heap_dist[end]     = tmpd;

        int tmpi           = heap_idx[0];
        heap_idx[0]        = heap_idx[end];
        heap_idx[end]      = tmpi;

        // Restore heap property on the reduced heap [0, end)
        heap_sift_down<K>(heap_dist, heap_idx, 0, end);
    }

    // Write out sorted results for this query.
    // result[q * K + j] = j-th nearest neighbor for this query.
    int result_base = q_global * K;
    for (int i = 0; i < K; ++i) {
        result[result_base + i].first  = heap_idx[i];
        result[result_base + i].second = heap_dist[i];
    }
}

// -------------------------------------------------------------------------------------------------
// Host interface:
//
// void run_knn(const float2 *query, int query_count,
//              const float2 *data,  int data_count,
//              std::pair<int, float> *result, int k);
//
// query, data, result are device pointers allocated via cudaMalloc.
// k is a power of two between 32 and 1024 inclusive.
// -------------------------------------------------------------------------------------------------
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    const int block_dim = KNN_BLOCK_DIM;
    dim3 block(block_dim);
    dim3 grid((query_count + block_dim - 1) / block_dim);

    // Dispatch to a specialized kernel based on k.
    // k is guaranteed to be a power of two between 32 and 1024 inclusive.
    switch (k) {
        case 32:
            knn_kernel_2d<32, KNN_TILE_DATA><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_kernel_2d<64, KNN_TILE_DATA><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_kernel_2d<128, KNN_TILE_DATA><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_kernel_2d<256, KNN_TILE_DATA><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_kernel_2d<512, KNN_TILE_DATA><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_kernel_2d<1024, KNN_TILE_DATA><<<grid, block>>>(query, query_count, data, data_count, result);
            break;
        default:
            // k is assumed valid by problem constraints; no fallback implemented.
            break;
    }

    // Optionally, the caller can synchronize and check for errors here:
    // cudaDeviceSynchronize();
    // cudaGetLastError();
}