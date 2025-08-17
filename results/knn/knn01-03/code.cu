#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <utility>

// Optimized CUDA k-NN for 2D points (squared Euclidean distance).
// Key design:
// - One thread block processes one query point (gridDim.x = query_count).
// - All threads collaboratively stream through the data points in tiles.
// - Each tile: every thread computes ITEMS_PER_THREAD distances; a warp-wide ballot
//   compacts only those candidates that beat the current block-wide threshold into a
//   shared "keeper" buffer.
// - A single thread (thread 0) maintains a block-wide max-heap of size k in shared
//   memory. Only "keeper" candidates are merged into the heap. This minimizes the
//   sequential heap maintenance work.
// - After all tiles, the heap is heap-sorted into ascending distance order and written
//   to the output result for the query.
//
// Notes:
// - No additional device memory is allocated; only shared memory is used.
// - k is a power of two in [32, 1024]. threads_per_block is chosen as min(256, k).
// - ITEMS_PER_THREAD is set to 16 for good arithmetic intensity and memory coalescing.
// - Shared memory per block usage: 8 * (T + k) bytes, where T = threads_per_block * ITEMS_PER_THREAD.
//   For worst-case threads_per_block=256, k<=1024, T=4096 -> smem ~ 8*(4096+1024)=40KB per block.
// - This code assumes modern NVIDIA datacenter GPUs (e.g., A100/H100) and a recent CUDA toolkit.

#ifndef KNN_ITEMS_PER_THREAD
#define KNN_ITEMS_PER_THREAD 16
#endif

// Device utility: compute squared L2 distance between a data point and query.
static __device__ __forceinline__ float squared_l2(const float2 a, const float qx, const float qy) {
    float dx = a.x - qx;
    float dy = a.y - qy;
    // FMA to improve throughput and precision: dx*dx + dy*dy
    return __fmaf_rn(dy, dy, dx * dx);
}

// Max-heap helpers for distances and indices stored in shared memory.
// Heap property: parent has distance >= children distance.
static __device__ __forceinline__ void heap_sift_up(float* hdist, int* hidx, int pos) {
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        float dp = hdist[parent];
        float dc = hdist[pos];
        if (dp < dc) {
            // Swap parent and child
            int ip = hidx[parent];
            hdist[parent] = dc; hidx[parent] = hidx[pos];
            hdist[pos] = dp;     hidx[pos] = ip;
            pos = parent;
        } else {
            break;
        }
    }
}

static __device__ __forceinline__ void heap_sift_down(float* hdist, int* hidx, int size, int pos = 0) {
    while (true) {
        int left = (pos << 1) + 1;
        if (left >= size) break;
        int right = left + 1;
        int maxc = left;
        if (right < size && hdist[right] > hdist[left]) {
            maxc = right;
        }
        if (hdist[pos] < hdist[maxc]) {
            float dp = hdist[pos];
            int   ip = hidx[pos];
            hdist[pos] = hdist[maxc]; hidx[pos] = hidx[maxc];
            hdist[maxc] = dp;         hidx[maxc] = ip;
            pos = maxc;
        } else {
            break;
        }
    }
}

static __device__ __forceinline__ void heap_push(float* hdist, int* hidx, int& size, float d, int i) {
    int pos = size;
    hdist[pos] = d;
    hidx[pos] = i;
    size++;
    heap_sift_up(hdist, hidx, pos);
}

static __device__ __forceinline__ void heap_replace_root(float* hdist, int* hidx, int size, float d, int i) {
    // Replace root and restore heap property
    hdist[0] = d;
    hidx[0] = i;
    heap_sift_down(hdist, hidx, size, 0);
}

static __device__ __forceinline__ void heap_sort_ascending(float* hdist, int* hidx, int size) {
    // In-place heapsort on a max-heap to produce ascending order.
    for (int i = size - 1; i > 0; --i) {
        // Swap root (current maximum) with the end
        float dr = hdist[0];
        int   ir = hidx[0];
        hdist[0] = hdist[i]; hidx[0] = hidx[i];
        hdist[i] = dr;       hidx[i] = ir;
        // Restore heap on the reduced heap [0..i-1]
        heap_sift_down(hdist, hidx, i, 0);
    }
}

// Kernel: One block per query. Uses dynamic shared memory for both the keeper buffer and the heap.
template<int ITEMS_PER_THREAD>
__global__ void knn_2d_kernel(const float2* __restrict__ query,
                              int query_count,
                              const float2* __restrict__ data,
                              int data_count,
                              std::pair<int, float>* __restrict__ result,
                              int k)
{
    const int qid = blockIdx.x;
    if (qid >= query_count) return;

    // Load query point into registers
    const float2 q = query[qid];
    const float qx = q.x;
    const float qy = q.y;

    // Dynamic shared memory layout:
    // [ keepDist | keepIdx | heapDist | heapIdx ]
    extern __shared__ unsigned char smem_raw[];
    float* keepDist = reinterpret_cast<float*>(smem_raw);
    int*   keepIdx  = reinterpret_cast<int*>(keepDist + blockDim.x * ITEMS_PER_THREAD);
    float* heapDist = reinterpret_cast<float*>(keepIdx + blockDim.x * ITEMS_PER_THREAD);
    int*   heapIdx  = reinterpret_cast<int*>(heapDist + k);

    // Shared scalars for coordination
    __shared__ int sKeepCount;
    __shared__ int sHeapSize;
    __shared__ float sThreshold;

    if (threadIdx.x == 0) {
        sKeepCount = 0;
        sHeapSize = 0;
        sThreshold = CUDART_INF_F; // Until heap is filled to size k
    }
    __syncthreads();

    // Process the dataset in tiles of T = blockDim.x * ITEMS_PER_THREAD points
    const int T = blockDim.x * ITEMS_PER_THREAD;
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = threadIdx.x & 31;

    for (int base = 0; base < data_count; base += T) {
        // Snapshot current threshold at tile start
        float tileThreshold = sThreshold;
        int tileStart = base;

        // Reset keeper counter for this tile
        if (threadIdx.x == 0) sKeepCount = 0;
        __syncthreads();

        // Each thread computes up to ITEMS_PER_THREAD distances and compacts "keepers"
        #pragma unroll
        for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
            int idx = tileStart + it * blockDim.x + threadIdx.x;
            bool in_range = (idx < data_count);

            float d = 0.0f;
            if (in_range) {
                float2 p = data[idx];
                d = squared_l2(p, qx, qy);
            }

            // Decide whether to keep this candidate based on the current block threshold.
            // If heap is not full yet, keep all valid candidates.
            bool heap_full = (sHeapSize >= k);
            bool keep = in_range && (!heap_full || (d < tileThreshold));

            // Warp-wide compaction: each warp atomically reserves space for its keepers
            unsigned mask = __ballot_sync(FULL_MASK, keep);
            int warpCount = __popc(mask);

            if (warpCount > 0) {
                int basePos = 0;
                if (lane == 0) {
                    basePos = atomicAdd(&sKeepCount, warpCount);
                }
                basePos = __shfl_sync(FULL_MASK, basePos, 0);

                if (keep) {
                    unsigned laneMask = mask & ((1u << lane) - 1u);
                    int offset = basePos + __popc(laneMask);
                    keepDist[offset] = d;
                    keepIdx[offset]  = idx;
                }
            }
        }

        __syncthreads();

        // Merge keepers into the heap (single-threaded to maintain heap invariants)
        if (threadIdx.x == 0) {
            int count = sKeepCount;
            int hsize = sHeapSize;

            // Fill the heap up to size k
            int i = 0;
            for (; i < count && hsize < k; ++i) {
                heap_push(heapDist, heapIdx, hsize, keepDist[i], keepIdx[i]);
            }

            // Update threshold if heap now full
            if (hsize == k) {
                sThreshold = heapDist[0];
            }

            // For remaining candidates, only keep if better than current threshold
            for (; i < count; ++i) {
                float d = keepDist[i];
                if (d < heapDist[0]) {
                    heap_replace_root(heapDist, heapIdx, hsize, d, keepIdx[i]);
                    sThreshold = heapDist[0]; // Update threshold to new root
                }
            }

            sHeapSize = hsize;
        }

        __syncthreads();
    }

    // At this point, heap contains the k nearest neighbors for this query.
    // Sort them in ascending distance order and write to output.
    if (threadIdx.x == 0) {
        int hsize = sHeapSize; // Should be k since data_count >= k
        heap_sort_ascending(heapDist, heapIdx, hsize);

        // Write results in ascending order: j-th nearest neighbor first
        int out_base = qid * k;
        for (int j = 0; j < k; ++j) {
            result[out_base + j].first  = heapIdx[j];
            result[out_base + j].second = heapDist[j];
        }
    }
}

// Host interface
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Threads per block: choose up to 256, but not exceeding k (and at least one warp).
    int threads_per_block = k < 256 ? (k >= 128 ? 128 : (k >= 64 ? 64 : 32)) : 256;

    // Launch one block per query
    dim3 grid(query_count);
    dim3 block(threads_per_block);

    // Shared memory size: 8 * (T + k) bytes, where T = threads_per_block * ITEMS_PER_THREAD
    const int T = threads_per_block * KNN_ITEMS_PER_THREAD;
    size_t smem_size = static_cast<size_t>(T + k) * (sizeof(float) + sizeof(int));

    // Increase the max dynamic shared memory size if needed (for Ampere/Hopper this can be large).
    cudaFuncSetAttribute(knn_2d_kernel<KNN_ITEMS_PER_THREAD>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_size));

    // Launch
    knn_2d_kernel<KNN_ITEMS_PER_THREAD><<<grid, block, smem_size>>>(
        query, query_count, data, data_count, result, k);
}