#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one CUDA block per query point.
// Each block scans all data points and maintains a top-K set using a shared-memory max-heap.
// Threads cooperatively initialize the heap with the first K distances, then stream through
// the remaining data, attempting to insert a candidate into the heap only if it beats
// the current worst (root of the heap). Concurrency on heap updates is controlled via
// a lightweight shared-memory spinlock. This approach is scalable for large data_count,
// and efficient because only O(K) successful heap updates are expected per query.
//
// At the end, the heap is heap-sorted in-place (O(K log K)) to produce ascending results,
// and the indices/distances are written to the output.
//
// Notes:
// - Distance is squared Euclidean: (dx*dx + dy*dy)
// - k is a power of two between 32 and 1024 inclusive, and data_count >= k
// - Uses dynamic shared memory sized as k*(sizeof(float)+sizeof(int)) bytes
// - Uses 256 threads per block by default, which provides a good balance on A100/H100

#ifndef KNN_THREADS_PER_BLOCK
#define KNN_THREADS_PER_BLOCK 256
#endif

// Sift-down operation for a max-heap stored in arrays (distances and corresponding indices).
// This function assumes the heap property may be violated at 'start' and restores it.
static __device__ __forceinline__
void heap_sift_down(float* dists, int* idxs, int start, int heap_size) {
    int i = start;
    float valDist = dists[i];
    int valIdx = idxs[i];

    while (true) {
        int left = (i << 1) + 1; // 2*i + 1
        if (left >= heap_size) break;
        int right = left + 1;

        // Select the larger child (by distance) for max-heap
        int larger = left;
        float largerDist = dists[left];
        if (right < heap_size) {
            float rightDist = dists[right];
            if (rightDist > largerDist) {
                larger = right;
                largerDist = rightDist;
            }
        }

        if (valDist >= largerDist) {
            break;
        }

        // Move child up
        dists[i] = largerDist;
        idxs[i] = idxs[larger];
        i = larger;
    }

    dists[i] = valDist;
    idxs[i] = valIdx;
}

// Bottom-up heap construction (Floyd's algorithm) for a max-heap.
static __device__ __forceinline__
void heap_build(float* dists, int* idxs, int heap_size) {
    for (int i = (heap_size >> 1) - 1; i >= 0; --i) {
        heap_sift_down(dists, idxs, i, heap_size);
    }
}

// In-place heap sort on a max-heap to produce ascending order (smallest distance first).
static __device__ __forceinline__
void heap_heapsort(float* dists, int* idxs, int heap_size) {
    for (int n = heap_size; n > 1; --n) {
        // Swap root (current max) with last element
        float tmpD = dists[0];
        int tmpI = idxs[0];
        dists[0] = dists[n - 1];
        idxs[0] = idxs[n - 1];
        dists[n - 1] = tmpD;
        idxs[n - 1] = tmpI;

        // Restore heap property on the reduced heap [0, n-2]
        heap_sift_down(dists, idxs, 0, n - 1);
    }
}

// Kernel: each block handles one query point.
__global__ void knn2d_block_streaming_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result,
    int k)
{
    int qid = blockIdx.x;
    if (qid >= query_count) return;

    // Load query point into registers.
    float2 q = query[qid];
    float qx = q.x;
    float qy = q.y;

    // Dynamic shared memory layout:
    // [0 .. k-1]: float distances (heap)
    // [k .. 2k-1]: int indices (heap)
    extern __shared__ unsigned char smem[];
    float* sDists = reinterpret_cast<float*>(smem);
    int* sIdx = reinterpret_cast<int*>(sDists + k);

    // Shared lock to serialize heap updates.
    __shared__ int heapLock;
    if (threadIdx.x == 0) {
        heapLock = 0;
    }
    __syncthreads();

    // Phase 1: Parallel initialization - compute distances for the first k data points.
    // Assumption: data_count >= k
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float2 dp = data[i];
        float dx = dp.x - qx;
        float dy = dp.y - qy;
        sDists[i] = dx * dx + dy * dy;
        sIdx[i] = i;
    }
    __syncthreads();

    // Build a max-heap of size k in shared memory (single thread; O(k)).
    if (threadIdx.x == 0) {
        heap_build(sDists, sIdx, k);
    }
    __syncthreads();

    // Phase 2: Stream through remaining data and attempt to insert better candidates.
    // Each thread processes indices in a strided fashion for coalesced global loads.
    for (int i = threadIdx.x + k; i < data_count; i += blockDim.x) {
        float2 dp = data[i];
        float dx = dp.x - qx;
        float dy = dp.y - qy;
        float dist = dx * dx + dy * dy;

        // Fast-path rejection using current heap root (worst among top-k).
        // Reading sDists[0] is a broadcast from shared memory.
        if (dist < sDists[0]) {
            // Try to acquire lock and insert if still better than current root.
            // This while loop spins only when a candidate is competitive and the lock is busy.
            while (true) {
                // Re-check before attempting lock to avoid unnecessary contention.
                if (dist >= sDists[0]) break;
                if (atomicCAS(&heapLock, 0, 1) == 0) {
                    // Inside critical section; validate condition again.
                    if (dist < sDists[0]) {
                        sDists[0] = dist;
                        sIdx[0] = i;
                        heap_sift_down(sDists, sIdx, 0, k);
                    }
                    __threadfence_block(); // Ensure shared updates are visible within the block
                    atomicExch(&heapLock, 0);
                    break;
                }
            }
        }
    }
    __syncthreads();

    // Phase 3: Sort the heap to ascending order for output (single thread; O(k log k)).
    if (threadIdx.x == 0) {
        heap_heapsort(sDists, sIdx, k);
    }
    __syncthreads();

    // Phase 4: Write results to global memory. For query qid, results occupy [qid*k .. qid*k + k - 1].
    int out_base = qid * k;
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        // Write index and distance to std::pair fields.
        result[out_base + j].first = sIdx[j];
        result[out_base + j].second = sDists[j];
    }
}

// Host function to launch the kernel.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Choose launch configuration.
    const int threads = KNN_THREADS_PER_BLOCK;
    const int blocks = query_count;

    // Dynamic shared memory size: k floats + k ints
    size_t shared_bytes = static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    // Launch the kernel: one block per query.
    knn2d_block_streaming_kernel<<<blocks, threads, shared_bytes>>>(query, query_count, data, data_count, result, k);

    // Ensure completion before returning (so that results are ready).
    cudaDeviceSynchronize();
}