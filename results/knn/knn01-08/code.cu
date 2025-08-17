#include <cuda_runtime.h>
#include <utility>

// Optimized CUDA k-NN for 2D points.
// Strategy:
// - One block per query point.
// - Multiple warps per block cooperatively traverse the data set with block-stride.
// - Each warp maintains its exact top-k (smallest distances) for its assigned subset as a max-heap in shared memory.
//   Only lane 0 of the warp updates the heap to avoid synchronization complexity.
// - After scanning all data, each warp converts its heap into a sorted ascending array via in-place heapsort.
// - Finally, thread 0 of the block performs a simple k-way merge across the warps' sorted lists to produce the query's global top-k.
// - This approach guarantees correctness because every data point is processed by exactly one warp, and each warp preserves
//   the k smallest values from its subset (k is up to 1024). No additional device memory allocations are required.
//
// Notes:
// - Assumes data_count >= k, k is a power-of-two in [32, 1024].
// - Uses blockDim.x = 512 by default (16 warps), which balances memory bandwidth and shared memory usage on A100/H100.
// - Shared memory per block = warps * k * (sizeof(float)+sizeof(int)) bytes. With 16 warps and k=1024, this is 131072 bytes, which fits on A100/H100.
// - Distances are squared Euclidean distances (no sqrt), as requested.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Max warps per block we use by default.
#ifndef KNN_WARPS_PER_BLOCK
#define KNN_WARPS_PER_BLOCK 16  // 16 warps = 512 threads per block
#endif

// Device utility: swap two elements in parallel arrays (distance and index)
__device__ __forceinline__ void swap_pair(float &ad, int &ai, float &bd, int &bi) {
    float td = ad; ad = bd; bd = td;
    int ti = ai; ai = bi; bi = ti;
}

// Device utility: sift-up operation for a max-heap represented by parallel arrays.
__device__ __forceinline__ void heap_sift_up(float* __restrict__ hd, int* __restrict__ hi, int pos) {
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        if (hd[parent] >= hd[pos]) break;
        swap_pair(hd[parent], hi[parent], hd[pos], hi[pos]);
        pos = parent;
    }
}

// Device utility: sift-down operation for a max-heap represented by parallel arrays.
__device__ __forceinline__ void heap_sift_down(float* __restrict__ hd, int* __restrict__ hi, int size, int pos) {
    while (true) {
        int left = (pos << 1) + 1;
        if (left >= size) break;
        int largest = pos;
        float best = hd[largest];

        float lval = hd[left];
        if (lval > best) {
            largest = left;
            best = lval;
        }
        int right = left + 1;
        if (right < size) {
            float rval = hd[right];
            if (rval > best) {
                largest = right;
                best = rval;
            }
        }
        if (largest == pos) break;
        swap_pair(hd[pos], hi[pos], hd[largest], hi[largest]);
        pos = largest;
    }
}

// In-place heapsort on a max-heap to produce ascending order in the same arrays.
// After calling, hd[0..size-1] will be in ascending order with corresponding hi entries.
__device__ __forceinline__ void heap_sort_ascending(float* __restrict__ hd, int* __restrict__ hi, int size) {
    // Standard heap sort: repeatedly place the current max at the end and re-heapify.
    for (int end = size - 1; end > 0; --end) {
        // Move current maximum (root) to the end.
        swap_pair(hd[0], hi[0], hd[end], hi[end]);
        // Restore heap property for reduced heap [0..end-1].
        heap_sift_down(hd, hi, end, 0);
    }
    // Now array is in ascending order (since we used a max-heap).
}

// Kernel implementing block-per-query k-NN with per-warp exact top-k heaps and final k-way merge.
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k)
{
    const int qid = blockIdx.x;
    if (qid >= query_count) return;

    // Broadcast the query point to the whole block through shared memory.
    __shared__ float2 s_query;
    if (threadIdx.x == 0) {
        s_query = query[qid];
    }
    __syncthreads();
    const float qx = s_query.x;
    const float qy = s_query.y;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    // Dynamic shared memory layout:
    // [warps * k floats for per-warp heap distances][warps * k ints for per-warp heap indices]
    extern __shared__ unsigned char smem[];
    float* s_heap_dists = reinterpret_cast<float*>(smem);
    int*   s_heap_index = reinterpret_cast<int*>(s_heap_dists + (size_t)warps * (size_t)k);

    // Pointers to this warp's heap storage
    float* my_hd = s_heap_dists + (size_t)warp * (size_t)k;
    int*   my_hi = s_heap_index + (size_t)warp * (size_t)k;

    // Lane 0 of each warp maintains the heap.
    int heap_size = 0;

    // Traverse the data in block-stride steps so all warps cover the entire set without gaps.
    // Each iteration, every lane computes one candidate distance; lane 0 collects all 32 via shuffles
    // and updates the warp's heap.
    for (int di = threadIdx.x; di < data_count; di += blockDim.x) {
        // Compute squared distance for this lane's assigned data index, if valid.
        int idx = -1;
        float dist = 0.0f;
        if (di < data_count) {
            float2 p = data[di];
            float dx = p.x - qx;
            float dy = p.y - qy;
            // Use FMA to improve performance/precision: dx*dx + dy*dy
            dist = fmaf(dx, dx, dy * dy);
            idx = di;
        }

        // Ensure all lanes have computed their candidate before shuffling.
        __syncwarp();

        // Lane 0 aggregates and maintains the heap.
        if (lane == 0) {
            // Local references for speed
            float* __restrict__ hd = my_hd;
            int*   __restrict__ hi = my_hi;

            // For each lane's candidate in the warp
            #pragma unroll
            for (int l = 0; l < WARP_SIZE; ++l) {
                int cidx  = __shfl_sync(0xFFFFFFFFu, idx, l);
                if (cidx < 0) continue; // invalid candidate (out of range in tail iteration)
                float cdst = __shfl_sync(0xFFFFFFFFu, dist, l);

                if (heap_size < k) {
                    // Insert into heap (max-heap): push back then sift up.
                    int pos = heap_size++;
                    hd[pos] = cdst;
                    hi[pos] = cidx;
                    heap_sift_up(hd, hi, pos);
                } else {
                    // Prune using current worst (root of max-heap).
                    if (cdst >= hd[0]) continue;
                    // Replace worst and sift down.
                    hd[0] = cdst;
                    hi[0] = cidx;
                    heap_sift_down(hd, hi, heap_size, 0);
                }
            }
        }

        // Synchronize warp before next iteration.
        __syncwarp();
    }

    // Convert each warp's heap to ascending order so we can perform an efficient k-way merge.
    if (lane == 0) {
        // Given data_count >= k, heap_size should be exactly k; if not, heap_sort_ascending still works with heap_size.
        heap_sort_ascending(my_hd, my_hi, heap_size);
    }

    __syncthreads();

    // Final k-way merge across warps' sorted lists, executed by a single thread.
    if (threadIdx.x == 0) {
        const int out_base = qid * k;

        // Positions within each warp's sorted list.
        int pos[KNN_WARPS_PER_BLOCK]; // Upper bound on warps per block; we launch with <= this.
        // warps may be less than KNN_WARPS_PER_BLOCK; initialize and use only first 'warps' entries.
        #pragma unroll
        for (int w = 0; w < KNN_WARPS_PER_BLOCK; ++w) pos[w] = 0;

        // Perform k selections by scanning current heads of all warps (warps is small: <= 16).
        for (int t = 0; t < k; ++t) {
            float best_d = CUDART_INF_F;
            int best_idx = -1;
            int best_w = -1;

            // Scan warps for the smallest current head
            for (int w = 0; w < warps; ++w) {
                int p = pos[w];
                if (p >= heap_size) continue; // Should not occur since heap_size==k, but kept for safety
                float d = s_heap_dists[(size_t)w * (size_t)k + p];
                int   i = s_heap_index[(size_t)w * (size_t)k + p];
                if (d < best_d) {
                    best_d = d;
                    best_idx = i;
                    best_w = w;
                }
            }

            // Write the next nearest neighbor for this query
            result[out_base + t].first  = best_idx;
            result[out_base + t].second = best_d;

            // Advance the chosen warp's pointer
            pos[best_w] += 1;
        }
    }
}

// Host interface: configures and launches the kernel.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Choose threads per block. 512 (16 warps) offers strong memory-level parallelism and fits shared memory
    // even for k=1024 on A100/H100.
    const int warps_per_block = KNN_WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    // Grid: one block per query
    dim3 grid(query_count);
    dim3 block(threads_per_block);

    // Dynamic shared memory size: per-warp heap of k floats + k ints.
    size_t smem_bytes = static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    knn_kernel<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k);
}