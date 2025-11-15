// Optimized k-NN (k-nearest neighbors) for 2D points on modern NVIDIA GPUs (A100/H100).
// 
// Algorithm overview (per query):
//   - One CUDA thread block handles one query point.
//   - The block is partitioned into WARPS_PER_BLOCK warps.
//   - Each warp processes a disjoint subset of all data points.
//   - Each warp maintains a max-heap (size K, in shared memory) of its locally best neighbors.
//   - Distance computation is fully parallel; heap updates are serialized within a warp
//     via a warp leader thread to avoid shared-memory races.
//   - After scanning all data points, each warp's heap is heap-sorted (ascending distances).
//   - The block then performs a multi-way merge of the warp-local sorted lists to produce
//     the final K nearest neighbors, sorted by distance.
//
// Design notes:
//   - K is limited to MAX_K=1024 (as per problem statement).
//   - THREADS_PER_BLOCK is set to 256 (8 warps), trading off occupancy vs shared memory
//     usage. Each warp has its own top-K heap;
//     total shared memory used for heaps is WARPS_PER_BLOCK * MAX_K * (sizeof(float)+sizeof(int)).
//     With WARPS_PER_BLOCK=8 and MAX_K=1024, this is 8 * 1024 * 8 = 64 KiB, which is
//     within per-block shared memory limits for A100/H100.
//   - Distance is squared Euclidean (L2) in 2D, i.e., (dx*dx + dy*dy).
//   - Results are returned sorted in ascending order by distance for each query.
//
// Assumptions:
//   - CUDA compute capability >= 8.0 (A100) / 9.0 (H100).
//   - data_count >= k, k in [32, 1024], power of two (but the code does not rely on power-of-two).
//   - query_count and data_count large enough to benefit from GPU parallelism.
//   - Input pointers (query, data, result) are device pointers allocated via cudaMalloc.
//
// Note: No additional device allocations are performed inside run_knn or the kernels.

#include <cuda_runtime.h>
#include <utility>
#include <float.h>

// Simple POD type mirroring std::pair<int,float> layout.
// We rely on standard-layout guarantees (first, second in order).
struct Pair {
    int   first;
    float second;
};

static_assert(sizeof(Pair) == sizeof(std::pair<int, float>),
              "Pair must match std::pair<int,float> layout");

// Tunable parameters for this implementation.
constexpr int THREADS_PER_BLOCK = 256;     // 8 warps per block on modern GPUs
constexpr int WARP_SIZE          = 32;
constexpr int WARPS_PER_BLOCK    = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int MAX_K              = 1024;   // Maximum supported k

// Device function: insert a candidate (dist, idx) into a warp-local max-heap top-K structure.
//
// Heap layout:
//   - For each warp w, heap_dist[w][0..k-1] and heap_idx[w][0..k-1] form a max-heap
//     on distances (root = largest distance in heap).
//   - heap_size[w] tracks the current number of entries in the heap (<= k).
//
// Insertion logic:
//   - If heap_size < k: insert at end and percolate up (O(log k)).
//   - If heap_size == k: compare to root; if candidate is not better (dist >= root), skip.
//     Otherwise, percolate down replacement of root with candidate (O(log k)).
template<int MAX_K_, int WARPS_PER_BLOCK_>
__device__ __forceinline__
void warp_heap_insert(
    int warp_id,
    float dist,
    int idx,
    float heap_dist[WARPS_PER_BLOCK_][MAX_K_],
    int   heap_idx [WARPS_PER_BLOCK_][MAX_K_],
    int   heap_size[WARPS_PER_BLOCK_],
    int k)
{
    int size = heap_size[warp_id];

    // Case 1: heap not full, just insert and percolate up.
    if (size < k) {
        int i = size;
        heap_size[warp_id] = size + 1;

        // Percolate up for max-heap on distance.
        while (i > 0) {
            int parent = (i - 1) >> 1;
            float parent_d = heap_dist[warp_id][parent];
            if (dist <= parent_d) break;
            heap_dist[warp_id][i] = parent_d;
            heap_idx [warp_id][i] = heap_idx[warp_id][parent];
            i = parent;
        }
        heap_dist[warp_id][i] = dist;
        heap_idx [warp_id][i] = idx;
    } else {
        // Case 2: heap full, check if candidate is better than worst (root).
        float root_d = heap_dist[warp_id][0];
        if (dist >= root_d) {
            // Not better than current worst neighbor; ignore.
            return;
        }

        // Replace root and percolate down.
        int i = 0;
        int left = 1;

        while (left < k) {
            int right  = left + 1;
            int largest = left;
            float largest_d = heap_dist[warp_id][left];

            if (right < k) {
                float right_d = heap_dist[warp_id][right];
                if (right_d > largest_d) {
                    largest   = right;
                    largest_d = right_d;
                }
            }

            // If candidate is larger or equal than the largest child, we found position.
            if (dist >= largest_d) break;

            // Move child up.
            heap_dist[warp_id][i] = largest_d;
            heap_idx [warp_id][i] = heap_idx[warp_id][largest];

            i    = largest;
            left = (i << 1) + 1;
        }

        heap_dist[warp_id][i] = dist;
        heap_idx [warp_id][i] = idx;
    }
}

// Device function: in-place heap sort of a single warp's max-heap into ascending order.
//
// After warp_heap_insert, the heap for warp_id is a max-heap within indices [0..size-1].
// Standard heapsort on a max-heap produces ascending order.
//
// Complexity: O(size log size), called once per warp per query (size <= k <= 1024).
template<int MAX_K_, int WARPS_PER_BLOCK_>
__device__ __forceinline__
void warp_heap_sort(
    int warp_id,
    float heap_dist[WARPS_PER_BLOCK_][MAX_K_],
    int   heap_idx [WARPS_PER_BLOCK_][MAX_K_],
    int size)
{
    if (size <= 1) return;

    // Heapsort: repeatedly move max element (root) to the end and re-heapify.
    for (int i = size - 1; i > 0; --i) {
        // Swap root with i.
        float td = heap_dist[warp_id][0];
        heap_dist[warp_id][0] = heap_dist[warp_id][i];
        heap_dist[warp_id][i] = td;

        int ti = heap_idx[warp_id][0];
        heap_idx[warp_id][0] = heap_idx[warp_id][i];
        heap_idx[warp_id][i] = ti;

        // Re-heapify subtree [0..i-1].
        int parent = 0;
        while (true) {
            int left = (parent << 1) + 1;
            if (left >= i) break;

            int right = left + 1;
            int largest = parent;
            float largest_d = heap_dist[warp_id][parent];
            float left_d = heap_dist[warp_id][left];

            if (left_d > largest_d) {
                largest   = left;
                largest_d = left_d;
            }
            if (right < i) {
                float right_d = heap_dist[warp_id][right];
                if (right_d > largest_d) {
                    largest   = right;
                    largest_d = right_d;
                }
            }
            if (largest == parent) break;

            float td2 = heap_dist[warp_id][parent];
            heap_dist[warp_id][parent] = heap_dist[warp_id][largest];
            heap_dist[warp_id][largest] = td2;

            int ti2 = heap_idx[warp_id][parent];
            heap_idx[warp_id][parent] = heap_idx[warp_id][largest];
            heap_idx[warp_id][largest] = ti2;

            parent = largest;
        }
    }
    // After this, heap_dist[warp_id][0..size-1] is in ascending order.
}

// Main k-NN kernel: one block per query.
template<int MAX_K_, int THREADS_PER_BLOCK_>
__global__ void knn_kernel_2d(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    Pair* __restrict__ result,
    int k)
{
    constexpr int WARP_SIZE_       = 32;
    constexpr int WARPS_PER_BLOCK_ = THREADS_PER_BLOCK_ / WARP_SIZE_;

    // Shared memory for warp-local heaps and query point.
    __shared__ float  warp_heap_dist[WARPS_PER_BLOCK_][MAX_K_];
    __shared__ int    warp_heap_idx [WARPS_PER_BLOCK_][MAX_K_];
    __shared__ int    warp_heap_size[WARPS_PER_BLOCK_];
    __shared__ float2 s_query;

    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE_;
    const int lane     = tid % WARP_SIZE_;
    const int query_id = blockIdx.x;

    if (query_id >= query_count || k <= 0) {
        return;
    }

    // Clamp k to MAX_K_ for safety (problem guarantees k <= MAX_K_).
    if (k > MAX_K_) k = MAX_K_;

    // Initialize heap sizes (one per warp) and load query into shared memory.
    if (tid < WARPS_PER_BLOCK_) {
        warp_heap_size[tid] = 0;
    }
    if (tid == 0) {
        s_query = query[query_id];
    }
    __syncthreads();

    const float qx = s_query.x;
    const float qy = s_query.y;

    // Each thread processes a subset of data points in a block-stride loop.
    // For each data point, the warp cooperatively considers inserting it into
    // the warp-local top-K heap.
    const int stride = THREADS_PER_BLOCK_;
    for (int data_idx = tid; data_idx < data_count; data_idx += stride) {
        // Load data point and compute squared distance to query.
        float2 p = data[data_idx];
        float dx = p.x - qx;
        float dy = p.y - qy;
        float dist = dx * dx + dy * dy;

        // Active lanes in this warp for the current loop iteration.
        unsigned int active_mask = __activemask();

        // Choose a warp-leader (lowest active lane) to perform heap updates.
        int leader_lane = __ffs(active_mask) - 1;

        // Leader reads heap size and current threshold (worst distance in heap).
        int   heap_size_val = 0;
        float threshold     = FLT_MAX;

        if (lane == leader_lane) {
            heap_size_val = warp_heap_size[warp_id];
            // If heap is not full, threshold is effectively +inf: any candidate may be inserted.
            if (heap_size_val >= k) {
                threshold = warp_heap_dist[warp_id][0];  // root contains current worst (largest) distance
            }
        }

        // Broadcast heap size and threshold to all active lanes in the warp.
        heap_size_val = __shfl_sync(active_mask, heap_size_val, leader_lane);
        threshold     = __shfl_sync(active_mask, threshold,     leader_lane);

        // Identify lanes with candidates possibly better than the current threshold.
        unsigned int candidate_mask = __ballot_sync(active_mask, dist < threshold || heap_size_val < k);

        // Process candidates one by one in order of lane ID, using the warp leader to update the heap.
        while (candidate_mask) {
            int cand_lane = __ffs(candidate_mask) - 1;

            // Broadcast candidate's distance and index from cand_lane to leader_lane.
            float cand_dist = __shfl_sync(active_mask, dist,     cand_lane);
            int   cand_idx  = __shfl_sync(active_mask, data_idx, cand_lane);

            if (lane == leader_lane) {
                // Leader inserts candidate into warp-local heap.
                warp_heap_insert<MAX_K_, WARPS_PER_BLOCK_>(
                    warp_id, cand_dist, cand_idx,
                    warp_heap_dist, warp_heap_idx, warp_heap_size, k);
            }

            // After insertion, update heap size and threshold for subsequent candidates.
            if (lane == leader_lane) {
                heap_size_val = warp_heap_size[warp_id];
                if (heap_size_val >= k) {
                    threshold = warp_heap_dist[warp_id][0];
                } else {
                    threshold = FLT_MAX;
                }
            }
            heap_size_val = __shfl_sync(active_mask, heap_size_val, leader_lane);
            threshold     = __shfl_sync(active_mask, threshold,     leader_lane);

            // Remove this lane from candidate mask and continue with remaining candidates.
            candidate_mask &= candidate_mask - 1;
        }
    }

    __syncthreads(); // Ensure all heaps are built before sorting.

    // Each warp-leader heap-sorts its own heap in ascending order.
    if (warp_id < WARPS_PER_BLOCK_ && lane == 0) {
        int size = warp_heap_size[warp_id];
        warp_heap_sort<MAX_K_, WARPS_PER_BLOCK_>(warp_id, warp_heap_dist, warp_heap_idx, size);
    }

    __syncthreads(); // Ensure warp-local lists are sorted before merging.

    // Single thread (tid == 0) performs a multi-way merge of all warp-local sorted lists
    // to produce the final top-K neighbors for this query, sorted by ascending distance.
    if (tid == 0) {
        int sizes[WARPS_PER_BLOCK_];
        int pos  [WARPS_PER_BLOCK_];

        // Initialize per-warp sizes and positions.
        for (int w = 0; w < WARPS_PER_BLOCK_; ++w) {
            sizes[w] = warp_heap_size[w];
            pos[w]   = 0;
        }

        const int base_out = query_id * k;

        for (int out = 0; out < k; ++out) {
            float best_d  = FLT_MAX;
            int   best_w  = -1;
            int   best_idx = -1;

            // Find the warp with the smallest next candidate.
            for (int w = 0; w < WARPS_PER_BLOCK_; ++w) {
                int p = pos[w];
                int s = sizes[w];
                if (p >= s) continue;  // this warp exhausted

                float d = warp_heap_dist[w][p];
                if (d < best_d) {
                    best_d   = d;
                    best_w   = w;
                    best_idx = warp_heap_idx[w][p];
                }
            }

            // Safety guard: should not happen when data_count >= k.
            if (best_w < 0) {
                break;
            }

            // Store result for this query and neighbor rank.
            result[base_out + out].first  = best_idx;
            result[base_out + out].second = best_d;

            // Advance position in the warp from which we took the best candidate.
            pos[best_w]++;
        }
    }
}

// Host interface as specified.
void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Reinterpret result pointer as Pair* (layout-compatible with std::pair<int,float>).
    Pair *result_pairs = reinterpret_cast<Pair*>(result);

    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid(query_count, 1, 1);

    knn_kernel_2d<MAX_K, THREADS_PER_BLOCK>
        <<<grid, block>>>(
            query,
            query_count,
            data,
            data_count,
            result_pairs,
            k);

    // No cudaDeviceSynchronize() here to allow asynchronous usage; caller can synchronize if needed.
}