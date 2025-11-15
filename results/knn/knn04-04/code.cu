// CUDA implementation of k-nearest neighbors (k-NN) for 2D points.
// -----------------------------------------------------------------------------
// Design overview (see inline comments for details):
//
// * One warp (32 threads) is assigned to a single query point.
// * A thread block contains WARPS_PER_BLOCK warps and therefore processes
//   WARPS_PER_BLOCK queries concurrently.
// * The entire block cooperatively loads a tile of data points into shared
//   memory; every warp then reuses this tile to compute distances to its query.
// * Each warp maintains a private max-heap of size k in shared memory which
//   stores the current k nearest neighbors (index + squared distance).
// * Distances to data points are computed in batches of 32 per warp. Each lane
//   computes one candidate; candidates are then broadcast via warp shuffles and
//   integrated into the heap by lane 0.
// * The heap provides O(log k) insertion / replacement, yielding
//       O(data_count * log k)
//   complexity per query, which is suitable for k up to 1024.
// * After all data points are processed, the heap is heap-sorted in place to
//   obtain neighbors ordered from nearest to farthest, and the result is written
//   to global memory.
// * No additional device memory is allocated; only dynamic shared memory is
//   used (per-block).
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Simple POD type that mirrors std::pair<int, float> layout.
//
// We use this type inside device code instead of std::pair because
// <utility>'s std::pair is not guaranteed to be __device__-compatible on all
// toolchains. On the host side we reinterpret the user-provided
// std::pair<int,float>* pointer to KnnResult*.
//
// The static_assert ensures that our layout matches std::pair<int,float>.
struct KnnResult {
    int   index;    // index of the neighbor in the data array
    float distance; // squared Euclidean distance
};

static_assert(sizeof(KnnResult) == sizeof(std::pair<int, float>),
              "KnnResult must have the same size as std::pair<int,float>");

// -----------------------------------------------------------------------------
// Tuning parameters for the kernel.
// -----------------------------------------------------------------------------

// We fix the warp size at 32, as required by the problem statement.
constexpr int WARP_SIZE        = 32;
constexpr int WARPS_PER_BLOCK  = 4;                 // 4 warps => 128 threads/block
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

// Number of data points per shared-memory tile.
//
// TILE_SIZE * sizeof(float2) + WARPS_PER_BLOCK * k * sizeof(Neighbor)
// must fit into the GPU's per-block shared memory. For TILE_SIZE = 2048 and
// WARPS_PER_BLOCK = 4 and k <= 1024 we use at most:
//
//   tile:             2048 * 8 bytes = 16 KB
//   4 warps, k=1024:  4 * 1024 * 8 bytes = 32 KB
//   total:            48 KB
//
// which is within the 48 KB default per-block limit and well below the
// 100+ KB limits of H100/A100 when configured.
constexpr int TILE_SIZE        = 2048;

static_assert(WARP_SIZE == 32, "This implementation assumes 32-thread warps.");
static_assert(TILE_SIZE % WARP_SIZE == 0,
              "TILE_SIZE must be a multiple of WARP_SIZE for warp tiling.");

// -----------------------------------------------------------------------------
// Per-warp neighbor structure stored in shared memory.
// -----------------------------------------------------------------------------

// Single neighbor entry: squared distance and index of the data point.
//
// This struct is used for the per-warp heap that stores the current k nearest
// candidates. The heap is a max-heap by "dist", so heap[0] always holds the
// farthest neighbor among the current k best; this is the rejection threshold
// for new candidates.
struct Neighbor {
    float dist;
    int   idx;
};

// Small utility to swap two Neighbor entries (used in heap operations).
__device__ __forceinline__ void swap_neighbors(Neighbor &a, Neighbor &b) {
    Neighbor tmp = a;
    a = b;
    b = tmp;
}

// Insert/replace a candidate in a per-warp max-heap of size up to max_size.
//
// * heap      : shared-memory heap array (size up to max_size)
// * heap_size : current number of elements in the heap; updated in place
// * max_size  : capacity (k)
// * dist, idx : candidate distance and index
// * current_max : approximate current maximum distance in the heap.
//
// Behavior:
//   - If heap_size < max_size: pushes the new element and percolates it upward
//     to maintain the max-heap property. When heap_size reaches max_size,
//     current_max is initialized to heap[0].dist.
//   - If heap_size == max_size and dist < current_max: the candidate is better
//     than the current worst neighbor; it replaces the root and percolates
//     downward. current_max is updated to the new heap[0].dist.
//   - Otherwise: the candidate is discarded.
__device__ __forceinline__ void heap_insert(Neighbor *heap,
                                            int      &heap_size,
                                            int       max_size,
                                            float     dist,
                                            int       idx,
                                            float    &current_max) {
    // Case 1: heap not full yet => push new element and bubble up.
    if (heap_size < max_size) {
        int pos = heap_size;
        heap[pos].dist = dist;
        heap[pos].idx  = idx;

        // Bubble up to maintain max-heap property.
        while (pos > 0) {
            int parent = (pos - 1) >> 1;
            if (heap[parent].dist >= heap[pos].dist)
                break;
            swap_neighbors(heap[parent], heap[pos]);
            pos = parent;
        }

        ++heap_size;

        // Initialize current_max when heap becomes full.
        if (heap_size == max_size) {
            current_max = heap[0].dist;
        }
    }
    // Case 2: heap full; candidate is better than current worst.
    else if (max_size > 0 && dist < current_max) {
        // Replace root (worst neighbor) with the new candidate.
        heap[0].dist = dist;
        heap[0].idx  = idx;

        // Percolate down to restore max-heap property.
        int parent = 0;
        while (true) {
            int left  = (parent << 1) + 1;
            if (left >= heap_size)
                break;
            int right = left + 1;
            int largest = left;
            if (right < heap_size && heap[right].dist > heap[left].dist)
                largest = right;
            if (heap[parent].dist >= heap[largest].dist)
                break;
            swap_neighbors(heap[parent], heap[largest]);
            parent = largest;
        }

        // Update threshold.
        current_max = heap[0].dist;
    }
    // Else: candidate is worse than current_max and is discarded.
}

// Convert a max-heap of Neighbor entries into an array sorted in ascending
// order by distance, in-place, using heapsort.
//
// After completion, heap[0] will be the closest neighbor and heap[heap_size-1]
// the farthest. This is run once per query (per warp) after all candidates
// have been processed.
__device__ __forceinline__ void heap_sort_max_to_ascending(Neighbor *heap,
                                                           int       heap_size) {
    // Standard in-place heapsort on a max-heap.
    for (int end = heap_size - 1; end > 0; --end) {
        // Move current maximum (root) to the end.
        swap_neighbors(heap[0], heap[end]);

        // Restore max-heap property on the reduced heap [0, end).
        int parent = 0;
        while (true) {
            int left = (parent << 1) + 1;
            if (left >= end)
                break;
            int right = left + 1;
            int largest = left;
            if (right < end && heap[right].dist > heap[left].dist)
                largest = right;
            if (heap[parent].dist >= heap[largest].dist)
                break;
            swap_neighbors(heap[parent], heap[largest]);
            parent = largest;
        }
    }
}

// -----------------------------------------------------------------------------
// CUDA kernel.
// -----------------------------------------------------------------------------

// Each warp processes one query:
//
// * The block cooperatively loads tiles of data points into shared memory.
// * Every warp reads the tile from shared memory and computes distances to its
//   own query point.
// * Candidate neighbors are integrated into a per-warp max-heap.
//
// The kernel expects:
//   - 'queries'  : array of 'query_count' float2 points
//   - 'data'     : array of 'data_count' float2 points
//   - 'k'        : number of nearest neighbors (power of two between 32 and 1024)
//   - 'results'  : array of size query_count * k, storing k results per query.
__global__ void knn_kernel(const float2 * __restrict__ queries,
                           int                   query_count,
                           const float2 * __restrict__ data,
                           int                   data_count,
                           int                   k,
                           KnnResult * __restrict__ results) {
    // Dynamic shared memory layout:
    //   [0 .. TILE_SIZE-1]              : float2 tile_points
    //   [TILE_SIZE .. end]              : Neighbor heaps, one heap per warp
    //
    // Size is configured at launch:
    //   TILE_SIZE * sizeof(float2)
    //   + WARPS_PER_BLOCK * k * sizeof(Neighbor)
    extern __shared__ unsigned char shared_raw[];

    float2   *tile_points = reinterpret_cast<float2 *>(shared_raw);
    Neighbor *knn_heaps   = reinterpret_cast<Neighbor *>(tile_points + TILE_SIZE);

    const int lane_id          = threadIdx.x & (WARP_SIZE - 1);       // 0..31
    const int warp_id_in_block = threadIdx.x >> 5;                    // 0..WARPS_PER_BLOCK-1
    const int warp_global_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    const bool warp_has_query  = (warp_global_id < query_count);

    // Pointer to this warp's private heap region in shared memory.
    Neighbor *heap = knn_heaps + warp_id_in_block * k;

    // Per-warp heap state (live only in lane 0, but stored in all lanes'
    // registers; only lane 0 mutates these variables).
    int   heap_size   = 0;
    float current_max = FLT_MAX;  // approximate upper bound on the worst distance

    // Load query point and broadcast its coordinates within the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_has_query && lane_id == 0) {
        float2 q = queries[warp_global_id];
        qx = q.x;
        qy = q.y;
    }
    const unsigned int full_mask = 0xffffffffu;
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);

    // Process the 'data' array in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_size = TILE_SIZE;
        if (tile_start + tile_size > data_count) {
            tile_size = data_count - tile_start;
        }

        // Block-wide, coalesced load of the current tile into shared memory.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile_points[i] = data[tile_start + i];
        }

        // Ensure the tile is fully loaded before any warp uses it.
        __syncthreads();

        // Each warp now processes this tile for its own query.
        if (warp_has_query) {
            // Iterate over the tile in warp-sized chunks. In each iteration,
            // every lane computes the distance to one data point, yielding
            // up to 32 candidates at once.
            for (int base = 0; base < tile_size; base += WARP_SIZE) {
                const int data_idx_in_tile = base + lane_id;
                const int global_data_idx  = tile_start + data_idx_in_tile;

                // Snapshot of heap state for this batch of candidates.
                const int   heap_size_bcast = __shfl_sync(full_mask, heap_size, 0);
                const float threshold       = __shfl_sync(full_mask, current_max, 0);

                float cand_dist = FLT_MAX;
                int   cand_idx  = -1;

                // Each lane computes distance to one candidate point (if in range).
                if (data_idx_in_tile < tile_size) {
                    float2 p = tile_points[data_idx_in_tile];
                    const float dx = p.x - qx;
                    const float dy = p.y - qy;
                    cand_dist = dx * dx + dy * dy;

                    // Prefilter using heap status:
                    //
                    // * If heap not full yet (heap_size_bcast < k), we accept all
                    //   candidates from this batch (we need k neighbors anyway).
                    // * Once heap is full, we accept only those with dist < threshold.
                    //   Note that 'threshold' is a stale upper bound on heap[0].dist
                    //   from the previous batch; using it can only admit extra
                    //   candidates, never reject true neighbors.
                    if (heap_size_bcast < k || cand_dist < threshold) {
                        cand_idx = global_data_idx;
                    } else {
                        cand_idx = -1;
                    }
                }

                // Integrate this batch of up to 32 candidates into the heap.
                //
                // Each candidate (cand_dist, cand_idx) is broadcast from its
                // owning lane 'src' to all lanes using shuffles; only lane 0
                // performs the heap update, but all lanes cooperate in producing
                // the inputs concurrently.
                for (int src = 0; src < WARP_SIZE; ++src) {
                    const float d   = __shfl_sync(full_mask, cand_dist, src);
                    const int   idx = __shfl_sync(full_mask, cand_idx,  src);

                    if (lane_id == 0 && idx >= 0) {
                        heap_insert(heap, heap_size, k, d, idx, current_max);
                    }
                }
            }
        }

        // Ensure all warps are done using this tile before loading the next one.
        __syncthreads();
    }

    // Finalize results for this warp's query.
    if (warp_has_query) {
        // Sort heap in ascending distance order (closest first).
        if (lane_id == 0 && heap_size > 0) {
            heap_sort_max_to_ascending(heap, heap_size);
        }

        // Ensure sorted heap is visible to all lanes in the warp.
        __syncwarp(full_mask);

        // Write k nearest neighbors to global memory; each lane writes a strided
        // subset for coalesced stores.
        KnnResult *out = results + warp_global_id * k;
        for (int j = lane_id; j < heap_size; j += WARP_SIZE) {
            out[j].index    = heap[j].idx;
            out[j].distance = heap[j].dist;
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface: run_knn
// -----------------------------------------------------------------------------

// Public interface as requested:
//
//   void run_knn(const float2 *query, int query_count,
//                const float2 *data,  int data_count,
//                std::pair<int, float> *result, int k);
//
// The pointers 'query', 'data', and 'result' are assumed to point to device
// memory allocated with cudaMalloc. No additional device memory is allocated
// inside this function; only per-block dynamic shared memory is used.
//
// NOTE:
//   The device kernel uses a custom KnnResult struct instead of std::pair, but
//   we reinterpret_cast the 'result' pointer here and rely on the fact that
//   KnnResult has the same size and binary layout as std::pair<int,float>.
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Reinterpret result pointer to device-friendly struct type.
    KnnResult *result_raw =
        reinterpret_cast<KnnResult *>(result);

    // One warp handles one query, so we need enough blocks to cover all queries.
    const dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    const dim3 grid_dim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                        1, 1);

    // Dynamic shared memory size:
    //
    //   TILE_SIZE * sizeof(float2)          : tile of data points
    // + WARPS_PER_BLOCK * k * sizeof(Neighbor) : per-warp heaps
    //
    // This respects the constraint k <= 1024 for the chosen TILE_SIZE and
    // WARPS_PER_BLOCK.
    const size_t shared_bytes =
        static_cast<size_t>(TILE_SIZE) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) *
        static_cast<size_t>(k) * sizeof(Neighbor);

    knn_kernel<<<grid_dim, block_dim, shared_bytes>>>(
        query, query_count, data, data_count, k, result_raw
    );

    // No cudaDeviceSynchronize() or error checking is performed here to keep
    // this routine lightweight. The caller may synchronize and check errors
    // as appropriate for their application.
}