#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include <utility>

// High-performance k-NN for 2D points using a warp-per-query approach, a shared-memory data tile per block,
// and a per-warp max-heap of size k stored in shared memory for intermediate results.
//
// - Each warp (32 threads) handles one query point.
// - The block cooperatively loads a tile of data points into shared memory.
// - All warps in the block compute distances from their query to the cached tile points.
// - For each warp, lane 0 maintains a max-heap (size k) of the current best neighbors (smallest distances).
// - Threads in the warp propose candidates; lane 0 filters and inserts them via heap operations.
// - After processing all tiles, lane 0 performs a heap sort to produce ascending order and writes results.
//
// Design choices:
// - 8 warps per block (256 threads) is a good compromise on A100/H100: high parallelism while leaving shared memory
//   for both the data tile and per-warp heaps (k up to 1024 => ~8KB/warp).
// - The data tile size is chosen at runtime to fit within the available opt-in dynamic shared memory.
// - Distances are squared Euclidean (no sqrt); k is power-of-two in [32, 1024].
// - No device allocations; all temporary storage is in shared memory.
// - Warp-level shuffles and ballots are used for intra-warp coordination, with warp barriers for clarity.

namespace {

// POD type with the same memory layout as std::pair<int, float>.
struct PairIF {
    int   first;
    float second;
};

// Tunable constant: number of warps per block (must be >=1). 8 warps => 256 threads per block.
constexpr int WARPS_PER_BLOCK = 8;
constexpr int WARP_SIZE       = 32;
constexpr unsigned FULL_MASK  = 0xffffffffu;

// Swap helper for heap operations (pairs of distance/index).
__device__ __forceinline__ void swap_pair(float &ad, int &ai, float &bd, int &bi) {
    float td = ad; ad = bd; bd = td;
    int   ti = ai; ai = bi; bi = ti;
}

// Sift-up for max-heap on (distances, indices).
__device__ __forceinline__ void heap_sift_up(float* __restrict__ dists,
                                             int*   __restrict__ idxs,
                                             int pos)
{
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        if (dists[parent] >= dists[pos]) break;
        swap_pair(dists[parent], idxs[parent], dists[pos], idxs[pos]);
        pos = parent;
    }
}

// Sift-down for max-heap on (distances, indices).
__device__ __forceinline__ void heap_sift_down(float* __restrict__ dists,
                                               int*   __restrict__ idxs,
                                               int size,
                                               int pos = 0)
{
    int left = (pos << 1) + 1;
    while (left < size) {
        int right = left + 1;
        int largest = (right < size && dists[right] > dists[left]) ? right : left;
        if (dists[pos] >= dists[largest]) break;
        swap_pair(dists[pos], idxs[pos], dists[largest], idxs[largest]);
        pos = largest;
        left = (pos << 1) + 1;
    }
}

// Push a value/index into the max-heap (size increases by 1).
__device__ __forceinline__ void heap_push_max(float* __restrict__ dists,
                                              int*   __restrict__ idxs,
                                              int& size,
                                              int   capacity,
                                              float val,
                                              int   id)
{
    // Precondition: size < capacity
    int pos = size;
    dists[pos] = val;
    idxs[pos]  = id;
    size++;
    heap_sift_up(dists, idxs, pos);
}

// Replace the root of the max-heap with (val, id) and restore heap property.
__device__ __forceinline__ void heap_replace_root(float* __restrict__ dists,
                                                  int*   __restrict__ idxs,
                                                  int    size,
                                                  float  val,
                                                  int    id)
{
    dists[0] = val;
    idxs[0]  = id;
    heap_sift_down(dists, idxs, size, 0);
}

// Heapsort (ascending) in-place on max-heap arrays of length 'k'.
// After this, dists[0..k-1] is ascending and idxs aligned accordingly.
__device__ __forceinline__ void heap_sort_ascending(float* __restrict__ dists,
                                                    int*   __restrict__ idxs,
                                                    int k)
{
    // Standard in-place heapsort on max-heap: repeatedly move the max to the end,
    // shrink the heap, and sift down to restore max-heap property.
    for (int end = k - 1; end > 0; --end) {
        // Move current max to end
        swap_pair(dists[0], idxs[0], dists[end], idxs[end]);
        // Restore heap property on the reduced heap [0, end)
        heap_sift_down(dists, idxs, end, 0);
    }
    // Result: ascending order (smallest at dists[0], largest at dists[k-1])
}

// CUDA kernel: warp-per-query k-NN for 2D points.
template <int WarpsPerBlock>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           PairIF* __restrict__ result,
                           int k,
                           int tile_capacity)
{
    // Shared memory layout (dynamic):
    // [ float2 tile_points[tile_capacity] ][ float dists[WarpsPerBlock*k] ][ int idxs[WarpsPerBlock*k] ]
    extern __shared__ unsigned char smem_raw[];
    float2* s_points = reinterpret_cast<float2*>(smem_raw);

    size_t points_bytes = static_cast<size_t>(tile_capacity) * sizeof(float2);
    float* s_dists = reinterpret_cast<float*>(smem_raw + points_bytes);
    int*   s_idxs  = reinterpret_cast<int*>(s_dists + static_cast<size_t>(WarpsPerBlock) * k);

    const int tid      = threadIdx.x;
    const int lane     = tid & (WARP_SIZE - 1);
    const int warp     = tid >> 5; // warp index within block
    const int warp_id  = blockIdx.x * WarpsPerBlock + warp; // global warp id ==> query index
    const bool active  = (warp_id < query_count);

    float* warp_dists = s_dists + warp * k;
    int*   warp_idxs  = s_idxs  + warp * k;

    // Load query coordinates to lane 0 and broadcast
    float qx = 0.0f, qy = 0.0f;
    if (lane == 0 && active) {
        float2 q = query[warp_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    int heap_size = 0; // maintained by lane 0 only

    // Iterate over data in tiles cached in shared memory
    for (int tile_base = 0; tile_base < data_count; tile_base += tile_capacity) {
        const int tile_count = min(tile_capacity, data_count - tile_base);

        // Block-cooperative load of data tile into shared memory
        for (int i = tid; i < tile_count; i += blockDim.x) {
            s_points[i] = data[tile_base + i];
        }
        __syncthreads(); // ensure tile loaded before use

        // Each active warp processes the cached tile
        if (active) {
            // Process candidates in groups of 32 (one per lane), then filter+insert into heap
            for (int t = 0; t < tile_count; t += WARP_SIZE) {
                int local_idx = t + lane;
                int global_idx = tile_base + local_idx;

                // Compute squared Euclidean distance for this lane's candidate in the tile
                float cand_dist = FLT_MAX;
                int   cand_idx  = -1;
                if (local_idx < tile_count) {
                    float2 p = s_points[local_idx]; // shared cached point
                    float dx = qx - p.x;
                    float dy = qy - p.y;
                    cand_dist = fmaf(dx, dx, dy * dy); // squared L2 distance
                    cand_idx  = global_idx;
                }

                // Snapshot heap state (lane 0) and broadcast to lanes
                int filled = __shfl_sync(FULL_MASK, heap_size, 0);
                float root_val = FLT_MAX;
                if (lane == 0) {
                    root_val = (heap_size >= k) ? warp_dists[0] : FLT_MAX; // current worst in top-k
                }
                root_val = __shfl_sync(FULL_MASK, root_val, 0);

                // Lanes propose candidates if either heap not full OR cand < current worst
                unsigned propose_mask = __ballot_sync(FULL_MASK,
                                                      (local_idx < tile_count) &&
                                                      ((filled < k) || (cand_dist < root_val)));

                // Serialize insertions to the shared per-warp heap through lane 0
                while (propose_mask) {
                    int src_lane = __ffs(propose_mask) - 1; // next lane to serve
                    float d = __shfl_sync(FULL_MASK, cand_dist, src_lane);
                    int   id = __shfl_sync(FULL_MASK, cand_idx,  src_lane);

                    if (lane == 0) {
                        if (heap_size < k) {
                            heap_push_max(warp_dists, warp_idxs, heap_size, k, d, id);
                        } else if (d < warp_dists[0]) {
                            heap_replace_root(warp_dists, warp_idxs, heap_size, d, id);
                        }
                    }
                    // Clear served lane
                    propose_mask &= (propose_mask - 1);
                }
                __syncwarp(); // warp-level barrier for clarity before next 32-candidate group
            }
        }

        __syncthreads(); // allow tile buffer to be reused/overwritten in next iteration
    }

    // Finalize: for each active warp, sort its heap ascending and write to result
    if (active && lane == 0) {
        // In case data_count < k (shouldn't happen per problem statement), heap_size may be < k.
        // We proceed with whatever is in the heap; if smaller than k, we will only write heap_size entries.
        int out_k = min(heap_size, k);

        // Convert max-heap into ascending order via in-place heapsort
        heap_sort_ascending(warp_dists, warp_idxs, out_k);

        // Write to global memory in ascending order of distance
        PairIF* out = result + static_cast<size_t>(warp_id) * k;
        for (int j = 0; j < out_k; ++j) {
            out[j].first  = warp_idxs[j];
            out[j].second = warp_dists[j];
        }

        // If out_k < k (unlikely here), fill remaining entries with sentinel values
        for (int j = out_k; j < k; ++j) {
            out[j].first  = -1;
            out[j].second = FLT_MAX;
        }
    }
}

} // namespace

// Host entry point. Launches the kernel with appropriate shared memory and tiling configuration.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Constants based on the chosen block configuration
    constexpr int warps_per_block = WARPS_PER_BLOCK;
    constexpr int threads_per_block = warps_per_block * WARP_SIZE;

    // Determine the number of blocks (one warp per query)
    int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;
    if (num_blocks <= 0) return;

    // Compute required shared memory for per-warp heaps
    // Each warp needs k floats + k ints = k * 8 bytes.
    size_t heap_bytes_per_block = static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    // Query device to opt-in to maximum dynamic shared memory (A100/H100 support > 96KB)
    int dev = 0;
    cudaGetDevice(&dev);
    int max_optin_bytes = 0;
    cudaDeviceGetAttribute(&max_optin_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    // Opt-in the kernel to use the maximum available dynamic shared memory
    cudaFuncSetAttribute(knn_kernel<warps_per_block>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_optin_bytes);

    // Choose a tile capacity that fits into the remaining shared memory
    // Total dynamic shared memory per block = heap_bytes_per_block + tile_capacity * sizeof(float2)
    // Ensure at least 1 element in tile (practically, we want more, but this guarantees correctness).
    size_t max_tile_bytes = (heap_bytes_per_block < static_cast<size_t>(max_optin_bytes))
                          ? static_cast<size_t>(max_optin_bytes) - heap_bytes_per_block
                          : size_t(0);

    int tile_capacity = 0;
    if (max_tile_bytes >= sizeof(float2)) {
        // Limit to data_count to avoid over-requesting dynamic memory unnecessarily
        size_t max_tile_elems = max_tile_bytes / sizeof(float2);
        if (max_tile_elems > static_cast<size_t>(data_count)) {
            max_tile_elems = static_cast<size_t>(data_count);
        }
        // For coalesced loads, using a multiple of blockDim.x can be beneficial
        size_t preferred = (max_tile_elems / threads_per_block) * threads_per_block;
        if (preferred == 0) preferred = max_tile_elems; // if data_count < threads_per_block
        tile_capacity = static_cast<int>(preferred);
        if (tile_capacity <= 0) tile_capacity = 1;
    } else {
        // Fallback minimal tiling if shared memory is extremely constrained (shouldn't happen on A100/H100)
        tile_capacity = 1;
    }

    // Compute the actual dynamic shared memory size for the kernel launch
    size_t dynamic_smem_bytes = heap_bytes_per_block + static_cast<size_t>(tile_capacity) * sizeof(float2);

    // Launch the kernel
    PairIF* result_cast = reinterpret_cast<PairIF*>(result);
    knn_kernel<warps_per_block><<<num_blocks, threads_per_block, dynamic_smem_bytes>>>(
        query, query_count,
        data, data_count,
        result_cast, k, tile_capacity
    );

    // Optional: synchronize or error-check can be added by the caller if desired.
}