#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// This implementation assigns exactly one warp (32 threads) to process one query point.
// The algorithm proceeds in batches (tiles) of data points that are cached in shared memory.
// Each warp maintains:
//   - An "intermediate result" of top-k neighbors in registers, distributed across lanes.
//   - A shared-memory per-warp candidate buffer for k candidates.
// Whenever the candidate buffer fills up, the warp merges its candidates with the intermediate result.
// The merge is implemented as a parallel bitonic sort over 2k elements (k old + k new), using the warp cooperatively.
// At the end, any remaining candidates are merged, and the final top-k (sorted by distance asc) is written to the result.
//
// Notes:
//   - k is assumed to be a power of two between 32 and 1024, inclusive.
//   - data_count >= k.
//   - Input arrays (query, data) and the output array (result) are device allocations.
//   - The kernel allocates no additional device memory, only dynamic shared memory.
//   - We set a large dynamic shared memory carveout on the kernel launch to accommodate per-warp buffers and cache tiles.

#ifndef KNN_WARPS_PER_BLOCK
#define KNN_WARPS_PER_BLOCK 4  // 4 warps per block => 128 threads per block; balanced for H100/A100 shared memory limits.
#endif

// Pair structure used in shared memory (distance as key, index as value).
// Packed into 8 bytes for efficient shared-memory bandwidth.
struct __align__(8) PairDF {
    float dist;
    int   idx;
};

// Utility: warp-wide bitonic sort on PairDF array in shared memory.
// The array length n must be a power-of-two. We sort ascending by 'dist'.
// Threads of a single warp cooperatively process the sorting network by striding over indices.
__device__ __forceinline__ void warp_bitonic_sort(PairDF* arr, int n) {
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & 31;

    // Standard bitonic sort network
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each lane processes multiple indices spaced by warpSize
            for (int i = lane; i < n; i += warpSize) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = ((i & size) == 0);
                    PairDF a = arr[i];
                    PairDF b = arr[j];
                    // Compare-exchange with direction 'up'
                    bool swap = (a.dist > b.dist);
                    if (!up) swap = !swap; // for descending, swap when a < b
                    if (swap) {
                        arr[i] = b;
                        arr[j] = a;
                    }
                }
            }
            __syncwarp(full_mask);
        }
    }
}

// Merge step: combines current top-k (in registers) and k candidates (in shared memory) into new top-k.
// Implementation:
//   - Copy top-k from registers into shared union buffer positions [0, k).
//   - The candidates are already in union buffer positions [k, 2k).
//   - Perform a bitonic sort on the 2k union.
//   - Copy the first k items (best k) back into registers, preserving lane-striping order.
template <int WARPS_PER_BLOCK_T>
__device__ __forceinline__ void warp_merge_fullsort(
    PairDF* union_base_all, int k,
    float* topkDist, int* topkIdx, int slots_per_lane)
{
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;

    // This warp's union base: [0, k) holds current top-k, [k, 2k) holds candidates.
    PairDF* union_base = union_base_all + warp_in_block * (2 * k);

    // Write current top-k from registers into shared union buffer [0, k)
    for (int t = 0; t < slots_per_lane; ++t) {
        int pos = lane + t * warpSize; // lane-striped positions
        union_base[pos].dist = topkDist[t];
        union_base[pos].idx  = topkIdx[t];
    }
    __syncwarp(full_mask);

    // Sort all 2k items in ascending order by distance
    warp_bitonic_sort(union_base, 2 * k);

    // Copy the best k back to registers, preserving lane-striped order
    for (int t = 0; t < slots_per_lane; ++t) {
        int pos = lane + t * warpSize;
        topkDist[t] = union_base[pos].dist;
        topkIdx[t]  = union_base[pos].idx;
    }
    __syncwarp(full_mask);
}

// Kernel: one warp per query. Each block has WARPS_PER_BLOCK warps.
// Dynamic shared memory layout (per block):
//   - float2 tile[TilePointsCapacity]
//   - PairDF union_mem[WARPS_PER_BLOCK * (2*k)]  // per-warp union area; second half is the candidate buffer.
// We pass TilePointsCapacity as a parameter chosen at launch time.
template <int WARPS_PER_BLOCK_T>
__global__ void knn2d_warp_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result,
    int k,
    int tile_points_capacity)
{
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp_global = blockIdx.x * WARPS_PER_BLOCK_T + warp_in_block;
    const bool active = (warp_global < query_count);

    // Shared memory layout setup
    extern __shared__ unsigned char smem_raw[];
    // Cache of data points for the block
    float2* sh_tile = reinterpret_cast<float2*>(smem_raw);
    // Union memory for all warps (2k per warp); each warp uses [0,k) for current top-k snapshot and [k,2k) for candidates.
    PairDF* sh_union = reinterpret_cast<PairDF*>(sh_tile + tile_points_capacity);

    // Per-warp pointers into shared memory
    PairDF* warp_union = sh_union + warp_in_block * (2 * k);
    PairDF* warp_cand  = warp_union + k; // candidate buffer region [k, 2k)

    // Per-warp in-register top-k storage (distributed across lanes)
    const int slots_per_lane = k >> 5; // k / 32; k is power of 2 >= 32, so divisible by 32
    float topkDist[slots_per_lane];
    int   topkIdx[slots_per_lane];

    // Initialize top-k to +inf and invalid index
    #pragma unroll
    for (int t = 0; t < slots_per_lane; ++t) {
        topkDist[t] = CUDART_INF_F;
        topkIdx[t]  = -1;
    }

    // Load this warp's query point and broadcast to all lanes
    float qx = 0.0f, qy = 0.0f;
    if (active) {
        if (lane == 0) {
            float2 q = query[warp_global];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(full_mask, qx, 0);
        qy = __shfl_sync(full_mask, qy, 0);
    }

    // Threshold for pruning: current k-th (worst) distance in the intermediate result.
    // Starts at +inf (no pruning), updated after each merge.
    float worstDist = CUDART_INF_F;

    // Per-warp candidate buffer count tracked by lane 0 and broadcast when needed.
    int candCount = 0;

    // Process the data set in tiles cached by the whole block
    for (int base = 0; base < data_count; base += tile_points_capacity) {
        int tileN = data_count - base;
        if (tileN > tile_points_capacity) tileN = tile_points_capacity;

        // Load tile into shared memory cooperatively across the block
        for (int i = threadIdx.x; i < tileN; i += blockDim.x) {
            sh_tile[i] = data[base + i];
        }
        __syncthreads();

        if (active) {
            // Each warp computes distances from its query to all points in the tile.
            // Lanes stride across the tile to evaluate different points in parallel.
            for (int tIdx = lane; tIdx < tileN; tIdx += warpSize) {
                float2 p = sh_tile[tIdx];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = fmaf(dx, dx, dy * dy); // dx*dx + dy*dy

                // Apply pruning based on current worstDist
                bool isCandidate = (dist < worstDist);
                unsigned mask = __ballot_sync(full_mask, isCandidate);
                int pending = __popc(mask);

                // Insert the current batch of candidate lanes into the warp's shared buffer,
                // handling possible buffer-full condition and performing merges as needed.
                // We may need at most two iterations for a 32-lane batch:
                //   - fill remaining space, possibly trigger merge
                //   - insert leftover lanes after merge
                while (pending > 0) {
                    int baseOff = 0;
                    int take = 0;
                    if (lane == 0) {
                        int space = k - candCount;
                        take = (pending < space) ? pending : space;
                        baseOff = candCount;
                        candCount += take;
                    }
                    baseOff = __shfl_sync(full_mask, baseOff, 0);
                    take    = __shfl_sync(full_mask, take,    0);

                    // Compute rank of this lane within the current 'pending' mask
                    unsigned laneMaskLt = (1u << lane) - 1u;
                    int rank = __popc(mask & laneMaskLt);

                    // Write selected candidates into shared buffer
                    if (isCandidate && rank < take) {
                        int slot = baseOff + rank;
                        warp_cand[slot].dist = dist;
                        warp_cand[slot].idx  = base + tIdx;
                    }
                    __syncwarp(full_mask);

                    // If buffer filled, merge with intermediate top-k
                    bool doMerge = false;
                    if (lane == 0) doMerge = (candCount == k);
                    doMerge = __shfl_sync(full_mask, doMerge ? 1 : 0, 0);

                    if (doMerge) {
                        // No need to fill candidates (is exactly k)
                        // Merge: place current top-k into shared [0,k), candidates already in [k,2k)
                        warp_merge_fullsort<WARPS_PER_BLOCK_T>(sh_union, k, topkDist, topkIdx, slots_per_lane);
                        candCount = 0;
                        // Update pruning threshold (k-th distance is in lane 31, last slot)
                        float newWorst = (lane == 31) ? topkDist[slots_per_lane - 1] : 0.0f;
                        worstDist = __shfl_sync(full_mask, newWorst, 31);
                    }

                    // Remove the first 'take' lanes from the pending set
                    bool stillPending = (isCandidate && (rank >= take));
                    mask = __ballot_sync(full_mask, stillPending);
                    pending = __popc(mask);
                    isCandidate = stillPending;
                }
            }
        }

        __syncthreads(); // Make sure all warps are done before overwriting the tile
    }

    if (active) {
        // After the last tile, if the candidate buffer is non-empty, fill the rest with +inf and merge.
        if (candCount > 0) {
            for (int pos = lane + candCount; pos < k; pos += warpSize) {
                warp_cand[pos].dist = CUDART_INF_F;
                warp_cand[pos].idx  = -1;
            }
            __syncwarp(full_mask);
            // Merge current top-k with padded candidates
            warp_merge_fullsort<WARPS_PER_BLOCK_T>(sh_union, k, topkDist, topkIdx, slots_per_lane);
            candCount = 0;
            float newWorst = (lane == 31) ? topkDist[slots_per_lane - 1] : 0.0f;
            worstDist = __shfl_sync(full_mask, newWorst, 31);
        }

        // Write final top-k for this query to global memory in ascending order by distance.
        // We output to result[query_id * k + j], where each j is the lane-striped index over k.
        int outBase = warp_global * k;
        #pragma unroll
        for (int t = 0; t < slots_per_lane; ++t) {
            int j = lane + t * warpSize; // position within [0, k)
            result[outBase + j].first  = topkIdx[t];
            result[outBase + j].second = topkDist[t];
        }
    }
}

// Host API: orchestrates shared memory sizing and kernel launch.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Choose execution configuration
    constexpr int WARPS_PER_BLOCK = KNN_WARPS_PER_BLOCK;
    const int block_threads = WARPS_PER_BLOCK * 32;
    const int total_warps = query_count;
    const int grid_blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Inspect device shared memory capacity and set dynamic shared memory size accordingly.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    // Maximum dynamic shared memory per block we can opt-in for.
    size_t max_optin = (prop.sharedMemPerBlockOptin ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock);

    // Per-block shared memory usage components:
    //   - Per-warp union buffer: 2*k PairDF per warp
    //   - Cached tile of float2 points
    const size_t per_warp_union_bytes = static_cast<size_t>(2) * static_cast<size_t>(k) * sizeof(PairDF);
    const size_t per_block_union_bytes = static_cast<size_t>(WARPS_PER_BLOCK) * per_warp_union_bytes;

    // Reserve space for the tile: use as much as available, but keep some headroom (none strictly needed).
    size_t tile_bytes_available = (max_optin > per_block_union_bytes) ? (max_optin - per_block_union_bytes) : 0;
    // Choose tile size (in number of points) as a multiple of block_threads for coalesced loading.
    size_t tile_points_capacity = tile_bytes_available / sizeof(float2);
    if (tile_points_capacity < static_cast<size_t>(block_threads)) {
        // Fallback: ensure at least one warp's worth; but by problem constraints we should not hit this path.
        tile_points_capacity = block_threads;
    }
    // Align capacity down to a multiple of block_threads
    tile_points_capacity = (tile_points_capacity / block_threads) * block_threads;
    if (tile_points_capacity == 0) {
        tile_points_capacity = block_threads; // safety
    }

    // Compute dynamic shared memory size to pass at launch.
    size_t dynamic_smem_bytes = per_block_union_bytes + tile_points_capacity * sizeof(float2);

    // Increase the kernel's dynamic shared memory limit if necessary.
    cudaFuncSetAttribute(
        knn2d_warp_kernel<KNN_WARPS_PER_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem_bytes));

    // Launch the kernel
    knn2d_warp_kernel<KNN_WARPS_PER_BLOCK>
        <<<grid_blocks, block_threads, dynamic_smem_bytes>>>(
            query, query_count,
            data, data_count,
            result,
            k,
            static_cast<int>(tile_points_capacity));

    // Optional: synchronize or check for errors (omitted to avoid extra host overhead).
    // cudaDeviceSynchronize();
}