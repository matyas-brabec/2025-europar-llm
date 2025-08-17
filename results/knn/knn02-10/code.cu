#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one warp (32 threads) to each query point.
// Each warp maintains a private max-heap (size k) of the current best neighbors
// in shared memory and updates it while iterating over the data points cached
// in shared memory tiles. Warp-aggregated updates minimize contention: for each
// batch of 32 candidates (one per lane), only the best candidates that beat the
// current worst in the heap are inserted, one at a time, using warp-synchronous
// operations. After processing all tiles, lane 0 of each warp heap-sorts its
// heap to produce the k nearest neighbors in ascending order by distance and
// writes the results to global memory.
//
// The code relies only on registers and shared memory; it does not allocate any
// additional device memory. The number of threads per block is a multiple of 32,
// and the number of warps per block is chosen on the host depending on k and the
// available dynamic shared memory. Tiles of data points are cached in shared memory
// and processed iteratively.

namespace {

// Constants
constexpr int WARP_SIZE = 32;

// Compute squared Euclidean distance between two float2 points.
__device__ __forceinline__ float squared_distance(const float2 a, const float2 b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    // fmaf can improve precision slightly and may map to FMA instruction.
    return fmaf(dx, dx, dy * dy);
}

// Swap utility for paired arrays (distances and indices).
__device__ __forceinline__ void swap_pair(float &ad, int &ai, float &bd, int &bi) {
    float td = ad; ad = bd; bd = td;
    int ti = ai; ai = bi; bi = ti;
}

// Sift-down operation for a max-heap stored in (dists, idxs) with length n.
// Compares by distance; larger distances are "greater".
__device__ __forceinline__ void heap_sift_down(float* dists, int* idxs, int i, int n) {
    while (true) {
        int l = (i << 1) + 1;
        if (l >= n) break;
        int r = l + 1;
        int s = l;
        if (r < n && dists[r] > dists[l]) s = r; // pick larger child
        if (dists[i] >= dists[s]) break;
        swap_pair(dists[i], idxs[i], dists[s], idxs[s]);
        i = s;
    }
}

// Build a max-heap from arrays (dists, idxs) of length n.
__device__ __forceinline__ void heap_build(float* dists, int* idxs, int n) {
    for (int i = (n >> 1) - 1; i >= 0; --i) {
        heap_sift_down(dists, idxs, i, n);
    }
}

// Heapsort in-place using max-heap to produce ascending order by distance.
__device__ __forceinline__ void heap_sort_ascending(float* dists, int* idxs, int n) {
    // Assume (dists, idxs) is already a valid max-heap of length n.
    for (int end = n - 1; end > 0; --end) {
        // Move current max to the end.
        swap_pair(dists[0], idxs[0], dists[end], idxs[end]);
        heap_sift_down(dists, idxs, 0, end);
    }
    // Now dists[0..n-1] is in ascending order.
}

// Warp-level argmin: among lanes with is_candidate and value 'val', find the lane
// with the minimal value (ties broken by lower lane id). Lanes not participating
// should pass val = +inf. Returns the winning lane id; broadcasts are performed via shuffles.
__device__ __forceinline__ int warp_argmin_lane(float val, unsigned mask, int lane_id) {
    float best_val = val;
    int best_lane = lane_id;
    // Reduce to lane 0
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float oth_val = __shfl_down_sync(mask, best_val, offset);
        int oth_lane = __shfl_down_sync(mask, best_lane, offset);
        // Prefer strictly smaller value; if equal, prefer lower lane id.
        if (oth_val < best_val || (oth_val == best_val && oth_lane < best_lane)) {
            best_val = oth_val;
            best_lane = oth_lane;
        }
    }
    // Broadcast the winning lane id from lane 0 to all lanes.
    int win_lane = __shfl_sync(mask, best_lane, 0);
    return win_lane;
}

} // anonymous namespace

// Kernel: one warp processes one query. Each warp maintains a private max-heap
// of size k in shared memory: heap_dists[warp_id_in_block][k], heap_idxs[warp_id_in_block][k].
// The block cooperatively loads tiles of data points into shared memory.
__global__ void knn_warp_kernel(const float2* __restrict__ query,
                                int query_count,
                                const float2* __restrict__ data,
                                int data_count,
                                std::pair<int, float>* __restrict__ result,
                                int k,
                                int tile_points)
{
    // Identify warp and lane.
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id_in_block = threadIdx.x >> 5; // 0..warps_per_block-1
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global_id = blockIdx.x * warps_per_block + warp_id_in_block;
    if (warp_global_id >= query_count) return;

    // Dynamic shared memory layout:
    // [0 .. tile_points-1]: float2 tile of data points (shared across warps)
    // next: warps_per_block * k floats for distances (heap storage per warp)
    // next: warps_per_block * k ints for indices (heap storage per warp)
    extern __shared__ __align__(16) unsigned char smem[];
    float2* sh_data = reinterpret_cast<float2*>(smem);
    unsigned char* ptr = smem + static_cast<size_t>(tile_points) * sizeof(float2);

    float* all_heap_dists = reinterpret_cast<float*>(ptr);
    ptr += static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * sizeof(float);

    int* all_heap_idxs = reinterpret_cast<int*>(ptr);
    // Pointers to this warp's private heap arrays
    float* heap_dists = all_heap_dists + static_cast<size_t>(warp_id_in_block) * static_cast<size_t>(k);
    int* heap_idxs = all_heap_idxs + static_cast<size_t>(warp_id_in_block) * static_cast<size_t>(k);

    // Load the query point and broadcast to the warp.
    float2 q = (lane_id == 0) ? query[warp_global_id] : make_float2(0.f, 0.f);
    q.x = __shfl_sync(0xffffffffu, q.x, 0);
    q.y = __shfl_sync(0xffffffffu, q.y, 0);

    // Initialize the heap:
    // Lane 0 computes distances to the first k data points and builds a max-heap.
    if (lane_id == 0) {
        // Seed with the first k data points (data_count >= k by assumption).
        for (int i = 0; i < k; ++i) {
            float2 p = data[i];
            float d = squared_distance(p, q);
            heap_dists[i] = d;
            heap_idxs[i] = i;
        }
        // Build max-heap so that heap_dists[0] is the current worst (largest) distance in the top-k.
        heap_build(heap_dists, heap_idxs, k);
    }
    __syncwarp();

    // Process data in tiles cached in shared memory.
    // All warps in the block cooperate to load the tile; each warp then iterates over the tile.
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        const int count = min(tile_points, data_count - tile_start);

        // Block-wide load of 'count' points into shared memory tile.
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            sh_data[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes the tile in batches of 32 points (one per lane).
        for (int base = 0; base < count; base += WARP_SIZE) {
            const int j = base + lane_id;

            // Compute candidate distance and global index for this lane (or +inf if out of bounds).
            float cand_dist = (j < count) ? squared_distance(sh_data[j], q) : CUDART_INF_F;
            int cand_idx = tile_start + j;

            // Warp-aggregated heap update:
            // Only candidates that beat the current worst (heap root) need to be inserted.
            // Insert them one at a time in ascending order of distance among lanes.
            while (true) {
                // Read current heap worst distance (max).
                const float heap_worst = heap_dists[0];
                unsigned mask = __activemask();
                // Identify lanes with candidates that beat the current worst.
                unsigned active_mask = __ballot_sync(mask, cand_dist < heap_worst);
                if (active_mask == 0u) {
                    break; // No lane has a candidate that beats current worst.
                }

                // Compute the lane with the minimal candidate distance among the active lanes.
                // Lanes not active contribute +inf and never win.
                float val = (cand_dist < heap_worst) ? cand_dist : CUDART_INF_F;
                int winner_lane = warp_argmin_lane(val, mask, lane_id);

                // The winning lane updates the heap root and sifts down.
                if (lane_id == winner_lane) {
                    heap_dists[0] = cand_dist;
                    heap_idxs[0] = cand_idx;
                    heap_sift_down(heap_dists, heap_idxs, 0, k);
                    // Mark this candidate as consumed to avoid re-inserting it.
                    cand_dist = CUDART_INF_F;
                }
                __syncwarp(mask);
                // Loop continues to see if other lanes still beat the new worst.
            }
        }

        __syncthreads();
    }

    // Finalize: lane 0 of the warp sorts the heap ascending and writes out results.
    if (lane_id == 0) {
        // Heapsort to ascending order.
        heap_sort_ascending(heap_dists, heap_idxs, k);

        // Write results for this query in row-major order: result[q * k + j]
        const int out_base = warp_global_id * k;
        for (int j = 0; j < k; ++j) {
            const int idx = heap_idxs[j];
            const float d = heap_dists[j];
            result[out_base + j].first = idx;
            result[out_base + j].second = d;
        }
    }
}

// Host interface.
// Determines a good launch configuration based on k and the device's dynamic shared memory,
// launches the kernel, and ensures the shared memory budget is sufficient.
//
// - Each warp processes one query.
// - The number of warps per block is chosen to keep the per-block shared memory within
//   the opt-in maximum and leave room for a reasonably large tile.
// - The tile size is chosen to maximize use of shared memory while leaving space for the
//   per-warp heaps. It is also limited to a reasonable upper bound to avoid excessive
//   per-block shared memory that could hurt occupancy.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    if (query_count <= 0) return;

    int device = 0;
    cudaGetDevice(&device);

    int max_default_smem = 0;
    int max_optin_smem = 0;
    cudaDeviceGetAttribute(&max_default_smem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Fall back to default if opt-in is unavailable (shouldn't happen on A100/H100).
    if (max_optin_smem <= 0) max_optin_smem = max_default_smem;

    // Choose warps per block. Start from 4 and adjust down if needed to fit shared memory.
    int warps_per_block = 4; // 128 threads per block by default (good balance).
    const int max_warps = 8; // Cap to 8 warps/block (256 threads).
    // Increase warps per block if k is small to improve latency hiding, but stay within smem.
    if (k <= 128) warps_per_block = 8;
    else if (k <= 256) warps_per_block = 6;

    if (warps_per_block > max_warps) warps_per_block = max_warps;
    if (warps_per_block < 1) warps_per_block = 1;

    // Compute per-block shared memory requirements: per-warp heaps + tile.
    auto compute_smem = [&](int wpb, int tile_pts) -> size_t {
        size_t heap_bytes = static_cast<size_t>(wpb) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));
        size_t tile_bytes = static_cast<size_t>(tile_pts) * sizeof(float2);
        return heap_bytes + tile_bytes;
    };

    // Target a reasonably large tile to amortize loads. We'll adjust to fit in shared memory.
    int tile_points = 8192; // 8192 points -> 64 KiB tile
    // Ensure at least some tile size.
    if (tile_points < WARP_SIZE) tile_points = WARP_SIZE;

    // Adjust warps_per_block and tile_points to fit within opt-in shared memory.
    while (true) {
        size_t heap_bytes = static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));
        // Ensure tile points non-negative.
        int max_tile_points_fit = 0;
        if (max_optin_smem > static_cast<int>(heap_bytes)) {
            max_tile_points_fit = (max_optin_smem - static_cast<int>(heap_bytes)) / static_cast<int>(sizeof(float2));
        }
        // Clamp tile_points to what fits, but try to keep it reasonably large.
        int desired_tile_points = tile_points;
        if (desired_tile_points > max_tile_points_fit) desired_tile_points = max_tile_points_fit;
        if (desired_tile_points < WARP_SIZE) {
            // Reduce warps per block if tile cannot fit reasonably.
            if (warps_per_block > 1) {
                warps_per_block >>= 1;
                continue;
            } else {
                desired_tile_points = max_tile_points_fit > 0 ? max_tile_points_fit : WARP_SIZE;
            }
        }

        tile_points = desired_tile_points;
        break;
    }

    // Limit tile_points to avoid excessive per-block SMEM that might reduce occupancy too much,
    // but ensure it's at least a few warps worth.
    const int threads_per_block = warps_per_block * WARP_SIZE;
    // Round tile_points down to a multiple of warp size to simplify batches (optional).
    if (tile_points > WARP_SIZE) {
        tile_points = (tile_points / WARP_SIZE) * WARP_SIZE;
    } else {
        tile_points = WARP_SIZE;
    }

    size_t smem_bytes = compute_smem(warps_per_block, tile_points);

    // Opt-in to the requested dynamic shared memory size.
    cudaFuncSetAttribute(knn_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));

    // Launch configuration: one warp per query.
    const int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    knn_warp_kernel<<<num_blocks, threads_per_block, smem_bytes>>>(
        query, query_count, data, data_count, result, k, tile_points
    );
}