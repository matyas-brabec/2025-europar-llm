#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <utility>

// -----------------------------------------------------------------------------
// k-NN for 2D points using one warp (32 threads) per query.
// - Each block has WARPS_PER_BLOCK warps.
// - All warps in a block cooperatively cache a tile of data points in shared
//   memory, then each warp computes distances from its own query point.
// - Each warp maintains:
//     * A private intermediate top-k neighbor list (indices + squared distances).
//       This list is kept sorted in ascending distance and stored per-warp in
//       shared memory plus per-thread register fragments.
//     * A per-warp candidate buffer in shared memory with capacity k.
//       Only points closer than the current k-th neighbor are inserted.
// - When the candidate buffer fills (or at the end), the warp merges its
//   candidates with its intermediate top-k using a warp-wide radix sort
//   (CUB::WarpRadixSort), which utilizes warp shuffles and shared memory.
// - After processing all data, each warp writes out its sorted k nearest
//   neighbors for its query to the result array.
//
// Assumptions:
// - data_count >= k
// - k is a power of two in [32, 1024]
// - query, data, result are all device pointers allocated with cudaMalloc.
// - Target hardware: modern NVIDIA data-center GPUs (e.g., A100, H100).
// -----------------------------------------------------------------------------

// Tunable constants (chosen for good performance on A100/H100).
static constexpr int WARP_SIZE          = 32;
static constexpr int WARPS_PER_BLOCK    = 4;       // 4 warps => 128 threads per block
static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int TILE_POINTS        = 1024;    // data tile size cached per block
static constexpr int MAX_K              = 1024;    // maximum supported k (power of two)
static constexpr int MAX_ITEMS_PER_THREAD = (2 * MAX_K) / WARP_SIZE; // for union size up to 2*MAX_K

using WarpSortT = cub::WarpRadixSort<float, WARP_SIZE, MAX_ITEMS_PER_THREAD, int>;

// -----------------------------------------------------------------------------
// Merge candidate buffer with current top-k for one warp using warp-wide
// radix sort.
//
// The union of current top-k (size k) and candidate buffer (size candCount)
// has size up to 2k <= 2*MAX_K. We map that union into per-thread register
// arrays (sortKeys/sortVals) of size MAX_ITEMS_PER_THREAD such that the whole
// warp covers up to 2*MAX_K elements. We pad remaining entries with +INF so
// they drift to the end of the sorted sequence.
//
// After sorting with WarpRadixSort, we write the globally smallest k elements
// back to the per-warp "best" arrays in shared memory, then reload the
// per-thread register fragments and update the warp's k-th (worst) distance.
// -----------------------------------------------------------------------------
__device__ __forceinline__
void merge_buffer_warp(
    int                    warp_id_in_block,
    int                    lane_id,
    int                    k,
    int                    items_per_thread,
    float*                 warp_best_dist,   // size k, per-warp slice in shared memory
    int*                   warp_best_idx,    // size k, per-warp slice in shared memory
    float*                 warp_cand_dist,   // size k, per-warp slice in shared memory
    int*                   warp_cand_idx,    // size k, per-warp slice in shared memory
    int                    cand_count,       // number of valid candidates in buffer
    float&                 worst_dist,       // in/out: updated k-th distance
    WarpSortT::TempStorage warp_sort_temp[],// shared temp storage array [WARPS_PER_BLOCK]
    float*                 best_local_dist,  // per-thread register fragment [items_per_thread]
    int*                   best_local_idx    // per-thread register fragment [items_per_thread]
) {
    const unsigned FULL_MASK = 0xffffffffu;

    // 1. Store current top-k (from registers) into per-warp shared arrays.
    //    Global index in [0, k) for entry j of lane lane_id is:
    //      g = lane_id * items_per_thread + j
    for (int j = 0; j < items_per_thread; ++j) {
        int g = lane_id * items_per_thread + j;
        warp_best_dist[g] = best_local_dist[j];
        warp_best_idx[g]  = best_local_idx[j];
    }
    __syncwarp(FULL_MASK);

    // 2. Prepare union of best[k] and candidates[cand_count] into per-thread
    //    register arrays for warp-wide radix sort.
    float sort_keys[MAX_ITEMS_PER_THREAD];
    int   sort_vals[MAX_ITEMS_PER_THREAD];

    const int thread_base = lane_id * MAX_ITEMS_PER_THREAD;
    const int union_size  = k + cand_count; // <= 2*k <= 2*MAX_K

    for (int i = 0; i < MAX_ITEMS_PER_THREAD; ++i) {
        int u = thread_base + i;  // global union index
        if (u < k) {
            // From current best list.
            sort_keys[i] = warp_best_dist[u];
            sort_vals[i] = warp_best_idx[u];
        } else {
            int c = u - k;
            if (c < cand_count) {
                // From candidate buffer.
                sort_keys[i] = warp_cand_dist[c];
                sort_vals[i] = warp_cand_idx[c];
            } else if (u < 2 * k && u < union_size) {
                // Should not happen (union_size <= k + cand_count <= 2k),
                // but keep for safety: if we ever had less than 2k entries
                // but u < union_size, fill with +INF anyway.
                sort_keys[i] = CUDART_INF_F;
                sort_vals[i] = -1;
            } else {
                // Padding with +INF so these entries end at the end of the sort.
                sort_keys[i] = CUDART_INF_F;
                sort_vals[i] = -1;
            }
        }
    }

    // 3. Warp-wide radix sort of union (in registers) by distance.
    WarpSortT(warp_sort_temp[warp_id_in_block]).SortPairs(sort_keys, sort_vals);

    // 4. Write globally smallest k elements back to per-warp best arrays in shared memory.
    for (int i = 0; i < MAX_ITEMS_PER_THREAD; ++i) {
        int u = thread_base + i;
        if (u < k) {
            warp_best_dist[u] = sort_keys[i];
            warp_best_idx[u]  = sort_vals[i];
        }
    }
    __syncwarp(FULL_MASK);

    // 5. Reload per-thread register fragments from shared memory.
    for (int j = 0; j < items_per_thread; ++j) {
        int g = lane_id * items_per_thread + j;
        best_local_dist[j] = warp_best_dist[g];
        best_local_idx[j]  = warp_best_idx[g];
    }

    // 6. Update k-th (worst) distance for the warp and broadcast to all lanes.
    float w = 0.0f;
    if (lane_id == 0) {
        w = warp_best_dist[k - 1];
    }
    w = __shfl_sync(FULL_MASK, w, 0);
    worst_dist = w;
}

// -----------------------------------------------------------------------------
// Main CUDA kernel: one warp processes one query.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int                        query_count,
    const float2* __restrict__ data,
    int                        data_count,
    std::pair<int, float>* __restrict__ result,
    int                        k
) {
    const unsigned FULL_MASK = 0xffffffffu;

    // Warp and lane identification.
    const int tid              = threadIdx.x;
    const int warp_id_in_block = tid / WARP_SIZE;
    const int lane_id          = tid % WARP_SIZE;
    const int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    const bool valid_warp      = (global_warp_id < query_count);

    // Shared memory layout (dynamic).
    extern __shared__ unsigned char shared_raw[];
    float2* s_points = reinterpret_cast<float2*>(shared_raw);
    unsigned char* ptr = shared_raw + TILE_POINTS * sizeof(float2);

    // Per-warp shared arrays: best distances/indices and candidate buffer.
    float* s_best_dist = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(float);

    int* s_best_idx = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(int);

    float* s_cand_dist = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(float);

    int* s_cand_idx = reinterpret_cast<int*>(ptr);
    // ptr now points to end of dynamic shared memory.

    // Per-warp slices into shared arrays.
    float* warp_best_dist = s_best_dist + warp_id_in_block * k;
    int*   warp_best_idx  = s_best_idx  + warp_id_in_block * k;
    float* warp_cand_dist = s_cand_dist + warp_id_in_block * k;
    int*   warp_cand_idx  = s_cand_idx  + warp_id_in_block * k;

    // Shared temp storage for CUB warp radix sort: one per warp.
    __shared__ typename WarpSortT::TempStorage warp_sort_temp[WARPS_PER_BLOCK];

    // Per-warp query coordinates (broadcast within warp).
    float qx = 0.0f;
    float qy = 0.0f;
    if (valid_warp) {
        float2 q;
        if (lane_id == 0) {
            q = query[global_warp_id];
        }
        qx = __shfl_sync(FULL_MASK, q.x, 0);
        qy = __shfl_sync(FULL_MASK, q.y, 0);
    }

    // Each lane keeps a fragment of the intermediate top-k in registers.
    // Since k is a power of two >= 32, k / WARP_SIZE is an integer in [1, 32].
    const int items_per_thread = k / WARP_SIZE;

    // Local fragments of best distances and indices.
    float best_local_dist[/*max*/ MAX_K / WARP_SIZE];
    int   best_local_idx[/*max*/ MAX_K / WARP_SIZE];

    // Initialize intermediate best list to "infinite" distances.
    float worst_dist = CUDART_INF_F;
    int cand_count   = 0;

    if (valid_warp) {
        for (int j = 0; j < items_per_thread; ++j) {
            best_local_dist[j] = CUDART_INF_F;
            best_local_idx[j]  = -1;
        }
    }

    // Process data in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        const int tile_size = (tile_start + TILE_POINTS <= data_count)
                            ? TILE_POINTS
                            : (data_count - tile_start);

        // Load tile of data points into shared memory cooperatively by the block.
        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_points[i] = data[tile_start + i];
        }
        __syncthreads();

        if (valid_warp) {
            // For each point in tile, compute distance to query and insert into
            // per-warp candidate buffer if closer than current k-th neighbor.
            for (int i = lane_id; i < tile_size; i += WARP_SIZE) {
                float2 p = s_points[i];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = dx * dx + dy * dy;

                bool is_good = (dist < worst_dist);
                unsigned mask = __ballot_sync(FULL_MASK, is_good);
                if (mask == 0u) {
                    continue;
                }

                int good_count = __popc(mask);

                // If candidate buffer would overflow, merge it first.
                if (cand_count + good_count > k) {
                    if (cand_count > 0) {
                        merge_buffer_warp(
                            warp_id_in_block,
                            lane_id,
                            k,
                            items_per_thread,
                            warp_best_dist,
                            warp_best_idx,
                            warp_cand_dist,
                            warp_cand_idx,
                            cand_count,
                            worst_dist,
                            warp_sort_temp,
                            best_local_dist,
                            best_local_idx
                        );
                    }
                    cand_count = 0;
                }

                // Reserve slots in candidate buffer for the new candidates.
                int lane_rank = __popc(mask & ((1u << lane_id) - 1u));
                if (is_good) {
                    int pos = cand_count + lane_rank;
                    warp_cand_dist[pos] = dist;
                    warp_cand_idx[pos]  = tile_start + i; // global data index
                }

                cand_count += good_count;
            }
        }

        __syncthreads(); // Ensure tile is not reused before all warps finish.
    }

    // Merge remaining candidates after the last tile.
    if (valid_warp && cand_count > 0) {
        merge_buffer_warp(
            warp_id_in_block,
            lane_id,
            k,
            items_per_thread,
            warp_best_dist,
            warp_best_idx,
            warp_cand_dist,
            warp_cand_idx,
            cand_count,
            worst_dist,
            warp_sort_temp,
            best_local_dist,
            best_local_idx
        );
        cand_count = 0;
    }

    // Write final k nearest neighbors (sorted by distance) to result array.
    if (valid_warp) {
        const int query_idx = global_warp_id;
        const int base_out  = query_idx * k;
        for (int j = 0; j < items_per_thread; ++j) {
            int g = lane_id * items_per_thread + j;
            std::pair<int, float> out;
            out.first  = best_local_idx[j];
            out.second = best_local_dist[j];
            result[base_out + g] = out;
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface: run_knn
// -----------------------------------------------------------------------------
void run_knn(
    const float2*            query,
    int                      query_count,
    const float2*            data,
    int                      data_count,
    std::pair<int, float>*   result,
    int                      k
) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Ensure k is within supported range.
    if (k > MAX_K || (k & (k - 1)) != 0 || k < WARP_SIZE) {
        // Unsupported k; in production code, handle this more gracefully.
        return;
    }

    // Compute execution configuration.
    const int warps_per_block   = WARPS_PER_BLOCK;
    const int threads_per_block = THREADS_PER_BLOCK;
    const int total_warps       = query_count;
    const int blocks            = (total_warps + warps_per_block - 1) / warps_per_block;

    // Dynamic shared memory size:
    // - TILE_POINTS float2 for cached data
    // - For each warp:
    //     * k floats  for best_dist
    //     * k ints    for best_idx
    //     * k floats  for cand_dist
    //     * k ints    for cand_idx
    const size_t per_warp_bytes =
        static_cast<size_t>(k) * (sizeof(float) + sizeof(int)) * 2; // 4 arrays of size k
    const size_t shared_bytes =
        TILE_POINTS * sizeof(float2) +
        WARPS_PER_BLOCK * per_warp_bytes;

    // Opt-in to enough dynamic shared memory if needed (Ampere/Hopper).
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Clamp requested shared memory to device capability.
    int max_dynamic = prop.sharedMemPerBlockOptin;
    int requested   = static_cast<int>(shared_bytes);
    if (requested > max_dynamic) {
        // In production, we might fall back to smaller tiles or fewer warps.
        // Here, just clamp to the maximum allowed.
        requested = max_dynamic;
    }

    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        requested
    );

    // Prefer shared memory over L1 cache if possible.
    cudaFuncSetCacheConfig(knn_kernel, cudaFuncCachePreferShared);

    // Launch kernel.
    knn_kernel<<<blocks, threads_per_block, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k
    );

    // Optional: synchronize or check errors here in production code.
    // cudaDeviceSynchronize();
}