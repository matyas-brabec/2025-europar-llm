#include <cuda_runtime.h>
#include <utility>
#include <algorithm>
#include <limits>

// This implementation assigns one warp (32 threads) per query point.
// Each warp maintains two per-query arrays in shared memory of length k:
//  - intermediate result (sorted, ascending) [indices + distances]
//  - candidate buffer (unsorted until flushed) [indices + distances]
// Additionally, each warp has a shared counter for the number of candidates
// currently stored in its buffer and a shared scalar "max_distance" (the
// distance of the current k-th nearest neighbor).
//
// The data points are processed in tiles. Each block cooperatively loads a tile
// of data points into shared memory. Then each warp computes distances from its
// own query point to the cached data points. Distances less than max_distance
// are added to the warp's candidate buffer using atomicAdd on the shared counter.
// Whenever the buffer is about to become full, it is flushed and merged with the
// intermediate result using a bitonic-sort-based merge as described.
//
// After processing all tiles, any remaining candidates are flushed and merged.
// Finally, the top-k indices and distances for each query are written out.
//
// Notes:
// - k is assumed to be a power-of-two in [32, 1024].
// - No additional device memory is allocated.
// - We use warp-scope synchronization (__syncwarp) and block-scope synchronization
//   (__syncthreads) where appropriate.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable constant: warps per block. We choose 6 to fit large k and a sizeable tile
// comfortably within the shared memory limits of A100/H100 while maintaining decent occupancy.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 6
#endif

// Helper POD matching the memory layout of std::pair<int,float> for device writes.
struct PairIF {
    int first;
    float second;
};

// Utility: round up 'x' to the next multiple of 'a' (a must be power-of-two).
static inline size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

// Device-side compare-and-swap for a pair (distance + index).
// The ordering key is 'dist'; ties are not specially handled (unspecified tie-breaking).
__device__ __forceinline__ void compare_swap_pair_asc(float &dist_i, int &idx_i, float &dist_j, int &idx_j, bool dir_up) {
    // dir_up == true  -> ascending within this subsequence
    // dir_up == false -> descending within this subsequence
    bool do_swap = (dir_up ? (dist_i > dist_j) : (dist_i < dist_j));
    if (do_swap) {
        float td = dist_i; dist_i = dist_j; dist_j = td;
        int ti = idx_i; idx_i = idx_j; idx_j = ti;
    }
}

// Warp-cooperative bitonic sort on pairs of arrays 'dists' and 'idxs' of length 'n'.
// Assumes n is a power-of-two (k constraint).
// All 32 threads in the warp participate. Each thread processes indices in [0,n)
// with a stride of warp size, filter l>i to ensure each pair is handled once.
// Synchronize with __syncwarp between inner stages.
__device__ void bitonic_sort_pairs_warp(float *dists, int *idxs, int n, unsigned mask) {
    // Bitonic sort network
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x & (WARP_SIZE - 1); i < n; i += WARP_SIZE) {
                int l = i ^ j;
                if (l > i) {
                    // Determine direction: ascending if (i & k) == 0
                    bool dir_up = ((i & k) == 0);
                    float di = dists[i];
                    float dl = dists[l];
                    int ii = idxs[i];
                    int il = idxs[l];
                    compare_swap_pair_asc(di, ii, dl, il, dir_up);
                    // Write back
                    dists[i] = di; idxs[i] = ii;
                    dists[l] = dl; idxs[l] = il;
                }
            }
            __syncwarp(mask);
        }
    }
}

// Flush-and-merge routine as specified:
// 0. The intermediate result is kept sorted ascending (invariant).
// 1. Sort the candidate buffer ascending (bitonic sort).
// 2. Merge by taking min(buffer[i], intermediate[k-1-i]) into the buffer (bitonic sequence).
// 3. Sort the merged sequence ascending (bitonic sort) -> updated intermediate.
// After finishing, swap roles of 'inter' and 'buf', reset count=0, and update max_distance.
// All operations are warp-cooperative and use __syncwarp for warp-scope synchronization.
__device__ void flush_and_merge_warp(float *&inter_d, int *&inter_i,
                                     float *&buf_d,  int *&buf_i,
                                     volatile int *buf_count_ptr,
                                     volatile float *maxdist_ptr,
                                     int k, unsigned mask)
{
    int count = *buf_count_ptr;
    if (count <= 0) return;

    // Pad the remainder of the buffer with +inf so we can bitonic sort exactly k elements.
    for (int t = threadIdx.x & (WARP_SIZE - 1), i = t; i < k; i += WARP_SIZE) {
        if (i >= count) {
            buf_d[i] = CUDART_INF_F;
            buf_i[i] = -1;
        }
    }
    __syncwarp(mask);

    // 1. Sort the buffer ascending.
    bitonic_sort_pairs_warp(buf_d, buf_i, k, mask);

    // 2. Merge into a bitonic sequence: buf[i] = min(buf[i], inter[k-1-i]).
    for (int t = threadIdx.x & (WARP_SIZE - 1), i = t; i < k; i += WARP_SIZE) {
        int j = k - 1 - i;
        float db = buf_d[i];
        int   ib = buf_i[i];
        float di = inter_d[j];
        int   ii = inter_i[j];
        if (di < db) { // take inter[j]
            buf_d[i] = di;
            buf_i[i] = ii;
        } else {
            // keep buf[i]
        }
    }
    __syncwarp(mask);

    // 3. Sort the merged sequence ascending (in-place in buf_*).
    bitonic_sort_pairs_warp(buf_d, buf_i, k, mask);
    __syncwarp(mask);

    // Swap roles: inter <-> buf
    float *tmpd = inter_d; inter_d = buf_d; buf_d = tmpd;
    int   *tmpi = inter_i; inter_i = buf_i; buf_i = tmpi;
    __syncwarp(mask);

    // Update max_distance and reset buffer count.
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        *maxdist_ptr = inter_d[k - 1];
        *buf_count_ptr = 0;
    }
    __syncwarp(mask);
}

// Kernel implementing k-NN for 2D points with one warp per query.
// Shared memory layout (dynamically sized at launch):
// [ float2 tile_points[tile_points_count] ]
// [ float  inter_dists[WARPS_PER_BLOCK][k] ]
// [ int    inter_idxs [WARPS_PER_BLOCK][k] ]
// [ float  buf_dists  [WARPS_PER_BLOCK][k] ]
// [ int    buf_idxs   [WARPS_PER_BLOCK][k] ]
// [ int    buf_counts [WARPS_PER_BLOCK]    ]
// [ float  max_dists  [WARPS_PER_BLOCK]    ]
/// @FIXED
/// extern "C" __global__
__global__
void knn_kernel_warp(const float2 * __restrict__ query,
                     int query_count,
                     const float2 * __restrict__ data,
                     int data_count,
                     PairIF * __restrict__ result,
                     int k,
                     int tile_points_count)
{
    extern __shared__ unsigned char shmem_raw[];
    unsigned char *sp = shmem_raw;

    // Shared memory partitioning.
    // Tile of data points
    float2 *s_tile = reinterpret_cast<float2*>(sp);
    sp += static_cast<size_t>(tile_points_count) * sizeof(float2);

    // Per-warp intermediate result arrays
    float *s_inter_dists = reinterpret_cast<float*>(sp);
    sp += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(float);

    int *s_inter_idxs = reinterpret_cast<int*>(sp);
    sp += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(int);

    // Per-warp candidate buffer arrays
    float *s_buf_dists = reinterpret_cast<float*>(sp);
    sp += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(float);

    int *s_buf_idxs = reinterpret_cast<int*>(sp);
    sp += static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(int);

    // Per-warp counters and max distances
    int *s_buf_counts = reinterpret_cast<int*>(sp);
    sp += static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(int);

    float *s_max_dists = reinterpret_cast<float*>(sp);
    sp += static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(float);

    // Thread identifiers
    const int lane_id  = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_gid = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned mask = __activemask();

    // Early exit if this warp has no query to process.
    if (warp_gid >= query_count) {
        return;
    }

    // Per-warp shared memory slices
    float *inter_d = s_inter_dists + warp_id * k;
    int   *inter_i = s_inter_idxs  + warp_id * k;
    float *buf_d   = s_buf_dists   + warp_id * k;
    int   *buf_i   = s_buf_idxs    + warp_id * k;
    volatile int   *buf_count_ptr  = s_buf_counts + warp_id;
    volatile float *maxdist_ptr    = s_max_dists  + warp_id;

    // Initialize intermediate result to +inf, indices to -1; buffer count=0; max distance=+inf.
    for (int i = lane_id; i < k; i += WARP_SIZE) {
        inter_d[i] = CUDART_INF_F;
        inter_i[i] = -1;
    }
    if (lane_id == 0) {
        *buf_count_ptr = 0;
        *maxdist_ptr   = CUDART_INF_F;
    }
    __syncwarp(mask);

    // Load query point (broadcast within warp).
    float2 qtmp = make_float2(0.0f, 0.0f);
    if (lane_id == 0) {
        qtmp = query[warp_gid];
    }
    const float qx = __shfl_sync(mask, qtmp.x, 0);
    const float qy = __shfl_sync(mask, qtmp.y, 0);

    // Process data in tiles: cooperative block load then warp computes distances to its query.
    for (int tile_base = 0; tile_base < data_count; tile_base += tile_points_count) {
        int tile_count = data_count - tile_base;
        if (tile_count > tile_points_count) tile_count = tile_points_count;

        // Load tile to shared memory: all threads in the block participate.
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            s_tile[t] = data[tile_base + t];
        }
        __syncthreads();

        // Iterate over tile: each warp processes one element per thread per iteration.
        for (int t = lane_id; t < tile_count; t += WARP_SIZE) {
            // Proactive flush to avoid overflow: if buffer usage >= k - WARP_SIZE, flush now.
            int need_flush = 0;
            if (lane_id == 0) {
                int c = *buf_count_ptr;
                need_flush = (c >= (k - WARP_SIZE)) ? 1 : 0;
            }
            need_flush = __shfl_sync(mask, need_flush, 0);
            if (need_flush) {
                flush_and_merge_warp(inter_d, inter_i, buf_d, buf_i, buf_count_ptr, maxdist_ptr, k, mask);
            }

            // Compute squared Euclidean distance from query to data point s_tile[t].
            float2 p = s_tile[t];
            float dx = p.x - qx;
            float dy = p.y - qy;
            float dist = fmaf(dy, dy, dx * dx); // dist = dx*dx + dy*dy

            // Read current max_distance (distance of the k-th nearest neighbor).
            float md = *maxdist_ptr;

            // If this point is closer than current max_distance, add it to candidate buffer.
            if (dist < md) {
                int pos = atomicAdd((int*)buf_count_ptr, 1);
                if (pos < k) {
                    buf_d[pos] = dist;
                    buf_i[pos] = tile_base + t;
                }
                // No else needed due to proactive flush; pos >= k should be rare to impossible here.
            }
        }

        // After processing the tile, flush remaining candidates (if any).
        int do_flush = 0;
        if (lane_id == 0) {
            do_flush = (*buf_count_ptr > 0) ? 1 : 0;
        }
        do_flush = __shfl_sync(mask, do_flush, 0);
        if (do_flush) {
            flush_and_merge_warp(inter_d, inter_i, buf_d, buf_i, buf_count_ptr, maxdist_ptr, k, mask);
        }

        __syncthreads(); // Ensure all warps finish using s_tile before the next load.
    }

    // Write out final top-k for this query (inter_d/inter_i are sorted ascending).
    const int out_base = warp_gid * k;
    for (int i = lane_id; i < k; i += WARP_SIZE) {
        PairIF out;
        out.first  = inter_i[i];
        out.second = inter_d[i];
        result[out_base + i] = out;
    }
}

// Compute the required dynamic shared memory size and a tile size that fits in the available
// per-block shared memory (opt-in). We prefer a large tile (up to 4096 points) but will reduce
// as needed given k and WARPS_PER_BLOCK.
// Returns the chosen tile_points and the total dynamic shared memory bytes required.
static inline void choose_tile_and_shmem(int k, int &tile_points, size_t &shmem_bytes) {
    // Baseline preferred tile size (points), each point is a float2 (8 bytes).
    int preferred_tile = 4096;

    // Per-warp shared arrays: 2 (inter + buffer) * k * (float + int) = 16 * k bytes.
    size_t per_warp_arrays = static_cast<size_t>(k) * (sizeof(float) + sizeof(int)) * 2;
    size_t per_warps_total = static_cast<size_t>(WARPS_PER_BLOCK) * per_warp_arrays;

    // Per-warp scalars: counts (int) + max_distance (float)
    size_t per_warp_scalars = sizeof(int) + sizeof(float);
    size_t scalars_total = static_cast<size_t>(WARPS_PER_BLOCK) * per_warp_scalars;

    // Available opt-in shared memory per block depends on device; we will query and clamp externally.
    // Here we set an upper bound for design (A100/H100 typical): 163840 or larger. We'll adjust tile later.

    // We'll start with preferred tile and compute shmem usage; if it exceeds what the kernel
    // is allowed (set via cudaFuncSetAttribute externally), we can reduce tile in run_knn before launch.
    tile_points = preferred_tile;

    // Compute shared memory footprint for a given tile_points
    auto compute_shmem = [&](int tile_pts) -> size_t {
        size_t bytes = 0;
        bytes += static_cast<size_t>(tile_pts) * sizeof(float2); // tile of points
        bytes += per_warps_total;                                // per-warp arrays
        bytes += scalars_total;                                   // per-warp counters + max_dists
        return bytes;
    };

    shmem_bytes = compute_shmem(tile_points);
}

// Host API: run_knn
// Launches the kernel with one warp per query.
// query: device pointer to float2 queries, size query_count
// data:  device pointer to float2 data points, size data_count
// result: device pointer to std::pair<int,float> (will be written as PairIF)
// k: power-of-two in [32, 1024]
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Determine grid/block configuration.
    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int total_warps = (query_count + 0) / 1; // one warp per query
    int grid_blocks = (query_count + warps_per_block - 1) / warps_per_block;
    if (grid_blocks <= 0) grid_blocks = 1;

    // Choose tile size and compute dynamic shared memory usage.
    int tile_points = 0;
    size_t shmem_bytes = 0;
    choose_tile_and_shmem(k, tile_points, shmem_bytes);

    // Query device shared memory limits and adjust tile to fit within opt-in limit.
    int device = 0;
    cudaGetDevice(&device);
    int max_optin_bytes = 0;
    cudaDeviceGetAttribute(&max_optin_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_optin_bytes <= 0) {
        // Fallback to legacy per-block limit if opt-in attribute is not reported.
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        max_optin_bytes = static_cast<int>(prop.sharedMemPerBlock);
    }

    // Reduce tile_points if needed to fit into available shared memory.
    auto calc_total_shmem = [&](int tile_pts) -> size_t {
        size_t per_warp_arrays = static_cast<size_t>(k) * (sizeof(float) + sizeof(int)) * 2;
        size_t per_warps_total = static_cast<size_t>(WARPS_PER_BLOCK) * per_warp_arrays;
        size_t per_warp_scalars = sizeof(int) + sizeof(float);
        size_t scalars_total = static_cast<size_t>(WARPS_PER_BLOCK) * per_warp_scalars;
        size_t bytes = 0;
        bytes += static_cast<size_t>(tile_pts) * sizeof(float2);
        bytes += per_warps_total;
        bytes += scalars_total;
        return bytes;
    };

    // Clamp tile_points so that shmem_bytes <= max_optin_bytes (with a small safety margin).
    // We use a conservative safety margin of 0 to allow maximal utilization on A100/H100.
    while (shmem_bytes > static_cast<size_t>(max_optin_bytes) && tile_points > 256) {
        tile_points >>= 1; // halve tile size until it fits
        shmem_bytes = calc_total_shmem(tile_points);
    }
    if (shmem_bytes > static_cast<size_t>(max_optin_bytes)) {
        // As an extreme fallback, set tile_points to minimal 256 if still too big.
        tile_points = 256;
        shmem_bytes = calc_total_shmem(tile_points);
    }

    // Opt-in to the required dynamic shared memory size for this kernel.
    cudaFuncSetAttribute(knn_kernel_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_bytes));

    // Reinterpret result pointer to POD with identical layout.
    PairIF *result_pairs = reinterpret_cast<PairIF*>(result);

    // Launch the kernel.
    dim3 grid(grid_blocks);
    dim3 block(threads_per_block);
    knn_kernel_warp<<<grid, block, shmem_bytes>>>(query, query_count, data, data_count, result_pairs, k, tile_points);

    // The caller is expected to perform error checks or synchronize as needed.
}
