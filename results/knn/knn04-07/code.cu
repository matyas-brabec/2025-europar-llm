#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>

// Optimized k-NN for 2D points using one warp (32 threads) per query.
// - Loads data points in shared memory in tiles cooperatively by the block.
// - Each warp computes distances from its query to all cached points.
// - Per warp, maintains a private top-k buffer in shared memory (double-buffered).
// - Updates the intermediate top-k with 32 candidates at a time via a warp-cooperative
//   merge using the merge-path algorithm.
// - Uses warp-level shuffles to sort the 32 candidates and to access them during merging.

static inline __device__ int lane_id() {
    return threadIdx.x & 31;
}

// Warp-wide bitonic sort of 32 (dist, idx) pairs in ascending order by dist.
// Each thread contributes one element; after sort, lane i holds the i-th smallest element.
// This uses the classic XOR-shuffle bitonic network with branchless min/max selection.
static inline __device__ void warp_bitonic_sort32_asc(float &dist, int &idx, unsigned mask) {
    int lane = lane_id();
    // k is the size of the sequence being merged/sorted at this stage
    for (unsigned k = 2; k <= 32; k <<= 1) {
        // j is the distance to the partner in the compare-exchange within stage k
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            float oth_dist = __shfl_xor_sync(mask, dist, j);
            int   oth_idx  = __shfl_xor_sync(mask, idx,  j);

            bool up = ((lane & k) == 0); // upper/lower half determines direction
            bool comp = (dist > oth_dist);
            // Select min/max without branches
            float min_d = comp ? oth_dist : dist;
            float max_d = comp ? dist : oth_dist;
            int   min_i = comp ? oth_idx  : idx;
            int   max_i = comp ? idx      : oth_idx;

            dist = up ? min_d : max_d;
            idx  = up ? min_i : max_i;
        }
    }
}

// Binary-search the merge path partition for diagonal D when merging A (length m)
// and B (length n). A is in memory (ascending). B elements are held one per lane
// in registers and accessed via shuffles from b_dist_local.
// Returns the number of elements to take from A (denoted 'a'), so that b = D - a.
// Reference: "GPU Merge Path" (Green, McColl, Bader).
static inline __device__ int mergepath_search_A_for_diag(
    int D, const float* A, int m, int n, float b_dist_local, unsigned mask)
{
    int low  = max(0, D - n);
    int high = min(D, m);
    // We search for the smallest 'a' such that A[a-1] <= B[D-a]
    while (low < high) {
        int a = (low + high) >> 1;
        int b = D - a;

        float A_im1 = (a > 0) ? A[a - 1] : -CUDART_INF_F;
        float B_j   = (b < n) ? __shfl_sync(mask, b_dist_local, b) : CUDART_INF_F;

        if (A_im1 > B_j) {
            high = a;
        } else {
            low = a + 1;
        }
    }
    return low;
}

// Merge the current warp-private top-k (arrays curDist/curIdx, length k, ascending)
// with 32 new sorted candidates B (one per lane, sorted ascending via bitonic) into
// the next buffer (nextDist/nextIdx), keeping only the smallest k of (k + 32).
// Uses the merge-path algorithm to partition the k outputs across the 32 lanes.
// Each lane writes a disjoint contiguous segment [diag_start, diag_end) of the result.
static inline __device__ void warp_merge_topk_with_32(
    const float* curDist, const int* curIdx,
    float* nextDist, int* nextIdx,
    int k,
    float b_dist_local, int b_idx_local,
    unsigned mask)
{
    const int lane = lane_id();
    const int n = 32;

    // Partition the first k outputs evenly across the 32 lanes
    int diag_start = (int)(((long long)lane * k) >> 5);              // floor(lane * k / 32)
    int diag_end   = (int)((((long long)lane + 1) * k) >> 5);        // floor((lane+1)*k/32)

    // Find (a_start, b_start) and (a_end, b_end) for the two diagonals
    int a_start = mergepath_search_A_for_diag(diag_start, curDist, k, n, b_dist_local, mask);
    int b_start = diag_start - a_start;

    int a_end = mergepath_search_A_for_diag(diag_end, curDist, k, n, b_dist_local, mask);
    int b_end = diag_end - a_end;

    int i = a_start;
    int j = b_start;
    int o = diag_start;

    // Sequentially merge this lane's chunk
    while (o < diag_end) {
        float a_val = (i < k) ? curDist[i] : CUDART_INF_F;
        float b_val;
        if (j < n) {
            b_val = __shfl_sync(mask, b_dist_local, j);
        } else {
            b_val = CUDART_INF_F;
        }

        // Select the smaller element
        if (b_val < a_val) {
            // Taking from B
            int b_idx = __shfl_sync(mask, b_idx_local, j);
            nextDist[o] = b_val;
            nextIdx[o]  = b_idx;
            ++j;
        } else {
            // Taking from A
            nextDist[o] = a_val;
            nextIdx[o]  = curIdx[i];
            ++i;
        }
        ++o;
    }
}

// The CUDA kernel that performs k-NN for 2D points.
// Each warp (32 threads) handles one query point.
// Shared memory layout (dynamic):
//   - float2 sData[tile_size]                      // cached data points
//   - float  topk_dist0[warps_per_block * k]       // per-warp top-k distances (buffer 0)
//   - int    topk_idx0 [warps_per_block * k]       // per-warp top-k indices   (buffer 0)
//   - float  topk_dist1[warps_per_block * k]       // per-warp top-k distances (buffer 1)
//   - int    topk_idx1 [warps_per_block * k]       // per-warp top-k indices   (buffer 1)
__global__ void knn2d_kernel(const float2* __restrict__ query, int query_count,
                             const float2* __restrict__ data,  int data_count,
                             std::pair<int, float>* __restrict__ result,
                             int k, int tile_size)
{
    extern __shared__ unsigned char smem_raw[];
    float2* sData = reinterpret_cast<float2*>(smem_raw);

    const int warps_per_block = blockDim.x >> 5;
    const int warp_id_in_block = threadIdx.x >> 5;
    const int lane = lane_id();
    const int warp_global_id = blockIdx.x * warps_per_block + warp_id_in_block;
    const bool warp_active = (warp_global_id < query_count);

    // Compute shared memory pointers for per-warp top-k buffers
    size_t offset_bytes = (size_t)tile_size * sizeof(float2);

    float* topk_dist0 = reinterpret_cast<float*>(smem_raw + offset_bytes);
    offset_bytes += (size_t)warps_per_block * k * sizeof(float);

    int*   topk_idx0  = reinterpret_cast<int*>(smem_raw + offset_bytes);
    offset_bytes += (size_t)warps_per_block * k * sizeof(int);

    float* topk_dist1 = reinterpret_cast<float*>(smem_raw + offset_bytes);
    offset_bytes += (size_t)warps_per_block * k * sizeof(float);

    int*   topk_idx1  = reinterpret_cast<int*>(smem_raw + offset_bytes);
    // offset_bytes += warps_per_block * k * sizeof(int); // not needed further

    // Warp-local base pointers for the two buffers
    float* my_dist0 = topk_dist0 + warp_id_in_block * k;
    int*   my_idx0  = topk_idx0  + warp_id_in_block * k;
    float* my_dist1 = topk_dist1 + warp_id_in_block * k;
    int*   my_idx1  = topk_idx1  + warp_id_in_block * k;

    // Initialize both buffers to +inf distances and -1 indices.
    // This provides a correct starting state for the merge operations.
    for (int pos = lane; pos < k; pos += 32) {
        my_dist0[pos] = CUDART_INF_F;
        my_idx0 [pos] = -1;
        my_dist1[pos] = CUDART_INF_F;
        my_idx1 [pos] = -1;
    }
    __syncwarp(); // ensure initialization is done within the warp

    // Current/next buffer toggle
    bool use_buf0 = true;

    // Load the query point into registers and broadcast within the warp
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        if (lane == 0) {
            float2 q = query[warp_global_id];
            qx = q.x; qy = q.y;
        }
        unsigned mask = __activemask();
        qx = __shfl_sync(mask, qx, 0);
        qy = __shfl_sync(mask, qy, 0);
    }

    // Process the dataset in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_size) {
        int tile_count = min(tile_size, data_count - tile_start);

        // Cooperative loading of the tile into shared memory
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            sData[i] = data[tile_start + i];
        }

        __syncthreads(); // ensure the whole tile is cached

        if (warp_active) {
            // Current and next top-k buffers for this warp
            float* curDist = use_buf0 ? my_dist0 : my_dist1;
            int*   curIdx  = use_buf0 ? my_idx0  : my_idx1;
            float* nxtDist = use_buf0 ? my_dist1 : my_dist0;
            int*   nxtIdx  = use_buf0 ? my_idx1  : my_idx0;

            const unsigned full_mask = __activemask(); // all 32 threads in this warp are active here

            // Process the tile in groups of 32 points; each lane handles one point per group
            int group_count = (tile_count + 31) >> 5;
            for (int g = 0; g < group_count; ++g) {
                int idx_in_tile = (g << 5) + lane;
                float cand_dist;
                int   cand_idx;

                if (idx_in_tile < tile_count) {
                    float2 p = sData[idx_in_tile];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    // Squared Euclidean distance
                    cand_dist = __fmaf_rn(dx, dx, dy * dy);
                    cand_idx  = tile_start + idx_in_tile;
                } else {
                    // Inactive lanes in the last group contribute +inf (ignored by merges)
                    cand_dist = CUDART_INF_F;
                    cand_idx  = -1;
                }

                // Sort the 32 candidates in ascending order by distance (across the warp)
                warp_bitonic_sort32_asc(cand_dist, cand_idx, full_mask);

                // Merge the sorted 32 candidates with the current top-k into the next buffer
                warp_merge_topk_with_32(curDist, curIdx, nxtDist, nxtIdx, k, cand_dist, cand_idx, full_mask);

                __syncwarp(full_mask); // ensure all lanes completed writes before swapping buffers

                // Swap buffers for the next group
                float* tmpD = curDist; curDist = nxtDist; nxtDist = tmpD;
                int*   tmpI = curIdx;  curIdx  = nxtIdx;  nxtIdx  = tmpI;
                use_buf0 = (curDist == my_dist0); // track current buffer for consistency
            }
        }

        __syncthreads(); // allow all warps to finish with this tile before loading the next
    }

    // Write out the final top-k for this query in ascending order (j-th nearest first)
    if (warp_active) {
        float* curDist = use_buf0 ? my_dist0 : my_dist1;
        int*   curIdx  = use_buf0 ? my_idx0  : my_idx1;

        size_t out_base = (size_t)warp_global_id * (size_t)k;
        for (int pos = lane; pos < k; pos += 32) {
            // Accessing std::pair fields directly is safe and avoids constructing temporaries
            result[out_base + pos].first  = curIdx[pos];
            result[out_base + pos].second = curDist[pos];
        }
    }
}

// Host interface to launch the k-NN kernel.
// - Decides the number of warps per block (default 8).
// - Computes an appropriate tile size based on available shared memory and k.
// - Allocates sufficient dynamic shared memory for the data tile and per-warp top-k buffers.
// - Uses cudaFuncSetAttribute to allow large dynamic shared memory on A100/H100.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose number of warps per block. 8 warps (256 threads) gives good balance on A100/H100.
    int warps_per_block = 8;
    int threads_per_block = warps_per_block * 32;

    // Determine the maximum dynamic shared memory per block supported (opt-in).
    int device = 0;
    cudaGetDevice(&device);

    int smem_max_optin = 0;
    cudaDeviceGetAttribute(&smem_max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (smem_max_optin == 0) {
        // Fallback to non-optin limit if opt-in not supported (should not happen on A100/H100).
        cudaDeviceGetAttribute(&smem_max_optin, cudaDevAttrMaxSharedMemoryPerBlock, device);
    }

    // Compute per-block shared memory usage for per-warp top-k buffers (double-buffered).
    size_t topk_bytes_per_block = (size_t)warps_per_block * (size_t)k * (sizeof(float) + sizeof(int)) * 2;

    // Reduce warps per block if necessary to fit shared memory constraints.
    while (topk_bytes_per_block + (size_t)32 * sizeof(float2) > (size_t)smem_max_optin && warps_per_block > 1) {
        warps_per_block >>= 1; // halve the warps per block
        threads_per_block = warps_per_block * 32;
        topk_bytes_per_block = (size_t)warps_per_block * (size_t)k * (sizeof(float) + sizeof(int)) * 2;
    }

    // Compute maximum tile size that fits into the remaining shared memory budget.
    size_t max_tile_bytes = (topk_bytes_per_block < (size_t)smem_max_optin)
                          ? ((size_t)smem_max_optin - topk_bytes_per_block)
                          : 0;
    // Prefer a large tile (e.g., up to 4096 points), but cap to what's available.
    int desired_tile_points = 4096;
    int tile_points = (int)(max_tile_bytes / sizeof(float2));
    if (tile_points > desired_tile_points) tile_points = desired_tile_points;
    if (tile_points < 32) tile_points = 32; // ensure at least one full warp of points
    // Round down to a multiple of 32 for simplicity (complete candidate groups)
    tile_points = (tile_points / 32) * 32;
    if (tile_points < 32) tile_points = 32;

    // Recompute final dynamic shared memory size
    size_t dynamic_smem_bytes = (size_t)tile_points * sizeof(float2) + topk_bytes_per_block;

    // Opt-in to large dynamic shared memory if necessary
    cudaFuncSetAttribute(knn2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dynamic_smem_bytes);

    // Compute grid configuration
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch the kernel
    knn2d_kernel<<<blocks, threads_per_block, dynamic_smem_bytes>>>(
        query, query_count, data, data_count, result, k, tile_points
    );

    // Optional: synchronize or check errors here if desired by the caller.
    // cudaDeviceSynchronize();
}