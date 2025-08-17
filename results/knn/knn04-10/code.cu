#include <cuda_runtime.h>
#include <utility>

// KNN for 2D points using warp-per-query with shared-memory tiling.
// - Each warp (32 threads) processes one query point.
// - The block cooperatively loads tiles of data points into shared memory.
// - Each warp computes distances from its query to the tile points and updates a per-warp top-k buffer in shared memory.
// - Candidates from each warp are processed in sorted ascending order per 32-point batch using warp-level bitonic sort with shuffles.
// - The per-warp intermediate top-k buffer is updated with multiple candidates per batch in a warp-synchronous manner.
// - After scanning all data points, each warp sorts its k results (bitonic sort in shared memory) and writes them to the output.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable parameters:
// Number of warps per block. 8 warps (256 threads) is a good balance for H100/A100.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif

// Tile size: number of data points (float2) loaded into shared memory at once.
// 4096 points => 32KB shared memory for tile, leaving room for per-warp top-k buffers even for k=1024.
#ifndef TILE_POINTS
#define TILE_POINTS 4096
#endif

// Full warp mask
#ifndef FULL_MASK
#define FULL_MASK 0xFFFFFFFFu
#endif

// Device utility: warp-level bitonic sort of 32 keys with associated values, ascending order.
// Each lane holds one key/value; after the sort, lane 'i' holds the i-th smallest key/value among the warp.
__device__ __forceinline__
void warp_bitonic_sort_asc_32(float &key, int &val, unsigned lane_id)
{
    // Standard warp-level bitonic sorting network using XOR shuffles.
    for (int k = 2; k <= WARP_SIZE; k <<= 1)
    {
        // For each stage, do log2(k) steps with decreasing 'j'
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            float other_key = __shfl_xor_sync(FULL_MASK, key, j);
            int other_val   = __shfl_xor_sync(FULL_MASK, val, j);

            bool up = ((lane_id & k) == 0);
            // If up: keep smaller on lower lane; If down: keep larger on lower lane.
            bool cond = (key > other_key) == up;
            float selected_key = cond ? other_key : key;
            int   selected_val = cond ? other_val : val;

            // The partner lane must keep the complementary element, but since all lanes execute
            // the same code, this formulation ensures consistency.
            key = selected_key;
            val = selected_val;
        }
    }
}

// Device utility: find the maximum value and its index in arr[0..len-1] across a warp cooperatively.
// Each lane scans a strided subset of the array (indices lane, lane+32, lane+64, ...), then a warp reduction finds the maximum.
// The results (max value and index) are broadcast to all lanes of the warp via shuffles.
__device__ __forceinline__
void warp_find_max_with_index(const float* __restrict__ arr, int len, unsigned lane_id, float &out_max_val, int &out_max_idx)
{
    float local_max = -CUDART_INF_F;
    int local_idx = -1;

    // Strided scan across the per-warp array in shared memory
    for (int i = lane_id; i < len; i += WARP_SIZE)
    {
        float v = arr[i];
        if (v > local_max)
        {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp-wide reduction to find global max and its index
    // We reduce with shfl_down; after reduction, lane 0 has the final result.
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
    {
        float other_val = __shfl_down_sync(FULL_MASK, local_max, offset);
        int other_idx   = __shfl_down_sync(FULL_MASK, local_idx, offset);
        if (other_val > local_max)
        {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Broadcast to all lanes
    out_max_val = __shfl_sync(FULL_MASK, local_max, 0);
    out_max_idx = __shfl_sync(FULL_MASK, local_idx, 0);
}

// CUDA kernel implementing k-NN for 2D points with squared Euclidean distances.
__global__ void knn_kernel_2d_warp(const float2* __restrict__ query,
                                   int query_count,
                                   const float2* __restrict__ data,
                                   int data_count,
                                   std::pair<int, float>* __restrict__ result,
                                   int k)
{
    // Thread, warp, and block configuration
    const unsigned tid       = threadIdx.x;
    const unsigned lane_id   = tid & (WARP_SIZE - 1);
    const unsigned warp_id   = tid >> 5; // warp index within the block
    const unsigned warps_per_block = blockDim.x >> 5;

    // The query index for this warp
    const int qidx = blockIdx.x * warps_per_block + warp_id;
    const bool has_query = (qidx < query_count);

    // Dynamic shared memory layout:
    // [0 .. TILE_POINTS-1]                -> float2 tile points
    // [next .. next + warps*k - 1]        -> float distances (per-warp top-k buffers)
    // [next .. next + warps*k - 1]        -> int indices (per-warp top-k buffers)
    extern __shared__ unsigned char s_mem[];
    unsigned char* smem_ptr = s_mem;

    // Shared tile of data points
    float2* s_tile = reinterpret_cast<float2*>(smem_ptr);
    smem_ptr += TILE_POINTS * sizeof(float2);

    // Per-warp top-k buffers in shared memory
    float* s_best_d = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += (warps_per_block * k) * sizeof(float);

    int* s_best_i = reinterpret_cast<int*>(smem_ptr);
    // smem_ptr += (warps_per_block * k) * sizeof(int); // not needed further

    // Pointers to this warp's private top-k buffers
    float* warp_best_d = s_best_d + warp_id * k;
    int*   warp_best_i = s_best_i + warp_id * k;

    // Initialize this warp's top-k buffers
    int count = 0; // number of elements currently in the top-k buffer (initially empty)
    if (has_query)
    {
        for (int i = lane_id; i < k; i += WARP_SIZE)
        {
            warp_best_d[i] = CUDART_INF_F;
            warp_best_i[i] = -1;
        }
        __syncwarp();
    }

    // Load the query point into registers, one lane loads and broadcasts
    float qx = 0.0f, qy = 0.0f;
    if (lane_id == 0 && has_query)
    {
        float2 q = query[qidx];
        qx = q.x; qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Iterate over data in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS)
    {
        const int tile_count = min(TILE_POINTS, data_count - tile_start);

        // Block-wide cooperative load of the tile into shared memory
        for (int t = tid; t < tile_count; t += blockDim.x)
        {
            s_tile[t] = data[tile_start + t];
        }
        __syncthreads();

        if (has_query)
        {
            // Process the tile in batches of 32 points (one warp batch)
            for (int base = 0; base < tile_count; base += WARP_SIZE)
            {
                const int j = base + lane_id;
                const bool valid = (j < tile_count);

                // Compute candidate distance for this lane (or INF if invalid)
                float cand_d = CUDART_INF_F;
                int   cand_i = -1;
                if (valid)
                {
                    float2 p = s_tile[j];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    // Squared Euclidean distance
                    cand_d = __fmaf_rn(dy, dy, dx * dx);
                    cand_i = tile_start + j; // global index
                }

                // Sort the 32 lane-local candidates ascending to process the best first.
                // Invalid lanes (cand_d = INF) move to the end.
                float sort_key = cand_d;
                int   sort_val = cand_i;
                warp_bitonic_sort_asc_32(sort_key, sort_val, lane_id);

                // After sorting: lane 't' holds the t-th smallest candidate among this batch.
                // Fill phase: append as many as needed to reach k.
                int valid_count = __popc(__ballot_sync(FULL_MASK, sort_key < CUDART_INF_F));
                int remaining = k - count;
                int to_fill = remaining > 0 ? min(remaining, valid_count) : 0;

                // Write the first 'to_fill' candidates into the top-k buffer at positions [count, count+to_fill)
                if (to_fill > 0)
                {
                    if ((int)lane_id < to_fill)
                    {
                        warp_best_d[count + lane_id] = sort_key;
                        warp_best_i[count + lane_id] = sort_val;
                    }
                    __syncwarp();
                    if (lane_id == 0) count += to_fill;
                    count = __shfl_sync(FULL_MASK, count, 0);
                }

                // Replacement phase: for remaining candidates in ascending order, try to insert if better than current worst.
                if (count == k && valid_count > to_fill)
                {
                    // Compute current tau (maximum distance in the buffer) and its index cooperatively
                    float tau;
                    int tau_idx;
                    warp_find_max_with_index(warp_best_d, k, lane_id, tau, tau_idx);

                    // Iterate over remaining sorted candidates (from best to worst)
                    for (int t = to_fill; t < valid_count; ++t)
                    {
                        // Broadcast the t-th candidate from lane 't' to all lanes
                        float c_d = __shfl_sync(FULL_MASK, sort_key, t);
                        int   c_i = __shfl_sync(FULL_MASK, sort_val, t);

                        // Early exit if candidate is not better than current worst
                        if (!(c_d < tau))
                            break;

                        // Replace the current worst with this candidate
                        if (lane_id == 0)
                        {
                            warp_best_d[tau_idx] = c_d;
                            warp_best_i[tau_idx] = c_i;
                        }
                        __syncwarp();

                        // Recompute tau for the next insertion attempt
                        warp_find_max_with_index(warp_best_d, k, lane_id, tau, tau_idx);
                    }
                }
            } // end for base
        } // end if has_query

        __syncthreads(); // Ensure all warps are done before loading the next tile
    } // end tiles loop

    // Finalize: sort the top-k buffer for this query ascending by distance and write to output
    if (has_query)
    {
        // In-place bitonic sort across k elements in this warp's private shared memory buffers.
        // All 32 threads in the warp cooperate, each handling a strided subset of indices.
        for (int size = 2; size <= k; size <<= 1)
        {
            for (int stride = size >> 1; stride > 0; stride >>= 1)
            {
                for (int i = lane_id; i < k; i += WARP_SIZE)
                {
                    int j = i ^ stride;
                    if (j > i)
                    {
                        bool up = ((i & size) == 0);
                        float ai = warp_best_d[i];
                        float aj = warp_best_d[j];
                        int bi = warp_best_i[i];
                        int bj = warp_best_i[j];
                        bool do_swap = (ai > aj) == up;
                        if (do_swap)
                        {
                            warp_best_d[i] = aj;
                            warp_best_d[j] = ai;
                            warp_best_i[i] = bj;
                            warp_best_i[j] = bi;
                        }
                    }
                }
                __syncwarp();
            }
        }

        // Write out the sorted results for this query
        const int out_base = qidx * k;
        for (int i = lane_id; i < k; i += WARP_SIZE)
        {
            result[out_base + i].first  = warp_best_i[i];
            result[out_base + i].second = warp_best_d[i]; // squared Euclidean distances
        }
    }
}

// Host-side interface
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Launch configuration
    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    // Grid dimension: one warp per query
    const int grid_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Dynamic shared memory size:
    // - TILE_POINTS * sizeof(float2) for the data tile
    // - warps_per_block * k * (sizeof(float) + sizeof(int)) for per-warp top-k buffers
    size_t smem_bytes = static_cast<size_t>(TILE_POINTS) * sizeof(float2)
                      + static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    // Launch the kernel
    knn_kernel_2d_warp<<<grid_blocks, threads_per_block, smem_bytes>>>(query, query_count, data, data_count, result, k);
}