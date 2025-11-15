// Optimized CUDA implementation of k-NN for 2D points.
// Each query is processed by a single warp (32 threads).
// The kernel maintains a per-query intermediate top-k result in registers
// (distributed across the 32 threads of the warp) and a per-query candidate
// buffer of size k in shared memory. The data points are processed in tiles
// loaded into shared memory by the entire block. Candidate buffers are
// periodically merged into the intermediate result using a warp-cooperative
// bitonic sort over up to 2k elements.
//
// This code targets modern NVIDIA data center GPUs (e.g., A100/H100) and
// assumes compilation with a recent CUDA toolkit.

#include <cuda_runtime.h>
#include <utility>
#include <limits>

// Convenience alias for the result pair type.
using PairIF = std::pair<int, float>;

// Constants for kernel configuration.
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_BLOCK = 4;       // 4 warps per block -> 128 threads
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
// Tile size in shared memory for caching data points.
// 4096 points * 8 bytes = 32 KB for the tile.
static constexpr int TILE_POINTS = 4096;

// Warp-synchronous bitonic sort of N elements in shared memory, ascending order by dist.
// N must be a power of two. The sort is performed cooperatively by a single warp.
// Each thread processes elements with indices i = lane + t * WARP_SIZE.
template <int N>
__device__ __forceinline__
void bitonic_sort_ascending(float *dist, int *idx)
{
    static_assert((N & (N - 1)) == 0, "N must be a power of two");

    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Standard bitonic sort network.
    for (int k = 2; k <= N; k <<= 1)
    {
        // For each stage, stride j halves until 1.
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            // Each thread processes a subset of indices.
            for (int i = lane; i < N; i += WARP_SIZE)
            {
                int ixj = i ^ j;
                if (ixj > i)
                {
                    bool up = ((i & k) == 0); // direction of comparison

                    float di = dist[i];
                    float dj = dist[ixj];
                    int ii = idx[i];
                    int ij = idx[ixj];

                    // For ascending order: when up==true, keep smaller at i.
                    if ((di > dj) == up)
                    {
                        dist[i] = dj;
                        dist[ixj] = di;
                        idx[i] = ij;
                        idx[ixj] = ii;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

// Merge the per-warp candidate buffer (size K) with the intermediate top-K result
// held in registers. The union of candidates and current result (up to 2K elements)
// is sorted via bitonic sort, and the best K elements are written back to the
// intermediate result. The k-th (worst) distance in the updated result is then
// stored in worst_dist.
//
// Template parameter K is the desired number of neighbors.
// - cand_count: number of valid entries currently in warp_dist[0..cand_count-1].
// - warp_dist/warp_idx: shared memory buffers of length 2K per warp:
//     * positions [0, K)   : candidate buffer
//     * positions [K, 2K)  : scratch for copying current result
// - res_dist/res_idx: register arrays holding the current intermediate result,
//   distributed across the warp; each thread owns K / WARP_SIZE entries.
template <int K>
__device__ __forceinline__
void merge_candidates_into_result(
    int cand_count,
    float *warp_dist,
    int   *warp_idx,
    float (&res_dist)[K / WARP_SIZE],
    int   (&res_idx)[K / WARP_SIZE],
    float &worst_dist)
{
    static_assert(K % WARP_SIZE == 0, "K must be a multiple of warp size (32)");
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const float INF = CUDART_INF_F;

    // Copy current intermediate result from registers into the second half
    // of the per-warp shared memory buffer: positions [K, 2K).
    #pragma unroll
    for (int i = 0; i < K / WARP_SIZE; ++i)
    {
        int pos = K + i * WARP_SIZE + lane;
        warp_dist[pos] = res_dist[i];
        warp_idx[pos]  = res_idx[i];
    }
    __syncwarp(FULL_MASK);

    // For candidate buffer positions [cand_count, K) that are not filled yet,
    // set them to "infinite" distance so they will sort to the end.
    for (int pos = lane + cand_count; pos < K; pos += WARP_SIZE)
    {
        warp_dist[pos] = INF;
        warp_idx[pos]  = -1;
    }
    __syncwarp(FULL_MASK);

    // Now warp_dist/warp_idx contain up to 2K elements:
    //   - warp_dist[0..K-1]   : candidates (valid up to cand_count)
    //   - warp_dist[K..2K-1]  : previous result
    // Some entries may be INF; bitonic sort will push them to the end.
    bitonic_sort_ascending<2 * K>(warp_dist, warp_idx);

    // After sorting ascending by distance, the first K elements are the new top-K.
    // Copy them back into the per-warp intermediate result arrays in registers.
    #pragma unroll
    for (int i = 0; i < K / WARP_SIZE; ++i)
    {
        int pos = i * WARP_SIZE + lane;
        res_dist[i] = warp_dist[pos];
        res_idx[i]  = warp_idx[pos];
    }

    // Compute the new worst_dist as the maximum distance among the K results.
    // Each thread computes a local max over its owned subset, then a warp-wide max.
    float local_max = res_dist[0];
    #pragma unroll
    for (int i = 1; i < K / WARP_SIZE; ++i)
    {
        local_max = fmaxf(local_max, res_dist[i]);
    }

    // Warp-wide reduction for maximum.
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(FULL_MASK, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    worst_dist = __shfl_sync(FULL_MASK, local_max, 0);
}

// Main k-NN kernel.
// Template parameter K is the number of neighbors to find per query (power of two,
// between 32 and 1024 inclusive, and divisible by 32).
// Each warp in a block handles one query, processing all data points in tiles.
template <int K>
__global__ void knn_kernel(
    const float2 *__restrict__ query,
    int query_count,
    const float2 *__restrict__ data,
    int data_count,
    PairIF *__restrict__ result)
{
    static_assert(K % WARP_SIZE == 0, "K must be a multiple of 32 (warp size)");

    extern __shared__ unsigned char smem[];
    // Shared memory layout:
    // [0, TILE_POINTS) float2 : cached data tile
    // followed by per-warp buffers for candidates + merge scratch:
    // float warp_dist[WARPS_PER_BLOCK][2*K];
    // int   warp_idx [WARPS_PER_BLOCK][2*K];
    float2 *tile = reinterpret_cast<float2 *>(smem);
    float  *all_warp_dist = reinterpret_cast<float *>(tile + TILE_POINTS);
    int    *all_warp_idx  = reinterpret_cast<int *>(all_warp_dist + WARPS_PER_BLOCK * 2 * K);

    const int thread_id = threadIdx.x;
    const int warp_id_in_block = thread_id / WARP_SIZE;
    const int lane = thread_id & (WARP_SIZE - 1);

    const unsigned FULL_MASK = 0xffffffffu;

    // Compute the global query index for this warp.
    const int block_first_query = blockIdx.x * WARPS_PER_BLOCK;
    if (block_first_query >= query_count)
    {
        return; // No queries assigned to this block.
    }
    const int query_idx = block_first_query + warp_id_in_block;
    const bool warp_active = (query_idx < query_count);

    // Per-warp shared memory slice for candidate/merge buffer.
    // warp_dist/warp_idx have length 2*K for this warp.
    float *warp_dist = all_warp_dist + warp_id_in_block * 2 * K;
    int   *warp_idx  = all_warp_idx  + warp_id_in_block * 2 * K;

    // Load the query point for this warp and broadcast within the warp.
    float2 q;
    if (warp_active)
    {
        if (lane == 0)
        {
            q = query[query_idx];
        }
        q.x = __shfl_sync(FULL_MASK, q.x, 0);
        q.y = __shfl_sync(FULL_MASK, q.y, 0);
    }

    // Per-warp intermediate top-K result stored in registers, distributed across lanes.
    // Each lane owns K / WARP_SIZE entries; global position = i * WARP_SIZE + lane.
    float res_dist[K / WARP_SIZE];
    int   res_idx[K / WARP_SIZE];

    const float INF = CUDART_INF_F;

    // Initialize intermediate result with infinite distances and invalid indices.
    #pragma unroll
    for (int i = 0; i < K / WARP_SIZE; ++i)
    {
        res_dist[i] = INF;
        res_idx[i]  = -1;
    }

    // Current k-th (worst) distance in the intermediate result.
    // Initially infinite so all candidates are accepted.
    float worst_dist = INF;

    // Per-warp candidate buffer occupancy (number of valid entries in warp_dist[0..cand_count-1]).
    int cand_count = 0;

    // Process the data points in tiles cached into shared memory.
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS)
    {
        int tile_size = data_count - tile_base;
        if (tile_size > TILE_POINTS)
        {
            tile_size = TILE_POINTS;
        }

        // Load tile of data points into shared memory with all threads in the block.
        for (int i = thread_id; i < tile_size; i += blockDim.x)
        {
            tile[i] = data[tile_base + i];
        }
        __syncthreads();

        if (warp_active)
        {
            // Number of iterations needed so that each warp processes the entire tile.
            const int iterations = (tile_size + WARP_SIZE - 1) / WARP_SIZE;

            // Each iteration: each lane processes at most one data point.
            for (int it = 0; it < iterations; ++it)
            {
                const int j = it * WARP_SIZE + lane;
                const bool inside = (j < tile_size);

                float d2 = 0.0f;
                int data_idx_global = -1;

                if (inside)
                {
                    float2 p = tile[j];
                    float dx = p.x - q.x;
                    float dy = p.y - q.y;
                    d2 = dx * dx + dy * dy; // squared Euclidean distance
                    data_idx_global = tile_base + j;
                }

                // Decide whether this point is a candidate based on current worst_dist.
                bool is_candidate = inside && (d2 < worst_dist);

                // Insert candidate into the per-warp buffer, flushing (merging) when full.
                while (true)
                {
                    // Identify lanes that have a candidate in this iteration.
                    unsigned mask = __ballot_sync(FULL_MASK, is_candidate);
                    int n_new = __popc(mask);

                    int cand_count_shared = __shfl_sync(FULL_MASK, cand_count, 0);

                    if (cand_count_shared + n_new <= K)
                    {
                        // Enough space in the candidate buffer for all new candidates.
                        // Each candidate lane obtains a unique position via prefix sum.
                        if (is_candidate)
                        {
                            unsigned lanes_before = mask & ((1u << lane) - 1u);
                            int offset = __popc(lanes_before);
                            int pos = cand_count_shared + offset;

                            warp_dist[pos] = d2;
                            warp_idx[pos]  = data_idx_global;
                        }

                        if (lane == 0)
                        {
                            cand_count = cand_count_shared + n_new;
                        }
                        break; // Done with this point.
                    }
                    else
                    {
                        // Candidate buffer would overflow; merge current candidates with
                        // intermediate result to free space, then reconsider this point.
                        if (cand_count_shared > 0)
                        {
                            merge_candidates_into_result<K>(
                                cand_count_shared,
                                warp_dist,
                                warp_idx,
                                res_dist,
                                res_idx,
                                worst_dist);
                        }
                        else
                        {
                            // This case should not normally occur because cand_count_shared
                            // must be > 0 if cand_count_shared + n_new > K and n_new <= WARP_SIZE <= K.
                            // Still, handle it safely by not merging and proceeding.
                        }

                        if (lane == 0)
                        {
                            cand_count = 0;
                        }

                        // Re-evaluate candidacy with the updated worst_dist.
                        is_candidate = inside && (d2 < worst_dist);
                        // Loop back and attempt insertion again; now buffer is empty,
                        // and since n_new <= WARP_SIZE <= K, insertion will succeed.
                    }
                } // end while candidate insertion
            } // end iterations over tile
        } // if warp_active

        // Ensure all warps have finished using the current tile before loading the next one.
        __syncthreads();
    } // end tile loop

    if (warp_active)
    {
        // After processing all tiles, merge any remaining candidates in the buffer.
        int final_cand_count = __shfl_sync(FULL_MASK, cand_count, 0);
        if (final_cand_count > 0)
        {
            merge_candidates_into_result<K>(
                final_cand_count,
                warp_dist,
                warp_idx,
                res_dist,
                res_idx,
                worst_dist);
        }

        // Write the final sorted top-K results for this query to global memory.
        // res_dist/res_idx contain the K smallest distances/indices in ascending order.
        const int out_base = query_idx * K;

        #pragma unroll
        for (int i = 0; i < K / WARP_SIZE; ++i)
        {
            int pos = i * WARP_SIZE + lane;
            if (pos < K)
            {
                PairIF p;
                p.first  = res_idx[i];
                p.second = res_dist[i];
                result[out_base + pos] = p;
            }
        }
    }
}

// Helper to launch the kernel for a given K at runtime.
template <int K>
void launch_knn_kernel(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    PairIF *result)
{
    // Grid and block configuration: each warp handles one query.
    const int num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    const dim3 grid_dim(num_blocks, 1, 1);

    // Shared memory size:
    //   TILE_POINTS * sizeof(float2) for the data tile
    // + WARPS_PER_BLOCK * 2 * K * (sizeof(float) + sizeof(int)) for per-warp buffers.
    const size_t shared_mem_bytes =
        TILE_POINTS * sizeof(float2)
        + static_cast<size_t>(WARPS_PER_BLOCK) * 2 * K * (sizeof(float) + sizeof(int));

    knn_kernel<K><<<grid_dim, block_dim, shared_mem_bytes>>>(
        query, query_count, data, data_count, result);
}

// Public C++ interface.
// query       : pointer to device array of query_count float2 points.
// data        : pointer to device array of data_count float2 points.
// result      : pointer to device array of size query_count * k, where each element is
//               a std::pair<int, float> storing (index, squared_distance).
// k           : number of nearest neighbors (power of two between 32 and 1024 inclusive).
void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    PairIF *result,
    int k)
{
    // Dispatch to a kernel specialization based on k.
    // Supported values of k: 32, 64, 128, 256, 512, 1024.
    switch (k)
    {
        case 32:
            launch_knn_kernel<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            launch_knn_kernel<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            launch_knn_kernel<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            launch_knn_kernel<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            launch_knn_kernel<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            launch_knn_kernel<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem statement guarantees that k is always a power of two between
            // 32 and 1024 inclusive, so this default path should never be taken.
            // For safety, we do nothing here.
            break;
    }
}