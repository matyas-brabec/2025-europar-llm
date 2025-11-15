#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Simple POD type with the same layout as std::pair<int,float>
struct PairIF {
    int   first;
    float second;
};

/**
 * Warp-wide bitonic sort of K elements, where each thread in the warp owns
 * ITEMS_PER_THREAD = K / 32 consecutive elements in registers.
 *
 * The global logical index of an element is:
 *   i = lane * ITEMS_PER_THREAD + local
 *
 * For a given bitonic (k, j) stage, comparisons may be:
 *   - Within a single thread (when j < ITEMS_PER_THREAD)
 *   - Across threads (when j >= ITEMS_PER_THREAD), but always between
 *     elements with the same local index in each thread.
 */
template <int K>
__device__ __forceinline__
void bitonic_sort_k(float (&dist)[K / 32], int (&idx)[K / 32]) {
    constexpr unsigned int FULL_MASK       = 0xffffffffu;
    constexpr int          WARP_SIZE       = 32;
    constexpr int          ITEMS_PER_THREAD = K / WARP_SIZE;

    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Standard bitonic sort network over K = WARP_SIZE * ITEMS_PER_THREAD elements
    #pragma unroll
    for (int seqLen = 2; seqLen <= K; seqLen <<= 1) {
        #pragma unroll
        for (int j = seqLen >> 1; j > 0; j >>= 1) {
            if (j < ITEMS_PER_THREAD) {
                // All comparisons for this (seqLen, j) step are within a thread
                #pragma unroll
                for (int local = 0; local < ITEMS_PER_THREAD; ++local) {
                    int i            = lane * ITEMS_PER_THREAD + local;
                    int partnerLocal = local ^ j;

                    // Only process each pair once
                    if (partnerLocal > local && partnerLocal < ITEMS_PER_THREAD) {
                        int l = lane * ITEMS_PER_THREAD + partnerLocal;

                        bool up           = ((i & seqLen) == 0);
                        bool smallerIndex = ((i & j)     == 0);

                        float a    = dist[local];
                        float b    = dist[partnerLocal];
                        int   ia   = idx[local];
                        int   ib   = idx[partnerLocal];

                        float minVal, maxVal;
                        int   minIdx, maxIdx;
                        if (a <= b) {
                            minVal = a; maxVal = b;
                            minIdx = ia; maxIdx = ib;
                        } else {
                            minVal = b; maxVal = a;
                            minIdx = ib; maxIdx = ia;
                        }

                        // If (smallerIndex == up) we keep the min at i, max at l; otherwise swapped
                        if (smallerIndex == up) {
                            dist[local]        = minVal;
                            idx[local]         = minIdx;
                            dist[partnerLocal] = maxVal;
                            idx[partnerLocal]  = maxIdx;
                        } else {
                            dist[local]        = maxVal;
                            idx[local]         = maxIdx;
                            dist[partnerLocal] = minVal;
                            idx[partnerLocal]  = minIdx;
                        }
                    }
                }
            } else {
                // Comparisons across threads. For j >= ITEMS_PER_THREAD,
                // j is a multiple of ITEMS_PER_THREAD. The partner lane offset is:
                int partnerOffset = j / ITEMS_PER_THREAD;

                #pragma unroll
                for (int local = 0; local < ITEMS_PER_THREAD; ++local) {
                    int   i       = lane * ITEMS_PER_THREAD + local;
                    float selfVal = dist[local];
                    int   selfIdx = idx[local];

                    // Fetch the partner element at the same local index in a different lane
                    float otherVal = __shfl_xor_sync(FULL_MASK, selfVal, partnerOffset);
                    int   otherIdx = __shfl_xor_sync(FULL_MASK, selfIdx, partnerOffset);

                    bool up           = ((i & seqLen) == 0);
                    bool smallerIndex = ((i & j)      == 0);

                    float minVal, maxVal;
                    int   minIdx, maxIdx;
                    if (selfVal <= otherVal) {
                        minVal = selfVal; maxVal = otherVal;
                        minIdx = selfIdx; maxIdx = otherIdx;
                    } else {
                        minVal = otherVal; maxVal = selfVal;
                        minIdx = otherIdx; maxIdx = selfIdx;
                    }

                    // Compute the new value for index i only; index (i^j) is handled in its own thread
                    if (smallerIndex == up) {
                        dist[local] = minVal;
                        idx[local]  = minIdx;
                    } else {
                        dist[local] = maxVal;
                        idx[local]  = maxIdx;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
    __syncwarp(FULL_MASK);
}

/**
 * Merge the current intermediate result (stored in registers across the warp)
 * with the per-warp candidate buffer in shared memory.
 *
 * Steps (per the specification):
 *  0. The intermediate result in registers is sorted ascending.
 *  1. Swap the contents of the buffer and the intermediate result so that
 *     the buffer is now in registers.
 *  2. Sort the buffer (now in registers) in ascending order using bitonic sort.
 *  3. Merge the buffer and the intermediate result in shared memory into the
 *     registers: for each global position i, take the minimum of
 *        buffer[i] and intermediate[K - 1 - i].
 *     This produces a bitonic sequence of length K in registers.
 *  4. Sort the merged sequence in ascending order using bitonic sort.
 *     The result is the updated intermediate result.
 *
 * The candidate buffer may contain fewer than K elements; the remaining
 * positions are filled with values >= current max_distance so that they
 * cannot influence the top-K result.
 */
template <int K>
__device__ __forceinline__
void merge_buffer_with_intermediate(float (&dist_reg)[K / 32],
                                    int   (&idx_reg)[K / 32],
                                    float *cand_dist,
                                    int   *cand_idx,
                                    int    cand_count,
                                    float &max_distance) {
    constexpr unsigned int FULL_MASK        = 0xffffffffu;
    constexpr int          WARP_SIZE        = 32;
    constexpr int          ITEMS_PER_THREAD = K / WARP_SIZE;

    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Fill unused slots in the candidate buffer with distances >= current max_distance
    // so that they cannot enter the top-K after merging.
    if (cand_count < K) {
        int num_to_fill = K - cand_count;
        for (int t = lane; t < num_to_fill; t += WARP_SIZE) {
            cand_dist[cand_count + t] = max_distance;
            cand_idx[cand_count + t]  = -1;
        }
    }
    __syncwarp(FULL_MASK);

    // Swap the K elements between registers (intermediate result) and shared memory (buffer)
    #pragma unroll
    for (int local = 0; local < ITEMS_PER_THREAD; ++local) {
        int   global_index = lane * ITEMS_PER_THREAD + local;
        float tmp_dist     = dist_reg[local];
        int   tmp_idx      = idx_reg[local];

        float buf_dist     = cand_dist[global_index];
        int   buf_idx      = cand_idx[global_index];

        dist_reg[local]    = buf_dist;
        idx_reg[local]     = buf_idx;

        cand_dist[global_index] = tmp_dist;
        cand_idx[global_index]  = tmp_idx;
    }
    __syncwarp(FULL_MASK);

    // Sort the buffer now stored in registers
    bitonic_sort_k<K>(dist_reg, idx_reg);
    __syncwarp(FULL_MASK);

    // Merge: for each position i, keep the minimum of
    //   buffer[i] (in registers) and intermediate[K - 1 - i] (in shared memory)
    #pragma unroll
    for (int local = 0; local < ITEMS_PER_THREAD; ++local) {
        int   global_index = lane * ITEMS_PER_THREAD + local;
        int   mirror_index = K - 1 - global_index;

        float a_dist = dist_reg[local];
        int   a_idx  = idx_reg[local];

        float b_dist = cand_dist[mirror_index];
        int   b_idx  = cand_idx[mirror_index];

        if (b_dist < a_dist) {
            dist_reg[local] = b_dist;
            idx_reg[local]  = b_idx;
        }
    }
    __syncwarp(FULL_MASK);

    // The registers now contain a bitonic sequence of length K; sort it
    bitonic_sort_k<K>(dist_reg, idx_reg);
    __syncwarp(FULL_MASK);

    // Update max_distance as the k-th (last) element of the sorted intermediate result
    float kth = dist_reg[ITEMS_PER_THREAD - 1];
    kth       = __shfl_sync(FULL_MASK, kth, WARP_SIZE - 1);
    max_distance = kth;
}

/**
 * Main k-NN kernel: one warp processes one query point.
 * Each warp maintains a distributed top-K list in registers, and a
 * per-warp candidate buffer of size K in shared memory.
 */
template <int K>
__global__
void knn_kernel(const float2 *__restrict__ query,
                int                         query_count,
                const float2 *__restrict__ data,
                int                         data_count,
                PairIF      *__restrict__  result) {
    constexpr unsigned int FULL_MASK        = 0xffffffffu;
    constexpr int          WARP_SIZE        = 32;
    constexpr int          WARPS_PER_BLOCK  = 4;
    constexpr int          THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int          ITEMS_PER_THREAD = K / WARP_SIZE;
    // Size of data tile loaded into shared memory per block
    constexpr int          DATA_TILE_SIZE   = 4096;

    // Shared memory:
    //  - sh_data:   cached data points for the current tile
    //  - sh_cand_*: per-warp candidate buffers of size K
    __shared__ float2 sh_data[DATA_TILE_SIZE];
    __shared__ int    sh_cand_idx[WARPS_PER_BLOCK * K];
    __shared__ float  sh_cand_dist[WARPS_PER_BLOCK * K];

    const int thread_id   = threadIdx.x;
    const int warp_id     = thread_id / WARP_SIZE;      // warp index within block
    const int lane        = thread_id & (WARP_SIZE - 1);
    const int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    const bool warp_active = (global_warp < query_count);

    // Per-warp views into shared candidate buffers
    int   *cand_idx  = sh_cand_idx  + warp_id * K;
    float *cand_dist = sh_cand_dist + warp_id * K;

    // Per-thread top-K intermediate result stored in registers
    float dist_reg[ITEMS_PER_THREAD];
    int   idx_reg[ITEMS_PER_THREAD];

    // Initialize intermediate result with "infinite" distances
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        dist_reg[i] = FLT_MAX;
        idx_reg[i]  = -1;
    }

    // Current distance threshold: distance of the k-th nearest neighbor
    float max_distance = FLT_MAX;

    // Number of candidates currently stored in this warp's shared buffer
    int cand_count = 0;

    // Query point for this warp
    float2 q;
    if (warp_active) {
        q = query[global_warp];
    }

    // Iterate over data points in tiles cached in shared memory
    for (int tile_start = 0; tile_start < data_count; tile_start += DATA_TILE_SIZE) {
        int tile_size = data_count - tile_start;
        if (tile_size > DATA_TILE_SIZE) tile_size = DATA_TILE_SIZE;

        // Load tile into shared memory cooperatively by the whole block
        for (int idx = thread_id; idx < tile_size; idx += THREADS_PER_BLOCK) {
            sh_data[idx] = data[tile_start + idx];
        }
        __syncthreads();

        if (warp_active) {
            // Each warp processes all points in the tile for its own query
            for (int j = lane; j < tile_size; j += WARP_SIZE) {
                float2 p = sh_data[j];
                float dx = q.x - p.x;
                float dy = q.y - p.y;
                float dist = dx * dx + dy * dy;

                // Filter by current max_distance
                bool is_candidate = (dist < max_distance);
                unsigned int mask = __ballot_sync(FULL_MASK, is_candidate);
                int num_new       = __popc(mask);

                if (num_new > 0) {
                    int old_count = cand_count;
                    int new_total = old_count + num_new;

                    // If the buffer would overflow, merge existing candidates first,
                    // then re-evaluate the current distance against the updated max_distance.
                    if (new_total > K) {
                        if (old_count > 0) {
                            merge_buffer_with_intermediate<K>(
                                dist_reg, idx_reg,
                                cand_dist, cand_idx,
                                old_count, max_distance);
                        }
                        cand_count = 0;

                        // Recheck with updated max_distance
                        is_candidate = (dist < max_distance);
                        mask         = __ballot_sync(FULL_MASK, is_candidate);
                        num_new      = __popc(mask);

                        if (num_new == 0) {
                            continue;
                        }

                        old_count = 0;
                        new_total = num_new;
                    }

                    int warp_offset = old_count;

                    // Reserve positions [warp_offset, warp_offset + num_new)
                    if (is_candidate) {
                        unsigned int lane_mask = mask & ((1u << lane) - 1u);
                        int prefix = __popc(lane_mask);
                        int pos    = warp_offset + prefix;

                        cand_idx[pos]  = tile_start + j; // global index of data point
                        cand_dist[pos] = dist;
                    }

                    cand_count = new_total;

                    // If the buffer is now full, merge it with the intermediate result
                    if (cand_count == K) {
                        merge_buffer_with_intermediate<K>(
                            dist_reg, idx_reg,
                            cand_dist, cand_idx,
                            cand_count, max_distance);
                        cand_count = 0;
                    }
                }
            }
        }

        __syncthreads();
    }

    // After processing all tiles, merge any remaining candidates
    if (warp_active && cand_count > 0) {
        merge_buffer_with_intermediate<K>(
            dist_reg, idx_reg,
            cand_dist, cand_idx,
            cand_count, max_distance);
        cand_count = 0;
    }

    // Write out the K nearest neighbors for this query
    if (warp_active) {
        #pragma unroll
        for (int local = 0; local < ITEMS_PER_THREAD; ++local) {
            int global_k_index = lane * ITEMS_PER_THREAD + local;
            int out_index      = global_warp * K + global_k_index;

            PairIF out;
            out.first  = idx_reg[local];
            out.second = dist_reg[local];

            result[out_index] = out;
        }
    }
}

/**
 * Host-side dispatcher. Launches a specialized kernel for each supported K.
 */
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k) {
    constexpr int WARP_SIZE        = 32;
    constexpr int WARPS_PER_BLOCK  = 4;
    constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

    // reinterpret std::pair<int,float>* as PairIF* (same memory layout)
    PairIF *result_cast = reinterpret_cast<PairIF *>(result);

    int num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 block_dim(THREADS_PER_BLOCK, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    switch (k) {
        case 32:
            knn_kernel<32><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result_cast);
            break;
        case 64:
            knn_kernel<64><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result_cast);
            break;
        case 128:
            knn_kernel<128><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result_cast);
            break;
        case 256:
            knn_kernel<256><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result_cast);
            break;
        case 512:
            knn_kernel<512><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result_cast);
            break;
        case 1024:
            knn_kernel<1024><<<grid_dim, block_dim>>>(
                query, query_count, data, data_count, result_cast);
            break;
        default:
            // Unsupported k (should not happen given the problem constraints)
            return;
    }

    // Synchronize to ensure completion before returning
    cudaDeviceSynchronize();
}