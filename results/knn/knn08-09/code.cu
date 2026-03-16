#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace knn_detail {

// Fixed warp size mandated by CUDA on NVIDIA GPUs.
constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// Per-k compile-time parameters.
// - REGS_PER_THREAD: each lane stores k/32 consecutive neighbors in registers.
// - SHARED_STRIDE: one extra padding slot per lane-chunk for k > 32 to avoid
//   worst-case 32-way bank conflicts when all lanes access the same register slot.
// - WARP_BUF_ELEMS: physical shared-memory footprint of one warp-local candidate buffer.
// - TILE_POINTS: block-wide data tile cached in shared memory. Larger tiles are used
//   when k is small; for k >= 512 the tile is reduced to preserve occupancy.
template <int K>
struct KnnTraits {
    static_assert(K >= 32 && K <= 1024, "k must be in [32, 1024]");
    static_assert((K & (K - 1)) == 0, "k must be a power of two");
    static_assert((K % kWarpSize) == 0, "k must be divisible by 32");

    static constexpr int REGS_PER_THREAD = K / kWarpSize;
    static constexpr int SHARED_STRIDE = (REGS_PER_THREAD == 1) ? 1 : (REGS_PER_THREAD + 1);
    static constexpr int WARP_BUF_ELEMS = kWarpSize * SHARED_STRIDE;
    static constexpr int TILE_POINTS = (K >= 512) ? 1024 : 2048;
};

// Swap one (index, distance) pair.
__device__ __forceinline__ void swap_pair(float &a_dist, int &a_idx, float &b_dist, int &b_idx) {
    const float td = a_dist;
    a_dist = b_dist;
    b_dist = td;

    const int ti = a_idx;
    a_idx = b_idx;
    b_idx = ti;
}

// Select the other element if it is smaller/larger in distance.
// Ties are intentionally left unresolved, as permitted by the specification.
__device__ __forceinline__ void select_other_by_distance(
    float &self_dist, int &self_idx,
    const float other_dist, const int other_idx,
    const bool take_min)
{
    if (take_min) {
        if (other_dist < self_dist) {
            self_dist = other_dist;
            self_idx = other_idx;
        }
    } else {
        if (self_dist < other_dist) {
            self_dist = other_dist;
            self_idx = other_idx;
        }
    }
}

// Map a logical buffer position in [0, K) to the padded shared-memory layout.
template <int K>
__device__ __forceinline__ int shared_phys_index_from_logical(const int logical) {
    constexpr int M = KnnTraits<K>::REGS_PER_THREAD;
    constexpr int STRIDE = KnnTraits<K>::SHARED_STRIDE;

    const int lane_chunk = logical / M;
    const int off = logical - lane_chunk * M;
    return lane_chunk * STRIDE + off;
}

// One Bitonic Sort stage for a k-element array distributed over a warp:
// each lane owns M = k/32 consecutive elements.
template <int K>
__device__ __forceinline__ void bitonic_sort_stage(
    float (&dist)[KnnTraits<K>::REGS_PER_THREAD],
    int (&idx)[KnnTraits<K>::REGS_PER_THREAD],
    const int stage,
    const int j,
    const int lane)
{
    constexpr int M = KnnTraits<K>::REGS_PER_THREAD;
    const int base = lane * M;

    if (j < M) {
        // Intra-lane compare-exchange: both elements are already in registers.
        #pragma unroll
        for (int off = 0; off < M; ++off) {
            const int partner = off ^ j;
            if (partner > off) {
                const int logical = base + off;
                const bool ascending = ((logical & stage) == 0);

                if (ascending) {
                    if (dist[partner] < dist[off]) {
                        swap_pair(dist[off], idx[off], dist[partner], idx[partner]);
                    }
                } else {
                    if (dist[off] < dist[partner]) {
                        swap_pair(dist[off], idx[off], dist[partner], idx[partner]);
                    }
                }
            }
        }
    } else {
        // Inter-lane compare-exchange: partner element has the same local register slot.
        const int lane_delta = j / M;
        const int partner_lane = lane ^ lane_delta;
        const bool lower_lane = (lane < partner_lane);

        #pragma unroll
        for (int off = 0; off < M; ++off) {
            const float other_dist = __shfl_xor_sync(kFullMask, dist[off], lane_delta);
            const int other_idx = __shfl_xor_sync(kFullMask, idx[off], lane_delta);

            const int logical = base + off;
            const bool ascending = ((logical & stage) == 0);
            const bool take_min = ascending ? lower_lane : !lower_lane;

            select_other_by_distance(dist[off], idx[off], other_dist, other_idx, take_min);
        }
    }
}

// Full Bitonic Sort in ascending order.
template <int K>
__device__ __forceinline__ void bitonic_sort_ascending(
    float (&dist)[KnnTraits<K>::REGS_PER_THREAD],
    int (&idx)[KnnTraits<K>::REGS_PER_THREAD],
    const int lane)
{
    #pragma unroll
    for (int stage = 2; stage <= K; stage <<= 1) {
        #pragma unroll
        for (int j = stage >> 1; j > 0; j >>= 1) {
            bitonic_sort_stage<K>(dist, idx, stage, j, lane);
        }
    }
}

// Bitonic merge of an already bitonic sequence into ascending order.
template <int K>
__device__ __forceinline__ void bitonic_merge_ascending(
    float (&dist)[KnnTraits<K>::REGS_PER_THREAD],
    int (&idx)[KnnTraits<K>::REGS_PER_THREAD],
    const int lane)
{
    constexpr int M = KnnTraits<K>::REGS_PER_THREAD;

    if (M == 1) {
        // Specialized fast path for k == 32.
        #pragma unroll
        for (int j = K >> 1; j > 0; j >>= 1) {
            const float other_dist = __shfl_xor_sync(kFullMask, dist[0], j);
            const int other_idx = __shfl_xor_sync(kFullMask, idx[0], j);
            const bool take_min = (lane < (lane ^ j));
            select_other_by_distance(dist[0], idx[0], other_dist, other_idx, take_min);
        }
        return;
    }

    #pragma unroll
    for (int j = K >> 1; j > 0; j >>= 1) {
        if (j < M) {
            // Intra-lane ascending merge.
            #pragma unroll
            for (int off = 0; off < M; ++off) {
                const int partner = off ^ j;
                if (partner > off) {
                    if (dist[partner] < dist[off]) {
                        swap_pair(dist[off], idx[off], dist[partner], idx[partner]);
                    }
                }
            }
        } else {
            // Inter-lane ascending merge.
            const int lane_delta = j / M;
            const int partner_lane = lane ^ lane_delta;
            const bool lower_lane = (lane < partner_lane);

            #pragma unroll
            for (int off = 0; off < M; ++off) {
                const float other_dist = __shfl_xor_sync(kFullMask, dist[off], lane_delta);
                const int other_idx = __shfl_xor_sync(kFullMask, idx[off], lane_delta);

                select_other_by_distance(dist[off], idx[off], other_dist, other_idx, lower_lane);
            }
        }
    }
}

// Merge the shared candidate buffer with the sorted intermediate result in registers.
// Required sequence:
// 1) swap buffer and result so the buffer is in registers,
// 2) sort the buffer,
// 3) build the top-k bitonic sequence via min(buffer[i], result[k-i-1]),
// 4) bitonic-merge that sequence into a new sorted intermediate result.
template <int K>
__device__ __forceinline__ void merge_buffer_with_result(
    float (&best_dist)[KnnTraits<K>::REGS_PER_THREAD],
    int (&best_idx)[KnnTraits<K>::REGS_PER_THREAD],
    float &max_distance,
    const int buffer_count,
    float *const s_buf_dist,
    int *const s_buf_idx,
    const int lane)
{
    constexpr int M = KnnTraits<K>::REGS_PER_THREAD;
    constexpr int STRIDE = KnnTraits<K>::SHARED_STRIDE;

    __syncwarp(kFullMask);

    // Step 1: move the candidate buffer into registers and spill the current result
    // back into the same shared-memory area. Unused candidate slots are padded with +inf.
    #pragma unroll
    for (int off = 0; off < M; ++off) {
        const int logical = lane * M + off;
        const int phys = lane * STRIDE + off;

        const float cand_dist = (logical < buffer_count) ? s_buf_dist[phys] : CUDART_INF_F;
        const int cand_idx = (logical < buffer_count) ? s_buf_idx[phys] : -1;

        s_buf_dist[phys] = best_dist[off];
        s_buf_idx[phys] = best_idx[off];

        best_dist[off] = cand_dist;
        best_idx[off] = cand_idx;
    }

    __syncwarp(kFullMask);

    // Step 2: sort the buffer in ascending order.
    bitonic_sort_ascending<K>(best_dist, best_idx, lane);

    // Step 3: merge sorted buffer (ascending) with previous result (ascending, read in reverse)
    // into a bitonic sequence containing the k smallest elements of the union.
    #pragma unroll
    for (int off = 0; off < M; ++off) {
        const int phys_rev = (kWarpSize - 1 - lane) * STRIDE + (M - 1 - off);
        const float other_dist = s_buf_dist[phys_rev];
        const int other_idx = s_buf_idx[phys_rev];

        if (other_dist < best_dist[off]) {
            best_dist[off] = other_dist;
            best_idx[off] = other_idx;
        }
    }

    // Step 4: the sequence is bitonic; a bitonic merge restores ascending order.
    bitonic_merge_ascending<K>(best_dist, best_idx, lane);

    // Refresh the pruning threshold with the k-th neighbor distance.
    const float tail = (lane == (kWarpSize - 1)) ? best_dist[M - 1] : 0.0f;
    max_distance = __shfl_sync(kFullMask, tail, kWarpSize - 1);
}

// Append one warp ballot worth of candidates to the shared buffer, and flush+merge
// whenever the buffer reaches size k.
template <int K>
__device__ __forceinline__ void append_candidates_mask(
    const float cand_dist,
    const int cand_idx,
    const unsigned mask,
    float (&best_dist)[KnnTraits<K>::REGS_PER_THREAD],
    int (&best_idx)[KnnTraits<K>::REGS_PER_THREAD],
    float *const s_buf_dist,
    int *const s_buf_idx,
    const int lane,
    const unsigned lane_mask_lt,
    int &buffer_count,
    float &max_distance)
{
    if (mask == 0u) {
        return;
    }

    const bool is_candidate = ((mask >> lane) & 1u) != 0u;
    const int rank = __popc(mask & lane_mask_lt);
    const int num = __popc(mask);
    const int old_count = buffer_count;

    if (old_count + num < K) {
        // Fast path: the warp-local buffer still has space.
        if (is_candidate) {
            const int logical = old_count + rank;
            const int phys = shared_phys_index_from_logical<K>(logical);
            s_buf_dist[phys] = cand_dist;
            s_buf_idx[phys] = cand_idx;
        }

        __syncwarp(kFullMask);
        buffer_count = old_count + num;
        return;
    }

    // Fill the remaining slots, merge, then re-test any overflow candidates against
    // the tightened max_distance before storing them in the now-empty buffer.
    const int fill = K - old_count;

    if (is_candidate && rank < fill) {
        const int logical = old_count + rank;
        const int phys = shared_phys_index_from_logical<K>(logical);
        s_buf_dist[phys] = cand_dist;
        s_buf_idx[phys] = cand_idx;
    }

    __syncwarp(kFullMask);

    buffer_count = K;
    merge_buffer_with_result<K>(best_dist, best_idx, max_distance, buffer_count, s_buf_dist, s_buf_idx, lane);
    buffer_count = 0;

    if (old_count + num > K) {
        const bool keep_remainder = is_candidate && (rank >= fill) && (cand_dist < max_distance);
        const unsigned rem_mask = __ballot_sync(kFullMask, keep_remainder);
        const int rem_num = __popc(rem_mask);

        if (keep_remainder) {
            const int rem_rank = __popc(rem_mask & lane_mask_lt);
            const int phys = shared_phys_index_from_logical<K>(rem_rank);
            s_buf_dist[phys] = cand_dist;
            s_buf_idx[phys] = cand_idx;
        }

        __syncwarp(kFullMask);
        buffer_count = rem_num;
    }
}

// One warp computes the k-NN result for one query.
// The block cooperatively loads data tiles into shared memory for reuse across all warps.
template <int K, int WARPS_PER_BLOCK>
__launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
__global__ void knn2d_kernel(
    const float2 * __restrict__ query,
    const int query_count,
    const float2 * __restrict__ data,
    const int data_count,
    std::pair<int, float> * __restrict__ result)
{
    constexpr int M = KnnTraits<K>::REGS_PER_THREAD;
    constexpr int TILE_POINTS = KnnTraits<K>::TILE_POINTS;
    constexpr int WARP_BUF_ELEMS = KnnTraits<K>::WARP_BUF_ELEMS;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    const bool valid_query = (query_idx < query_count);

    // Use a 16-byte aligned dynamic shared-memory base.
    extern __shared__ uint4 smem_vec[];
    unsigned char *const smem_raw = reinterpret_cast<unsigned char *>(smem_vec);

    float2 *const s_data = reinterpret_cast<float2 *>(smem_raw);
    float *const s_buf_dist_all =
        reinterpret_cast<float *>(smem_raw + static_cast<std::size_t>(TILE_POINTS) * sizeof(float2));
    int *const s_buf_idx_all =
        reinterpret_cast<int *>(reinterpret_cast<unsigned char *>(s_buf_dist_all) +
                                static_cast<std::size_t>(WARPS_PER_BLOCK) * WARP_BUF_ELEMS * sizeof(float));

    float *const s_buf_dist = s_buf_dist_all + warp_in_block * WARP_BUF_ELEMS;
    int *const s_buf_idx = s_buf_idx_all + warp_in_block * WARP_BUF_ELEMS;

    // Lane-local prefix mask for ballot compaction.
    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    // Load query point once per warp and broadcast it.
    float qx = 0.0f;
    float qy = 0.0f;
    if (valid_query && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Sorted intermediate result: each lane owns M consecutive neighbors in registers.
    float best_dist[M];
    int best_idx[M];
    #pragma unroll
    for (int off = 0; off < M; ++off) {
        best_dist[off] = CUDART_INF_F;
        best_idx[off] = -1;
    }

    float max_distance = CUDART_INF_F;
    int buffer_count = 0;

    // Iterate over the dataset in block-cached tiles.
    for (int batch_start = 0; batch_start < data_count; batch_start += TILE_POINTS) {
        int batch_size = data_count - batch_start;
        if (batch_size > TILE_POINTS) {
            batch_size = TILE_POINTS;
        }

        // Cooperative, coalesced tile load.
        for (int t = threadIdx.x; t < batch_size; t += blockDim.x) {
            s_data[t] = data[batch_start + t];
        }

        __syncthreads();

        if (valid_query) {
            // Each warp consumes the tile 32 points at a time so that one ballot compactly
            // appends one warp-width group of candidates.
            #pragma unroll 1
            for (int tile_off = 0; tile_off < batch_size; tile_off += kWarpSize) {
                const int local = tile_off + lane;
                const bool active = (local < batch_size);

                float dist = CUDART_INF_F;
                int cand_global_idx = 0;

                if (active) {
                    const float2 p = s_data[local];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);
                    cand_global_idx = batch_start + local;
                }

                const unsigned mask = __ballot_sync(kFullMask, active && (dist < max_distance));

                if (mask != 0u) {
                    append_candidates_mask<K>(
                        dist, cand_global_idx, mask,
                        best_dist, best_idx,
                        s_buf_dist, s_buf_idx,
                        lane, lane_mask_lt,
                        buffer_count, max_distance);
                }
            }
        }

        // Ensure all warps are done with the tile before overwriting shared memory.
        __syncthreads();
    }

    if (valid_query) {
        // Final partial flush, if any candidates remain buffered.
        if (buffer_count > 0) {
            merge_buffer_with_result<K>(best_dist, best_idx, max_distance, buffer_count, s_buf_dist, s_buf_idx, lane);
        }

        // Write the sorted result back in the required row-major layout.
        std::pair<int, float> *const out =
            result + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int off = 0; off < M; ++off) {
            const int logical = lane * M + off;
            out[logical].first = best_idx[off];
            out[logical].second = best_dist[off];
        }
    }
}

template <int K, int WARPS_PER_BLOCK>
constexpr std::size_t shared_bytes_for_kernel() {
    return static_cast<std::size_t>(KnnTraits<K>::TILE_POINTS) * sizeof(float2) +
           static_cast<std::size_t>(WARPS_PER_BLOCK) * KnnTraits<K>::WARP_BUF_ELEMS * (sizeof(float) + sizeof(int));
}

template <int K, int WARPS_PER_BLOCK>
inline void launch_knn_kernel(
    const float2 *query,
    const int query_count,
    const float2 *data,
    const int data_count,
    std::pair<int, float> *result)
{
    const std::size_t shmem_bytes = shared_bytes_for_kernel<K, WARPS_PER_BLOCK>();

    // Opt in to the required dynamic shared memory and bias the SM partitioning
    // toward shared memory, which is critical for tile reuse and warp-local buffers.
    (void)cudaFuncSetAttribute(
        knn2d_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes));

    (void)cudaFuncSetAttribute(
        knn2d_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const dim3 block(WARPS_PER_BLOCK * kWarpSize);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn2d_kernel<K, WARPS_PER_BLOCK><<<grid, block, shmem_bytes>>>(
        query, query_count, data, data_count, result);
}

template <int K>
inline void select_and_launch(
    const float2 *query,
    const int query_count,
    const float2 *data,
    const int data_count,
    std::pair<int, float> *result,
    const int sm_count,
    const int max_optin_shared)
{
    // Choose the largest warp-group count per block that still leaves enough blocks
    // to cover all SMs. This maximizes data-tile reuse within a block while avoiding
    // obvious under-subscription when the query count is only moderately large.
    const int blocks16 = (query_count + 15) / 16;
    const int blocks8 = (query_count + 7) / 8;

    if (shared_bytes_for_kernel<K, 16>() <= static_cast<std::size_t>(max_optin_shared) &&
        blocks16 >= sm_count) {
        launch_knn_kernel<K, 16>(query, query_count, data, data_count, result);
    } else if (shared_bytes_for_kernel<K, 8>() <= static_cast<std::size_t>(max_optin_shared) &&
               blocks8 >= sm_count) {
        launch_knn_kernel<K, 8>(query, query_count, data, data_count, result);
    } else {
        launch_knn_kernel<K, 4>(query, query_count, data, data_count, result);
    }
}

} // namespace knn_detail

void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k)
{
    if (query_count <= 0) {
        return;
    }

    int device = 0;
    int sm_count = 0;
    int max_optin_shared = 0;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    (void)cudaDeviceGetAttribute(&max_optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Conservative fallbacks match the A100-class minimum among the intended targets.
    if (sm_count <= 0) {
        sm_count = 1;
    }
    if (max_optin_shared <= 0) {
        max_optin_shared = 163840;
    }

    switch (k) {
        case 32:
            knn_detail::select_and_launch<32>(query, query_count, data, data_count, result, sm_count, max_optin_shared);
            break;
        case 64:
            knn_detail::select_and_launch<64>(query, query_count, data, data_count, result, sm_count, max_optin_shared);
            break;
        case 128:
            knn_detail::select_and_launch<128>(query, query_count, data, data_count, result, sm_count, max_optin_shared);
            break;
        case 256:
            knn_detail::select_and_launch<256>(query, query_count, data, data_count, result, sm_count, max_optin_shared);
            break;
        case 512:
            knn_detail::select_and_launch<512>(query, query_count, data, data_count, result, sm_count, max_optin_shared);
            break;
        case 1024:
            knn_detail::select_and_launch<1024>(query, query_count, data, data_count, result, sm_count, max_optin_shared);
            break;
        default:
            // Inputs are specified to be valid; this path is only a defensive guard.
            break;
    }
}