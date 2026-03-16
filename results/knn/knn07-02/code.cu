#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace {

// Warp-centric 2D k-NN for modern NVIDIA GPUs.
//
// Design choices tuned for A100/H100-class hardware:
// - One warp owns one query.
// - A block cooperatively stages a tile of `data` into shared memory.
// - The current top-K ("intermediate result") is kept privately in registers,
//   distributed across the 32 lanes of the owning warp.
// - Each warp also owns a shared-memory candidate buffer of size K plus a
//   shared counter. Distances below the current K-th distance are appended to
//   that buffer via atomicAdd (warp-aggregated to cut atomic traffic).
// - When the buffer becomes full, it is merged into the private top-K with the
//   exact bitonic-sort / bitonic-merge procedure requested in the prompt.
// - K is template-specialized for the six valid powers of two, so per-lane
//   storage and the key loops are compile-time constants.
//
// Tile sizes:
// - K <= 512  -> 4096 staged points/block
// - K == 1024 -> 2048 staged points/block
//
// With 8 warps/block this keeps the shared-memory footprint in the ~35-80 KiB
// range, which is a good occupancy / reuse trade-off on Ampere/Hopper.
// Fallbacks to fewer warps/block are provided if the device reports a smaller
// dynamic shared-memory budget.

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ bool pair_greater(float da, int ia, float db, int ib) {
    return (da > db) || ((da == db) && (ia > ib));
}

__device__ __forceinline__ float squared_l2(float qx, float qy, const float2 &p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

// Select the lowest `n` set bits from `mask`.
// Any subset would work; lowest lanes keep rank computation trivial.
__device__ __forceinline__ unsigned select_lowest_n_bits(unsigned mask, int n) {
    unsigned out = 0u;
#pragma unroll
    for (int i = 0; i < kWarpSize; ++i) {
        if (i >= n) break;
        const int bit = __ffs(mask) - 1;
        out |= (1u << bit);
        mask &= (mask - 1u);
    }
    return out;
}

// After a merge, max_distance can only decrease. Re-filter the still-pending
// candidates so obviously-too-far points are dropped immediately.
__device__ __forceinline__ unsigned refresh_pending_mask(
    unsigned pending_mask,
    float lane_distance,
    float max_distance,
    int lane) {
    const bool pending = ((pending_mask >> lane) & 1u) != 0u;
    return __ballot_sync(kFullMask, pending && (lane_distance < max_distance));
}

// Bitonic sort of a warp-owned shared-memory array of power-of-two length N.
// Each lane processes the positions lane, lane+32, lane+64, ...
template <int N>
__device__ __forceinline__ void bitonic_sort_shared(float *dist, int *idx, int lane) {
    static_assert((N & (N - 1)) == 0, "Bitonic sort size must be a power of two");
    static_assert(N >= kWarpSize, "Sort size must be at least one warp");

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
            for (int pos = lane; pos < N; pos += kWarpSize) {
                const int partner = pos ^ j;
                if (partner > pos) {
                    const float di = dist[pos];
                    const float dj = dist[partner];
                    const int   ii = idx[pos];
                    const int   ij = idx[partner];

                    const bool ascending = ((pos & k) == 0);
                    const bool do_swap = ascending
                        ? pair_greater(di, ii, dj, ij)
                        : pair_less(di, ii, dj, ij);

                    if (do_swap) {
                        dist[pos] = dj;
                        dist[partner] = di;
                        idx[pos] = ij;
                        idx[partner] = ii;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge the warp-private sorted top-K in registers with the shared candidate
// buffer using the exact required sequence:
//
// 1) sort buffer ascending,
// 2) take pointwise min(buffer[i], best[K-1-i]) to build a bitonic sequence
//    containing the K smallest elements of both arrays,
// 3) bitonic-sort that merged sequence ascending.
//
// The candidate buffer is padded with (+inf, INT_MAX) so the same fixed-size
// bitonic sort works for both full and partially filled buffers.
template <int K>
__device__ __forceinline__ void merge_buffer_into_best(
    int *cand_count,
    int *cand_idx,
    float *cand_dist,
    int (&best_idx)[K / kWarpSize],
    float (&best_dist)[K / kWarpSize],
    float &max_distance,
    int lane) {

    constexpr int SlotsPerLane = K / kWarpSize;
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0, "Unsupported K");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size");

    int count = 0;
    if (lane == 0) count = *cand_count;
    count = __shfl_sync(kFullMask, count, 0);
    if (count == 0) return;

#pragma unroll
    for (int pos = lane; pos < K; pos += kWarpSize) {
        if (pos >= count) {
            cand_dist[pos] = CUDART_INF_F;
            cand_idx[pos] = INT_MAX;
        }
    }
    __syncwarp();

    // Step 1: sort the candidate buffer ascending.
    bitonic_sort_shared<K>(cand_dist, cand_idx, lane);

    // Step 2: build the required bitonic sequence of the best K elements.
#pragma unroll
    for (int s = 0; s < SlotsPerLane; ++s) {
        const int pos = lane + s * kWarpSize;

        const float rev_best_dist =
            __shfl_sync(kFullMask, best_dist[SlotsPerLane - 1 - s], kWarpSize - 1 - lane);
        const int rev_best_idx =
            __shfl_sync(kFullMask, best_idx[SlotsPerLane - 1 - s], kWarpSize - 1 - lane);

        const float cd = cand_dist[pos];
        const int ci = cand_idx[pos];

        if (!pair_less(cd, ci, rev_best_dist, rev_best_idx)) {
            cand_dist[pos] = rev_best_dist;
            cand_idx[pos] = rev_best_idx;
        }
    }
    __syncwarp();

    // Step 3: sort the merged bitonic sequence ascending.
    bitonic_sort_shared<K>(cand_dist, cand_idx, lane);

#pragma unroll
    for (int s = 0; s < SlotsPerLane; ++s) {
        const int pos = lane + s * kWarpSize;
        best_idx[s] = cand_idx[pos];
        best_dist[s] = cand_dist[pos];
    }

    const float lane_tail = best_dist[SlotsPerLane - 1];
    max_distance = __shfl_sync(kFullMask, lane_tail, kWarpSize - 1);

    // Only one warp touches its own counter, so a plain shared write is enough.
    if (lane == 0) *cand_count = 0;
    __syncwarp();
}

// Insert the current 32-lane candidate chunk into the shared candidate buffer.
// The prompt requires atomicAdd to update the candidate count; this uses a
// warp-aggregated variant so one atomic reserves a contiguous block of slots
// for many qualifying lanes at once.
template <int K>
__device__ __forceinline__ void insert_candidate_chunk(
    unsigned qualifying_mask,
    int lane_candidate_idx,
    float lane_candidate_dist,
    int *cand_count,
    int *cand_idx,
    float *cand_dist,
    int (&best_idx)[K / kWarpSize],
    float (&best_dist)[K / kWarpSize],
    float &max_distance,
    int lane) {

    const unsigned lane_bit = 1u << lane;

    while (qualifying_mask) {
        int current_count = 0;
        if (lane == 0) current_count = *cand_count;
        current_count = __shfl_sync(kFullMask, current_count, 0);

        if (current_count == K) {
            merge_buffer_into_best<K>(
                cand_count, cand_idx, cand_dist, best_idx, best_dist, max_distance, lane);
            qualifying_mask = refresh_pending_mask(
                qualifying_mask, lane_candidate_dist, max_distance, lane);
            continue;
        }

        const int available = K - current_count;
        const int pending = __popc(qualifying_mask);
        const int to_insert = (pending < available) ? pending : available;

        unsigned insert_mask = 0u;
        if (lane == 0) {
            insert_mask = select_lowest_n_bits(qualifying_mask, to_insert);
        }
        insert_mask = __shfl_sync(kFullMask, insert_mask, 0);

        int base = 0;
        if (lane == 0) {
            // Required atomicAdd on the shared candidate count.
            base = atomicAdd(cand_count, to_insert);
        }
        base = __shfl_sync(kFullMask, base, 0);

        if (insert_mask & lane_bit) {
            const unsigned prior_bits = (lane == 0) ? 0u : (lane_bit - 1u);
            const int rank = __popc(insert_mask & prior_bits);
            const int pos = base + rank;
            cand_idx[pos] = lane_candidate_idx;
            cand_dist[pos] = lane_candidate_dist;
        }
        __syncwarp();

        qualifying_mask &= ~insert_mask;
        current_count += to_insert;

        if (current_count == K) {
            merge_buffer_into_best<K>(
                cand_count, cand_idx, cand_dist, best_idx, best_dist, max_distance, lane);
            qualifying_mask = refresh_pending_mask(
                qualifying_mask, lane_candidate_dist, max_distance, lane);
        }
    }
}

template <int K, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 2)
void knn_kernel(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int batch_points) {

    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0, "Unsupported K");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size");
    static_assert(WARPS_PER_BLOCK >= 1 && WARPS_PER_BLOCK <= 8, "Unexpected block shape");

    // Shared-memory layout:
    // [batch_points x float2 staged data]
    // [WARPS_PER_BLOCK x int candidate counts]
    // [WARPS_PER_BLOCK x K x int candidate indices]
    // [WARPS_PER_BLOCK x K x float candidate distances]
    extern __shared__ unsigned char shared_raw[];

    float2 *sh_data = reinterpret_cast<float2 *>(shared_raw);
    unsigned char *ptr =
        shared_raw + static_cast<std::size_t>(batch_points) * sizeof(float2);

    int *sh_cand_count = reinterpret_cast<int *>(ptr);
    ptr += static_cast<std::size_t>(WARPS_PER_BLOCK) * sizeof(int);

    int *sh_cand_idx = reinterpret_cast<int *>(ptr);
    ptr += static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(int);

    float *sh_cand_dist = reinterpret_cast<float *>(ptr);

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int query_idx = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool query_active = (query_idx < query_count);

    int *cand_count = &sh_cand_count[warp_id];
    int *cand_idx = &sh_cand_idx[warp_id * K];
    float *cand_dist = &sh_cand_dist[warp_id * K];

    if (lane == 0) *cand_count = 0;
    __syncwarp();

    constexpr int SlotsPerLane = K / kWarpSize;
    int best_idx[SlotsPerLane];
    float best_dist[SlotsPerLane];
    bool best_initialized = false;
    float max_distance = CUDART_INF_F;

    float qx = 0.0f;
    float qy = 0.0f;
    if (query_active) {
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);
    }

    for (int batch_start = 0; batch_start < data_count; batch_start += batch_points) {
        int current_batch = data_count - batch_start;
        if (current_batch > batch_points) current_batch = batch_points;

        // Whole block stages the next tile of data points into shared memory.
        for (int i = threadIdx.x; i < current_batch; i += blockDim.x) {
            sh_data[i] = data[batch_start + i];
        }
        __syncthreads();

        if (query_active) {
            int local_begin = 0;

            // Initialization uses the first K points of the first tile.
            // Host-side launch selection guarantees batch_points >= K.
            if (!best_initialized) {
#pragma unroll
                for (int s = 0; s < SlotsPerLane; ++s) {
                    const int pos = lane + s * kWarpSize;
                    cand_idx[pos] = pos;  // batch_start == 0 here
                    cand_dist[pos] = squared_l2(qx, qy, sh_data[pos]);
                }
                __syncwarp();

                bitonic_sort_shared<K>(cand_dist, cand_idx, lane);

#pragma unroll
                for (int s = 0; s < SlotsPerLane; ++s) {
                    const int pos = lane + s * kWarpSize;
                    best_idx[s] = cand_idx[pos];
                    best_dist[s] = cand_dist[pos];
                }

                const float lane_tail = best_dist[SlotsPerLane - 1];
                max_distance = __shfl_sync(kFullMask, lane_tail, kWarpSize - 1);

                best_initialized = true;
                local_begin = K;
            }

            // Process the tile in warp-sized chunks. Strict `< max_distance`
            // matches the prompt; equal distances may be broken arbitrarily.
            for (int local_base = local_begin; local_base < current_batch; local_base += kWarpSize) {
                const int local = local_base + lane;

                int data_idx = -1;
                float dist = CUDART_INF_F;
                bool qualifies = false;

                if (local < current_batch) {
                    data_idx = batch_start + local;
                    dist = squared_l2(qx, qy, sh_data[local]);
                    qualifies = (dist < max_distance);
                }

                const unsigned qualifying_mask =
                    __ballot_sync(kFullMask, qualifies);

                if (qualifying_mask) {
                    insert_candidate_chunk<K>(
                        qualifying_mask,
                        data_idx,
                        dist,
                        cand_count,
                        cand_idx,
                        cand_dist,
                        best_idx,
                        best_dist,
                        max_distance,
                        lane);
                }
            }
        }

        // All warps must finish reading this tile before the block overwrites it.
        __syncthreads();
    }

    if (query_active) {
        int tail_count = 0;
        if (lane == 0) tail_count = *cand_count;
        tail_count = __shfl_sync(kFullMask, tail_count, 0);

        if (tail_count > 0) {
            merge_buffer_into_best<K>(
                cand_count, cand_idx, cand_dist, best_idx, best_dist, max_distance, lane);
        }

        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

#pragma unroll
        for (int s = 0; s < SlotsPerLane; ++s) {
            const int pos = lane + s * kWarpSize;
            result[out_base + pos].first = best_idx[s];
            result[out_base + pos].second = best_dist[s];
        }
    }
}

template <int K, int WARPS_PER_BLOCK>
constexpr std::size_t shared_overhead_bytes() {
    return static_cast<std::size_t>(WARPS_PER_BLOCK) * sizeof(int) +
           static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(int) +
           static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(float);
}

template <int K>
constexpr int preferred_batch_points() {
    return (K == 1024) ? 2048 : 4096;
}

template <int K, int WARPS_PER_BLOCK>
bool launch_variant(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int max_dynamic_smem) {

    constexpr std::size_t overhead = shared_overhead_bytes<K, WARPS_PER_BLOCK>();

    if (max_dynamic_smem <= 0) return false;
    if (static_cast<std::size_t>(max_dynamic_smem) <= overhead) return false;

    const std::size_t available_batch_bytes =
        static_cast<std::size_t>(max_dynamic_smem) - overhead;

    int max_batch_points = static_cast<int>(
        (available_batch_bytes / sizeof(float2)) &
        ~static_cast<std::size_t>(kWarpSize - 1));

    if (max_batch_points < K) return false;

    int batch_points = preferred_batch_points<K>();
    if (batch_points > max_batch_points) batch_points = max_batch_points;
    if (batch_points < K) batch_points = K;

    const std::size_t shared_bytes =
        overhead + static_cast<std::size_t>(batch_points) * sizeof(float2);

    cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));

    cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    cudaFuncSetCacheConfig(knn_kernel<K, WARPS_PER_BLOCK>, cudaFuncCachePreferShared);

    const dim3 block(WARPS_PER_BLOCK * kWarpSize);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn_kernel<K, WARPS_PER_BLOCK><<<grid, block, shared_bytes>>>(
        query, query_count, data, data_count, result, batch_points);

    return true;
}

template <int K>
void dispatch_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result) {

    int device = 0;
    cudaGetDevice(&device);

    int max_smem_default = 0;
    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_default, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    int max_dynamic_smem = max_smem_default;
    if (max_smem_optin > max_dynamic_smem) max_dynamic_smem = max_smem_optin;
    if (max_dynamic_smem <= 0) max_dynamic_smem = 48 * 1024;

    // Prefer 8 warps/block for maximum staged-data reuse across queries, but
    // fall back if the device's dynamic shared-memory budget is smaller.
    if (launch_variant<K, 8>(query, query_count, data, data_count, result, max_dynamic_smem)) return;
    if (launch_variant<K, 4>(query, query_count, data, data_count, result, max_dynamic_smem)) return;
    if (launch_variant<K, 2>(query, query_count, data, data_count, result, max_dynamic_smem)) return;
    (void)launch_variant<K, 1>(query, query_count, data, data_count, result, max_dynamic_smem);
}

} // namespace

void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k) {

    if (query_count <= 0) return;

    switch (k) {
        case 32:   dispatch_knn<32>(query, query_count, data, data_count, result); break;
        case 64:   dispatch_knn<64>(query, query_count, data, data_count, result); break;
        case 128:  dispatch_knn<128>(query, query_count, data, data_count, result); break;
        case 256:  dispatch_knn<256>(query, query_count, data, data_count, result); break;
        case 512:  dispatch_knn<512>(query, query_count, data, data_count, result); break;
        case 1024: dispatch_knn<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Inputs are guaranteed valid by the prompt.
    }
}