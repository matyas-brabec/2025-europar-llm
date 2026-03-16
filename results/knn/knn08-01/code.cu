#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>

namespace {

// One warp owns one query. The current sorted top-k is kept entirely in registers, split
// evenly across the warp: each lane stores k/32 consecutive entries.
//
// A block also stages a tile of the data set in shared memory so that several warps reuse
// each global load. Separate shared-memory regions hold per-warp candidate buffers.
//
// The runtime launches one of three block shapes:
//
//   4 warps / block,  tile cap 1024  -> 40,960 B shared at k=1024, allows 4 resident
//                                       blocks on a 164 KB A100 SM.
//   8 warps / block,  tile cap 2048  -> 81,920 B shared at k=1024, allows 2 resident
//                                       blocks on a 164 KB A100 SM.
//  16 warps / block,  tile cap 3584  -> 159,744 B shared at k=1024, allows 1 resident
//                                       block on a 164 KB A100 SM.
//
// The launcher picks the largest reuse factor (warps per block) that still leaves roughly
// 75% as many blocks as SMs for the current query count. This balances:
//   * more warps/block  -> better reuse of the staged data tile,
//   * fewer warps/block -> more independent blocks when the query set is smaller.
//
// Ties are allowed to resolve arbitrarily by the problem statement. The sorting network
// uses the data index as a deterministic secondary key only to make the total order strict.
constexpr int      kWarpSize     = 32;
constexpr unsigned kFullMask     = 0xFFFFFFFFu;
constexpr int      kInvalidIndex = INT_MAX;

template <int WARPS_PER_BLOCK>
struct BlockConfig;

template <>
struct BlockConfig<4> {
    static constexpr int BLOCK_THREADS = 4 * kWarpSize;
    static constexpr int TILE_CAP      = 1024;
};

template <>
struct BlockConfig<8> {
    static constexpr int BLOCK_THREADS = 8 * kWarpSize;
    static constexpr int TILE_CAP      = 2048;
};

template <>
struct BlockConfig<16> {
    static constexpr int BLOCK_THREADS = 16 * kWarpSize;
    static constexpr int TILE_CAP      = 3584;
};

__device__ __forceinline__ bool item_less(float a_dist, int a_idx, float b_dist, int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

__device__ __forceinline__ void swap_items(float& a_dist, int& a_idx, float& b_dist, int& b_idx) {
    const float td = a_dist;
    const int   ti = a_idx;
    a_dist = b_dist;
    a_idx  = b_idx;
    b_dist = td;
    b_idx  = ti;
}

// Distributed bitonic sort over K items, where each lane owns ITEMS_PER_THREAD consecutive
// items. Because ITEMS_PER_THREAD is a power of two and the lane owns a consecutive range,
// any inter-lane exchange always targets the same register slot in the partner lane.
// For j < ITEMS_PER_THREAD, the exchange is purely intra-lane and becomes a simple register swap.
template <int K, int ITEMS_PER_THREAD>
__device__ __forceinline__ void bitonic_sort_warp(float (&dist)[ITEMS_PER_THREAD],
                                                  int (&idx)[ITEMS_PER_THREAD],
                                                  int lane) {
#pragma unroll
    for (int seq = 2; seq <= K; seq <<= 1) {
#pragma unroll
        for (int stride = seq >> 1; stride > 0; stride >>= 1) {
            if (stride >= ITEMS_PER_THREAD) {
                const int lane_xor = stride / ITEMS_PER_THREAD;

#pragma unroll
                for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
                    const int  global_i  = lane * ITEMS_PER_THREAD + item;
                    const bool up        = ((global_i & seq) == 0);
                    const bool lower_half = ((lane & lane_xor) == 0);
                    const bool keep_min  = (lower_half == up);

                    const float other_dist = __shfl_xor_sync(kFullMask, dist[item], lane_xor);
                    const int   other_idx  = __shfl_xor_sync(kFullMask, idx[item],  lane_xor);

                    if (keep_min) {
                        if (item_less(other_dist, other_idx, dist[item], idx[item])) {
                            dist[item] = other_dist;
                            idx[item]  = other_idx;
                        }
                    } else {
                        if (item_less(dist[item], idx[item], other_dist, other_idx)) {
                            dist[item] = other_dist;
                            idx[item]  = other_idx;
                        }
                    }
                }
            } else {
#pragma unroll
                for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
                    const int partner = item ^ stride;
                    if (partner > item) {
                        const int  global_i = lane * ITEMS_PER_THREAD + item;
                        const bool up       = ((global_i & seq) == 0);
                        const bool do_swap  = up
                            ? item_less(dist[partner], idx[partner], dist[item], idx[item])
                            : item_less(dist[item], idx[item], dist[partner], idx[partner]);

                        if (do_swap) {
                            swap_items(dist[item], idx[item], dist[partner], idx[partner]);
                        }
                    }
                }
            }
        }
    }
}

// Merge the shared-memory candidate buffer with the register-resident intermediate result,
// exactly in the order requested by the prompt:
//
//   0. intermediate result in registers is already sorted ascending,
//   1. swap intermediate result <-> shared buffer, so the buffer is now in registers,
//   2. bitonic-sort the buffer in ascending order,
//   3. create the bitonic "first K of the union" sequence by taking
//      min(buffer[i], intermediate[K-1-i]),
//   4. bitonic-sort again to obtain the updated intermediate result.
//
// Invalid entries in a partially filled buffer are padded with (+inf, INT_MAX) so they sort last.
template <int K, int ITEMS_PER_THREAD>
__device__ __forceinline__ void flush_candidate_buffer(float (&reg_dist)[ITEMS_PER_THREAD],
                                                       int (&reg_idx)[ITEMS_PER_THREAD],
                                                       float* warp_buf_dist,
                                                       int* warp_buf_idx,
                                                       int& buf_count,
                                                       float& max_distance,
                                                       int lane) {
    if (buf_count == 0) {
        return;
    }

    __syncwarp(kFullMask);

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int global_i = lane * ITEMS_PER_THREAD + item;
        if (global_i >= buf_count) {
            warp_buf_dist[global_i] = CUDART_INF_F;
            warp_buf_idx[global_i]  = kInvalidIndex;
        }
    }

    __syncwarp(kFullMask);

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int global_i = lane * ITEMS_PER_THREAD + item;

        const float cand_dist = warp_buf_dist[global_i];
        const int   cand_idx  = warp_buf_idx[global_i];

        warp_buf_dist[global_i] = reg_dist[item];
        warp_buf_idx[global_i]  = reg_idx[item];

        reg_dist[item] = cand_dist;
        reg_idx[item]  = cand_idx;
    }

    __syncwarp(kFullMask);

    bitonic_sort_warp<K, ITEMS_PER_THREAD>(reg_dist, reg_idx, lane);

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int global_i = lane * ITEMS_PER_THREAD + item;

        const float other_dist = warp_buf_dist[K - 1 - global_i];
        const int   other_idx  = warp_buf_idx[K - 1 - global_i];

        if (item_less(other_dist, other_idx, reg_dist[item], reg_idx[item])) {
            reg_dist[item] = other_dist;
            reg_idx[item]  = other_idx;
        }
    }

    bitonic_sort_warp<K, ITEMS_PER_THREAD>(reg_dist, reg_idx, lane);

    buf_count = 0;

    // The k-th nearest neighbor is the last element of the sorted register file layout:
    // lane 31, last local slot.
    max_distance = __shfl_sync(kFullMask, reg_dist[ITEMS_PER_THREAD - 1], kWarpSize - 1);
}

template <int K, int WARPS_PER_BLOCK>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int tile_points) {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0, "K must be a power of two in [32, 1024].");
    static_assert(K % kWarpSize == 0, "K must be divisible by 32.");

    constexpr int ITEMS_PER_THREAD = K / kWarpSize;
    constexpr int BLOCK_THREADS    = BlockConfig<WARPS_PER_BLOCK>::BLOCK_THREADS;

    // Dynamic shared memory layout:
    //   [block-wide data tile][warp0 buf dist][warp1 buf dist]...[warpN buf dist]
    //   [warp0 buf idx ][warp1 buf idx ]...[warpN buf idx ]
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float2* sh_data     = reinterpret_cast<float2*>(smem_raw);
    float*  sh_buf_dist = reinterpret_cast<float*>(sh_data + tile_points);
    int*    sh_buf_idx  = reinterpret_cast<int*>(sh_buf_dist + WARPS_PER_BLOCK * K);

    const int warp_id  = threadIdx.x >> 5;
    const int lane     = threadIdx.x & (kWarpSize - 1);
    const int query_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool valid_query = (query_id < query_count);

    float* warp_buf_dist = sh_buf_dist + warp_id * K;
    int*   warp_buf_idx  = sh_buf_idx  + warp_id * K;

    // Private top-k result kept in registers, distributed across the warp.
    float reg_dist[ITEMS_PER_THREAD];
    int   reg_idx[ITEMS_PER_THREAD];

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        reg_dist[item] = CUDART_INF_F;
        reg_idx[item]  = kInvalidIndex;
    }

    float qx = 0.0f;
    float qy = 0.0f;

    if (valid_query) {
        float2 q = make_float2(0.0f, 0.0f);
        if (lane == 0) {
            q = query[query_id];
        }
        qx = __shfl_sync(kFullMask, q.x, 0);
        qy = __shfl_sync(kFullMask, q.y, 0);
    }

    // Warp-uniform candidate count. It is replicated in every lane because that is cheaper
    // than a shared scalar and avoids extra memory traffic.
    int   buf_count     = 0;
    float max_distance  = CUDART_INF_F;
    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    for (int tile_base = 0; tile_base < data_count; tile_base += tile_points) {
        int tile_count = data_count - tile_base;
        if (tile_count > tile_points) {
            tile_count = tile_points;
        }

        // Whole block loads the current data tile into shared memory.
        for (int i = threadIdx.x; i < tile_count; i += BLOCK_THREADS) {
            sh_data[i] = data[tile_base + i];
        }

        __syncthreads();

        if (valid_query) {
            const int rounds = (tile_count + kWarpSize - 1) / kWarpSize;

#pragma unroll 1
            for (int round = 0; round < rounds; ++round) {
                const int local_data_idx = round * kWarpSize + lane;
                const bool active = (local_data_idx < tile_count);

                float dist = CUDART_INF_F;
                int   data_idx = tile_base + local_data_idx;

                if (active) {
                    const float2 p  = sh_data[local_data_idx];
                    const float  dx = qx - p.x;
                    const float  dy = qy - p.y;

                    // Squared Euclidean distance. A NaN can only arise from invalid input data;
                    // map it to +inf so that it naturally sinks to the end of the ordering.
                    dist = __fmaf_rn(dx, dx, dy * dy);
                    if (!(dist >= 0.0f)) {
                        dist = CUDART_INF_F;
                    }
                }

                bool keep = active && (dist < max_distance);
                unsigned mask = __ballot_sync(kFullMask, keep);
                int selected = __popc(mask);

                if (selected) {
                    // If the packed append would overflow the shared candidate buffer, merge the
                    // current buffer first and re-evaluate the current candidate against the new
                    // (usually tighter) max_distance.
                    if (buf_count + selected > K) {
                        flush_candidate_buffer<K, ITEMS_PER_THREAD>(
                            reg_dist, reg_idx, warp_buf_dist, warp_buf_idx, buf_count, max_distance, lane);

                        keep = active && (dist < max_distance);
                        mask = __ballot_sync(kFullMask, keep);
                        selected = __popc(mask);
                    }

                    if (selected) {
                        const int base = buf_count;
                        const int rank = __popc(mask & lane_mask_lt);

                        if (keep) {
                            warp_buf_dist[base + rank] = dist;
                            warp_buf_idx[base + rank]  = data_idx;
                        }

                        buf_count = base + selected;

                        if (buf_count == K) {
                            flush_candidate_buffer<K, ITEMS_PER_THREAD>(
                                reg_dist, reg_idx, warp_buf_dist, warp_buf_idx, buf_count, max_distance, lane);
                        }
                    }
                }
            }
        }

        // The data-tile region in shared memory is reused by the next iteration.
        __syncthreads();
    }

    if (valid_query) {
        if (buf_count) {
            flush_candidate_buffer<K, ITEMS_PER_THREAD>(
                reg_dist, reg_idx, warp_buf_dist, warp_buf_idx, buf_count, max_distance, lane);
        }

        std::pair<int, float>* out =
            result + static_cast<std::size_t>(query_id) * K + lane * ITEMS_PER_THREAD;

#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            out[item].first  = reg_idx[item];
            out[item].second = reg_dist[item];
        }
    }
}

inline int query_threshold_for_warps_per_block(int sm_count, int warps_per_block) {
    // ceil(0.75 * sm_count * warps_per_block)
    return (3 * sm_count * warps_per_block + 3) / 4;
}

template <int K, int WARPS_PER_BLOCK>
bool try_launch_knn(const float2* query,
                    int query_count,
                    const float2* data,
                    int data_count,
                    std::pair<int, float>* result,
                    int max_shared_optin) {
    constexpr int BLOCK_THREADS = BlockConfig<WARPS_PER_BLOCK>::BLOCK_THREADS;
    constexpr int TILE_CAP      = BlockConfig<WARPS_PER_BLOCK>::TILE_CAP;

    const std::size_t fixed_bytes =
        static_cast<std::size_t>(WARPS_PER_BLOCK) * K * (sizeof(float) + sizeof(int));

    if (fixed_bytes >= static_cast<std::size_t>(max_shared_optin)) {
        return false;
    }

    const int max_tile_points =
        static_cast<int>((static_cast<std::size_t>(max_shared_optin) - fixed_bytes) / sizeof(float2));

    if (max_tile_points <= 0) {
        return false;
    }

    // tile_limit is the largest full-tile size this variant will ever ask for on this device.
    // The actual launch may use fewer points if data_count is smaller.
    int tile_limit = TILE_CAP;
    if (tile_limit > max_tile_points) {
        tile_limit = max_tile_points;
        if (tile_limit > BLOCK_THREADS) {
            tile_limit = (tile_limit / BLOCK_THREADS) * BLOCK_THREADS;
        }
    }

    if (tile_limit <= 0) {
        return false;
    }

    const int tile_points = std::min(tile_limit, data_count);
    const std::size_t max_dynamic_smem =
        fixed_bytes + static_cast<std::size_t>(tile_limit) * sizeof(float2);
    const std::size_t launch_dynamic_smem =
        fixed_bytes + static_cast<std::size_t>(tile_points) * sizeof(float2);

    // Allow >48 KB dynamic shared memory and bias the SM partition toward shared memory.
    if (cudaFuncSetAttribute(knn_kernel<K, WARPS_PER_BLOCK>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(max_dynamic_smem)) != cudaSuccess) {
        return false;
    }

    (void)cudaFuncSetAttribute(knn_kernel<K, WARPS_PER_BLOCK>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               static_cast<int>(cudaSharedmemCarveoutMaxShared));

    const int grid_x = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    knn_kernel<K, WARPS_PER_BLOCK><<<grid_x, BLOCK_THREADS, launch_dynamic_smem>>>(
        query, query_count, data, data_count, result, tile_points);

    return true;
}

template <int K>
void launch_for_k(const float2* query,
                  int query_count,
                  const float2* data,
                  int data_count,
                  std::pair<int, float>* result) {
    // Conservative A100 defaults in case attribute queries fail; the actual device values are
    // used when available.
    int device            = 0;
    int sm_count          = 108;
    int max_shared_optin  = 163840;

    if (cudaGetDevice(&device) == cudaSuccess) {
        int value = 0;
        if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
            sm_count = value;
        }
        if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess) {
            max_shared_optin = value;
        }
    }

    // Prefer more reuse only when the resulting number of blocks is still large enough to
    // keep the GPU well occupied.
    if (query_count >= query_threshold_for_warps_per_block(sm_count, 16)) {
        if (try_launch_knn<K, 16>(query, query_count, data, data_count, result, max_shared_optin)) {
            return;
        }
        if (try_launch_knn<K, 8>(query, query_count, data, data_count, result, max_shared_optin)) {
            return;
        }
        (void)try_launch_knn<K, 4>(query, query_count, data, data_count, result, max_shared_optin);
        return;
    }

    if (query_count >= query_threshold_for_warps_per_block(sm_count, 8)) {
        if (try_launch_knn<K, 8>(query, query_count, data, data_count, result, max_shared_optin)) {
            return;
        }
        (void)try_launch_knn<K, 4>(query, query_count, data, data_count, result, max_shared_optin);
        return;
    }

    (void)try_launch_knn<K, 4>(query, query_count, data, data_count, result, max_shared_optin);
}

}  // namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:
            launch_for_k<32>(query, query_count, data, data_count, result);
            return;
        case 64:
            launch_for_k<64>(query, query_count, data, data_count, result);
            return;
        case 128:
            launch_for_k<128>(query, query_count, data, data_count, result);
            return;
        case 256:
            launch_for_k<256>(query, query_count, data, data_count, result);
            return;
        case 512:
            launch_for_k<512>(query, query_count, data, data_count, result);
            return;
        case 1024:
            launch_for_k<1024>(query, query_count, data, data_count, result);
            return;
        default:
            // Inputs are documented to be valid, so this path is only a defensive no-op.
            return;
    }
}