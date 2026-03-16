#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

// Targeted tuning for modern data-center GPUs (A100/H100 class):
// - 8 warps/block: one warp per query, so a block processes 8 queries and reuses one
//   shared-memory data tile across all 8.
// - 4096-point tiles: 32 KiB of shared memory for the cached dataset tile; large
//   enough to amortize block barriers and global-memory latency.
// - Overflow buffer capped at 256 for large K: this keeps the warp-local prune sort
//   small while preserving exactness via an in-place merge into the exact top-K list.
constexpr int kWarpSize        = 32;
constexpr int kWarpsPerBlock   = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;
constexpr int kDataTileSize    = 4096;
constexpr unsigned kFullMask   = 0xffffffffu;

// Distances are non-negative squared L2 values. For non-negative IEEE-754 floats,
// unsigned integer order of the bit pattern matches numerical order. Packing
// [distance_bits | index] into one 64-bit key therefore gives a sortable total
// order without carrying two separate arrays during selection and merge.
constexpr uint64_t kInfKey = (uint64_t{0x7f800000u} << 32) | uint64_t{0xffffffffu};

template <int K, int BUF_CAP>
constexpr std::size_t shared_storage_bytes() {
    return std::size_t(kWarpsPerBlock) * std::size_t(K + BUF_CAP) * sizeof(uint64_t) +
           std::size_t(2) * std::size_t(kDataTileSize) * sizeof(float);
}

__device__ __forceinline__ uint64_t pack_key(float dist, int idx) {
    return (uint64_t(__float_as_uint(dist)) << 32) | static_cast<uint32_t>(idx);
}

__device__ __forceinline__ float unpack_dist(uint64_t key) {
    return __uint_as_float(static_cast<unsigned>(key >> 32));
}

__device__ __forceinline__ int unpack_idx(uint64_t key) {
    return static_cast<int>(static_cast<uint32_t>(key));
}

template <int N>
__device__ __forceinline__ void warp_bitonic_sort(uint64_t* arr, int lane) {
    static_assert((N & (N - 1)) == 0, "Bitonic sort size must be a power of two.");

    // One warp sorts one fixed-size shared-memory array. Each lane owns N/32 slots.
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int idx = lane; idx < N; idx += kWarpSize) {
                const int partner = idx ^ stride;
                if (partner > idx) {
                    const bool up = ((idx & size) == 0);
                    const uint64_t a = arr[idx];
                    const uint64_t b = arr[partner];
                    const bool a_le_b = (a <= b);
                    arr[idx]     = (up == a_le_b) ? a : b;
                    arr[partner] = (up == a_le_b) ? b : a;
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

template <int K, int BUF_CAP>
__device__ __forceinline__ void warp_prune(
    uint64_t* top,
    uint64_t* buf,
    int& top_count,
    int& buf_count,
    float& threshold,
    int lane) {

    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert((BUF_CAP & (BUF_CAP - 1)) == 0, "Buffer size must be a power of two.");

    if (buf_count == 0) {
        threshold = (top_count == K) ? unpack_dist(top[K - 1]) : CUDART_INF_F;
        return;
    }

    // Pad the inactive suffix with +inf so a fixed-size bitonic sort can be used.
#pragma unroll
    for (int i = buf_count + lane; i < BUF_CAP; i += kWarpSize) {
        buf[i] = kInfKey;
    }
    __syncwarp(kFullMask);

    warp_bitonic_sort<BUF_CAP>(buf, lane);

    const int merged_count = top_count + buf_count;
    const int new_top_count = (merged_count < K) ? merged_count : K;

    float new_threshold = CUDART_INF_F;

    if (lane == 0) {
        // Exact in-place merge without extra storage:
        // 1) Drop the largest merged_count - new_top_count items by walking the tails
        //    of the two sorted arrays.
        // 2) Merge the remaining prefixes from the back into top[].
        int i = top_count - 1;
        int j = buf_count - 1;
        int to_drop = merged_count - new_top_count;

        while (to_drop > 0) {
            if (j < 0 || (i >= 0 && top[i] >= buf[j])) {
                --i;
            } else {
                --j;
            }
            --to_drop;
        }

        int keep_top = i + 1;
        int keep_buf = j + 1;

        i = keep_top - 1;
        j = keep_buf - 1;
        int out = new_top_count - 1;

        while (j >= 0) {
            if (i >= 0 && top[i] > buf[j]) {
                top[out--] = top[i--];
            } else {
                top[out--] = buf[j--];
            }
        }

        if (new_top_count == K) {
            new_threshold = unpack_dist(top[K - 1]);
        }
    }

    __syncwarp(kFullMask);
    top_count = new_top_count;
    buf_count = 0;
    threshold = __shfl_sync(kFullMask, new_threshold, 0);
}

template <int K, int BUF_CAP>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result) {

    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert((BUF_CAP & (BUF_CAP - 1)) == 0, "Buffer size must be a power of two.");

    const int lane    = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int qid     = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_id;
    const bool active = (qid < query_count);

    extern __shared__ uint64_t smem_u64[];

    // Layout of dynamic shared memory:
    // [ per-warp exact top-K keys ][ per-warp overflow buffer ][ tile_x ][ tile_y ]
    uint64_t* const all_top = smem_u64;
    uint64_t* const all_buf = all_top + kWarpsPerBlock * K;

    // The data tile is cached as SoA in shared memory to avoid the 2-way bank
    // conflicts that contiguous float2 accesses would create on 32-bank hardware.
    float* const tile_x = reinterpret_cast<float*>(all_buf + kWarpsPerBlock * BUF_CAP);
    float* const tile_y = tile_x + kDataTileSize;

    uint64_t* const top = all_top + warp_id * K;
    uint64_t* const buf = all_buf + warp_id * BUF_CAP;

    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[qid];
        qx = q.x;
        qy = q.y;
    }
    if (active) {
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);
    }

    int top_count = 0;
    int buf_count = 0;
    float threshold = CUDART_INF_F;

    for (int tile_base = 0; tile_base < data_count; tile_base += kDataTileSize) {
        const int remaining  = data_count - tile_base;
        const int tile_count = (remaining < kDataTileSize) ? remaining : kDataTileSize;

        // Cooperative block-wide load of the current data tile.
#pragma unroll
        for (int i = threadIdx.x; i < kDataTileSize; i += kThreadsPerBlock) {
            if (i < tile_count) {
                const float2 p = data[tile_base + i];
                tile_x[i] = p.x;
                tile_y[i] = p.y;
            }
        }
        __syncthreads();

        if (active) {
            // Process the shared-memory tile in warp-sized batches.
            for (int base = 0; base < tile_count; base += kWarpSize) {
                const int local = base + lane;
                const bool lane_valid = (local < tile_count);

                float dist = 0.0f;
                int global_idx = 0;

                if (lane_valid) {
                    const float dx = qx - tile_x[local];
                    const float dy = qy - tile_y[local];
                    dist = fmaf(dx, dx, dy * dy);
                    global_idx = tile_base + local;
                }

                // Exact thresholded admission:
                // - Before top[] is full, every candidate must be retained.
                // - Once top[] already stores K exact nearest items among the processed
                //   prefix, any candidate with dist >= threshold cannot possibly enter
                //   the final answer, even if buf[] has not been merged yet.
                bool accept = lane_valid && ((top_count < K) || (dist < threshold));
                unsigned mask = __ballot_sync(kFullMask, accept);
                int n = __popc(mask);

                if (n != 0) {
                    // If the current batch would overflow the buffer, first merge the
                    // buffered candidates into the exact top-K list and then reevaluate
                    // the current batch against the tighter threshold.
                    if (buf_count + n > BUF_CAP) {
                        warp_prune<K, BUF_CAP>(top, buf, top_count, buf_count, threshold, lane);
                        accept = lane_valid && ((top_count < K) || (dist < threshold));
                        mask = __ballot_sync(kFullMask, accept);
                        n = __popc(mask);
                    }

                    if (n != 0) {
                        const unsigned prefix_mask = (1u << lane) - 1u;
                        const int rank = __popc(mask & prefix_mask);

                        if (accept) {
                            buf[buf_count + rank] = pack_key(dist, global_idx);
                        }

                        __syncwarp(kFullMask);
                        buf_count += n;

                        if (buf_count == BUF_CAP) {
                            warp_prune<K, BUF_CAP>(top, buf, top_count, buf_count, threshold, lane);
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    if (active) {
        // Final exact merge of the residual buffer.
        warp_prune<K, BUF_CAP>(top, buf, top_count, buf_count, threshold, lane);

        // The output type is std::pair<int,float>; write only plain data members.
        const std::size_t out_base = static_cast<std::size_t>(qid) * static_cast<std::size_t>(K);
        for (int j = lane; j < K; j += kWarpSize) {
            const uint64_t key = top[j];
            result[out_base + j].first  = unpack_idx(key);
            result[out_base + j].second = unpack_dist(key);
        }
    }
}

template <int K, int BUF_CAP>
void launch_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result) {

    constexpr std::size_t kSmemBytes = shared_storage_bytes<K, BUF_CAP>();

    // Keep the configuration compatible with A100-class hardware as requested.
    static_assert(kSmemBytes <= 163840, "Shared-memory usage exceeds A100 per-block limit.");

    cudaFuncSetAttribute(
        knn_kernel<K, BUF_CAP>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));

    cudaFuncSetAttribute(
        knn_kernel<K, BUF_CAP>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    knn_kernel<K, BUF_CAP><<<blocks, kThreadsPerBlock, kSmemBytes>>>(
        query, query_count, data, data_count, result);
}

} // namespace

void run_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k) {

    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:   launch_knn<32,   32 >(query, query_count, data, data_count, result); break;
        case 64:   launch_knn<64,   64 >(query, query_count, data, data_count, result); break;
        case 128:  launch_knn<128,  128>(query, query_count, data, data_count, result); break;
        case 256:  launch_knn<256,  256>(query, query_count, data, data_count, result); break;
        case 512:  launch_knn<512,  256>(query, query_count, data, data_count, result); break;
        case 1024: launch_knn<1024, 256>(query, query_count, data, data_count, result); break;
        default:
            // Contract says k is always valid; keep the default branch side-effect free.
            break;
    }
}