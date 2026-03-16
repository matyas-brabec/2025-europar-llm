#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

using ResultPair = std::pair<int, float>;

// Tuned for A100/H100-class GPUs as requested.
// - 256 threads/block = 8 warps/block, one query per warp.
// - 4096-point shared-memory data tile.
//   For K=1024, the per-warp top-k state plus the tile uses exactly 160 KiB,
//   i.e. A100's opt-in per-block dynamic shared-memory limit.
constexpr int kWarpSize       = 32;
constexpr int kWarpsPerBlock  = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kBatchPoints    = 4096;
constexpr unsigned kFullMask  = 0xFFFFFFFFu;

// Packed (distance, index) key layout:
//   high 32 bits : IEEE-754 bits of non-negative squared distance
//   low  32 bits : unsigned representation of the point index
// Squared Euclidean distances are non-negative, so unsigned integer ordering on the
// high word matches numeric float ordering. This lets the merge path compare one
// 64-bit key instead of separate (distance, index) fields.
constexpr uint64_t kPackedInf = (uint64_t{0x7F800000u} << 32) | 0xFFFFFFFFu;

static_assert(kBatchPoints % kWarpSize == 0, "Batch size must be a multiple of warp size.");

template <int K>
constexpr std::size_t shared_bytes_for_k() {
    return static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(2 * K) * sizeof(uint64_t) +
           static_cast<std::size_t>(2 * kBatchPoints) * sizeof(float);
}

static_assert(shared_bytes_for_k<1024>() == 160u * 1024u,
              "Worst-case K=1024 variant must fit A100's 160 KiB opt-in shared memory.");

__device__ __forceinline__ uint64_t pack_key(const float dist, const int idx) {
    return (static_cast<uint64_t>(__float_as_uint(dist)) << 32) |
           static_cast<uint32_t>(idx);
}

__device__ __forceinline__ float unpack_dist(const uint64_t key) {
    return __uint_as_float(static_cast<uint32_t>(key >> 32));
}

__device__ __forceinline__ int unpack_idx(const uint64_t key) {
    return static_cast<int>(static_cast<uint32_t>(key));
}

// One warp sorts its own 2*K packed keys in shared memory.
// Loops are intentionally kept rolled to avoid massive code size for the 2048-element case.
template <int N>
__device__ __forceinline__ void warp_bitonic_sort(uint64_t* keys) {
    const int lane = threadIdx.x & (kWarpSize - 1);

    #pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += kWarpSize) {
                const int ixj = i ^ stride;
                if (ixj > i) {
                    const uint64_t a = keys[i];
                    const uint64_t b = keys[ixj];
                    const bool ascending = ((i & size) == 0);
                    if (ascending ? (a > b) : (a < b)) {
                        keys[i] = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

// Merge the current intermediate result [0, K) with the candidate buffer [K, 2K).
// If the candidate buffer is only partially filled, pad the unused tail with +inf so the
// same 2*K network handles both full and partial merges.
template <int K>
__device__ __forceinline__ void merge_candidates(uint64_t* warp_keys, int& cand_count, float& kth_dist) {
    if (cand_count == 0) return;

    const int lane = threadIdx.x & (kWarpSize - 1);

    // Ensure all prior candidate writes are visible to the warp before sorting.
    __syncwarp(kFullMask);

    if (cand_count < K) {
        #pragma unroll
        for (int pos = K + cand_count + lane; pos < 2 * K; pos += kWarpSize) {
            warp_keys[pos] = kPackedInf;
        }
    }

    __syncwarp(kFullMask);
    warp_bitonic_sort<2 * K>(warp_keys);

    float new_kth = CUDART_INF_F;
    if (lane == 0) {
        new_kth = unpack_dist(warp_keys[K - 1]);
    }
    kth_dist = __shfl_sync(kFullMask, new_kth, 0);
    cand_count = 0;
}

template <int K>
__global__ __launch_bounds__(kThreadsPerBlock)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                ResultPair* __restrict__ result) {
    extern __shared__ __align__(16) unsigned char smem_raw[];

    // Shared-memory layout:
    //   topk_keys: 8 warps * (2*K packed keys/warp)
    //     [0, K)  : warp-private intermediate top-k result
    //     [K, 2K) : warp-private candidate buffer
    //   batch_x / batch_y: block-wide cached data tile in SoA layout
    //
    // Using SoA for the cached data tile avoids the 2-way bank conflicts that would arise
    // if consecutive lanes read consecutive float2 values directly from shared memory.
    uint64_t* const topk_keys = reinterpret_cast<uint64_t*>(smem_raw);
    float* const batch_x = reinterpret_cast<float*>(topk_keys + kWarpsPerBlock * (2 * K));
    float* const batch_y = batch_x + kBatchPoints;

    const int tid      = threadIdx.x;
    const int lane     = tid & (kWarpSize - 1);
    const int warp     = tid >> 5;
    const int query_idx = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const bool active  = (query_idx < query_count);

    uint64_t* const warp_keys = topk_keys + warp * (2 * K);

    // The prompt asks for a private copy of the intermediate top-k per query.
    // Here it is a warp-private shared-memory region; this avoids local-memory spills
    // when K is as large as 1024.
    if (active) {
        #pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            warp_keys[i] = kPackedInf;
        }
    }

    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    int cand_count = 0;
    float kth_dist = CUDART_INF_F;

    for (int batch_start = 0; batch_start < data_count; batch_start += kBatchPoints) {
        int batch_count = data_count - batch_start;
        if (batch_count > kBatchPoints) batch_count = kBatchPoints;

        // Whole block cooperatively loads the next data tile into shared memory.
        for (int i = tid; i < batch_count; i += kThreadsPerBlock) {
            const float2 p = data[batch_start + i];
            batch_x[i] = p.x;
            batch_y[i] = p.y;
        }
        __syncthreads();

        if (active) {
            // Warp processes the tile 32 points at a time so ballot+prefix can compact
            // accepted candidates into the shared candidate buffer.
            for (int tile = 0; tile < batch_count; tile += kWarpSize) {
                const int point = tile + lane;
                const bool valid = (point < batch_count);

                float dist = CUDART_INF_F;
                int idx = -1;
                if (valid) {
                    const float dx = qx - batch_x[point];
                    const float dy = qy - batch_y[point];
                    dist = __fmaf_rn(dx, dx, dy * dy);
                    idx = batch_start + point;
                }

                bool accept = valid && (dist < kth_dist);
                unsigned mask = __ballot_sync(kFullMask, accept);
                int n_accept = __popc(mask);

                // The warp discovers up to 32 candidates at once. If appending the whole set
                // would overflow the K-sized candidate buffer, merge the current buffer first,
                // tighten the threshold, and then re-test the just computed distances.
                if (cand_count + n_accept > K) {
                    merge_candidates<K>(warp_keys, cand_count, kth_dist);

                    accept = valid && (dist < kth_dist);
                    mask = __ballot_sync(kFullMask, accept);
                    n_accept = __popc(mask);
                }

                if (accept) {
                    const unsigned prior_lanes = (lane == 0) ? 0u : ((1u << lane) - 1u);
                    const int rank = __popc(mask & prior_lanes);
                    warp_keys[K + cand_count + rank] = pack_key(dist, idx);
                }

                cand_count += n_accept;

                // Immediate merge as soon as the candidate buffer becomes full.
                if (cand_count == K) {
                    merge_candidates<K>(warp_keys, cand_count, kth_dist);
                }
            }
        }

        // No warp may start overwriting the shared data tile for the next batch until all
        // warps are done reading the current tile.
        __syncthreads();
    }

    if (active) {
        // Final partial merge after the last cached batch.
        merge_candidates<K>(warp_keys, cand_count, kth_dist);

        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            const uint64_t key = warp_keys[i];
            // Write fields separately to avoid relying on any device-side std::pair
            // constructors or assignment operators.
            result[out_base + static_cast<std::size_t>(i)].first  = unpack_idx(key);
            result[out_base + static_cast<std::size_t>(i)].second = unpack_dist(key);
        }
    }
}

template <int K>
inline void launch_knn_specialized(const float2* query,
                                   const int query_count,
                                   const float2* data,
                                   const int data_count,
                                   ResultPair* result) {
    constexpr std::size_t smem_bytes = shared_bytes_for_k<K>();

    // This kernel is intentionally shared-memory heavy. Request the full dynamic shared-memory
    // budget and bias the SM carveout toward shared memory on A100/H100-class devices.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);
    knn_kernel<K><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result);
}

}  // namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Dispatch to one of the six legal K specializations so K is compile-time constant inside
    // the hot loop and inside the merge network.
    switch (k) {
        case 32:
            launch_knn_specialized<32>(query, query_count, data, data_count, result);
            return;
        case 64:
            launch_knn_specialized<64>(query, query_count, data, data_count, result);
            return;
        case 128:
            launch_knn_specialized<128>(query, query_count, data, data_count, result);
            return;
        case 256:
            launch_knn_specialized<256>(query, query_count, data, data_count, result);
            return;
        case 512:
            launch_knn_specialized<512>(query, query_count, data, data_count, result);
            return;
        case 1024:
            launch_knn_specialized<1024>(query, query_count, data, data_count, result);
            return;
        default:
            return;
    }
}