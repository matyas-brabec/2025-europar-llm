#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <utility>

namespace knn_detail {

// Optimized design summary:
// - One warp computes one query.
// - The current top-K for that query is kept in registers, distributed so each lane owns K/32
//   consecutive entries.
// - A warp-private shared-memory candidate buffer stores up to K newly found closer points.
// - The database is scanned in large shared-memory tiles loaded cooperatively by the full block.
// - Whenever the candidate buffer fills, it is merged into the register-resident intermediate
//   result using the exact bitonic-sort-based procedure requested by the prompt.
//
// Hyper-parameters chosen for A100/H100-class GPUs:
// - 8 warps/block = 256 threads/block. This maximizes reuse of each shared-memory tile across
//   multiple queries while still fitting a large tile and the per-warp candidate buffers into
//   shared memory.
// - ~79.75 KiB dynamic shared memory/block (80 KiB minus 256 B headroom). On A100 this allows
//   two resident blocks/SM within the 163840 B shared-memory budget, while keeping the tile as
//   large as possible.

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockThreads = kWarpsPerBlock * kWarpSize;
constexpr std::size_t kTargetDynamicSmemBytes = 80u * 1024u - 256u;
constexpr int kInvalidIndex = INT_MAX;

// Device code should not rely on std::pair constructors being __device__-callable.
// The result buffer comes from cudaMalloc, so we reinterpret it as a POD with the same ABI.
struct alignas(std::pair<int, float>) PairIFPOD {
    int first;
    float second;
};

static_assert(sizeof(PairIFPOD) == sizeof(std::pair<int, float>),
              "std::pair<int,float> must be ABI-compatible with {int,float}.");
static_assert(alignof(PairIFPOD) == alignof(std::pair<int, float>),
              "std::pair<int,float> alignment must match {int,float}.");

// Total order used inside the sorting network.
// Ties on distance may be resolved arbitrarily by the problem statement; using the index as a
// secondary key simply makes the network deterministic and keeps sentinel entries at the end.
__device__ __forceinline__ bool pair_less(const float a_dist, const int a_idx,
                                          const float b_dist, const int b_idx) {
    return (a_dist < b_dist) || ((a_dist == b_dist) && (a_idx < b_idx));
}

__device__ __forceinline__ void swap_pair(float& a_dist, int& a_idx,
                                          float& b_dist, int& b_idx) {
    const float td = a_dist;
    a_dist = b_dist;
    b_dist = td;

    const int ti = a_idx;
    a_idx = b_idx;
    b_idx = ti;
}

// Warp-wide bitonic sort over K elements distributed across the 32 lanes.
// Each lane owns LOCAL_K = K/32 consecutive values.
// Because LOCAL_K is a power of two and the ownership is consecutive, XOR partners work out as:
// - stride < LOCAL_K   : partner stays in the same lane, so registers can be swapped directly.
// - stride >= LOCAL_K  : partner is in another lane, but always at the same local register index,
//                        so a single __shfl_xor_sync per local slot suffices.
template <int K>
__device__ __forceinline__ void bitonic_sort_warp(float* dist, int* idx, const int lane) {
    static_assert((K & (K - 1)) == 0 && K >= 32 && K <= 1024,
                  "K must be a supported power of two.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");

    constexpr int LOCAL_K = K / kWarpSize;
    const int lane_base = lane * LOCAL_K;

    #pragma unroll
    for (int stage = 2; stage <= K; stage <<= 1) {
        #pragma unroll
        for (int stride = stage >> 1; stride > 0; stride >>= 1) {
            if (stride < LOCAL_K) {
                #pragma unroll
                for (int r = 0; r < LOCAL_K; ++r) {
                    const int partner = r ^ stride;
                    if (partner > r) {
                        const int global_i = lane_base + r;
                        const bool sort_asc = ((global_i & stage) == 0);

                        if (sort_asc) {
                            if (pair_less(dist[partner], idx[partner], dist[r], idx[r])) {
                                swap_pair(dist[r], idx[r], dist[partner], idx[partner]);
                            }
                        } else {
                            if (pair_less(dist[r], idx[r], dist[partner], idx[partner])) {
                                swap_pair(dist[r], idx[r], dist[partner], idx[partner]);
                            }
                        }
                    }
                }
            } else {
                const int lane_delta = stride / LOCAL_K;

                #pragma unroll
                for (int r = 0; r < LOCAL_K; ++r) {
                    const float other_d = __shfl_xor_sync(kFullMask, dist[r], lane_delta);
                    const int other_i = __shfl_xor_sync(kFullMask, idx[r], lane_delta);

                    const int global_i = lane_base + r;
                    const bool sort_asc = ((global_i & stage) == 0);
                    const bool lower = ((global_i & stride) == 0);

                    // In ascending subsequences the lower index keeps the smaller element.
                    // In descending subsequences the lower index keeps the larger element.
                    const bool keep_small = (sort_asc == lower);

                    if (keep_small) {
                        if (pair_less(other_d, other_i, dist[r], idx[r])) {
                            dist[r] = other_d;
                            idx[r] = other_i;
                        }
                    } else {
                        if (pair_less(dist[r], idx[r], other_d, other_i)) {
                            dist[r] = other_d;
                            idx[r] = other_i;
                        }
                    }
                }
            }

            // The next network layer depends on the outputs of the current one. Shuffles already
            // synchronize the exchange itself; this warp barrier makes the layer-to-layer
            // dependency explicit and keeps the warp converged on Volta+ schedulers.
            __syncwarp(kFullMask);
        }
    }
}

// Merges the shared-memory candidate buffer into the sorted register-resident intermediate result
// using the exact sequence requested:
//
// 1. Swap buffer <-> intermediate result so the buffer moves into registers.
// 2. Bitonic-sort the register buffer.
// 3. Build the bitonic sequence C[i] = min(A[i], B[K-1-i]) where A is the sorted buffer in
//    registers and B is the old intermediate result in shared memory.
// 4. Bitonic-sort C to obtain the updated intermediate result.
//
// shared_buffer_* is warp-private even though it resides in shared memory.
template <int K>
__device__ __forceinline__ void merge_buffer_into_result(float* result_dist,
                                                         int* result_idx,
                                                         float* shared_buffer_dist,
                                                         int* shared_buffer_idx,
                                                         int& buffer_count,
                                                         float& max_distance,
                                                         const int lane) {
    constexpr int LOCAL_K = K / kWarpSize;
    const int lane_base = lane * LOCAL_K;

    // Order all prior writes to the warp-private shared buffer before this merge reads them.
    __syncwarp(kFullMask);

    // Step 1: swap buffer <-> intermediate result.
    #pragma unroll
    for (int r = 0; r < LOCAL_K; ++r) {
        const int pos = lane_base + r;

        const float buffer_d = (pos < buffer_count) ? shared_buffer_dist[pos] : CUDART_INF_F;
        const int buffer_i = (pos < buffer_count) ? shared_buffer_idx[pos] : kInvalidIndex;

        shared_buffer_dist[pos] = result_dist[r];
        shared_buffer_idx[pos] = result_idx[r];

        result_dist[r] = buffer_d;
        result_idx[r] = buffer_i;
    }

    // The old intermediate result is now in shared memory and will be consumed in reverse order.
    __syncwarp(kFullMask);

    // Step 2: sort the buffer now resident in registers.
    bitonic_sort_warp<K>(result_dist, result_idx, lane);

    // Step 3: build the bitonic sequence containing the smallest K elements of the 2K union.
    #pragma unroll
    for (int r = 0; r < LOCAL_K; ++r) {
        const int pos = lane_base + r;
        const int mirror = K - 1 - pos;

        const float other_d = shared_buffer_dist[mirror];
        const int other_i = shared_buffer_idx[mirror];

        if (pair_less(other_d, other_i, result_dist[r], result_idx[r])) {
            result_dist[r] = other_d;
            result_idx[r] = other_i;
        }
    }

    // Step 4: sort the bitonic sequence to restore the ascending invariant.
    bitonic_sort_warp<K>(result_dist, result_idx, lane);

    // The K-th nearest neighbor is the globally last element, which resides in lane 31 at the
    // last local slot because ownership is consecutive.
    const float local_last = result_dist[LOCAL_K - 1];
    max_distance = __shfl_sync(kFullMask, local_last, kWarpSize - 1);
    buffer_count = 0;
}

template <int K, int WARPS_PER_BLOCK, int TILE_POINTS>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize, 2)
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                std::pair<int, float>* __restrict__ result) {
    static_assert((K & (K - 1)) == 0 && K >= 32 && K <= 1024,
                  "K must be a supported power of two.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");
    static_assert(WARPS_PER_BLOCK > 0, "At least one warp is required.");
    static_assert((TILE_POINTS % kWarpSize) == 0, "Tile size must be a multiple of the warp size.");

    constexpr int LOCAL_K = K / kWarpSize;

    extern __shared__ unsigned char shared_raw[];

    // The input is AoS in global memory (float2), but the shared-memory tile is stored as SoA
    // (x[] and y[]) so every warp streams through conflict-free 32-bit shared accesses.
    float* tile_x = reinterpret_cast<float*>(shared_raw);
    float* tile_y = tile_x + TILE_POINTS;
    float* candidate_dist = tile_y + TILE_POINTS;
    int* candidate_idx = reinterpret_cast<int*>(candidate_dist + WARPS_PER_BLOCK * K);

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool warp_active = (query_idx < query_count);

    const unsigned lower_lane_mask = (1u << lane) - 1u;
    const int lane_base = lane * LOCAL_K;

    // One lane loads the query point and broadcasts it to the rest of the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (warp_active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Warp-private candidate buffer slices in shared memory.
    float* warp_buffer_dist = candidate_dist + warp_id * K;
    int* warp_buffer_idx = candidate_idx + warp_id * K;

    // Intermediate result kept in registers; initially K sentinels, which is already sorted.
    float result_dist[LOCAL_K];
    int result_idx[LOCAL_K];
    #pragma unroll
    for (int r = 0; r < LOCAL_K; ++r) {
        result_dist[r] = CUDART_INF_F;
        result_idx[r] = kInvalidIndex;
    }

    // Warp-uniform state replicated in registers.
    int buffer_count = 0;
    float max_distance = CUDART_INF_F;

    for (int base = 0; base < data_count; base += TILE_POINTS) {
        const int batch_count = (base + TILE_POINTS <= data_count) ? TILE_POINTS : (data_count - base);

        // Cooperative block-wide load of the current database tile.
        for (int t = threadIdx.x; t < batch_count; t += WARPS_PER_BLOCK * kWarpSize) {
            const float2 p = data[base + t];
            tile_x[t] = p.x;
            tile_y[t] = p.y;
        }
        __syncthreads();

        if (warp_active) {
            // The loop is written in fixed 32-point rounds so every lane participates in every
            // ballot, even in the last partial round of the tile.
            const int rounds = (batch_count + kWarpSize - 1) >> 5;

            for (int round = 0; round < rounds; ++round) {
                const int tile_pos = (round << 5) + lane;
                const bool valid = (tile_pos < batch_count);

                float dist = CUDART_INF_F;
                int data_idx = kInvalidIndex;

                if (valid) {
                    const float dx = qx - tile_x[tile_pos];
                    const float dy = qy - tile_y[tile_pos];
                    dist = fmaf(dx, dx, dy * dy);  // squared Euclidean distance
                    data_idx = base + tile_pos;
                }

                bool is_candidate = valid && (dist < max_distance);
                unsigned mask = __ballot_sync(kFullMask, is_candidate);
                int num = __popc(mask);

                if (num != 0) {
                    // If this ballot would overflow the shared candidate buffer, flush the current
                    // buffer first, then re-test the same 32 distances against the tightened
                    // threshold before storing them.
                    if (buffer_count + num > K) {
                        merge_buffer_into_result<K>(result_dist,
                                                    result_idx,
                                                    warp_buffer_dist,
                                                    warp_buffer_idx,
                                                    buffer_count,
                                                    max_distance,
                                                    lane);

                        is_candidate = valid && (dist < max_distance);
                        mask = __ballot_sync(kFullMask, is_candidate);
                        num = __popc(mask);
                    }

                    if (num != 0) {
                        const int base_pos = buffer_count;

                        if (is_candidate) {
                            const int offset = __popc(mask & lower_lane_mask);
                            warp_buffer_dist[base_pos + offset] = dist;
                            warp_buffer_idx[base_pos + offset] = data_idx;
                        }

                        buffer_count += num;

                        // The prompt requires merging whenever the warp-private buffer is full.
                        if (buffer_count == K) {
                            merge_buffer_into_result<K>(result_dist,
                                                        result_idx,
                                                        warp_buffer_dist,
                                                        warp_buffer_idx,
                                                        buffer_count,
                                                        max_distance,
                                                        lane);
                        }
                    }
                }
            }
        }

        // Prevent any warp from overwriting the shared tile for the next batch before all warps
        // are done consuming the current one.
        __syncthreads();
    }

    if (warp_active) {
        // Flush the last partially filled candidate buffer.
        if (buffer_count != 0) {
            merge_buffer_into_result<K>(result_dist,
                                        result_idx,
                                        warp_buffer_dist,
                                        warp_buffer_idx,
                                        buffer_count,
                                        max_distance,
                                        lane);
        }

        // Write the final sorted top-K results back to global memory.
        PairIFPOD* const out = reinterpret_cast<PairIFPOD*>(result);
        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int r = 0; r < LOCAL_K; ++r) {
            const std::size_t pos = out_base + static_cast<std::size_t>(lane_base + r);
            out[pos] = PairIFPOD{result_idx[r], result_dist[r]};
        }
    }
}

// K takes one of only six legal values. Specializing on K lets the compiler fully unroll the
// local register loops and the bitonic network, which is much faster than a runtime-generic path.
template <int K>
inline void launch_knn_impl(const float2* query,
                            int query_count,
                            const float2* data,
                            int data_count,
                            std::pair<int, float>* result) {
    constexpr std::size_t kCandidateBytes =
        static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(K) *
        (sizeof(float) + sizeof(int));
    constexpr std::size_t kBytesPerSharedPoint = 2 * sizeof(float);  // x[] + y[]
    constexpr int kRawTilePoints =
        static_cast<int>((kTargetDynamicSmemBytes - kCandidateBytes) / kBytesPerSharedPoint);
    constexpr int kTilePoints = kRawTilePoints & ~31;
    constexpr std::size_t kSmemBytes =
        static_cast<std::size_t>(kTilePoints) * kBytesPerSharedPoint + kCandidateBytes;

    static_assert(kCandidateBytes < kTargetDynamicSmemBytes,
                  "Shared-memory budget too small for the candidate buffers.");
    static_assert(kTilePoints >= 32, "Shared-memory budget leaves no room for a tile.");
    static_assert((kTilePoints % 32) == 0, "Tile size must be warp-aligned.");
    static_assert(kSmemBytes <= kTargetDynamicSmemBytes,
                  "Computed shared-memory usage exceeds the target budget.");

    const dim3 block(kBlockThreads);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

    // Dynamic shared memory above 48 KiB requires opt-in. Prefer the shared-heavy carveout since
    // this kernel is intentionally shared-memory-centric.
    (void)cudaFuncSetAttribute(knn_kernel<K, kWarpsPerBlock, kTilePoints>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(kSmemBytes));
    (void)cudaFuncSetAttribute(knn_kernel<K, kWarpsPerBlock, kTilePoints>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);
    (void)cudaFuncSetCacheConfig(knn_kernel<K, kWarpsPerBlock, kTilePoints>,
                                 cudaFuncCachePreferShared);

    knn_kernel<K, kWarpsPerBlock, kTilePoints>
        <<<grid, block, kSmemBytes>>>(query, query_count, data, data_count, result);
}

}  // namespace knn_detail

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
            knn_detail::launch_knn_impl<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_detail::launch_knn_impl<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_detail::launch_knn_impl<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_detail::launch_knn_impl<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_detail::launch_knn_impl<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_detail::launch_knn_impl<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // The contract guarantees a supported power-of-two K, but keep the function total.
            break;
    }
}