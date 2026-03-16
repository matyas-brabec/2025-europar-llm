#include <cuda_runtime.h>
#include <cstdint>
#include <utility>

namespace knn2d_internal {

// One warp owns one query.
constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// Packed representation of one (index, distance) pair.
// High 32 bits store the IEEE-754 bits of the non-negative squared distance,
// low 32 bits store the data-point index. Because all distances here are
// squared Euclidean distances, they are non-negative, so integer ordering of
// the float bits matches numerical ordering. This lets the merge path sort a
// single 64-bit key per neighbor candidate.
constexpr uint32_t kInfBits = 0x7f800000u;
constexpr uint32_t kEmptyIndexBits = 0xffffffffu;
constexpr uint64_t kEmptyKey =
    (static_cast<uint64_t>(kInfBits) << 32) | static_cast<uint64_t>(kEmptyIndexBits);

// The public interface uses std::pair<int, float>. On device we write a POD
// with the same binary footprint so we do not rely on std::pair being fully
// device-annotated by the host standard library implementation.
struct ResultPairRaw {
    int first;
    float second;
};

static_assert(sizeof(ResultPairRaw) == 8, "Unexpected ResultPairRaw size.");
static_assert(sizeof(std::pair<int, float>) == sizeof(ResultPairRaw),
              "std::pair<int,float> must be 8 bytes for this implementation.");

// Kernel configuration:
// - 8 warps/block gives good data reuse because one shared-memory batch of data
//   points is reused by 8 simultaneous queries, while still leaving enough block
//   granularity for query counts in the low thousands.
// - K is specialized at compile time to keep each warp's private top-k storage
//   fixed-size and efficiently addressable.
template <int K>
struct KernelConfig {
    static_assert(K >= 32 && K <= 1024, "K out of supported range.");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");

    static constexpr int WARPS_PER_BLOCK = 8;
    static constexpr int THREADS = WARPS_PER_BLOCK * kWarpSize;
    static constexpr int ITEMS_PER_LANE = K / kWarpSize;

    // Per warp we need 2*K packed keys in shared memory:
    //   [0, K)   : scratch area for the current top-k
    //   [K, 2*K) : candidate buffer
    static constexpr int REGION_ENTRIES = 2 * K;
    static constexpr int FIXED_BYTES =
        WARPS_PER_BLOCK * REGION_ENTRIES * static_cast<int>(sizeof(uint64_t)) +
        WARPS_PER_BLOCK * static_cast<int>(sizeof(int));
};

__device__ __forceinline__ uint64_t pack_key(float dist, int idx) {
    return (static_cast<uint64_t>(__float_as_uint(dist)) << 32) |
           static_cast<uint32_t>(idx);
}

__device__ __forceinline__ float unpack_distance(uint64_t key) {
    return __uint_as_float(static_cast<uint32_t>(key >> 32));
}

__device__ __forceinline__ int unpack_index(uint64_t key) {
    return static_cast<int>(static_cast<uint32_t>(key));
}

// Classic bitonic sort on a per-warp shared-memory segment of power-of-two size.
// N is 2*K, so it is always a power of two by construction.
template <int N>
__device__ __forceinline__ void bitonic_sort_shared(uint64_t* keys) {
    const int lane = threadIdx.x & (kWarpSize - 1);

#pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
#pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll 1
            for (int i = lane; i < N; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    const uint64_t a = keys[i];
                    const uint64_t b = keys[j];
                    const bool ascending = ((i & size) == 0);
                    const bool do_swap = ascending ? (a > b) : (a < b);
                    if (do_swap) {
                        keys[i] = b;
                        keys[j] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge the shared candidate buffer [K, 2*K) with the warp-private top-k state.
// The result is written back into the warp-private striped layout:
//   top_keys[t] holds global rank lane + 32*t.
template <int K>
__device__ __forceinline__ void merge_candidate_buffer(
    uint64_t* warp_keys,
    int* warp_count,
    int candidate_count,
    uint64_t (&top_keys)[KernelConfig<K>::ITEMS_PER_LANE],
    float& max_distance) {
    constexpr int ITEMS_PER_LANE = KernelConfig<K>::ITEMS_PER_LANE;
    const int lane = threadIdx.x & (kWarpSize - 1);

    // Copy current top-k into the first half of the per-warp shared region.
#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const int pos = lane + t * kWarpSize;
        warp_keys[pos] = top_keys[t];
    }

    // Pad the unused part of the candidate buffer with +inf so a single sort of
    // 2*K elements can perform the full merge/select.
#pragma unroll
    for (int pos = lane; pos < K; pos += kWarpSize) {
        if (pos >= candidate_count) {
            warp_keys[K + pos] = kEmptyKey;
        }
    }

    __syncwarp();

    bitonic_sort_shared<2 * K>(warp_keys);

    // Reload the K smallest keys back into the warp-private striped layout.
#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const int pos = lane + t * kWarpSize;
        top_keys[t] = warp_keys[pos];
    }

    // The k-th nearest neighbor is the last element of the sorted top-k.
    float lane_last = 0.0f;
    if (lane == (kWarpSize - 1)) {
        lane_last = unpack_distance(top_keys[ITEMS_PER_LANE - 1]);
    }
    max_distance = __shfl_sync(kFullMask, lane_last, kWarpSize - 1);

    // Reset the shared candidate count after the merge.
    if (lane == 0) {
        *warp_count = 0;
    }
    __syncwarp();
}

template <int K>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    ResultPairRaw* __restrict__ result,
    int batch_points) {
    using Cfg = KernelConfig<K>;
    constexpr int ITEMS_PER_LANE = Cfg::ITEMS_PER_LANE;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & (kWarpSize - 1);
    const int query_idx = static_cast<int>(blockIdx.x) * Cfg::WARPS_PER_BLOCK + warp;
    const bool active = (query_idx < query_count);

    // Shared layout:
    //   [0, batch_points)                              : cached data points
    //   [batch_points, batch_points + warps*2*K)      : packed per-warp key regions
    //   [..]                                           : per-warp candidate counts
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* sh_data = reinterpret_cast<float2*>(shared_storage);
    auto* sh_keys = reinterpret_cast<uint64_t*>(sh_data + batch_points);
    auto* sh_count =
        reinterpret_cast<int*>(sh_keys + Cfg::WARPS_PER_BLOCK * Cfg::REGION_ENTRIES);

    uint64_t* const warp_keys = sh_keys + warp * Cfg::REGION_ENTRIES;
    int* const warp_count = sh_count + warp;

    if (lane == 0) {
        *warp_count = 0;
    }
    __syncthreads();

    // Warp-private intermediate result: K packed (index, distance) pairs,
    // striped across the 32 lanes, i.e. lane owns positions lane + 32*t.
    uint64_t top_keys[ITEMS_PER_LANE];
#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        top_keys[t] = kEmptyKey;
    }

    float max_distance = CUDART_INF_F;
    int buffer_count = 0;  // local mirror of the shared candidate count

    float qx = 0.0f;
    float qy = 0.0f;
    if (active) {
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);
    }

    // Iterate over the data set in shared-memory batches.
    for (int batch_base = 0; batch_base < data_count; batch_base += batch_points) {
        const int remaining = data_count - batch_base;
        const int current_batch = (remaining < batch_points) ? remaining : batch_points;

        // Cooperative block-wide global->shared load of the current batch.
        for (int i = tid; i < current_batch; i += Cfg::THREADS) {
            sh_data[i] = data[batch_base + i];
        }
        __syncthreads();

        if (active) {
            // Process the cached batch in warp-sized tiles. Each lane handles one
            // candidate point per tile.
            for (int tile = 0; tile < current_batch; tile += kWarpSize) {
                const int local_idx = tile + lane;

                float dist = 0.0f;
                int data_idx = 0;
                bool take = false;

                if (local_idx < current_batch) {
                    const float2 p = sh_data[local_idx];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);  // squared L2; no sqrt
                    data_idx = batch_base + local_idx;
                    take = (dist < max_distance);
                }

                const unsigned candidate_mask = __ballot_sync(kFullMask, take);
                const int candidate_lanes = __popc(candidate_mask);

                if (candidate_lanes != 0) {
                    // If the current tile would overflow the candidate buffer, first
                    // merge the existing buffer into the intermediate top-k.
                    if (buffer_count + candidate_lanes > K) {
                        merge_candidate_buffer<K>(
                            warp_keys, warp_count, buffer_count, top_keys, max_distance);
                        buffer_count = 0;
                    }

                    // Required by the problem statement: add each surviving candidate
                    // with atomicAdd on the shared candidate count.
                    if (take) {
                        const int pos = atomicAdd(warp_count, 1);
                        warp_keys[K + pos] = pack_key(dist, data_idx);
                    }

                    buffer_count += candidate_lanes;
                    __syncwarp();

                    // Merge immediately once the candidate buffer becomes full.
                    if (buffer_count == K) {
                        merge_candidate_buffer<K>(
                            warp_keys, warp_count, buffer_count, top_keys, max_distance);
                        buffer_count = 0;
                    }
                }
            }
        }

        __syncthreads();
    }

    if (active) {
        // Final merge for the partially filled candidate buffer, if any.
        if (buffer_count != 0) {
            merge_candidate_buffer<K>(
                warp_keys, warp_count, buffer_count, top_keys, max_distance);
        }

        // Write the final sorted top-k for this query.
        const int out_base = query_idx * K;
#pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const int out_pos = out_base + lane + t * kWarpSize;
            const uint64_t key = top_keys[t];

            ResultPairRaw out;
            out.first = unpack_index(key);
            out.second = unpack_distance(key);
            result[out_pos] = out;
        }
    }
}

// Batch-size heuristic:
// - If the fixed per-query shared state is small enough, target two resident
//   blocks/SM by budgeting half of the SM shared memory per block.
// - If the fixed state already forces a single block/SM (notably K=1024 on SM80),
//   use the full per-block shared-memory limit to maximize data reuse.
// - Round to a multiple of one warp so the per-warp processing loop stays aligned.
template <int K>
inline int compute_batch_points(int max_block_smem, int max_sm_smem) {
    using Cfg = KernelConfig<K>;

    int budget = max_block_smem;
    const int two_block_budget = max_sm_smem / 2;
    if (Cfg::FIXED_BYTES <= two_block_budget) {
        budget = (max_block_smem < two_block_budget) ? max_block_smem : two_block_budget;
    }

    const int available = budget - Cfg::FIXED_BYTES;
    if (available <= 0) {
        return 0;
    }

    const int raw_points = available / static_cast<int>(sizeof(float2));
    return (raw_points / kWarpSize) * kWarpSize;
}

template <int K>
inline void launch_knn_kernel(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    ResultPairRaw* result) {
    using Cfg = KernelConfig<K>;

    // Default to SM80-safe values; overwrite with runtime values when available.
    int device = 0;
    int max_block_smem = 163840;
    int max_sm_smem = 163840;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(
        &max_block_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    (void)cudaDeviceGetAttribute(
        &max_sm_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

    const int batch_points = compute_batch_points<K>(max_block_smem, max_sm_smem);
    if (batch_points <= 0) {
        return;
    }

    const int shared_bytes =
        Cfg::FIXED_BYTES + batch_points * static_cast<int>(sizeof(float2));
    const dim3 block(Cfg::THREADS);
    const dim3 grid((query_count + Cfg::WARPS_PER_BLOCK - 1) / Cfg::WARPS_PER_BLOCK);

    (void)cudaFuncSetAttribute(
        knn_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    (void)cudaFuncSetAttribute(
        knn_kernel<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    knn_kernel<K><<<grid, block, static_cast<size_t>(shared_bytes)>>>(
        query, query_count, data, data_count, result, batch_points);

    // Surface launch/configuration errors without forcing device synchronization.
    (void)cudaPeekAtLastError();
}

}  // namespace knn2d_internal

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

    auto* raw_result = reinterpret_cast<knn2d_internal::ResultPairRaw*>(result);

    // Dispatch on K so the compiler sees fixed-size warp-private top-k storage.
    switch (k) {
        case 32:
            knn2d_internal::launch_knn_kernel<32>(
                query, query_count, data, data_count, raw_result);
            break;
        case 64:
            knn2d_internal::launch_knn_kernel<64>(
                query, query_count, data, data_count, raw_result);
            break;
        case 128:
            knn2d_internal::launch_knn_kernel<128>(
                query, query_count, data, data_count, raw_result);
            break;
        case 256:
            knn2d_internal::launch_knn_kernel<256>(
                query, query_count, data, data_count, raw_result);
            break;
        case 512:
            knn2d_internal::launch_knn_kernel<512>(
                query, query_count, data, data_count, raw_result);
            break;
        case 1024:
            knn2d_internal::launch_knn_kernel<1024>(
                query, query_count, data, data_count, raw_result);
            break;
        default:
            // The problem statement guarantees a valid power-of-two K in [32, 1024].
            break;
    }
}