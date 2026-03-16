#include <cuda_runtime.h>
#include <utility>

namespace {

// Exact brute-force 2D k-NN for k in {32, 64, 128, 256, 512, 1024}.
//
// Main design choices:
//
// 1) One warp computes one query, exactly as requested.
// 2) The thread block stages a contiguous tile of data points into shared memory;
//    all warps in the block reuse that tile, which is critical because data_count
//    is typically orders of magnitude larger than query_count.
// 3) Each warp owns a private shared-memory segment of 2*K entries:
//      - [0, K)   : current committed top-K, kept sorted ascending by distance
//      - [K, 2*K) : unsorted candidate buffer
//    When the candidate buffer fills, the warp bitonically sorts 2*K entries
//    and keeps the first K. This amortizes maintenance cost and avoids
//    per-candidate O(K) insertion work.
// 4) Distances are squared Euclidean distances and are non-negative. Therefore
//    IEEE-754 bitwise ordering of the float distance matches numeric ordering.
//    We pack {distance_bits, index} into one 64-bit key, so the warp sorts one
//    integer array instead of separate distance/index arrays.
// 5) No extra device allocations are used; all temporary state lives in shared
//    memory or registers.
//
// On A100/H100-class GPUs, 8 warps per block is preferred because each staged
// data tile is reused by 8 queries. If per-block shared memory is too small for
// a given K, the launcher falls back to 4 warps per block, then to 2 warps with
// a smaller tile that still stays within the default 48 KiB dynamic shared limit.

typedef unsigned long long Key;

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr Key kSentinelKey = (Key(0x7f800000u) << 32) | Key(0xffffffffu); // +inf distance, max index
constexpr size_t kDefaultDynamicSharedBytes = 48u * 1024u;

constexpr int kPreferredTilePoints = 2048; // 16 KiB shared tile of float2
constexpr int kFallbackTilePoints  = 1024; // 8 KiB fallback tile

template <int K, int WARPS_PER_BLOCK, int TILE_POINTS>
struct KernelTraits {
    static constexpr int kBlockThreads = WARPS_PER_BLOCK * kWarpSize;
    static constexpr size_t kSharedBytes =
        size_t(WARPS_PER_BLOCK) * size_t(2 * K) * sizeof(Key) +
        size_t(TILE_POINTS) * sizeof(float2);
};

__device__ __forceinline__ Key pack_key(float dist, unsigned idx) {
    return (Key(__float_as_uint(dist)) << 32) | Key(idx);
}

__device__ __forceinline__ float unpack_distance(Key key) {
    return __uint_as_float(static_cast<unsigned>(key >> 32));
}

__device__ __forceinline__ int unpack_index(Key key) {
    return static_cast<int>(static_cast<unsigned>(key));
}

__device__ __forceinline__ float squared_l2_distance(float qx, float qy, const float2 &p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return __fmaf_rn(dx, dx, dy * dy);
}

// Warp-local bitonic sort in shared memory.
//
// The network is intentionally kept as loops instead of fully unrolling the whole
// thing for large N (up to 2048). The scan over data points is the dominant cost;
// keeping the sort code compact reduces code size and register pressure while still
// being fast enough because flushes are comparatively infrequent.
template <int N>
__device__ __forceinline__ void bitonic_sort_keys(Key *keys, int lane) {
    static_assert((N & (N - 1)) == 0, "Bitonic sort length must be a power of two.");

    #pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll 1
            for (int i = lane; i < N; i += kWarpSize) {
                const int partner = i ^ stride;
                if (partner > i) {
                    const Key a = keys[i];
                    const Key b = keys[partner];
                    const bool ascending = ((i & size) == 0);
                    const bool do_swap = ascending ? (b < a) : (a < b);
                    if (do_swap) {
                        keys[i] = b;
                        keys[partner] = a;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

template <int K>
__device__ __forceinline__ void initialize_topk(
    float qx,
    float qy,
    const float2 *tile,
    Key *warp_keys,
    int lane)
{
    #pragma unroll
    for (int i = lane; i < K; i += kWarpSize) {
        const float dist = squared_l2_distance(qx, qy, tile[i]);
        warp_keys[i] = pack_key(dist, static_cast<unsigned>(i));
    }
    __syncwarp(kFullMask);
    bitonic_sort_keys<K>(warp_keys, lane);
}

template <int K>
__device__ __forceinline__ void flush_candidate_buffer(
    Key *warp_keys,
    int lane,
    int candidate_count)
{
    // Fill the unused tail of the second half with sentinels so sorting 2*K items
    // always leaves the real best K entries in [0, K).
    for (int pos = candidate_count + lane; pos < K; pos += kWarpSize) {
        warp_keys[K + pos] = kSentinelKey;
    }
    __syncwarp(kFullMask);
    bitonic_sort_keys<2 * K>(warp_keys, lane);
}

// Push one warp-wide batch of up to 32 candidate points into the candidate buffer.
//
// "worst" is the current kth element of the committed top-K in [0, K). Buffered
// candidates can only tighten the true threshold, never loosen it, so rejecting
// dist >= worst is always safe.
template <int K>
__device__ __forceinline__ void enqueue_candidate(
    Key key,
    float dist,
    bool valid,
    Key *warp_keys,
    int lane,
    unsigned lane_prefix_mask,
    int &buffer_count,
    float &worst)
{
    bool pass = valid && (dist < worst);
    unsigned mask = __ballot_sync(kFullMask, pass);
    int accepted = __popc(mask);

    if (accepted == 0) {
        return;
    }

    if (buffer_count + accepted > K) {
        flush_candidate_buffer<K>(warp_keys, lane, buffer_count);
        buffer_count = 0;
        worst = unpack_distance(warp_keys[K - 1]);

        // Re-evaluate against the tightened threshold after compaction.
        pass = valid && (dist < worst);
        mask = __ballot_sync(kFullMask, pass);
        accepted = __popc(mask);

        if (accepted == 0) {
            return;
        }
    }

    if (pass) {
        const int rank = __popc(mask & lane_prefix_mask);
        warp_keys[K + buffer_count + rank] = key;
    }

    __syncwarp(kFullMask);
    buffer_count += accepted;

    if (buffer_count == K) {
        flush_candidate_buffer<K>(warp_keys, lane, buffer_count);
        buffer_count = 0;
        worst = unpack_distance(warp_keys[K - 1]);
    }
}

template <int K, int TILE_POINTS, bool FULL_TILE>
__device__ __forceinline__ void process_tile_candidates(
    float qx,
    float qy,
    const float2 *tile,
    int tile_begin,
    int tile_count,
    int start,
    Key *warp_keys,
    int lane,
    unsigned lane_prefix_mask,
    int &buffer_count)
{
    float worst = unpack_distance(warp_keys[K - 1]);
    const int limit = FULL_TILE ? TILE_POINTS : tile_count;

    #pragma unroll 1
    for (int base = start; base < limit; base += kWarpSize) {
        const int local_idx = base + lane;
        const bool valid = FULL_TILE || (local_idx < tile_count);

        Key key = 0;
        float dist = 0.0f;

        if (valid) {
            const float2 p = tile[local_idx];
            dist = squared_l2_distance(qx, qy, p);
            key = pack_key(dist, static_cast<unsigned>(tile_begin + local_idx));
        }

        enqueue_candidate<K>(
            key,
            dist,
            valid,
            warp_keys,
            lane,
            lane_prefix_mask,
            buffer_count,
            worst);
    }
}

template <int K, int WARPS_PER_BLOCK, int TILE_POINTS>
__global__ void knn2d_kernel(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    std::pair<int, float> * __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024, "K out of supported range.");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");
    static_assert(TILE_POINTS >= K, "The first tile must contain at least K points.");

    extern __shared__ __align__(16) unsigned char shared_raw[];
    Key *s_keys = reinterpret_cast<Key *>(shared_raw);
    float2 *s_tile = reinterpret_cast<float2 *>(s_keys + WARPS_PER_BLOCK * (2 * K));

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const unsigned lane_prefix_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);

    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    const bool active = (query_idx < query_count);

    Key *warp_keys = s_keys + warp_in_block * (2 * K);

    // Load the query once per warp and broadcast its coordinates.
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

    int buffer_count = 0;

    // Cooperative global->shared staging of the data set, one contiguous tile at a time.
    for (int tile_begin = 0; tile_begin < data_count; tile_begin += TILE_POINTS) {
        int tile_count = data_count - tile_begin;
        if (tile_count > TILE_POINTS) {
            tile_count = TILE_POINTS;
        }

        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_tile[i] = data[tile_begin + i];
        }
        __syncthreads();

        if (active) {
            if (tile_begin == 0) {
                initialize_topk<K>(qx, qy, s_tile, warp_keys, lane);
            }

            const int start = (tile_begin == 0) ? K : 0;

            if (tile_count == TILE_POINTS) {
                process_tile_candidates<K, TILE_POINTS, true>(
                    qx, qy, s_tile, tile_begin, tile_count, start,
                    warp_keys, lane, lane_prefix_mask, buffer_count);
            } else {
                process_tile_candidates<K, TILE_POINTS, false>(
                    qx, qy, s_tile, tile_begin, tile_count, start,
                    warp_keys, lane, lane_prefix_mask, buffer_count);
            }
        }

        // All warps must finish reading the current tile before the block overwrites it.
        __syncthreads();
    }

    if (active) {
        if (buffer_count > 0) {
            flush_candidate_buffer<K>(warp_keys, lane, buffer_count);
        }

        const size_t out_base = size_t(query_idx) * size_t(K);

        // Direct member stores avoid relying on any device-callable std::pair constructors.
        #pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            const Key key = warp_keys[i];
            result[out_base + i].first  = unpack_index(key);
            result[out_base + i].second = unpack_distance(key);
        }
    }
}

template <int K, int WARPS_PER_BLOCK, int TILE_POINTS>
inline void launch_knn_variant(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result)
{
    constexpr size_t shared_bytes = KernelTraits<K, WARPS_PER_BLOCK, TILE_POINTS>::kSharedBytes;
    constexpr int block_threads = KernelTraits<K, WARPS_PER_BLOCK, TILE_POINTS>::kBlockThreads;

    if (shared_bytes > kDefaultDynamicSharedBytes) {
        cudaFuncSetAttribute(
            knn2d_kernel<K, WARPS_PER_BLOCK, TILE_POINTS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));

        cudaFuncSetAttribute(
            knn2d_kernel<K, WARPS_PER_BLOCK, TILE_POINTS>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100);
    }

    const dim3 block(block_threads);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn2d_kernel<K, WARPS_PER_BLOCK, TILE_POINTS>
        <<<grid, block, shared_bytes>>>(query, query_count, data, data_count, result);
}

template <int K>
inline void launch_knn_best(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result)
{
    int device = 0;
    int max_optin_shared = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    // Prefer 8 warps/block for maximum tile reuse on A100/H100-class GPUs.
    // Fall back to 4 warps if the shared-memory budget for this K is smaller,
    // then to 2 warps with a 1024-point tile that always fits inside 48 KiB.
    if (max_optin_shared >= static_cast<int>(KernelTraits<K, 8, kPreferredTilePoints>::kSharedBytes)) {
        launch_knn_variant<K, 8, kPreferredTilePoints>(query, query_count, data, data_count, result);
    } else if (max_optin_shared >= static_cast<int>(KernelTraits<K, 4, kPreferredTilePoints>::kSharedBytes)) {
        launch_knn_variant<K, 4, kPreferredTilePoints>(query, query_count, data, data_count, result);
    } else {
        launch_knn_variant<K, 2, kFallbackTilePoints>(query, query_count, data, data_count, result);
    }
}

} // anonymous namespace

// As usual for a CUDA launch wrapper, this function is asynchronous with respect
// to the host. The caller may synchronize explicitly or rely on subsequent CUDA
// operations (e.g. cudaMemcpy) that imply synchronization.
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

    switch (k) {
        case 32:
            launch_knn_best<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            launch_knn_best<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            launch_knn_best<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            launch_knn_best<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            launch_knn_best<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            launch_knn_best<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // Per the problem statement, k is always valid.
            break;
    }
}