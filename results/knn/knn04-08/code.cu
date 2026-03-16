#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace knn_detail {

// Optimized warp-per-query 2D k-NN for large query/data sets.
//
// Design choices tuned for modern data-center GPUs (A100/H100 class):
//   * One warp computes one query.
//   * A 256-thread block contains 8 warps, so each shared-memory data tile is
//     reused by 8 queries before the next tile is loaded.
//   * The block caches 1024 data points per tile. This is large enough to
//     materially reduce barrier overhead versus a 256-point tile, while still
//     fitting comfortably together with the K=1024 per-warp working set.
//   * K is specialized at compile time (32/64/128/256/512/1024) so that the
//     sort/merge networks remain static and branch-free with respect to K.
//
// Per-warp temporary state in shared memory:
//   neighbors[0 .. K-1]     : current sorted top-K
//   neighbors[K .. 2K-1]    : unsorted buffer of newly accepted candidates
//
// Multiple candidates are inserted simultaneously:
//   * Each round processes 32 candidates (one per lane).
//   * Passing lanes compact themselves into the buffer via ballot + prefix rank.
//   * When the buffer would overflow, it is flushed:
//       - sort buffer ascending
//       - first flush: copy buffer into top-K
//       - later flushes: reverse the buffer and bitonic-merge [top-K | buffer]
//
// The kth-best threshold only decreases as more points are seen. Therefore,
// keeping a stale threshold while candidates are buffered is safe: it can only
// admit extra candidates (more work), never reject a true neighbor.

using result_pair_t = std::pair<int, float>;
using packed_neighbor_t = std::uint64_t;

constexpr int kWarpSize       = 32;
constexpr int kBlockThreads   = 256;
constexpr int kWarpsPerBlock  = kBlockThreads / kWarpSize;
constexpr int kTilePoints     = 1024;   // 4 loads per thread, 32 rounds per full tile
constexpr unsigned kFullMask  = 0xffffffffu;

// Packed key = [distance_bits | index_bits].
// Squared Euclidean distances are non-negative, so IEEE-754 bit ordering for
// float is monotone under unsigned integer comparison. The index occupies the
// low 32 bits and acts as a deterministic tie-breaker.
constexpr packed_neighbor_t kInfPacked =
    (static_cast<packed_neighbor_t>(0x7f800000u) << 32) |
    static_cast<packed_neighbor_t>(0xffffffffu);

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of warp size.");
static_assert(kTilePoints % kWarpSize == 0, "Tile size must be a multiple of warp size.");
static_assert(kTilePoints == 4 * kBlockThreads,
              "The tile loader below is intentionally unrolled for four loads per thread.");

__device__ __forceinline__ packed_neighbor_t pack_neighbor(const float dist, const int idx) {
    return (static_cast<packed_neighbor_t>(__float_as_uint(dist)) << 32) |
           static_cast<std::uint32_t>(idx);
}

__device__ __forceinline__ float unpack_distance(const packed_neighbor_t packed) {
    return __uint_as_float(static_cast<unsigned int>(packed >> 32));
}

__device__ __forceinline__ int unpack_index(const packed_neighbor_t packed) {
    return static_cast<int>(static_cast<std::uint32_t>(packed));
}

// Standard in-place bitonic sort on a power-of-two number of elements in shared memory.
template <int N>
__device__ __forceinline__ void warp_bitonic_sort_asc(packed_neighbor_t* a) {
    static_assert((N & (N - 1)) == 0, "Bitonic sort size must be a power of two.");

    const int lane = threadIdx.x & (kWarpSize - 1);

    #pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += kWarpSize) {
                const int partner = i ^ stride;
                if (partner > i) {
                    const bool ascending = ((i & size) == 0);
                    packed_neighbor_t ai = a[i];
                    packed_neighbor_t aj = a[partner];

                    if ((ascending && ai > aj) || (!ascending && ai < aj)) {
                        a[i]       = aj;
                        a[partner] = ai;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

// Bitonic merge for a power-of-two array that is already bitonic and must end up ascending.
template <int N>
__device__ __forceinline__ void warp_bitonic_merge_asc(packed_neighbor_t* a) {
    static_assert((N & (N - 1)) == 0, "Bitonic merge size must be a power of two.");

    const int lane = threadIdx.x & (kWarpSize - 1);

    #pragma unroll 1
    for (int stride = N >> 1; stride > 0; stride >>= 1) {
        for (int i = lane; i < N; i += kWarpSize) {
            const int partner = i ^ stride;
            if (partner > i) {
                packed_neighbor_t ai = a[i];
                packed_neighbor_t aj = a[partner];

                if (ai > aj) {
                    a[i]       = aj;
                    a[partner] = ai;
                }
            }
        }
        __syncwarp(kFullMask);
    }
}

// Flush the warp-private buffer into the sorted top-K state.
template <int K>
__device__ __forceinline__ void flush_buffer(
    packed_neighbor_t* neighbors,
    int&               buf_count,
    bool&              top_ready,
    float&             threshold)
{
    if (buf_count == 0) {
        return;
    }

    const int lane = threadIdx.x & (kWarpSize - 1);

    // Pad the unused part of the buffer with +inf so that sorting/merging can
    // operate on fixed-size power-of-two arrays.
    __syncwarp(kFullMask);
    for (int i = lane + buf_count; i < K; i += kWarpSize) {
        neighbors[K + i] = kInfPacked;
    }
    __syncwarp(kFullMask);

    // Sort buffered candidates ascending.
    warp_bitonic_sort_asc<K>(neighbors + K);

    if (!top_ready) {
        // First flush: initialize the exact top-K from the first K points seen.
        for (int i = lane; i < K; i += kWarpSize) {
            neighbors[i] = neighbors[K + i];
        }
        top_ready = true;
    } else {
        // Later flushes: top-K is already ascending, so reverse the ascending
        // buffer into descending order. Concatenating [ascending | descending]
        // forms a bitonic sequence, which can be fully sorted with a bitonic merge.
        for (int i = lane; i < (K >> 1); i += kWarpSize) {
            packed_neighbor_t tmp            = neighbors[K + i];
            neighbors[K + i]                 = neighbors[(2 * K - 1) - i];
            neighbors[(2 * K - 1) - i]       = tmp;
        }
        __syncwarp(kFullMask);

        warp_bitonic_merge_asc<2 * K>(neighbors);
    }

    __syncwarp(kFullMask);

    // Publish the new exact kth threshold to the whole warp.
    float new_threshold = 0.0f;
    if (lane == 0) {
        new_threshold = unpack_distance(neighbors[K - 1]);
    }
    threshold = __shfl_sync(kFullMask, new_threshold, 0);
    buf_count = 0;
}

template <int K>
constexpr std::size_t shared_bytes_for() {
    return static_cast<std::size_t>(kTilePoints) * sizeof(float2) +
           static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(2 * K) * sizeof(packed_neighbor_t);
}

// Fits within the A100 opt-in per-block shared-memory limit (163,840 bytes).
static_assert(shared_bytes_for<1024>() <= 163840u,
              "The largest specialization must fit within A100/H100-class shared memory.");

template <int K>
__global__ __launch_bounds__(kBlockThreads)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    result_pair_t* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0,
                  "K must be a supported power of two in [32, 1024].");

    extern __shared__ unsigned char smem_raw[];

    float2* tile = reinterpret_cast<float2*>(smem_raw);
    packed_neighbor_t* neighbor_storage =
        reinterpret_cast<packed_neighbor_t*>(tile + kTilePoints);

    const int tid  = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    const int qid = blockIdx.x * kWarpsPerBlock + warp;
    const bool active_warp = (qid < query_count);

    packed_neighbor_t* warp_neighbors = neighbor_storage + warp * (2 * K);

    float qx = 0.0f;
    float qy = 0.0f;
    float threshold = CUDART_INF_F;
    int   buf_count = 0;
    bool  top_ready = false;

    if (active_warp) {
        if (lane == 0) {
            const float2 q = query[qid];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);
    }

    for (int tile_start = 0; tile_start < data_count; tile_start += kTilePoints) {
        const int remaining  = data_count - tile_start;
        const int batch_count = (remaining < kTilePoints) ? remaining : kTilePoints;

        // Cooperative block load: 1024 points per tile, 4 points per thread.
        const int load0 = tid;
        const int load1 = tid + kBlockThreads;
        const int load2 = tid + 2 * kBlockThreads;
        const int load3 = tid + 3 * kBlockThreads;

        if (load0 < batch_count) tile[load0] = data[tile_start + load0];
        if (load1 < batch_count) tile[load1] = data[tile_start + load1];
        if (load2 < batch_count) tile[load2] = data[tile_start + load2];
        if (load3 < batch_count) tile[load3] = data[tile_start + load3];

        __syncthreads();

        if (active_warp) {
            // Each round handles 32 candidates from the shared tile.
            #pragma unroll 1
            for (int base = 0; base < batch_count; base += kWarpSize) {
                const int  local_idx = base + lane;
                const bool valid     = (local_idx < batch_count);
                const int  data_idx  = tile_start + local_idx;

                float dist = CUDART_INF_F;
                if (valid) {
                    const float2 p  = tile[local_idx];
                    const float  dx = qx - p.x;
                    const float  dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);
                }

                bool pass = valid && (!top_ready || dist < threshold);
                unsigned mask = __ballot_sync(kFullMask, pass);
                int accepted_count = __popc(mask);

                // Flush only if this round would overflow the buffer. This keeps
                // the exact threshold reasonably fresh without forcing a flush
                // after every accepted candidate when K is small.
                if (buf_count + accepted_count > K) {
                    flush_buffer<K>(warp_neighbors, buf_count, top_ready, threshold);

                    // Re-evaluate the current round against the new exact threshold.
                    pass = valid && (!top_ready || dist < threshold);
                    mask = __ballot_sync(kFullMask, pass);
                    accepted_count = __popc(mask);
                }

                if (accepted_count) {
                    const unsigned lower_lanes = (lane == 0) ? 0u : ((1u << lane) - 1u);
                    const int rank = __popc(mask & lower_lanes);

                    if (pass) {
                        warp_neighbors[K + buf_count + rank] = pack_neighbor(dist, data_idx);
                    }

                    buf_count += accepted_count;
                }
            }
        }

        __syncthreads();
    }

    if (active_warp) {
        // Final exact top-K.
        flush_buffer<K>(warp_neighbors, buf_count, top_ready, threshold);

        const std::size_t out_base = static_cast<std::size_t>(qid) * static_cast<std::size_t>(K);

        // The top-K array is already sorted nearest-to-farthest.
        for (int i = lane; i < K; i += kWarpSize) {
            const packed_neighbor_t packed = warp_neighbors[i];
            result[out_base + static_cast<std::size_t>(i)].first  = unpack_index(packed);
            result[out_base + static_cast<std::size_t>(i)].second = unpack_distance(packed);
        }
    }
}

template <int K>
inline void launch_knn_specialized(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    result_pair_t* result)
{
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0,
                  "Unsupported K specialization.");

    constexpr int shared_bytes = static_cast<int>(shared_bytes_for<K>());

    // Opt into the large dynamic shared-memory footprint required by the warp-private
    // top-K state, and bias the SM carveout toward shared memory.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shared_bytes);
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    knn_kernel<K><<<blocks, kBlockThreads, shared_bytes>>>(
        query, query_count, data, data_count, result);
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
    // The launch is intentionally asynchronous; callers can choose their own
    // synchronization and error-handling policy around this API.
    if (query_count <= 0) {
        return;
    }

    // Monomorphize K so that the top-K networks are fully specialized.
    switch (k) {
        case 32:   knn_detail::launch_knn_specialized<32>(query, query_count, data, data_count, result);   return;
        case 64:   knn_detail::launch_knn_specialized<64>(query, query_count, data, data_count, result);   return;
        case 128:  knn_detail::launch_knn_specialized<128>(query, query_count, data, data_count, result);  return;
        case 256:  knn_detail::launch_knn_specialized<256>(query, query_count, data, data_count, result);  return;
        case 512:  knn_detail::launch_knn_specialized<512>(query, query_count, data, data_count, result);  return;
        case 1024: knn_detail::launch_knn_specialized<1024>(query, query_count, data, data_count, result); return;
        default:   return; // Unreachable for valid inputs, kept for interface robustness.
    }
}