#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace knn_detail {

// ----------------------------------------------------------------------------
// Exact warp-per-query k-NN for 2D points.
//
// Design choices tuned for modern data-center GPUs (A100/H100 class):
//   * One warp computes one query, as requested.
//   * A 256-thread block contains 8 warps, so each shared-memory data tile is
//     reused by 8 independent queries before the block moves on.
//   * No extra device memory is allocated. All temporary state lives in shared
//     memory and registers.
//   * Each warp owns a private top-k buffer in shared memory. Because k can be
//     as large as 1024, shared memory is the right place for the exact
//     intermediate result.
//   * Distances and indices are stored as a single packed 64-bit key:
//         high 32 bits = raw IEEE-754 bits of the non-negative distance,
//         low  32 bits = index.
//     For non-negative floats, unsigned integer order matches numeric order, so
//     one 64-bit comparison orders by (distance, index).
//   * The first k points bootstrap the top-k buffer, then a warp-wide bitonic
//     sort makes it ordered once.
//   * Later points are processed in shared-memory tiles. Each lane evaluates its
//     own subset of tile points, compacting all "better than current kth"
//     candidates from the tile into a warp-private candidate buffer. The warp
//     then sorts that candidate batch and merges it into the intermediate top-k
//     result.
//   * For k = 1024, a 128-point tile keeps the shared-memory footprint low
//     enough to preserve good occupancy on A100-class parts. For smaller k,
//     256-point tiles reduce synchronization and per-tile overhead.
// ----------------------------------------------------------------------------

constexpr int kWarpSize       = 32;
constexpr int kBlockThreads   = 256;
constexpr int kWarpsPerBlock  = kBlockThreads / kWarpSize;
constexpr unsigned kFullMask  = 0xffffffffu;

using Key = unsigned long long;

static_assert(sizeof(float2) == sizeof(Key), "float2 must occupy 8 bytes");

constexpr Key kInfKey =
    (static_cast<Key>(0x7f800000u) << 32) | static_cast<Key>(0xffffffffu);

template <int K, int TILE>
struct KernelTraits {
    static_assert((K & (K - 1)) == 0, "k must be a power of two");
    static_assert(K >= 32 && K <= 1024, "supported k range is [32, 1024]");
    static_assert(TILE == 128 || TILE == 256, "supported tile sizes are 128 and 256");
    static_assert((TILE & (TILE - 1)) == 0, "tile size must be a power of two");
    static_assert((K % kWarpSize) == 0, "k must be a multiple of 32");
    static_assert((TILE % kWarpSize) == 0, "tile size must be a multiple of 32");

    enum {
        ItemsPerThread = TILE / kWarpSize,
        SharedBytes =
            static_cast<int>(sizeof(float2) * TILE +
                             sizeof(Key) * kWarpsPerBlock * (K + TILE))
    };
};

template <int K>
struct DefaultTile {
    enum { Value = 256 };
};

template <>
struct DefaultTile<1024> {
    enum { Value = 128 };
};

__device__ __forceinline__ Key pack_key(const float dist, const unsigned idx) {
    // Distances are squared L2 norms, hence non-negative. For non-negative
    // floats, the raw IEEE-754 bit pattern is monotonic as an unsigned integer.
    return (static_cast<Key>(__float_as_uint(dist)) << 32) | static_cast<Key>(idx);
}

__device__ __forceinline__ float unpack_distance(const Key key) {
    return __uint_as_float(static_cast<unsigned int>(key >> 32));
}

__device__ __forceinline__ int unpack_index(const Key key) {
    return static_cast<int>(static_cast<unsigned int>(key));
}

__device__ __forceinline__ float l2_sq(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

__device__ __forceinline__ int warp_sum_int(int v) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(kFullMask, v, offset);
    }
    return __shfl_sync(kFullMask, v, 0);
}

__device__ __forceinline__ int warp_exclusive_scan_int(const int v) {
    int x = v;
    const int lane = threadIdx.x & (kWarpSize - 1);
    #pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int y = __shfl_up_sync(kFullMask, x, offset);
        if (lane >= offset) {
            x += y;
        }
    }
    return x - v;
}

template <int LEN>
__device__ __forceinline__ void warp_bitonic_sort_shared(Key* const buf) {
    // One warp sorts one power-of-two shared-memory buffer.
    const int lane = threadIdx.x & (kWarpSize - 1);

    #pragma unroll
    for (int size = 2; size <= LEN; size <<= 1) {
        #pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int i = lane; i < LEN; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    Key a = buf[i];
                    Key b = buf[j];
                    const bool ascending = ((i & size) == 0);
                    if (ascending) {
                        if (a > b) {
                            buf[i] = b;
                            buf[j] = a;
                        }
                    } else {
                        if (a < b) {
                            buf[i] = b;
                            buf[j] = a;
                        }
                    }
                }
            }
            __syncwarp();
        }
    }
}

template <int TILE>
__device__ __forceinline__ int candidate_sort_length(const int total);

template <>
__device__ __forceinline__ int candidate_sort_length<128>(const int total) {
    if (total <= 32) return 32;
    if (total <= 64) return 64;
    return 128;
}

template <>
__device__ __forceinline__ int candidate_sort_length<256>(const int total) {
    if (total <= 32)  return 32;
    if (total <= 64)  return 64;
    if (total <= 128) return 128;
    return 256;
}

template <int TILE>
__device__ __forceinline__ void sort_candidate_buffer(Key* const buf, const int len);

template <>
__device__ __forceinline__ void sort_candidate_buffer<128>(Key* const buf, const int len) {
    if (len == 32) {
        warp_bitonic_sort_shared<32>(buf);
    } else if (len == 64) {
        warp_bitonic_sort_shared<64>(buf);
    } else {
        warp_bitonic_sort_shared<128>(buf);
    }
}

template <>
__device__ __forceinline__ void sort_candidate_buffer<256>(Key* const buf, const int len) {
    if (len == 32) {
        warp_bitonic_sort_shared<32>(buf);
    } else if (len == 64) {
        warp_bitonic_sort_shared<64>(buf);
    } else if (len == 128) {
        warp_bitonic_sort_shared<128>(buf);
    } else {
        warp_bitonic_sort_shared<256>(buf);
    }
}

template <int K, int TILE>
__global__ __launch_bounds__(256, 2)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                std::pair<int, float>* __restrict__ result) {
    typedef KernelTraits<K, TILE> Traits;

    extern __shared__ Key smem[];

    // Shared-memory layout:
    //   [ data tile (float2[TILE]) ][ per-warp best[K] ][ per-warp candidates[TILE] ]
    float2* const s_data = reinterpret_cast<float2*>(smem);
    Key* const s_best    = reinterpret_cast<Key*>(s_data + TILE);
    Key* const s_cand    = s_best + kWarpsPerBlock * K;

    const int tid     = threadIdx.x;
    const int lane    = tid & (kWarpSize - 1);
    const int warp_id = tid >> 5;

    const int query_idx   = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool active_warp = (query_idx < query_count);

    Key* const best = s_best + warp_id * K;
    Key* const cand = s_cand + warp_id * TILE;

    // Broadcast the query point from lane 0 of the owning warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && active_warp) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // ------------------------------------------------------------------------
    // Phase 1: bootstrap the top-k list from the first K data points.
    // ------------------------------------------------------------------------
    for (int batch_start = 0; batch_start < K; batch_start += TILE) {
        int tile_count = K - batch_start;
        if (tile_count > TILE) tile_count = TILE;

        // The whole block cooperatively caches the current data tile.
        if (tid < tile_count) {
            s_data[tid] = data[batch_start + tid];
        }
        __syncthreads();

        if (active_warp) {
            #pragma unroll
            for (int item = 0; item < Traits::ItemsPerThread; ++item) {
                const int pos = lane + item * kWarpSize;
                if (pos < tile_count) {
                    const int data_idx = batch_start + pos;
                    best[data_idx] = pack_key(l2_sq(qx, qy, s_data[pos]),
                                              static_cast<unsigned>(data_idx));
                }
            }
        }
        __syncthreads();
    }

    if (active_warp) {
        warp_bitonic_sort_shared<K>(best);
    }
    __syncthreads();

    // ------------------------------------------------------------------------
    // Phase 2: stream the remainder of the data set in shared-memory tiles.
    //
    // Each lane evaluates its own subset of the tile and keeps every point that
    // beats the current kth distance. These lane-local candidates are compacted
    // into a contiguous warp-private candidate buffer, sorted, and merged into
    // the intermediate top-k result.
    // ------------------------------------------------------------------------
    for (int batch_start = K; batch_start < data_count; batch_start += TILE) {
        int tile_count = data_count - batch_start;
        if (tile_count > TILE) tile_count = TILE;

        if (tid < tile_count) {
            s_data[tid] = data[batch_start + tid];
        }
        __syncthreads();

        if (active_warp) {
            // Read the current kth key once and broadcast it within the warp.
            // Filtering by distance alone is sufficient because the problem
            // allows arbitrary tie handling.
            Key threshold_key = 0;
            if (lane == 0) {
                threshold_key = best[K - 1];
            }
            threshold_key = __shfl_sync(kFullMask, threshold_key, 0);
            const float threshold_dist = unpack_distance(threshold_key);

            // Keep the tile-local candidates in registers. The array index is a
            // compile-time constant thanks to full unrolling, so this remains
            // register-backed rather than spilling to local memory.
            Key local_keys[Traits::ItemsPerThread];
            unsigned valid_mask = 0u;
            int local_count = 0;

            #pragma unroll
            for (int item = 0; item < Traits::ItemsPerThread; ++item) {
                const int pos = lane + item * kWarpSize;
                Key key = kInfKey;

                if (pos < tile_count) {
                    const int data_idx = batch_start + pos;
                    const float dist = l2_sq(qx, qy, s_data[pos]);
                    if (dist < threshold_dist) {
                        key = pack_key(dist, static_cast<unsigned>(data_idx));
                        valid_mask |= (1u << item);
                        ++local_count;
                    }
                }

                local_keys[item] = key;
            }

            const int total_candidates = warp_sum_int(local_count);

            if (total_candidates > 0) {
                // Compact all lane-local candidates into a contiguous warp-
                // private shared buffer using an exclusive scan over counts.
                const int base = warp_exclusive_scan_int(local_count);
                int out = base;

                #pragma unroll
                for (int item = 0; item < Traits::ItemsPerThread; ++item) {
                    if (valid_mask & (1u << item)) {
                        cand[out++] = local_keys[item];
                    }
                }

                // Round the populated prefix up to the next power-of-two sort
                // network size, so sparse tiles do not pay for a full-tile sort.
                const int sort_len = candidate_sort_length<TILE>(total_candidates);

                for (int i = total_candidates + lane; i < sort_len; i += kWarpSize) {
                    cand[i] = kInfKey;
                }
                __syncwarp();

                sort_candidate_buffer<TILE>(cand, sort_len);
                __syncwarp();

                if (lane == 0) {
                    // Determine how many sorted candidates survive into the new
                    // top-k. Let r be that number. Since both arrays are sorted,
                    // r is the largest value such that:
                    //     cand[r - 1] < best[K - r]
                    int lo = 0;
                    int hi = (total_candidates < K) ? total_candidates : K;

                    while (lo < hi) {
                        const int mid = (lo + hi + 1) >> 1;
                        if (cand[mid - 1] < best[K - mid]) {
                            lo = mid;
                        } else {
                            hi = mid - 1;
                        }
                    }

                    const int r = lo;

                    // In-place backward merge.
                    //
                    // Only cand[0 .. r-1] survive. They displace exactly r
                    // elements from the tail of best[], so best[0 .. K-r-1] has
                    // exactly r free slots above it. That makes a backward merge
                    // into best[] safe without any extra storage.
                    int i = K - r - 1;
                    int j = r - 1;
                    int dst = K - 1;

                    while (j >= 0) {
                        if (i >= 0 && best[i] > cand[j]) {
                            best[dst--] = best[i--];
                        } else {
                            best[dst--] = cand[j--];
                        }
                    }
                }

                __syncwarp();
            }
        }

        __syncthreads();
    }

    // The top-k buffer is kept sorted ascending, so writing results is a simple
    // strided warp store.
    if (active_warp) {
        const size_t out_base =
            static_cast<size_t>(query_idx) * static_cast<size_t>(K);

        for (int i = lane; i < K; i += kWarpSize) {
            const Key key = best[i];
            result[out_base + static_cast<size_t>(i)].first  = unpack_index(key);
            result[out_base + static_cast<size_t>(i)].second = unpack_distance(key);
        }
    }
}

template <int K, int TILE>
inline void launch_knn_impl(const float2* query,
                            const int query_count,
                            const float2* data,
                            const int data_count,
                            std::pair<int, float>* result) {
    // Opt into the shared-memory size required by the chosen specialization.
    // The kernel itself remains asynchronous; the caller can synchronize if
    // host-side completion is needed.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, TILE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        KernelTraits<K, TILE>::SharedBytes);

    (void)cudaFuncSetAttribute(
        knn_kernel<K, TILE>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const dim3 block(kBlockThreads);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

    knn_kernel<K, TILE><<<grid, block, KernelTraits<K, TILE>::SharedBytes>>>(
        query, query_count, data, data_count, result);
}

template <int K>
inline void launch_knn_for_k(const float2* query,
                             const int query_count,
                             const float2* data,
                             const int data_count,
                             std::pair<int, float>* result) {
    launch_knn_impl<K, DefaultTile<K>::Value>(
        query, query_count, data, data_count, result);
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

    // Dispatch to a compile-time specialization so the sort network size, the
    // per-warp top-k storage, and the tile-local candidate storage are all
    // fixed constants. That lets the compiler unroll the hot inner loops.
    switch (k) {
        case 32:
            knn_detail::launch_knn_for_k<32>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_detail::launch_knn_for_k<64>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_detail::launch_knn_for_k<128>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_detail::launch_knn_for_k<256>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_detail::launch_knn_for_k<512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_detail::launch_knn_for_k<1024>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem statement guarantees a valid power-of-two k in
            // [32, 1024], so this path is unreachable for valid callers.
            break;
    }
}