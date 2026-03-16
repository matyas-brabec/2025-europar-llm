#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace {

// Exact brute-force 2D k-NN specialized for the only six legal K values.
//
// Mapping:
//   * one warp computes one query;
//   * one 256-thread block owns 8 warps / 8 queries;
//   * the block streams the data set through a shared-memory tile;
//   * each warp computes one 32-point candidate chunk at a time (one candidate per lane),
//     sorts that chunk in-warp, then exact-merges it with its current sorted top-K.
//
// The top-K buffer is kept in shared memory rather than registers because K can be as large
// as 1024. Shared memory also enables exact warp-parallel merge-path updates with arbitrary
// indexing while keeping register pressure reasonable.
//
// Shared-memory layout intentionally uses SoA:
//   * tile points are cached as x[] and y[] (not float2) to avoid 64-bit shared-memory bank
//     issues on the hot read path;
//   * top-K is stored as separate dist[] and idx[] arrays for the same reason;
//   * one pad slot is inserted every 32 logical top-K elements. During the merge, lane l writes
//     a contiguous segment of K/32 logical outputs; the padding maps those blocked writes onto
//     distinct shared-memory banks.
constexpr int kWarpSize            = 32;
constexpr int kBlockThreads        = 256;   // 8 warps/block: good tile reuse without starving the grid at ~1k queries.
constexpr int kWarpsPerBlock       = kBlockThreads / kWarpSize;
constexpr int kTilePoints          = 1536;  // 6 point loads/thread/tile; still allows 2 resident blocks/SM at K=1024 on A100/H100.
constexpr int kTileLoadsPerThread  = kTilePoints / kBlockThreads;
constexpr unsigned kFullMask       = 0xFFFFFFFFu;
constexpr std::size_t kSharedAlign = 128;
constexpr int kPartitionSteps      = 5;     // merge-path search range is <= 32 because the second list always has 32 items.

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of warp size.");
static_assert(kTilePoints % kWarpSize == 0, "Tile size must be a multiple of warp size.");
static_assert(kTileLoadsPerThread * kBlockThreads == kTilePoints, "Tile size must evenly distribute across the block.");

__host__ __device__ constexpr std::size_t align_up(std::size_t x, std::size_t a) {
    return (x + a - 1) & ~(a - 1);
}

template <int K>
struct KnnTraits {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    static constexpr int ItemsPerThread = K / kWarpSize;
    static constexpr int PaddedK        = K + (K / kWarpSize);                 // one pad every 32 logical elements
    static constexpr int WorstPhys      = (K - 1) + ((K - 1) >> 5);            // physical shared index of logical K-1
};

template <int K>
struct SharedLayout {
    static constexpr std::size_t XOffset    = 0;
    static constexpr std::size_t YOffset    = align_up(XOffset    + kTilePoints * sizeof(float), kSharedAlign);
    static constexpr std::size_t TopDOffset = align_up(YOffset    + kTilePoints * sizeof(float), kSharedAlign);
    static constexpr std::size_t TopIOffset = align_up(TopDOffset + kWarpsPerBlock * KnnTraits<K>::PaddedK * sizeof(float), kSharedAlign);
    static constexpr std::size_t Bytes      = align_up(TopIOffset + kWarpsPerBlock * KnnTraits<K>::PaddedK * sizeof(int),   kSharedAlign);
};

static_assert(SharedLayout<1024>::Bytes <= 163840,
              "Chosen tile size exceeds the shared-memory budget targeted for A100/H100.");

__device__ __forceinline__ int topk_phys_index(const int logical) {
    // logical -> physical with one pad slot after every 32 logical elements
    return logical + (logical >> 5);
}

__device__ __forceinline__ bool pair_less(const float a_d, const int a_i,
                                          const float b_d, const int b_i) {
    // Index is only an internal deterministic tie-breaker; the problem statement allows any tie order.
    return (a_d < b_d) || ((a_d == b_d) && (a_i < b_i));
}

__device__ __forceinline__ void warp_bitonic_sort_32(float& d, int& i) {
    // Full warp-wide bitonic sort of 32 candidate pairs in ascending order.
    const int lane = threadIdx.x & (kWarpSize - 1);

#pragma unroll
    for (int size = 2; size <= kWarpSize; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const float od = __shfl_xor_sync(kFullMask, d, stride);
            const int   oi = __shfl_xor_sync(kFullMask, i, stride);

            const bool ascending = ((lane & size) == 0);
            const bool lower     = ((lane & stride) == 0);
            const bool want_min  = (lower == ascending);

            if (want_min) {
                if (pair_less(od, oi, d, i)) {
                    d = od;
                    i = oi;
                }
            } else {
                if (pair_less(d, i, od, oi)) {
                    d = od;
                    i = oi;
                }
            }
        }
    }
}

template <int K>
__device__ __forceinline__ int merge_path_partition(const float* top_d,
                                                    const int*   top_i,
                                                    const float  cand_d,
                                                    const int    cand_i,
                                                    const int    diag) {
    // Merge-path partition of A[K] (current top-K) and B[32] (sorted candidate chunk) for the
    // merged prefix of length "diag". Because |B|=32 exactly, the search interval is at most 32
    // wide, so five unrolled binary-search steps are sufficient.
    int low  = diag - kWarpSize;
    int high = diag;
    if (low  < 0) low  = 0;
    if (high > K) high = K;

#pragma unroll
    for (int iter = 0; iter < kPartitionSteps; ++iter) {
        if (low < high) {
            const int mid = (low + high) >> 1;
            const int b   = diag - mid; // 0..32

            bool b_left_is_less_than_a_mid = true;
            if (b > 0) {
                const float bd = __shfl_sync(kFullMask, cand_d, b - 1);
                const int phys = topk_phys_index(mid);
                const float ad = top_d[phys];

                if (bd < ad) {
                    b_left_is_less_than_a_mid = true;
                } else if (bd > ad) {
                    b_left_is_less_than_a_mid = false;
                } else {
                    const int bi = __shfl_sync(kFullMask, cand_i, b - 1);
                    const int ai = top_i[phys];
                    b_left_is_less_than_a_mid = (bi < ai);
                }
            }

            if (b_left_is_less_than_a_mid) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
    }

    return low;
}

template <int K>
__device__ __forceinline__ void merge_sorted_chunk(float* top_d,
                                                   int*   top_i,
                                                   float  cand_d,
                                                   int    cand_i) {
    // Exact merge-select of A[K] and B[32], keeping the smallest K items.
    //
    // Each lane owns a contiguous logical output segment of length K/32. It computes that segment
    // entirely from the old A and the current B into registers, the warp synchronizes, and only
    // then writes back to shared memory. This guarantees that every lane reads the old top-K.
    constexpr int ItemsPerThread = KnnTraits<K>::ItemsPerThread;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int diag = lane * ItemsPerThread;

    int a = merge_path_partition<K>(top_d, top_i, cand_d, cand_i, diag);
    int b = diag - a;

    float out_d[ItemsPerThread];
    int   out_i[ItemsPerThread];

#pragma unroll
    for (int t = 0; t < ItemsPerThread; ++t) {
        const bool has_b = (b < kWarpSize);
        const bool has_a = (a < K);

        float bd = 0.0f;
        float ad = 0.0f;
        int phys_a = 0;

        if (has_b) {
            bd = __shfl_sync(kFullMask, cand_d, b);
        }
        if (has_a) {
            phys_a = topk_phys_index(a);
            ad = top_d[phys_a];
        }

        if (has_b && (!has_a || (bd < ad))) {
            out_d[t] = bd;
            out_i[t] = __shfl_sync(kFullMask, cand_i, b);
            ++b;
        } else if (!has_b || (has_a && (ad < bd))) {
            out_d[t] = ad;
            out_i[t] = top_i[phys_a];
            ++a;
        } else {
            // Equal distance (or NaN/other unordered case): use the same deterministic tie-break
            // as everywhere else so sorting/merge remain consistent.
            const int bi = has_b ? __shfl_sync(kFullMask, cand_i, b) : INT_MAX;
            const int ai = has_a ? top_i[phys_a] : INT_MAX;
            if (bi < ai) {
                out_d[t] = bd;
                out_i[t] = bi;
                ++b;
            } else {
                out_d[t] = ad;
                out_i[t] = ai;
                ++a;
            }
        }
    }

    __syncwarp(kFullMask);

#pragma unroll
    for (int t = 0; t < ItemsPerThread; ++t) {
        const int logical = diag + t;
        const int phys    = topk_phys_index(logical);
        top_d[phys] = out_d[t];
        top_i[phys] = out_i[t];
    }

    __syncwarp(kFullMask);
}

template <int K>
__global__ __launch_bounds__(kBlockThreads)
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                std::pair<int, float>* __restrict__ result) {
    constexpr int ItemsPerThread = KnnTraits<K>::ItemsPerThread;
    constexpr int PaddedK        = KnnTraits<K>::PaddedK;
    constexpr int WorstPhys      = KnnTraits<K>::WorstPhys;

    extern __shared__ __align__(128) unsigned char smem_raw[];

    float* const s_x     = reinterpret_cast<float*>(smem_raw + SharedLayout<K>::XOffset);
    float* const s_y     = reinterpret_cast<float*>(smem_raw + SharedLayout<K>::YOffset);
    float* const s_top_d = reinterpret_cast<float*>(smem_raw + SharedLayout<K>::TopDOffset);
    int*   const s_top_i = reinterpret_cast<int*>(smem_raw + SharedLayout<K>::TopIOffset);

    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & (kWarpSize - 1);
    const int query_id = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool active  = (query_id < query_count);

    float* const warp_top_d = s_top_d + warp_id * PaddedK;
    int*   const warp_top_i = s_top_i + warp_id * PaddedK;

    float qx = 0.0f;
    float qy = 0.0f;

    if (active) {
        const float2 q = query[query_id];
        qx = q.x;
        qy = q.y;

        // Bootstrap with sorted sentinels. Because (+inf, INT_MAX) is already sorted, the exact
        // same merge path used for steady-state updates also handles initialization: the first
        // K/32 chunks simply replace infinities.
#pragma unroll
        for (int t = 0; t < ItemsPerThread; ++t) {
            const int logical = lane + t * kWarpSize;
            const int phys    = topk_phys_index(logical);
            warp_top_d[phys] = CUDART_INF_F;
            warp_top_i[phys] = INT_MAX;
        }
    }

    __syncwarp(kFullMask);

    for (int tile_base = 0; tile_base < data_count; tile_base += kTilePoints) {
        int tile_points = data_count - tile_base;
        if (tile_points > kTilePoints) tile_points = kTilePoints;

        // Whole block loads the next data tile into shared memory.
#pragma unroll
        for (int it = 0; it < kTileLoadsPerThread; ++it) {
            const int local = tid + it * kBlockThreads;
            if (local < tile_points) {
                const float2 p = data[tile_base + local];
                s_x[local] = p.x;
                s_y[local] = p.y;
            }
        }

        __syncthreads();

        if (active) {
            for (int chunk = 0; chunk < tile_points; chunk += kWarpSize) {
                const float worst_d = warp_top_d[WorstPhys];

                float cand_d = CUDART_INF_F;
                int   cand_i = INT_MAX;

                const int local = chunk + lane;
                if (local < tile_points) {
                    // Exact cheap pre-filter: if dx^2 already exceeds the current top-K threshold,
                    // the full squared distance cannot improve the result, so we skip the y load/FMA.
                    const float dx  = qx - s_x[local];
                    const float dx2 = dx * dx;
                    if (dx2 <= worst_d) {
                        const float dy = qy - s_y[local];
                        cand_d = __fmaf_rn(dy, dy, dx2);
                        cand_i = tile_base + local;
                    }
                }

                bool better = (cand_d < worst_d);
                if (!better && (cand_d == worst_d)) {
                    const int worst_i = warp_top_i[WorstPhys];
                    better = (cand_i < worst_i);
                }

                // Candidates that cannot beat the current threshold are turned into sentinels so
                // the downstream sort/merge logic stays branch-free and exact.
                if (!better) {
                    cand_d = CUDART_INF_F;
                    cand_i = INT_MAX;
                }

                // If no lane in the warp has a viable candidate, skip the expensive sort+merge.
                if (__any_sync(kFullMask, better)) {
                    warp_bitonic_sort_32(cand_d, cand_i);
                    merge_sorted_chunk<K>(warp_top_d, warp_top_i, cand_d, cand_i);
                }
            }
        }

        // Ensure every warp is done consuming the shared tile before the block overwrites it.
        __syncthreads();
    }

    if (active) {
        // The top-K buffer is kept sorted ascending, so output is already nearest-first.
        const std::size_t out_base = static_cast<std::size_t>(query_id) * static_cast<std::size_t>(K);

#pragma unroll
        for (int t = 0; t < ItemsPerThread; ++t) {
            const int logical = lane + t * kWarpSize;
            const int phys    = topk_phys_index(logical);
            result[out_base + static_cast<std::size_t>(logical)].first  = warp_top_i[phys];
            result[out_base + static_cast<std::size_t>(logical)].second = warp_top_d[phys];
        }
    }
}

template <int K>
inline void launch_knn_impl(const float2* query,
                            int query_count,
                            const float2* data,
                            int data_count,
                            std::pair<int, float>* result) {
    if (query_count <= 0) return;

    constexpr std::size_t smem_bytes = SharedLayout<K>::Bytes;

    // The kernel uses a large dynamic shared-memory region; opt in explicitly and prefer the
    // shared-heavy carveout because the algorithm is shared-memory centric.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    knn_kernel<K><<<blocks, kBlockThreads, smem_bytes>>>(query, query_count, data, data_count, result);
}

} // namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) return;

    // Runtime dispatch over the only valid K values so the compiler sees K, loop trip counts,
    // shared-memory layout, and padding as compile-time constants.
    //
    // Intentionally asynchronous: callers can apply ordinary CUDA error checking/synchronization
    // outside this function if desired.
    switch (k) {
        case 32:   launch_knn_impl<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_impl<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_impl<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_impl<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_impl<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_impl<1024>(query, query_count, data, data_count, result); break;
        default:   break; // inputs are guaranteed valid by contract
    }
}