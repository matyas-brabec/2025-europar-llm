#include <cuda_runtime.h>
#include <utility>
#include <cstddef>
#include <cstdint>

namespace {

// Exact brute-force k-NN for 2D points, specialized for the problem regime:
// - k is one of {32, 64, 128, 256, 512, 1024}
// - query_count is typically O(10^3)
// - data_count is typically O(10^6)
//
// High-level mapping:
//   * one warp owns one query;
//   * one CTA owns 8 queries (8 warps / 256 threads);
//   * each CTA stages 1024 data points in shared memory, so every global load
//     of a data point is reused across 8 independent queries.
//
// Per-query selection strategy:
//   * keep the exact current top-k as a sorted shared-memory array of packed
//     (distance_bits, index) keys;
//   * because distances are non-negative IEEE-754 floats, unsigned bit order
//     matches numeric order, so the packed key is totally ordered;
//   * after every 32-point batch, the current worst key is exact, so any new
//     candidate with key >= worst_key can be discarded exactly (not
//     approximately);
//   * if only a few candidates beat the threshold, lane 0 inserts them directly
//     into the sorted array;
//   * otherwise, the warp sorts the 32 new keys with a register-resident
//     bitonic network and merges them with the current top-k using a merge-path
//     style partition.
//
// This design uses no extra device allocations; only registers and dynamic
// shared memory are used.

using Key = unsigned long long;

constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE; // 256
constexpr int TILE_POINTS       = 1024;                        // 8 KiB tile of float2
constexpr int SMALL_INSERT      = 8;                           // hybrid update threshold
constexpr unsigned FULL_MASK    = 0xFFFFFFFFu;

// Sentinel key: +inf distance with the largest possible index.
// Real finite keys are always smaller; real +inf distances are also smaller
// because their lower 32-bit index is < 0xFFFFFFFF.
constexpr Key INF_KEY = (static_cast<Key>(0x7F800000u) << 32) | static_cast<Key>(0xFFFFFFFFu);

__device__ __forceinline__ Key pack_key(const float dist, const int idx) {
    return (static_cast<Key>(__float_as_uint(dist)) << 32) |
           static_cast<Key>(static_cast<unsigned>(idx));
}

__device__ __forceinline__ float unpack_dist(const Key key) {
    return __uint_as_float(static_cast<unsigned>(key >> 32));
}

__device__ __forceinline__ int unpack_idx(const Key key) {
    return static_cast<int>(static_cast<unsigned>(key));
}

// 64-bit shuffles built from two 32-bit shuffles for portability.
__device__ __forceinline__ Key shfl_u64(const Key v, const int src_lane) {
    const unsigned lo = __shfl_sync(FULL_MASK, static_cast<unsigned>(v), src_lane);
    const unsigned hi = __shfl_sync(FULL_MASK, static_cast<unsigned>(v >> 32), src_lane);
    return (static_cast<Key>(hi) << 32) | static_cast<Key>(lo);
}

__device__ __forceinline__ Key shfl_xor_u64(const Key v, const int lane_mask) {
    const unsigned lo = __shfl_xor_sync(FULL_MASK, static_cast<unsigned>(v), lane_mask);
    const unsigned hi = __shfl_xor_sync(FULL_MASK, static_cast<unsigned>(v >> 32), lane_mask);
    return (static_cast<Key>(hi) << 32) | static_cast<Key>(lo);
}

// Sort 32 keys across one warp in ascending order.
// Each lane owns one key. Invalid entries should already be set to INF_KEY.
__device__ __forceinline__ Key warp_bitonic_sort32(Key key) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    #pragma unroll
    for (int k = 2; k <= WARP_SIZE; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const Key other = shfl_xor_u64(key, j);
            const bool up   = ((lane & k) == 0);
            const Key minv  = (key < other) ? key : other;
            const Key maxv  = (key < other) ? other : key;
            key = up ? minv : maxv;
        }
    }
    return key;
}

// Merge the current sorted top-k array A[K] with a sorted candidate array B[m],
// producing the smallest K keys into OUT[K]. A and OUT must not alias.
template <int K>
__device__ __forceinline__ void warp_merge_topk(const Key* __restrict__ a,
                                                const Key* __restrict__ b,
                                                const int m,
                                                Key* __restrict__ out) {
    static_assert(K >= WARP_SIZE && K <= 1024 && (K & (K - 1)) == 0, "Unsupported K");
    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;

    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const int out_begin = lane * ITEMS_PER_LANE;

    // Merge-path partition for this lane's contiguous output segment.
    int low  = (out_begin > K) ? (out_begin - K) : 0;
    int high = (out_begin < m) ? out_begin : m;

    while (low < high) {
        const int mid        = (low + high) >> 1;
        const int a_left_idx = out_begin - mid - 1;

        if (a_left_idx >= 0 && b[mid] < a[a_left_idx]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int b_idx = low;
    int a_idx = out_begin - b_idx;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const Key a_val = (a_idx < K) ? a[a_idx] : INF_KEY;
        const Key b_val = (b_idx < m) ? b[b_idx] : INF_KEY;
        const bool take_a = (a_val <= b_val);
        out[out_begin + t] = take_a ? a_val : b_val;
        a_idx += take_a ? 1 : 0;
        b_idx += take_a ? 0 : 1;
    }
}

template <int K>
__launch_bounds__(THREADS_PER_BLOCK, 1)
__global__ void knn_kernel(const float2* __restrict__ query,
                           const int query_count,
                           const float2* __restrict__ data,
                           const int data_count,
                           std::pair<int, float>* __restrict__ result) {
    static_assert(K >= WARP_SIZE && K <= 1024 && (K & (K - 1)) == 0, "Unsupported K");

    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;

    // Dynamic shared memory layout:
    //   topk_buf0[WARPS_PER_BLOCK][K]
    //   topk_buf1[WARPS_PER_BLOCK][K]
    //   batch    [WARPS_PER_BLOCK][32]
    //   data_tile[TILE_POINTS]
    extern __shared__ __align__(8) unsigned char shared_raw[];
    Key* shared_keys = reinterpret_cast<Key*>(shared_raw);

    Key* topk_buf0 = shared_keys;
    Key* topk_buf1 = topk_buf0 + static_cast<std::size_t>(WARPS_PER_BLOCK) * K;
    Key* batch_buf = topk_buf1 + static_cast<std::size_t>(WARPS_PER_BLOCK) * K;
    float2* data_tile = reinterpret_cast<float2*>(batch_buf + static_cast<std::size_t>(WARPS_PER_BLOCK) * WARP_SIZE);

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x >> 5;

    const int qid = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool active = (qid < query_count);

    // One query per warp; lane 0 loads and broadcasts.
    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && active) {
        const float2 q = query[qid];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    Key* const buf0 = topk_buf0 + static_cast<std::size_t>(warp) * K;
    Key* const buf1 = topk_buf1 + static_cast<std::size_t>(warp) * K;
    Key* const warp_batch = batch_buf + static_cast<std::size_t>(warp) * WARP_SIZE;

    // Initialize the current top-k buffer to +inf.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        buf0[lane + i * WARP_SIZE] = INF_KEY;
    }

    // Current buffer selector and exact current worst key for this warp/query.
    int cur_sel = 0;
    Key worst_key = INF_KEY;

    // Full-tile loop over the database.
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS) {
        // Stage one tile of data points into shared memory.
        #pragma unroll
        for (int i = threadIdx.x; i < TILE_POINTS; i += THREADS_PER_BLOCK) {
            const int data_idx = tile_base + i;
            if (data_idx < data_count) {
                data_tile[i] = data[data_idx];
            }
        }

        __syncthreads();

        if (active) {
            // Process the tile in 32-point warp batches.
            #pragma unroll 1
            for (int sub = 0; sub < TILE_POINTS; sub += WARP_SIZE) {
                const int data_idx = tile_base + sub + lane;

                Key key = INF_KEY;
                bool valid = false;

                if (data_idx < data_count) {
                    const float2 p = data_tile[sub + lane];

                    const float dx = p.x - qx;
                    const float dy = p.y - qy;

                    // Squared Euclidean distance.
                    const float dist = __fmaf_rn(dx, dx, dy * dy);

                    key = pack_key(dist, data_idx);
                    valid = (key < worst_key);
                }

                const unsigned valid_mask = __ballot_sync(FULL_MASK, valid);
                const int m = __popc(valid_mask);

                if (m == 0) {
                    continue;
                }

                Key* const cur = (cur_sel == 0) ? buf0 : buf1;

                if (m <= SMALL_INSERT) {
                    // Fast path: only a few records beat the current threshold.
                    // Compact them, sort them cheaply in lane 0, then insert them
                    // into the current sorted top-k buffer in place.
                    if (valid) {
                        const unsigned lower_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
                        const int write_pos = __popc(valid_mask & lower_mask);
                        warp_batch[write_pos] = key;
                    }

                    __syncwarp(FULL_MASK);

                    if (lane == 0) {
                        // Insertion sort the tiny compacted batch.
                        for (int i = 1; i < m; ++i) {
                            const Key v = warp_batch[i];
                            int j = i - 1;
                            while (j >= 0 && warp_batch[j] > v) {
                                warp_batch[j + 1] = warp_batch[j];
                                --j;
                            }
                            warp_batch[j + 1] = v;
                        }

                        // Insert sorted keys into the current top-k.
                        // Since the batch is sorted ascending, insertion positions
                        // are monotone non-decreasing; search_lo exploits that.
                        int search_lo = 0;

                        for (int i = 0; i < m; ++i) {
                            const Key v = warp_batch[i];

                            // Earlier insertions may already have tightened the
                            // threshold enough that the remaining keys are no
                            // longer competitive.
                            if (!(v < cur[K - 1])) {
                                break;
                            }

                            int lo = search_lo;
                            int hi = K;
                            while (lo < hi) {
                                const int mid = (lo + hi) >> 1;
                                if (cur[mid] < v) {
                                    lo = mid + 1;
                                } else {
                                    hi = mid;
                                }
                            }

                            const int pos = lo;

                            // Shift tail right by one and place the new key.
                            #pragma unroll 1
                            for (int j = K - 1; j > pos; --j) {
                                cur[j] = cur[j - 1];
                            }
                            cur[pos] = v;
                            search_lo = pos + 1;
                        }

                        worst_key = cur[K - 1];
                    }

                    // Ensure lane-0 writes to shared memory are visible to the warp.
                    __syncwarp(FULL_MASK);
                    worst_key = shfl_u64(worst_key, 0);
                } else {
                    // Heavy path: enough winning candidates that a full warp-level
                    // sort + merge is cheaper than serial in-place insertion.
                    if (!valid) {
                        key = INF_KEY;
                    }

                    key = warp_bitonic_sort32(key);
                    warp_batch[lane] = key;

                    __syncwarp(FULL_MASK);

                    Key* const out = (cur_sel == 0) ? buf1 : buf0;
                    warp_merge_topk<K>(cur, warp_batch, m, out);

                    __syncwarp(FULL_MASK);

                    if (lane == 0) {
                        worst_key = out[K - 1];
                    }

                    cur_sel ^= 1;
                    worst_key = shfl_u64(worst_key, 0);
                }
            }
        }

        // The tile buffer is shared across all warps in the CTA.
        __syncthreads();
    }

    // top-k is already sorted ascending, so the final writeout is straightforward.
    if (active) {
        const Key* const cur = (cur_sel == 0) ? buf0 : buf1;
        const std::size_t out_base = static_cast<std::size_t>(qid) * K;

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; ++i) {
            const int j = lane + i * WARP_SIZE;
            const Key key = cur[j];
            result[out_base + j].first  = unpack_idx(key);
            result[out_base + j].second = unpack_dist(key);
        }
    }
}

template <int K>
struct SharedBytes {
    static constexpr std::size_t value =
        // two K-sized top-k buffers per warp + one 32-key batch buffer per warp
        (static_cast<std::size_t>(WARPS_PER_BLOCK) *
         (static_cast<std::size_t>(2) * K + WARP_SIZE)) * sizeof(Key) +
        // one float2 tile shared across the whole CTA
        static_cast<std::size_t>(TILE_POINTS) * sizeof(float2);
};

template <int K>
inline void launch_knn(const float2* query,
                       const int query_count,
                       const float2* data,
                       const int data_count,
                       std::pair<int, float>* result) {
    constexpr std::size_t shared_bytes = SharedBytes<K>::value;

    // The chosen parameters are constrained by the A100 opt-in shared-memory
    // limit (163,840 bytes). They also keep the grid large enough for the
    // stated "thousands of queries" regime on 100+ SM GPUs.
    static_assert(shared_bytes <= 163840, "Per-block shared memory exceeds A100 opt-in limit.");

    // Opt in to the required dynamic shared-memory size and bias the L1/shared
    // partition toward shared memory; this kernel is explicitly shared-memory
    // heavy and stages the database tile in shared memory by design.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shared_bytes));
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Launch is intentionally asynchronous with respect to the host.
    knn_kernel<K><<<grid, block, shared_bytes>>>(query, query_count, data, data_count, result);
}

} // namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // K is runtime but comes from a tiny fixed set, so dispatching to
    // compile-time specializations gives the compiler full visibility into:
    //   * K/32 items per lane,
    //   * shared-memory footprint,
    //   * merge-loop trip counts.
    switch (k) {
        case 32:   launch_knn<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn<1024>(query, query_count, data, data_count, result); break;
        default:
            // Input contract says k is always valid, so this path should never run.
            break;
    }
}