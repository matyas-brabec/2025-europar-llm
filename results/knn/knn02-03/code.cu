#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace {

// Exact 2D squared-L2 k-NN tuned for A100/H100-class GPUs.
//
// Design summary:
//   * One warp computes one query.
//   * A block contains 8 warps = 8 concurrent queries, which lets each shared-memory
//     data tile be reused by 8 queries before the next global-memory fetch.
//   * The database is streamed in 2048-point tiles cached in shared memory.
//   * Each warp keeps a private top-K in shared memory plus a 32-entry candidate buffer.
//   * Distances and indices are packed into one 64-bit key: [distance_bits | index].
//     Because squared distances are non-negative, the IEEE-754 bit pattern is monotonic,
//     so plain integer comparison orders by distance first. The index provides a cheap,
//     deterministic tie-break (any tie resolution is acceptable per the problem statement).
//   * The expensive O(K) top-K merge is amortized by buffering candidates and only
//     sorting/merging when the 32-entry buffer fills or when the scan ends.
//   * No additional device memory is allocated; all scratch state lives in shared memory
//     or registers.

constexpr int kWarpSize        = 32;
constexpr int kWarpsPerBlock   = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;  // 256 threads/block
constexpr int kTilePoints      = 2048;                        // 16 KiB float2 tile
constexpr unsigned kFullMask   = 0xFFFFFFFFu;

using KnnKey = unsigned long long;

// Sentinel "larger than every real candidate" = [ +inf | 0xFFFFFFFF ].
// Every real point has a valid non-negative index, so even a real +inf distance is smaller.
constexpr KnnKey kInvalidKey =
    (static_cast<KnnKey>(0x7F800000u) << 32) | static_cast<KnnKey>(0xFFFFFFFFu);

static_assert(sizeof(KnnKey) == sizeof(float2), "This kernel expects float2 to be 8 bytes.");
static_assert(kTilePoints % kWarpSize == 0, "The shared tile must be a whole number of warps.");

// Shared-memory footprint in 8-byte slots:
//   tile[TILE_POINTS] + topk_cur[WARPS*K] + topk_tmp[WARPS*K] + candidate_buf[WARPS*32]
template <int K>
constexpr std::size_t shared_bytes() {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0,
                  "K must be a power of two in [32, 1024].");
    return static_cast<std::size_t>(kTilePoints + kWarpsPerBlock * (2 * K + kWarpSize)) *
           sizeof(KnnKey);
}

__device__ __forceinline__ KnnKey pack_key(float dist, int idx) {
    return (static_cast<KnnKey>(__float_as_uint(dist)) << 32) |
           static_cast<unsigned int>(idx);
}

__device__ __forceinline__ float unpack_dist(KnnKey key) {
    return __uint_as_float(static_cast<unsigned int>(key >> 32));
}

__device__ __forceinline__ int unpack_idx(KnnKey key) {
    return static_cast<int>(static_cast<unsigned int>(key));
}

// 32-wide bitonic sort using warp shuffles.
// Each lane owns one key; after the network, lane order is ascending.
__device__ __forceinline__ void warp_bitonic_sort_32(KnnKey &key, int lane) {
#pragma unroll
    for (int k = 2; k <= kWarpSize; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const KnnKey other = __shfl_xor_sync(kFullMask, key, j);

            // Standard bitonic direction logic for shuffle-xor compare/swap.
            const bool up         = ((lane & k) == 0);
            const bool keep_small = (((lane & j) == 0) == up);
            const bool take_other = keep_small ? (other < key) : (key < other);

            if (take_other) {
                key = other;
            }
        }
    }
}

// Merge-path partition for the first "diag" outputs of merging two sorted arrays a and b.
// Returns the number of elements taken from a on that diagonal.
__device__ __forceinline__ int merge_path_partition(
    const KnnKey *a, int a_count,
    const KnnKey *b, int b_count,
    int diag) {

    if (diag <= 0) {
        return 0;
    }
    if (diag >= a_count + b_count) {
        return a_count;
    }

    int low  = diag - b_count;
    int high = diag;

    if (low < 0) {
        low = 0;
    }
    if (high > a_count) {
        high = a_count;
    }

    // Binary search for the first valid partition.
    while (low < high) {
        const int mid = (low + high) >> 1;
        const int j   = diag - mid;

        if (j > 0 && mid < a_count && a[mid] < b[j - 1]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = diag - i;

    // One-step correction for the complementary boundary.
    while (i > 0 && j < b_count && b[j] < a[i - 1]) {
        --i;
        ++j;
    }

    return i;
}

// Sorts the warp-private candidate buffer, discards stale entries that are no longer
// better than the latest worst top-K key, then merges the surviving prefix into top-K.
// current_worst_key is an upper bound on the true threshold while the buffer is non-empty;
// that makes lazy buffering exact: we may keep extra candidates, but we never drop a valid one.
template <int K>
__device__ __forceinline__ void flush_candidate_buffer(
    KnnKey *buf,
    int &buf_count,
    KnnKey *&cur_topk,
    KnnKey *&tmp_topk,
    int &current_valid,
    KnnKey &current_worst_key,
    int lane) {

    if (buf_count == 0) {
        return;
    }

    // Make prior shared-memory appends visible within the warp before reading the buffer.
    __syncwarp(kFullMask);

    KnnKey key = (lane < buf_count) ? buf[lane] : kInvalidKey;
    warp_bitonic_sort_32(key, lane);

    // The buffer is now sorted in registers. Only the prefix strictly smaller than the
    // current worst top-K key can still matter.
    const unsigned useful_mask =
        __ballot_sync(kFullMask, (lane < buf_count) && (key < current_worst_key));
    const int useful_count = __popc(useful_mask);

    // Materialize the sorted buffer back to shared memory for indexed merge-path access.
    buf[lane] = key;
    __syncwarp(kFullMask);

    if (useful_count == 0) {
        buf_count = 0;
        return;
    }

    const int new_valid = ((current_valid + useful_count) < K)
                              ? (current_valid + useful_count)
                              : K;

    // Parallel merge of cur_topk[0:current_valid) and buf[0:useful_count),
    // writing the first new_valid outputs into tmp_topk[0:new_valid).
    const int diag_begin = (lane * new_valid) >> 5;          // / 32
    const int diag_end   = ((lane + 1) * new_valid) >> 5;   // / 32

    const int a_begin = merge_path_partition(cur_topk, current_valid, buf, useful_count, diag_begin);
    const int a_end   = merge_path_partition(cur_topk, current_valid, buf, useful_count, diag_end);

    const int b_begin = diag_begin - a_begin;
    const int b_end   = diag_end - a_end;

    int ai = a_begin;
    int bi = b_begin;

    for (int out = diag_begin; out < diag_end; ++out) {
        const bool take_b = (bi < b_end) && (ai >= a_end || buf[bi] < cur_topk[ai]);
        tmp_topk[out] = take_b ? buf[bi++] : cur_topk[ai++];
    }

    __syncwarp(kFullMask);

    KnnKey *old = cur_topk;
    cur_topk = tmp_topk;
    tmp_topk = old;

    current_valid = new_valid;
    buf_count = 0;

    // Threshold for future pruning: if top-K is not full yet, keep using the sentinel
    // so that every real point still qualifies.
    KnnKey new_worst = kInvalidKey;
    if (current_valid == K && lane == 0) {
        new_worst = cur_topk[K - 1];
    }
    current_worst_key = __shfl_sync(kFullMask, new_worst, 0);
}

template <int K>
__global__ __launch_bounds__(kThreadsPerBlock)
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                std::pair<int, float> * __restrict__ result) {
    // One dynamic shared-memory allocation is partitioned into:
    //   [data tile][topk current][topk scratch][candidate buffers]
    // All regions are 8-byte aligned because both float2 and KnnKey are 8 bytes.
    extern __shared__ KnnKey smem64[];

    float2 *tile   = reinterpret_cast<float2 *>(smem64);
    KnnKey *topk0  = smem64 + kTilePoints;
    KnnKey *topk1  = topk0 + kWarpsPerBlock * K;
    KnnKey *cbufs  = topk1 + kWarpsPerBlock * K;

    const int lane    = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int qid     = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_id;
    const bool active = (qid < query_count);

    // Warp-private regions in shared memory.
    KnnKey *cur_topk = topk0 + warp_id * K;
    KnnKey *tmp_topk = topk1 + warp_id * K;
    KnnKey *cbuf     = cbufs + warp_id * kWarpSize;

    int current_valid = 0;
    int cbuf_count = 0;
    KnnKey current_worst_key = kInvalidKey;

    // Load the query point once per warp, then broadcast via shuffles.
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

    // Stream the database in shared-memory tiles. Every block reuses each tile for 8 queries.
    for (int tile_begin = 0; tile_begin < data_count; tile_begin += kTilePoints) {
        int tile_count = data_count - tile_begin;
        if (tile_count > kTilePoints) {
            tile_count = kTilePoints;
        }

        // Cooperative global -> shared load of the current data tile.
        for (int i = threadIdx.x; i < tile_count; i += kThreadsPerBlock) {
            tile[i] = data[tile_begin + i];
        }
        __syncthreads();

        if (active) {
            // Each chunk maps one data point to one lane.
            for (int chunk = 0; chunk < tile_count; chunk += kWarpSize) {
                const int local_idx = chunk + lane;
                const bool lane_active = (local_idx < tile_count);

                KnnKey candidate_key = kInvalidKey;
                if (lane_active) {
                    const float2 p = tile[local_idx];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    const float dist = fmaf(dx, dx, dy * dy);  // squared L2
                    candidate_key = pack_key(dist, tile_begin + local_idx);
                }

                // A candidate is interesting iff it is strictly smaller than the current
                // worst top-K key. Before top-K is full, current_worst_key is kInvalidKey,
                // so every real point qualifies automatically.
                bool qualifies = lane_active && (candidate_key < current_worst_key);
                unsigned mask = __ballot_sync(kFullMask, qualifies);
                int qualifying_count = __popc(mask);

                if (qualifying_count == 0) {
                    continue;
                }

                // If the append would overflow the 32-entry buffer, first flush the existing
                // buffer, then re-test this chunk against the refreshed threshold.
                if (cbuf_count + qualifying_count > kWarpSize) {
                    flush_candidate_buffer<K>(
                        cbuf, cbuf_count, cur_topk, tmp_topk,
                        current_valid, current_worst_key, lane);

                    qualifies = lane_active && (candidate_key < current_worst_key);
                    mask = __ballot_sync(kFullMask, qualifies);
                    qualifying_count = __popc(mask);

                    if (qualifying_count == 0) {
                        continue;
                    }
                }

                if (qualifies) {
                    const unsigned rank = __popc(mask & ((1u << lane) - 1u));
                    cbuf[cbuf_count + static_cast<int>(rank)] = candidate_key;
                }

                cbuf_count += qualifying_count;

                if (cbuf_count == kWarpSize) {
                    flush_candidate_buffer<K>(
                        cbuf, cbuf_count, cur_topk, tmp_topk,
                        current_valid, current_worst_key, lane);
                }
            }
        }

        // Ensure all warps are done with the current shared tile before the block overwrites it.
        __syncthreads();
    }

    if (active) {
        // Final pending candidates.
        flush_candidate_buffer<K>(
            cbuf, cbuf_count, cur_topk, tmp_topk,
            current_valid, current_worst_key, lane);

        // cur_topk is sorted ascending, so it already matches the required output order.
        const std::size_t out_base = static_cast<std::size_t>(qid) * K;
        for (int i = lane; i < K; i += kWarpSize) {
            const KnnKey key = (i < current_valid) ? cur_topk[i] : kInvalidKey;
            result[out_base + i].first  = unpack_idx(key);
            result[out_base + i].second = unpack_dist(key);
        }
    }
}

template <int K>
inline void launch_knn_specialization(const float2 *query,
                                      int query_count,
                                      const float2 *data,
                                      int data_count,
                                      std::pair<int, float> *result) {
    constexpr std::size_t kSharedBytes = shared_bytes<K>();

    // Opt in to the larger shared-memory allocation used by the bigger K specializations.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    // Asynchronous launch; caller can synchronize/overlap as desired.
    knn_kernel<K><<<blocks, kThreadsPerBlock, kSharedBytes>>>(
        query, query_count, data, data_count, result);

    // Surface launch-configuration errors without forcing a device-wide sync.
    (void)cudaPeekAtLastError();
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

    switch (k) {
        case 32:   launch_knn_specialization<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_specialization<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_specialization<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_specialization<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_specialization<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_specialization<1024>(query, query_count, data, data_count, result); break;
        default:
            // Inputs are specified to be valid, so this is unreachable under the stated contract.
            break;
    }
}