#include <cuda_runtime.h>
#include <utility>
#include <cstddef>
#include <cstdint>

namespace {

// Tuned for modern datacenter GPUs (A100/H100):
// - 256 threads/CTA provides good memory throughput for the streaming pass over `data`.
// - Each thread examines 8 points per round; this keeps register pressure modest.
// - The shared candidate queue keeps K + 2 full rounds of candidates. That slack
//   materially reduces prune frequency while keeping the worst-case dynamic shared
//   memory exactly at 48 KiB when K == 1024.
constexpr int kBlockThreads     = 256;
constexpr int kLocalBuffer      = 8;
constexpr int kQueueSlackRounds = 2;
constexpr unsigned int kInfBits = 0x7f800000u;

template<int K, int BLOCK_THREADS, int LOCAL_BUFFER, int SLACK_ROUNDS>
struct KnnConfig {
    static_assert((K & (K - 1)) == 0 && K >= 32 && K <= 1024,
                  "K must be a power of two in [32, 1024].");
    static_assert(BLOCK_THREADS % 32 == 0,
                  "BLOCK_THREADS must be a multiple of warp size.");

    static constexpr int kRoundCandidates = BLOCK_THREADS * LOCAL_BUFFER;
    static constexpr int kCapacity        = K + SLACK_ROUNDS * kRoundCandidates;
    static constexpr int kPruneTrigger    = kCapacity - kRoundCandidates;

    // Dynamic shared memory layout:
    //   float cand_dist[kCapacity]
    //   int   cand_idx [kCapacity]
    //   float tmp_dist [K]
    //   int   tmp_idx  [K]
    static constexpr std::size_t kDynamicSharedBytes =
        sizeof(float) * static_cast<std::size_t>(kCapacity) +
        sizeof(int)   * static_cast<std::size_t>(kCapacity) +
        sizeof(float) * static_cast<std::size_t>(K) +
        sizeof(int)   * static_cast<std::size_t>(K);
};

static_assert(
    KnnConfig<1024, kBlockThreads, kLocalBuffer, kQueueSlackRounds>::kDynamicSharedBytes
        == 48u * 1024u,
    "The chosen tuning is intended to keep the worst-case dynamic shared memory at 48 KiB."
);

// Per-CTA static shared state. All sizable temporaries used for the candidate queue live
// in dynamic shared memory, so this structure is intentionally small.
template<int BLOCK_THREADS>
struct StaticShared {
    unsigned int hist[256];               // 8-bit radix histogram for exact shared-memory pruning.
    int warp_sums[BLOCK_THREADS / 32];    // Scratch for block-wide exclusive scans.
    int scan_total;                       // Block-wide scan aggregate.
    int queue_size;                       // Current number of elements in the candidate queue.
    unsigned int threshold_bits;          // Current K-th distance threshold (IEEE-754 ordered bits).
    int threshold_valid;                  // 0 before the first exact prune, then 1.
    int write_base;                       // Queue insertion base for the current round.

    // Scratch for radix selection.
    unsigned int prefix;
    unsigned int mask;
    int rank;
    int total_lt;
    int need_eq;
};

// For non-negative IEEE-754 floats, raw bit order matches numeric order. Squared Euclidean
// distances are non-negative, so we can select/sort by raw bits without extra transforms.
__device__ __forceinline__ unsigned int ordered_bits_nonnegative(float x) {
    return __float_as_uint(x);
}

__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    const unsigned int ba = ordered_bits_nonnegative(da);
    const unsigned int bb = ordered_bits_nonnegative(db);
    return (ba < bb) || ((ba == bb) && (ia < ib));
}

// Lightweight CTA-wide exclusive sum for integer counts. This replaces a heavier generic
// block scan because this kernel only needs scans over small integers.
template<int BLOCK_THREADS>
__device__ __forceinline__ int block_exclusive_sum(
    int x,
    int* block_total,
    int* warp_sums)
{
    static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be warp-aligned.");

    constexpr int kNumWarps = BLOCK_THREADS / 32;
    constexpr unsigned int kFullMask = 0xffffffffu;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int inclusive = x;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int y = __shfl_up_sync(kFullMask, inclusive, offset);
        if (lane >= offset) inclusive += y;
    }

    const int warp_exclusive = inclusive - x;
    if (lane == 31) {
        warp_sums[warp] = inclusive;
    }
    __syncthreads();

    if (warp == 0) {
        int warp_total = (lane < kNumWarps) ? warp_sums[lane] : 0;
        int warp_scan  = warp_total;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int y = __shfl_up_sync(kFullMask, warp_scan, offset);
            if (lane >= offset) warp_scan += y;
        }
        if (lane < kNumWarps) {
            warp_sums[lane] = warp_scan - warp_total;
        }
        if (lane == kNumWarps - 1) {
            *block_total = warp_scan;
        }
    }
    __syncthreads();

    return warp_sums[warp] + warp_exclusive;
}

// Exact prune of the shared candidate queue back down to K elements.
// The queue always contains:
//   - the exact top-K among all previously processed points, and
//   - additional candidates with distance < the last known K-th threshold.
// Therefore, selecting the K smallest values from the queue is sufficient to restore the
// exact invariant after each prune.
template<int K, int BLOCK_THREADS>
__device__ __forceinline__ void prune_queue(
    int q,
    float* cand_dist,
    int* cand_idx,
    float* tmp_dist,
    int* tmp_idx,
    StaticShared<BLOCK_THREADS>& ss)
{
    const int tid = threadIdx.x;

    // Exact K-th order statistic by 4-pass 8-bit radix selection over the queue.
    if (tid == 0) {
        ss.prefix = 0u;
        ss.mask   = 0u;
        ss.rank   = K; // 1-based rank of the K-th smallest value.
    }
    __syncthreads();

#pragma unroll
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += BLOCK_THREADS) {
            ss.hist[b] = 0u;
        }
        __syncthreads();

        const unsigned int prefix = ss.prefix;
        const unsigned int mask   = ss.mask;

        for (int i = tid; i < q; i += BLOCK_THREADS) {
            const unsigned int bits = ordered_bits_nonnegative(cand_dist[i]);
            if ((bits & mask) == prefix) {
                atomicAdd(&ss.hist[(bits >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            unsigned int accum  = 0u;
            unsigned int chosen = 0u;
            const unsigned int rank = static_cast<unsigned int>(ss.rank);

            while (chosen < 256u) {
                const unsigned int c = ss.hist[chosen];
                if (accum + c >= rank) break;
                accum += c;
                ++chosen;
            }

            ss.prefix = prefix | (chosen << shift);
            ss.mask   = mask   | (0xFFu << shift);
            ss.rank  -= static_cast<int>(accum);
        }
        __syncthreads();
    }

    const unsigned int thr_bits = ss.prefix;
    if (tid == 0) {
        ss.threshold_bits  = thr_bits;
        ss.threshold_valid = 1;
    }
    __syncthreads();

    // Count elements strictly smaller than the threshold and those exactly equal to it.
    int lt_i = 0;
    int eq_i = 0;
    for (int i = tid; i < q; i += BLOCK_THREADS) {
        const unsigned int bits = ordered_bits_nonnegative(cand_dist[i]);
        lt_i += static_cast<int>(bits <  thr_bits);
        eq_i += static_cast<int>(bits == thr_bits);
    }

    const int lt_base = block_exclusive_sum<BLOCK_THREADS>(lt_i, &ss.scan_total, ss.warp_sums);
    const int total_lt = ss.scan_total;
    const int eq_base = block_exclusive_sum<BLOCK_THREADS>(eq_i, &ss.scan_total, ss.warp_sums);

    if (tid == 0) {
        ss.total_lt = total_lt;
        ss.need_eq  = K - total_lt;
    }
    __syncthreads();

    // Compact the exact K smallest items into tmp_* (unsorted):
    //   all keys < threshold
    //   plus just enough keys == threshold to reach K
    int lt_out = lt_base;
    int eq_out = eq_base;
    for (int i = tid; i < q; i += BLOCK_THREADS) {
        const float d = cand_dist[i];
        const int idx = cand_idx[i];
        const unsigned int bits = ordered_bits_nonnegative(d);

        if (bits < thr_bits) {
            tmp_dist[lt_out] = d;
            tmp_idx[lt_out]  = idx;
            ++lt_out;
        } else if (bits == thr_bits) {
            if (eq_out < ss.need_eq) {
                const int pos = ss.total_lt + eq_out;
                tmp_dist[pos] = d;
                tmp_idx[pos]  = idx;
            }
            ++eq_out;
        }
    }
    __syncthreads();

    for (int i = tid; i < K; i += BLOCK_THREADS) {
        cand_dist[i] = tmp_dist[i];
        cand_idx[i]  = tmp_idx[i];
    }
    __syncthreads();
}

// Final in-CTA sort of the exact top-K before writing results. K is a power of two, so a
// standard bitonic network is simple and effective here. This is done only once per query.
template<int K, int BLOCK_THREADS>
__device__ __forceinline__ void sort_final_topk(float* dist, int* idx) {
    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < K; i += BLOCK_THREADS) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = ((i & size) == 0);

                    const float di = dist[i];
                    const int   ii = idx[i];
                    const float dj = dist[j];
                    const int   ij = idx[j];

                    const bool do_swap = up ? pair_less(dj, ij, di, ii)
                                            : pair_less(di, ii, dj, ij);

                    if (do_swap) {
                        dist[i] = dj;
                        idx[i]  = ij;
                        dist[j] = di;
                        idx[j]  = ii;
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<int K, int BLOCK_THREADS = kBlockThreads, int LOCAL_BUFFER = kLocalBuffer, int SLACK_ROUNDS = kQueueSlackRounds>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result)
{
    using Cfg = KnnConfig<K, BLOCK_THREADS, LOCAL_BUFFER, SLACK_ROUNDS>;

    const int qid = blockIdx.x;
    if (qid >= query_count) {
        return;
    }

    __shared__ StaticShared<BLOCK_THREADS> ss;

    // Dynamic shared-memory layout:
    //   [cand_dist | cand_idx | tmp_dist | tmp_idx]
    extern __shared__ unsigned char smem_raw[];
    unsigned char* ptr = smem_raw;

    float* cand_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * Cfg::kCapacity;

    int* cand_idx = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * Cfg::kCapacity;

    float* tmp_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * K;

    int* tmp_idx = reinterpret_cast<int*>(ptr);

    const int tid = threadIdx.x;

    const float2 q = query[qid];
    const float qx = q.x;
    const float qy = q.y;

    if (tid == 0) {
        ss.queue_size      = 0;
        ss.threshold_bits  = kInfBits;
        ss.threshold_valid = 0;
    }
    __syncthreads();

    float local_dist[LOCAL_BUFFER];
    int   local_idx[LOCAL_BUFFER];

    // Main streaming pass over the database.
    // Exactness invariant:
    //   after each prune, cand_[0:K) contains the exact K best points seen so far.
    // Therefore, once a valid threshold exists, any future point with distance >= threshold
    // cannot be required (ties may be broken arbitrarily), so we safely ignore equality.
    for (int base = 0; base < data_count; base += Cfg::kRoundCandidates) {
        int local_count = 0;

        const bool have_threshold   = (ss.threshold_valid != 0);
        const unsigned int thr_bits = ss.threshold_bits;

#pragma unroll
        for (int j = 0; j < LOCAL_BUFFER; ++j) {
            const int did = base + j * BLOCK_THREADS + tid;
            if (did < data_count) {
                const float2 p = data[did];
                const float dx = qx - p.x;
                const float dy = qy - p.y;
                const float d  = fmaf(dx, dx, dy * dy); // squared L2 distance

                if (have_threshold) {
                    if (ordered_bits_nonnegative(d) < thr_bits) {
                        local_dist[local_count] = d;
                        local_idx [local_count] = did;
                        ++local_count;
                    }
                } else {
                    // Before the first exact prune, accept everything so that +inf distances
                    // are still handled correctly.
                    local_dist[local_count] = d;
                    local_idx [local_count] = did;
                    ++local_count;
                }
            }
        }

        const int local_offset =
            block_exclusive_sum<BLOCK_THREADS>(local_count, &ss.scan_total, ss.warp_sums);
        const int round_total = ss.scan_total;

        if (tid == 0) {
            ss.write_base = ss.queue_size;
            ss.queue_size += round_total;
        }
        __syncthreads();

        const int out = ss.write_base + local_offset;
#pragma unroll
        for (int j = 0; j < LOCAL_BUFFER; ++j) {
            if (j < local_count) {
                cand_dist[out + j] = local_dist[j];
                cand_idx [out + j] = local_idx [j];
            }
        }
        __syncthreads();

        // Prune eagerly the first time we have more than K candidates, then only when the
        // shared queue approaches capacity. This cuts prune frequency while keeping the
        // queue safely bounded without any global scratch space.
        if ((ss.queue_size > K) &&
            ((ss.threshold_valid == 0) || (ss.queue_size > Cfg::kPruneTrigger))) {
            const int qsize = ss.queue_size;
            prune_queue<K, BLOCK_THREADS>(qsize, cand_dist, cand_idx, tmp_dist, tmp_idx, ss);
            if (tid == 0) {
                ss.queue_size = K;
            }
        }
        __syncthreads();
    }

    // Final exact prune if slack remains in the queue.
    if (ss.queue_size > K) {
        const int qsize = ss.queue_size;
        prune_queue<K, BLOCK_THREADS>(qsize, cand_dist, cand_idx, tmp_dist, tmp_idx, ss);
        if (tid == 0) {
            ss.queue_size = K;
        }
    }
    __syncthreads();

    sort_final_topk<K, BLOCK_THREADS>(cand_dist, cand_idx);

    // Store sorted (index, squared-distance) pairs for this query.
    // Use member assignment instead of constructing std::pair on device.
    std::pair<int, float>* const out =
        result + static_cast<std::size_t>(qid) * static_cast<std::size_t>(K);

    for (int i = tid; i < K; i += BLOCK_THREADS) {
        out[i].first  = cand_idx[i];
        out[i].second = cand_dist[i];
    }
}

template<int K>
inline void launch_knn_impl(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result)
{
    using Cfg = KnnConfig<K, kBlockThreads, kLocalBuffer, kQueueSlackRounds>;

    // These attribute calls are idempotent and keep the kernel in a high-shared-memory mode.
    // The wrapper intentionally remains asynchronous with respect to the host, matching
    // normal CUDA kernel launch semantics.
    static_cast<void>(cudaFuncSetAttribute(
        knn_kernel<K, kBlockThreads, kLocalBuffer, kQueueSlackRounds>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Cfg::kDynamicSharedBytes)));

    static_cast<void>(cudaFuncSetAttribute(
        knn_kernel<K, kBlockThreads, kLocalBuffer, kQueueSlackRounds>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    knn_kernel<K, kBlockThreads, kLocalBuffer, kQueueSlackRounds>
        <<<query_count, kBlockThreads, Cfg::kDynamicSharedBytes>>>(
            query, query_count, data, data_count, result);
}

} // namespace

void run_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k)
{
    if (query_count <= 0) {
        return;
    }

    // K is specialized at compile time so that:
    //   - the final sort network is fully specialized,
    //   - shared-memory sizes scale with K,
    //   - the compiler can optimize aggressively for the fixed top-K width.
    switch (k) {
        case 32:   launch_knn_impl<32>  (query, query_count, data, data_count, result); break;
        case 64:   launch_knn_impl<64>  (query, query_count, data, data_count, result); break;
        case 128:  launch_knn_impl<128> (query, query_count, data, data_count, result); break;
        case 256:  launch_knn_impl<256> (query, query_count, data, data_count, result); break;
        case 512:  launch_knn_impl<512> (query, query_count, data, data_count, result); break;
        case 1024: launch_knn_impl<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Inputs are guaranteed valid by contract.
    }
}