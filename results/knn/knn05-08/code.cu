#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

// One warp processes one query. A 256-thread block therefore handles 8 queries at once,
// which gives a good balance between occupancy and reuse of the shared-memory data batch.
constexpr int kWarpSize       = 32;
constexpr int kBlockThreads   = 256;
constexpr int kWarpsPerBlock  = kBlockThreads / kWarpSize;
constexpr unsigned kFullMask  = 0xffffffffu;

// Packed sortable key:
//   high 32 bits = IEEE-754 bits of the non-negative squared distance
//   low  32 bits = data index
//
// For non-negative floats, unsigned integer order matches floating-point order, so a single
// 64-bit integer comparison implements lexicographic ordering by (distance, index).
// This gives deterministic tie-breaking by smaller data index essentially for free.
using pair_key_t = unsigned long long;
constexpr pair_key_t kInfKey =
    (static_cast<pair_key_t>(0x7f800000u) << 32) | static_cast<pair_key_t>(0xffffffffu);

// Shared memory budget is tuned against A100's 163,840-byte opt-in shared-memory limit.
// H100 has more, so these choices also fit there.
constexpr std::size_t kA100MaxOptInShared = 163840u;

__device__ __forceinline__ pair_key_t pack_key(float dist, int idx) {
    return (static_cast<pair_key_t>(__float_as_uint(dist)) << 32) |
           static_cast<unsigned int>(idx);
}

__device__ __forceinline__ float unpack_dist(pair_key_t key) {
    return __uint_as_float(static_cast<unsigned int>(key >> 32));
}

__device__ __forceinline__ int unpack_idx(pair_key_t key) {
    // Real indices are non-negative, but the sentinel 0xffffffff becomes -1 here,
    // which is a sensible "invalid" value if it ever escapes (it should not at the end).
    return static_cast<int>(static_cast<unsigned int>(key));
}

// In-place bitonic sort of K packed (distance, index) keys in shared memory.
// K is always a power of two in [32, 1024].
template <int K>
__device__ __forceinline__ void bitonic_sort_shared(pair_key_t* keys) {
    const int lane = threadIdx.x & (kWarpSize - 1);

    #pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll 1
            for (int pos = lane; pos < K; pos += kWarpSize) {
                const int partner = pos ^ stride;
                if (partner > pos) {
                    const bool ascending = ((pos & size) == 0);
                    pair_key_t a = keys[pos];
                    pair_key_t b = keys[partner];
                    if (ascending ? (b < a) : (a < b)) {
                        keys[pos]     = b;
                        keys[partner] = a;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

// Merge-path partition for two sorted arrays A and B of equal length K.
// Returns i such that A[0..i) and B[0..diag-i) contain exactly the first "diag" merged elements.
template <int K>
__device__ __forceinline__ int merge_path_partition(const pair_key_t* a,
                                                    const pair_key_t* b,
                                                    int diag) {
    int low  = (diag > K) ? (diag - K) : 0;
    int high = (diag < K) ? diag : K;

    while (low <= high) {
        const int i = (low + high) >> 1;
        const int j = diag - i;

        if (i > 0 && j < K && a[i - 1] > b[j]) {
            high = i - 1;
        } else if (j > 0 && i < K && b[j - 1] > a[i]) {
            low = i + 1;
        } else {
            return i;
        }
    }

    return low;
}

// Merge the shared-memory candidate buffer with the warp-private top-K result.
// The warp-private top-K lives distributed across lane-local registers:
// each lane owns K/32 consecutive sorted keys.
// During the merge, the current top-K is staged to shared memory so the warp can do
// random accesses for merge-path partitioning without any extra device allocations.
template <int K>
__device__ __forceinline__ void merge_buffer(pair_key_t (&top)[K / kWarpSize],
                                             pair_key_t* warp_store,
                                             int& buf_count,
                                             pair_key_t& kth_key) {
    constexpr int CHUNK = K / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int base = lane * CHUNK;

    pair_key_t* curr = warp_store;      // [0, K)   : staging area for current top-K
    pair_key_t* cand = warp_store + K;  // [K, 2K)  : candidate buffer

    // Stage the current top-K into shared memory.
    #pragma unroll
    for (int t = 0; t < CHUNK; ++t) {
        curr[base + t] = top[t];
    }

    // Pad the unused tail of the candidate buffer with +inf so we can sort exactly K elements.
    #pragma unroll
    for (int pos = lane; pos < K; pos += kWarpSize) {
        if (pos >= buf_count) {
            cand[pos] = kInfKey;
        }
    }

    __syncwarp(kFullMask);

    // Sort only the candidate side; the current top-K is already sorted.
    bitonic_sort_shared<K>(cand);

    // Parallel merge of two sorted K-element lists. Each lane writes its own CHUNK outputs
    // directly back into its register-resident "top" segment.
    const int diag = base;
    int ai = merge_path_partition<K>(curr, cand, diag);
    int bi = diag - ai;

    #pragma unroll
    for (int t = 0; t < CHUNK; ++t) {
        const pair_key_t av = (ai < K) ? curr[ai] : kInfKey;
        const pair_key_t bv = (bi < K) ? cand[bi] : kInfKey;
        const bool take_a = (av <= bv);
        top[t] = take_a ? av : bv;
        ai += static_cast<int>(take_a);
        bi += static_cast<int>(!take_a);
    }

    buf_count = 0;

    // The final (K-1)-th key is owned by lane 31, local slot CHUNK-1.
    kth_key = __shfl_sync(kFullMask, top[CHUNK - 1], 31);
}

template <int K, int DATA_BATCH>
__global__ __launch_bounds__(kBlockThreads)
void knn2d_kernel(const float2* __restrict__ query,
                  int query_count,
                  const float2* __restrict__ data,
                  int data_count,
                  std::pair<int, float>* __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");
    static_assert((DATA_BATCH % kWarpSize) == 0, "Batch size must be a multiple of warp size.");

    constexpr int CHUNK = K / kWarpSize;

    const int lane    = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int q_idx   = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool active = (q_idx < query_count);

    // Shared-memory layout:
    //   sh_data[DATA_BATCH]                            : cached data points for the current batch
    //   sh_keys[kWarpsPerBlock * 2 * K]              : per-warp [current-top staging | candidate buffer]
    extern __shared__ float2 shmem[];
    float2* sh_data = shmem;
    pair_key_t* sh_keys = reinterpret_cast<pair_key_t*>(sh_data + DATA_BATCH);
    pair_key_t* warp_store = sh_keys + static_cast<std::size_t>(warp_id) * (2 * K);
    pair_key_t* cand = warp_store + K;

    // Warp-private intermediate result: each lane keeps CHUNK consecutive sorted entries.
    pair_key_t top[CHUNK];
    #pragma unroll
    for (int t = 0; t < CHUNK; ++t) {
        top[t] = kInfKey;
    }

    pair_key_t kth_key = kInfKey;
    int buf_count = 0;  // Uniform across the warp.

    // Load the query point once and broadcast it across the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[q_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Scan the database in shared-memory batches.
    for (int batch_begin = 0; batch_begin < data_count; batch_begin += DATA_BATCH) {
        const int remaining  = data_count - batch_begin;
        const int batch_size = (remaining < DATA_BATCH) ? remaining : DATA_BATCH;
        const float2* batch_ptr = data + batch_begin;

        // Cooperative block-wide load into shared memory.
        for (int i = threadIdx.x; i < batch_size; i += kBlockThreads) {
            sh_data[i] = batch_ptr[i];
        }
        __syncthreads();

        if (active) {
            // Each 32-point step maps naturally to one warp iteration:
            // lane t evaluates the t-th point in the step.
            #pragma unroll 1
            for (int base = 0; base < batch_size; base += kWarpSize) {
                const int step_remaining = batch_size - base;
                const int step_capacity  = (step_remaining < kWarpSize) ? step_remaining : kWarpSize;

                // To avoid a spill path for a partially full buffer, merge slightly early when
                // fewer than "step_capacity" slots remain. This is exact and usually beneficial:
                // it tightens the threshold sooner and therefore increases skipping.
                if (buf_count > K - step_capacity) {
                    merge_buffer<K>(top, warp_store, buf_count, kth_key);
                }

                const int local_idx = base + lane;

                pair_key_t candidate_key = kInfKey;
                bool keep = false;

                if (local_idx < batch_size) {
                    const float2 p = sh_data[local_idx];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    const float dist = fmaf(dx, dx, dy * dy);
                    candidate_key = pack_key(dist, batch_begin + local_idx);

                    // Skip anything that is not better than the current K-th best entry.
                    keep = (candidate_key < kth_key);
                }

                const unsigned mask = __ballot_sync(kFullMask, keep);
                const int num_kept = __popc(mask);

                if (keep) {
                    const unsigned lower = mask & ((1u << lane) - 1u);
                    const int rank = __popc(lower);
                    cand[buf_count + rank] = candidate_key;
                }

                buf_count += num_kept;

                // Exact "buffer full" merge.
                if (buf_count == K) {
                    merge_buffer<K>(top, warp_store, buf_count, kth_key);
                }
            }
        }

        // No need for a trailing barrier after the last batch.
        if (batch_begin + batch_size < data_count) {
            __syncthreads();
        }
    }

    if (active) {
        // Final partial-buffer merge, if any candidates remain.
        if (buf_count > 0) {
            merge_buffer<K>(top, warp_store, buf_count, kth_key);
        }

        // Write the sorted top-K for this query.
        std::pair<int, float>* out = result + static_cast<std::size_t>(q_idx) * K;
        const int base = lane * CHUNK;

        #pragma unroll
        for (int t = 0; t < CHUNK; ++t) {
            const pair_key_t key = top[t];
            out[base + t].first  = unpack_idx(key);
            out[base + t].second = unpack_dist(key);
        }
    }
}

template <int K, int DATA_BATCH>
void launch_knn_impl(const float2* query,
                     int query_count,
                     const float2* data,
                     int data_count,
                     std::pair<int, float>* result) {
    constexpr std::size_t shared_bytes =
        static_cast<std::size_t>(DATA_BATCH) * sizeof(float2) +
        static_cast<std::size_t>(kWarpsPerBlock) * 2ull * static_cast<std::size_t>(K) * sizeof(pair_key_t);

    static_assert(shared_bytes <= kA100MaxOptInShared,
                  "Chosen kernel configuration exceeds the A100 opt-in shared-memory limit.");

    // Opt in to the large shared-memory carveout needed by the tuned batch sizes.
    cudaFuncSetAttribute(knn2d_kernel<K, DATA_BATCH>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_bytes));
    cudaFuncSetAttribute(knn2d_kernel<K, DATA_BATCH>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
    knn2d_kernel<K, DATA_BATCH><<<blocks, kBlockThreads, shared_bytes>>>(
        query, query_count, data, data_count, result);
}

}  // namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Batch-size tuning:
    //   K = 32/64/128 : 8192-point batches keep 2 resident blocks/SM on A100.
    //   K = 256       : 4096-point batches keep 2 resident blocks/SM.
    //   K = 512       : 2048-point batches are the largest power-of-two choice that still keeps 2 blocks/SM.
    //   K = 1024      : occupancy is 1 block/SM regardless, so use 4096 points; this exactly matches
    //                   the A100 163,840-byte opt-in shared-memory limit.
    switch (k) {
        case 32:   launch_knn_impl<32,   8192>(query, query_count, data, data_count, result); break;
        case 64:   launch_knn_impl<64,   8192>(query, query_count, data, data_count, result); break;
        case 128:  launch_knn_impl<128,  8192>(query, query_count, data, data_count, result); break;
        case 256:  launch_knn_impl<256,  4096>(query, query_count, data, data_count, result); break;
        case 512:  launch_knn_impl<512,  2048>(query, query_count, data, data_count, result); break;
        case 1024: launch_knn_impl<1024, 4096>(query, query_count, data, data_count, result); break;
        default:   break;  // Per the problem statement, this path is unreachable.
    }
}