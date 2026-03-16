#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

// Exact brute-force k-NN for 2D points in squared Euclidean distance.
//
// Design choices tuned for modern data-center GPUs (A100/H100 class):
// - One warp computes one query.
// - A 256-thread block therefore processes 8 queries at once. This is a good compromise:
//   enough data-tile reuse inside a block, while still creating enough blocks for
//   query counts in the low thousands.
// - Legal k values are only {32, 64, 128, 256, 512, 1024}, so we specialize on K.
//   That gives fixed-size per-lane top-k storage and lets the compiler optimize
//   register usage and loop structure aggressively.
// - Each warp keeps the current top-k in a private register array.
// - Each warp also owns a k-entry candidate buffer in shared memory.
// - Data points are loaded tile-by-tile into shared memory by the whole block.
// - When the candidate buffer fills (or would overflow), the warp:
//     1) pads the candidate buffer with +inf,
//     2) sorts it with a warp-local bitonic network,
//     3) merges it with the current top-k using merge path.
//
// Distances are returned as squared L2 norms, exactly as requested.

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_THREADS = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;

// For non-negative floats, IEEE-754 bit order matches numeric order.  Packing the
// distance in the upper 32 bits and the index in the lower 32 bits lets us move
// (distance,index) as a single 64-bit word during shared-memory sorts/merges.
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
constexpr std::uint64_t INF_KEY = 0x7F800000FFFFFFFFull;  // +inf distance, index -1.
constexpr std::size_t A100_MAX_SMEM_BYTES = 163840;

// Device code should not depend on std::pair having device constructors/operators.
// The result buffer is raw cudaMalloc memory; we reinterpret it as this trivially
// assignable POD, while asserting the common ABI layout.
struct ResultPair {
    int first;
    float second;
};

static_assert(sizeof(ResultPair) == sizeof(std::pair<int, float>),
              "std::pair<int,float> layout mismatch");
static_assert(alignof(ResultPair) == alignof(std::pair<int, float>),
              "std::pair<int,float> alignment mismatch");

constexpr int ct_log2(int v) {
    return (v <= 1) ? 0 : 1 + ct_log2(v >> 1);
}

__device__ __forceinline__ std::uint64_t pack_key(int idx, float dist) {
    return (static_cast<std::uint64_t>(__float_as_uint(dist)) << 32) |
           static_cast<std::uint32_t>(idx);
}

__device__ __forceinline__ int unpack_idx(std::uint64_t key) {
    return static_cast<int>(static_cast<std::uint32_t>(key));
}

__device__ __forceinline__ float unpack_dist(std::uint64_t key) {
    return __uint_as_float(static_cast<std::uint32_t>(key >> 32));
}

// The warp-private top-k is kept in registers in "lane-chunked" logical order:
// lane l owns logical positions [l * ITEMS_PER_LANE, ..., l * ITEMS_PER_LANE + ITEMS_PER_LANE - 1].
//
// When spilling that array to shared memory for a merge, we store it in an interleaved
// physical layout best_sh[lane + 32 * t].  That spill pattern is shared-memory friendly.
// This helper maps a logical index back to that physical interleaved layout.
template <int K>
__device__ __forceinline__ std::uint64_t best_logical_load(
    const std::uint64_t* __restrict__ best_sh,
    int logical_idx) {
    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;
    constexpr int ITEMS_LOG2 = ct_log2(ITEMS_PER_LANE);
    constexpr int ITEMS_MASK = ITEMS_PER_LANE - 1;

    const int owner_lane = logical_idx >> ITEMS_LOG2;
    const int owner_off  = logical_idx & ITEMS_MASK;
    return best_sh[owner_lane + owner_off * WARP_SIZE];
}

// Warp-local bitonic sort for exactly K keys in shared memory.
// K is always a power of two and at most 1024.
template <int K>
__device__ __forceinline__ void sort_candidate_buffer(
    std::uint64_t* __restrict__ cand_sh,
    int lane) {
    static_assert(K >= WARP_SIZE && (K & (K - 1)) == 0, "K must be a power of two in [32,1024]");
    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;

    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int t = 0; t < ITEMS_PER_LANE; ++t) {
                const int i = lane + t * WARP_SIZE;
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = ((i & size) == 0);
                    const std::uint64_t a = cand_sh[i];
                    const std::uint64_t b = cand_sh[j];
                    const bool do_swap = ascending ? (a > b) : (a < b);
                    if (do_swap) {
                        cand_sh[i] = b;
                        cand_sh[j] = a;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

// Standard merge-path partition for two sorted arrays of equal length K.
// A is the current best set (stored in the interleaved shared-memory layout above),
// B is the sorted candidate buffer in contiguous shared memory.
template <int K>
__device__ __forceinline__ int merge_path_partition(
    const std::uint64_t* __restrict__ best_sh,
    const std::uint64_t* __restrict__ cand_sh,
    int diag) {
    int low  = (diag > K) ? (diag - K) : 0;
    int high = (diag < K) ? diag : K;

    while (low < high) {
        const int mid = (low + high) >> 1;
        const int b   = diag - mid;

        const std::uint64_t a_key  = (mid < K) ? best_logical_load<K>(best_sh, mid) : INF_KEY;
        const std::uint64_t b_left = (b > 0) ? cand_sh[b - 1] : 0ull;

        // We advance "mid" while A[mid] belongs before B[b-1].
        if ((b > 0) && (mid < K) && (a_key <= b_left)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Flush the warp's candidate buffer into the current best set.
// best_keys[] is the private register copy of the current top-k, lane-chunked.
// cand_sh[] is the shared candidate buffer, contiguous.
// best_sh[] is shared scratch for the current best set, written in interleaved layout.
// Returns the updated k-th distance, broadcast to the whole warp.
template <int K>
__device__ __forceinline__ float merge_candidate_buffer(
    std::uint64_t* __restrict__ best_sh,
    std::uint64_t* __restrict__ cand_sh,
    std::uint64_t (&best_keys)[K / WARP_SIZE],
    int cand_count,
    int lane) {
    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;

    // Spill current best to shared scratch in the interleaved physical layout.
    // At the same time, pad the unused tail of the candidate buffer with +inf,
    // so we can always sort/merge exactly K elements.
#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const int phys = lane + t * WARP_SIZE;
        best_sh[phys] = best_keys[t];
        if (phys >= cand_count) {
            cand_sh[phys] = INF_KEY;
        }
    }
    __syncwarp(FULL_MASK);

    // Sort candidates in-place.
    sort_candidate_buffer<K>(cand_sh, lane);

    // Each lane produces one contiguous chunk of the merged top-k.
    const int out_begin = lane * ITEMS_PER_LANE;
    int a = merge_path_partition<K>(best_sh, cand_sh, out_begin);
    int b = out_begin - a;

#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const std::uint64_t a_key = (a < K) ? best_logical_load<K>(best_sh, a) : INF_KEY;
        const std::uint64_t b_key = (b < K) ? cand_sh[b] : INF_KEY;
        const bool take_a = (a_key <= b_key);
        best_keys[t] = take_a ? a_key : b_key;
        a += static_cast<int>(take_a);
        b += static_cast<int>(!take_a);
    }

    // Lane 31 owns the last logical chunk, and its last register is the global k-th item.
    float kth_dist = unpack_dist(best_keys[ITEMS_PER_LANE - 1]);
    kth_dist = __shfl_sync(FULL_MASK, kth_dist, WARP_SIZE - 1);
    return kth_dist;
}

template <int K, int BATCH_POINTS>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    ResultPair* __restrict__ result) {
    static_assert(K >= WARP_SIZE && K <= 1024 && (K & (K - 1)) == 0,
                  "K must be a power of two in [32,1024]");
    static_assert(BATCH_POINTS % WARP_SIZE == 0, "BATCH_POINTS must be a multiple of 32");
    static_assert((BATCH_POINTS & 1) == 0, "BATCH_POINTS must be even for float4 tile loads");

    constexpr int ITEMS_PER_LANE = K / WARP_SIZE;

    // Shared-memory layout:
    //   [batch tile of float2 points][candidate buffers][best scratch buffers]
    //
    // uint4 gives us 16-byte alignment so float4 tile loads into shared memory are aligned.
    extern __shared__ uint4 smem_u4[];
    float2* batch = reinterpret_cast<float2*>(smem_u4);
    std::uint64_t* cand_all = reinterpret_cast<std::uint64_t*>(batch + BATCH_POINTS);
    std::uint64_t* best_all = cand_all + WARPS_PER_BLOCK * K;

    const int warp_id   = threadIdx.x >> 5;
    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const int query_idx = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active_query = (query_idx < query_count);

    std::uint64_t* cand_sh = cand_all + warp_id * K;
    std::uint64_t* best_sh = best_all + warp_id * K;

    // One query point per warp; load once and broadcast.
    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && active_query) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Private register-resident current top-k, lane-chunked.
    std::uint64_t best_keys[ITEMS_PER_LANE];
#pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        best_keys[t] = INF_KEY;
    }

    float kth_dist = CUDART_INF_F;
    int cand_count = 0;

    const unsigned lane_bit     = 1u << lane;
    const unsigned lane_mask_lt = lane_bit - 1u;

    for (int tile_base = 0; tile_base < data_count; tile_base += BATCH_POINTS) {
        int batch_count = data_count - tile_base;
        if (batch_count > BATCH_POINTS) {
            batch_count = BATCH_POINTS;
        }

        // Cooperative, vectorized tile load:
        // data is float2[], so loading two points at a time as float4 halves the number of
        // load instructions. cudaMalloc alignment plus even tile starts make this safe here.
        const int pair_count = batch_count >> 1;
        float4* batch4 = reinterpret_cast<float4*>(batch);
        const float4* data4 = reinterpret_cast<const float4*>(data + tile_base);

        for (int i = threadIdx.x; i < pair_count; i += BLOCK_THREADS) {
            batch4[i] = data4[i];
        }
        if ((batch_count & 1) && threadIdx.x == 0) {
            batch[batch_count - 1] = data[tile_base + batch_count - 1];
        }

        // Whole block must finish populating the shared tile before any warp consumes it.
        __syncthreads();

        if (active_query) {
            // Walk the tile in warp-sized groups so each lane handles one point per group.
            for (int group = 0; group < batch_count; group += WARP_SIZE) {
                const int local_idx = group + lane;
                const bool point_active = (local_idx < batch_count);

                float dist = 0.0f;
                const int data_idx = tile_base + local_idx;

                if (point_active) {
                    const float2 p = batch[local_idx];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = __fmaf_rn(dx, dx, dy * dy);
                }

                // Problem statement only requires pruning by the k-th distance, not by any
                // tie-break rule, so a strict distance comparison is sufficient.
                unsigned mask = __ballot_sync(FULL_MASK, point_active && (dist < kth_dist));

                // Rare overflow case: the current group would not fit into the candidate buffer.
                // Flush current candidates first, then re-evaluate the current group against the
                // tighter post-merge threshold.
                while (mask != 0u) {
                    const int accepted = __popc(mask);

                    if ((cand_count > 0) && (cand_count + accepted > K)) {
                        kth_dist = merge_candidate_buffer<K>(best_sh, cand_sh, best_keys, cand_count, lane);
                        cand_count = 0;
                        mask = __ballot_sync(FULL_MASK, point_active && (dist < kth_dist));
                        continue;
                    }

                    const int base = cand_count;

                    if (mask & lane_bit) {
                        const int rank = __popc(mask & lane_mask_lt);
                        cand_sh[base + rank] = pack_key(data_idx, dist);
                    }

                    cand_count = base + accepted;

                    if (cand_count == K) {
                        kth_dist = merge_candidate_buffer<K>(best_sh, cand_sh, best_keys, cand_count, lane);
                        cand_count = 0;
                    }
                    break;
                }
            }
        }

        // The block reuses the same shared tile buffer on the next iteration.
        __syncthreads();
    }

    if (active_query) {
        if (cand_count > 0) {
            kth_dist = merge_candidate_buffer<K>(best_sh, cand_sh, best_keys, cand_count, lane);
            (void)kth_dist;
        }

        // Write back the final lane-chunked top-k.
        // This store pattern is not the most coalesced possible, but the output traffic is tiny
        // relative to the O(query_count * data_count) distance work and data-tile traffic.
        const std::size_t out_base = static_cast<std::size_t>(query_idx) * K;
        const int chunk_base = lane * ITEMS_PER_LANE;

#pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const std::uint64_t key = best_keys[t];
            ResultPair& dst = result[out_base + chunk_base + t];
            dst.first  = unpack_idx(key);
            dst.second = unpack_dist(key);
        }
    }
}

template <int K, int BATCH_POINTS>
inline void launch_knn_specialized(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result) {
    constexpr std::size_t SHARED_BYTES =
        static_cast<std::size_t>(BATCH_POINTS) * sizeof(float2) +
        static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(std::uint64_t) * 2u;

    static_assert(SHARED_BYTES <= A100_MAX_SMEM_BYTES,
                  "Chosen kernel configuration exceeds A100 shared-memory capacity");

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Opt into the required dynamic shared memory size for this specialization.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, BATCH_POINTS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SHARED_BYTES));

    knn_kernel<K, BATCH_POINTS><<<grid, block, SHARED_BYTES>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<ResultPair*>(result));
}

}  // namespace

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

    // Batch-size heuristics:
    // - K <= 256: 4096-point tiles. Shared memory still allows 2-4 blocks/SM on A100,
    //   which is a good balance of tile reuse and occupancy.
    // - K >= 512: 2048-point tiles. Larger K consumes more warp-private/shared state,
    //   so a smaller tile keeps the block within the A100 shared-memory budget.
    switch (k) {
        case 32:
            launch_knn_specialized<32, 4096>(query, query_count, data, data_count, result);
            break;
        case 64:
            launch_knn_specialized<64, 4096>(query, query_count, data, data_count, result);
            break;
        case 128:
            launch_knn_specialized<128, 4096>(query, query_count, data, data_count, result);
            break;
        case 256:
            launch_knn_specialized<256, 4096>(query, query_count, data, data_count, result);
            break;
        case 512:
            launch_knn_specialized<512, 2048>(query, query_count, data, data_count, result);
            break;
        case 1024:
            launch_knn_specialized<1024, 2048>(query, query_count, data, data_count, result);
            break;
        default:
            // The problem statement guarantees valid k, so this path is only defensive.
            break;
    }
}