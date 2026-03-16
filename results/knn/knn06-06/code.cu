#include <cuda_runtime.h>
#include <utility>
#include <cstddef>
#include <cstdint>

namespace {

// Warp-per-query 2D k-NN for modern data-center NVIDIA GPUs.
//
// Design choices:
// - One warp computes one query exactly as requested.
// - A 256-thread block = 8 warps = 8 queries/block. This is a practical balance:
//   enough inter-query reuse of the shared data tile, while still yielding enough
//   blocks when query_count is only in the low-thousands on ~100+ SM GPUs.
// - A 2048-point data tile is the largest tile that still keeps 2 resident blocks/SM
//   possible for K <= 512 with this block size on A100/H100-class shared-memory budgets.
// - Each warp keeps its current top-k privately in registers as packed 64-bit keys.
// - Each warp owns a shared-memory candidate buffer of exactly k entries plus a shared
//   counter updated via atomicAdd as required.
// - Merge cost is reduced by exploiting the fact that the current top-k is already sorted:
//     1) sort only the candidate buffer,
//     2) reverse it,
//     3) bitonic-merge current-top (ascending) with candidates (descending).
//
// Packed-key optimization:
//   key = (float_bits(distance) << 32) | index
// Distances are squared Euclidean distances and therefore non-negative. For non-negative
// IEEE-754 floats, unsigned integer bit order matches numeric order, so unsigned 64-bit
// comparison on the packed key implements lexicographic ordering by (distance, index).
// This halves shared-memory traffic in compare/swap relative to separate dist/index arrays.

constexpr int kWarpSize       = 32;
constexpr unsigned kFullMask  = 0xffffffffu;
constexpr int kBlockThreads   = 256;
constexpr int kWarpsPerBlock  = kBlockThreads / kWarpSize;
constexpr int kDataTile       = 2048;

constexpr std::size_t kTileBytes  = static_cast<std::size_t>(kDataTile) * sizeof(float2);
constexpr std::size_t kCountBytes = static_cast<std::size_t>(kWarpsPerBlock) * sizeof(unsigned int);

// +inf distance, maximal valid tie-break index.
constexpr std::uint64_t kPackedSentinel =
    (static_cast<std::uint64_t>(0x7f800000u) << 32) | 0x7fffffffu;

// Shared-memory layout uses uint64_t as the base type so the dynamic shared-memory base
// is naturally 8-byte aligned for both float2 tiles and packed keys.
static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a whole number of warps.");
static_assert(kDataTile % kWarpSize == 0, "Tile size must be a whole number of warps.");
static_assert((kTileBytes % alignof(std::uint64_t)) == 0, "Tile must keep packed-key alignment.");
static_assert(((kTileBytes + kCountBytes) % alignof(std::uint64_t)) == 0,
              "Counts must keep packed-key alignment.");

template <int K>
constexpr std::size_t shared_bytes_for_kernel() {
    static_assert(K >= kWarpSize && K <= 1024 && ((K & (K - 1)) == 0),
                  "K must be a power of two in [32, 1024].");
    return kTileBytes + kCountBytes +
           static_cast<std::size_t>(kWarpsPerBlock) * 2u * static_cast<std::size_t>(K) *
               sizeof(std::uint64_t);
}

// A100/H100-class devices can opt in to this per-block dynamic shared-memory size.
// The chosen configuration stays below the A100 163 KiB per-block opt-in limit.
static_assert(shared_bytes_for_kernel<1024>() <= 166912,
              "Shared-memory configuration exceeds the target per-block limit.");

__device__ __forceinline__ std::uint64_t pack_key(const float dist, const int idx) {
    return (static_cast<std::uint64_t>(__float_as_uint(dist)) << 32) |
           static_cast<std::uint32_t>(idx);
}

__device__ __forceinline__ float unpack_dist(const std::uint64_t key) {
    return __uint_as_float(static_cast<unsigned int>(key >> 32));
}

__device__ __forceinline__ int unpack_idx(const std::uint64_t key) {
    return static_cast<int>(static_cast<std::uint32_t>(key));
}

template <int K>
__device__ __forceinline__
void merge_buffer_into_topk(
    const int lane,
    std::uint64_t (&top)[K / kWarpSize],
    float &max_distance,
    unsigned int &buffer_count,
    unsigned int *const __restrict__ shared_count,
    std::uint64_t *const __restrict__ work)
{
    constexpr int kItemsPerLane = K / kWarpSize;

    // The private top-k is stored in an interleaved lane-major layout:
    //   logical slot = item * 32 + lane
    // This makes staging to/from shared memory naturally coalesced at warp granularity.
#pragma unroll
    for (int item = 0; item < kItemsPerLane; ++item) {
        work[item * kWarpSize + lane] = top[item];
    }

    // Candidate buffer occupies work[K : 2K). Pad the unused tail with sentinels so the
    // sort/merge logic is uniform regardless of the current candidate count.
    for (int pos = lane; pos < K; pos += kWarpSize) {
        if (static_cast<unsigned int>(pos) >= buffer_count) {
            work[K + pos] = kPackedSentinel;
        }
    }
    __syncwarp();

    // Sort only the candidate half (ascending).
#pragma unroll 1
    for (unsigned int size = 2; size <= static_cast<unsigned int>(K); size <<= 1) {
#pragma unroll 1
        for (unsigned int stride = size >> 1; stride > 0; stride >>= 1) {
            for (unsigned int i = static_cast<unsigned int>(lane); i < static_cast<unsigned int>(K); i += kWarpSize) {
                const unsigned int partner = i ^ stride;
                if (partner > i) {
                    const unsigned int a_pos = K + i;
                    const unsigned int b_pos = K + partner;

                    const std::uint64_t a = work[a_pos];
                    const std::uint64_t b = work[b_pos];

                    const bool ascending = ((i & size) == 0u);
                    if (ascending ? (b < a) : (a < b)) {
                        work[a_pos] = b;
                        work[b_pos] = a;
                    }
                }
            }
            __syncwarp();
        }
    }

    // Reverse the candidate half so that [0:K) is ascending and [K:2K) is descending,
    // i.e. the whole 2K sequence is bitonic.
    for (unsigned int i = static_cast<unsigned int>(lane); i < static_cast<unsigned int>(K >> 1); i += kWarpSize) {
        const unsigned int a_pos = K + i;
        const unsigned int b_pos = (2u * static_cast<unsigned int>(K) - 1u) - i;

        const std::uint64_t a = work[a_pos];
        const std::uint64_t b = work[b_pos];
        work[a_pos] = b;
        work[b_pos] = a;
    }
    __syncwarp();

    // Bitonic merge of the already-bitonic 2K sequence into ascending order.
#pragma unroll 1
    for (unsigned int stride = static_cast<unsigned int>(K); stride > 0; stride >>= 1) {
        for (unsigned int i = static_cast<unsigned int>(lane); i < 2u * static_cast<unsigned int>(K); i += kWarpSize) {
            const unsigned int partner = i ^ stride;
            if (partner > i) {
                const std::uint64_t a = work[i];
                const std::uint64_t b = work[partner];
                if (b < a) {
                    work[i] = b;
                    work[partner] = a;
                }
            }
        }
        __syncwarp();
    }

    // Reload the first K logical slots as the new private top-k.
#pragma unroll
    for (int item = 0; item < kItemsPerLane; ++item) {
        top[item] = work[item * kWarpSize + lane];
    }

    // The worst retained distance sits at logical slot K-1, which maps to
    // lane 31, local item (K/32 - 1).
    float tail = 0.0f;
    if (lane == (kWarpSize - 1)) {
        tail = unpack_dist(top[kItemsPerLane - 1]);
    }
    max_distance = __shfl_sync(kFullMask, tail, kWarpSize - 1);

    // Keep the required shared count in sync with the now-empty candidate buffer.
    if (lane == 0) {
        *shared_count = 0u;
    }
    __syncwarp();
    buffer_count = 0u;
}

template <int K>
__device__ __forceinline__
void maybe_insert_candidate(
    const int lane,
    const unsigned int lane_lt_mask,
    const bool valid,
    const float dist,
    const int idx,
    std::uint64_t (&top)[K / kWarpSize],
    float &max_distance,
    unsigned int &buffer_count,
    unsigned int *const __restrict__ shared_count,
    std::uint64_t *const __restrict__ work)
{
    // Strictly lower than max_distance exactly as requested.
    bool accept = valid && (dist < max_distance);
    unsigned int mask = __ballot_sync(kFullMask, accept);
    if (mask == 0u) {
        return;
    }

    unsigned int add = __popc(mask);

    // If the whole warp's accepted points would overflow the candidate buffer,
    // flush the current buffer first, then re-test against the tighter threshold.
    if (buffer_count + add > static_cast<unsigned int>(K)) {
        merge_buffer_into_topk<K>(lane, top, max_distance, buffer_count, shared_count, work);

        accept = valid && (dist < max_distance);
        mask = __ballot_sync(kFullMask, accept);
        if (mask == 0u) {
            return;
        }
        add = __popc(mask);
    }

    // One shared atomicAdd per warp-round reserves a contiguous range of slots;
    // each accepting lane then computes its own offset with a warp-local prefix sum.
    unsigned int base = 0u;
    if (lane == 0) {
        base = atomicAdd(shared_count, add);
    }
    base = __shfl_sync(kFullMask, base, 0);

    if (accept) {
        const unsigned int offset = __popc(mask & lane_lt_mask);
        work[K + base + offset] = pack_key(dist, idx);
    }

    // Needed so a subsequent merge sees all candidate writes.
    __syncwarp();

    buffer_count = base + add;

    // Flush immediately when the buffer becomes full so max_distance tightens as early
    // as possible and later candidates can be rejected cheaply.
    if (buffer_count == static_cast<unsigned int>(K)) {
        merge_buffer_into_topk<K>(lane, top, max_distance, buffer_count, shared_count, work);
    }
}

template <int K>
__global__ __launch_bounds__(kBlockThreads, 1)
void knn_kernel(
    const float2 *const __restrict__ query,
    const int query_count,
    const float2 *const __restrict__ data,
    const int data_count,
    std::pair<int, float> *const __restrict__ result)
{
    constexpr int kItemsPerLane = K / kWarpSize;

    // Dynamic shared-memory layout:
    //   [0, kDataTile)                   : float2 data tile
    //   [kDataTile, kDataTile + warps)   : per-warp candidate counts
    //   [rest]                           : per-warp work area of 2*K packed keys
    //                                      [0:K)   = staged current top-k
    //                                      [K:2K)  = candidate buffer
    extern __shared__ std::uint64_t smem_u64[];

    float2 *const tile = reinterpret_cast<float2 *>(smem_u64);
    unsigned int *const counts = reinterpret_cast<unsigned int *>(tile + kDataTile);
    std::uint64_t *const work_all = reinterpret_cast<std::uint64_t *>(counts + kWarpsPerBlock);

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const unsigned int lane_lt_mask = (1u << lane) - 1u;

    const int query_idx = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool active = (query_idx < query_count);

    if (lane == 0) {
        counts[warp_id] = 0u;
    }

    // Private per-query intermediate top-k.
    std::uint64_t top[kItemsPerLane];
#pragma unroll
    for (int item = 0; item < kItemsPerLane; ++item) {
        top[item] = kPackedSentinel;
    }

    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    std::uint64_t *const work = work_all + static_cast<std::size_t>(warp_id) * (2u * static_cast<std::size_t>(K));
    unsigned int *const shared_count = counts + warp_id;

    unsigned int buffer_count = 0u;
    float max_distance = CUDART_INF_F;

    for (int tile_base = 0; tile_base < data_count; tile_base += kDataTile) {
        const int remaining = data_count - tile_base;
        const int tile_count = (remaining < kDataTile) ? remaining : kDataTile;

        // Cooperative load of the next data tile by the whole block.
        for (int i = threadIdx.x; i < tile_count; i += kBlockThreads) {
            tile[i] = data[tile_base + i];
        }
        __syncthreads();

        if (active) {
            // Warp collectives require all 32 lanes to participate with the same mask.
            // Therefore the tile is processed in warp-sized rounds and invalid lanes
            // on the partial tail simply contribute accept=false.
            const int rounds = (tile_count + kWarpSize - 1) / kWarpSize;
#pragma unroll 1
            for (int round = 0; round < rounds; ++round) {
                const int j = round * kWarpSize + lane;
                const bool valid = (j < tile_count);

                float dist = 0.0f;
                if (valid) {
                    const float2 p = tile[j];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = __fmaf_rn(dx, dx, dy * dy);
                }

                maybe_insert_candidate<K>(
                    lane,
                    lane_lt_mask,
                    valid,
                    dist,
                    tile_base + j,
                    top,
                    max_distance,
                    buffer_count,
                    shared_count,
                    work);
            }
        }

        // Prevent any warp from overwriting the shared tile while other warps still use it.
        __syncthreads();
    }

    if (active) {
        if (buffer_count != 0u) {
            merge_buffer_into_topk<K>(lane, top, max_distance, buffer_count, shared_count, work);
        }

        // Coalesced warp write-back of the final sorted top-k.
        const std::size_t out_base = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);
#pragma unroll
        for (int item = 0; item < kItemsPerLane; ++item) {
            const std::uint64_t key = top[item];
            const std::size_t pos = out_base + static_cast<std::size_t>(item) * kWarpSize + static_cast<std::size_t>(lane);
            result[pos].first = unpack_idx(key);
            result[pos].second = unpack_dist(key);
        }
    }
}

template <int K>
inline void launch_knn_kernel(
    const float2 *query,
    const int query_count,
    const float2 *data,
    const int data_count,
    std::pair<int, float> *result)
{
    constexpr std::size_t shared_bytes = shared_bytes_for_kernel<K>();

    // Opt in to the larger per-block shared-memory budget used by this kernel.
    cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const dim3 block(kBlockThreads);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

    knn_kernel<K><<<grid, block, shared_bytes>>>(query, query_count, data, data_count, result);
}

} // namespace

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

    // No extra device memory is allocated; all temporary state lives in registers/shared.
    switch (k) {
        case 32:   launch_knn_kernel<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_kernel<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_kernel<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_kernel<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_kernel<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_kernel<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Inputs are specified to be valid.
    }
}