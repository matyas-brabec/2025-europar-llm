#include <cuda_runtime.h>
#include <utility>

namespace {

// Warp size is fixed by the problem statement.
constexpr int kWarpThreads = 32;

// A single tile size is used across all launch variants.
// 2016 = 63 * 32, i.e. an integer number of warp-sized chunks.
// This value is deliberately chosen so that even the heaviest A100-compatible
// variants (e.g. K=512/W=18 and K=1024/W=9) fit under the opt-in shared-memory
// limit, while still being large enough to amortize global-memory loads.
constexpr int kDataTilePoints = 2016;

// The host-side launcher asks for enough blocks to cover roughly 75% of the SMs.
// This balances two competing goals:
//   1) larger blocks -> more queries reuse each shared-memory tile,
//   2) smaller blocks -> more grid-level parallelism when query_count is modest.
constexpr int kTargetGridCoverageNum = 3;
constexpr int kTargetGridCoverageDen = 4;

using packed_t = unsigned long long;
using output_pair_t = std::pair<int, float>;

// Distances are non-negative squared L2 distances, so IEEE-754 bit patterns
// preserve the numerical order for all finite values and +inf. Packing the
// distance into the high 32 bits makes a plain unsigned compare sort by
// distance first. The low 32 bits carry the index payload and provide a
// deterministic tie-breaker, although the problem statement does not require
// any specific tie resolution.
__device__ __forceinline__ packed_t pack_pair(const float dist, const int idx) {
    return (static_cast<packed_t>(__float_as_uint(dist)) << 32) |
           static_cast<unsigned int>(idx);
}

__device__ __forceinline__ float unpack_dist(const packed_t v) {
    return __uint_as_float(static_cast<unsigned int>(v >> 32));
}

__device__ __forceinline__ int unpack_idx(const packed_t v) {
    return static_cast<int>(static_cast<unsigned int>(v));
}

__device__ __forceinline__ float squared_l2_2d(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return __fmaf_rn(dx, dx, dy * dy);
}

// Warp-local bitonic sort on a K-element array stored in the caller warp's
// private shared-memory segment. The implementation follows the requested
// bitonic-sort structure exactly, with each lane owning a striped subset:
//   global position p  <->  lane = p & 31, slot = p >> 5.
template <int K>
__device__ __forceinline__ void bitonic_sort_shared(packed_t* arr) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two");
    static_assert(K >= kWarpThreads && K <= 1024, "K out of supported range");
    constexpr int ITEMS_PER_LANE = K / kWarpThreads;

    const int lane = threadIdx.x & (kWarpThreads - 1);

    __syncwarp();

    // Keep the stage loops dynamic to avoid excessive code size, but fully
    // unroll the per-lane striped accesses, which are short and compile-time
    // known.
#pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
#pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int item = 0; item < ITEMS_PER_LANE; ++item) {
                const int i = item * kWarpThreads + lane;
                const int l = i ^ stride;

                if (l > i) {
                    const bool ascending = ((i & size) == 0);

                    const packed_t a = arr[i];
                    const packed_t b = arr[l];

                    const bool do_swap = ascending ? (a > b) : (a < b);
                    if (do_swap) {
                        arr[i] = b;
                        arr[l] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge the warp-private top-k (kept in registers, striped across the warp)
// with the shared-memory candidate buffer exactly as requested:
//   1) sort buffer ascending with bitonic sort,
//   2) form the bitonic merge sequence via min(buffer[i], result[k-i-1]),
//   3) bitonic-sort that sequence to produce the updated top-k.
// The candidate counter is reset to zero on exit, and max_distance is updated.
template <int K>
__device__ __forceinline__ void merge_candidate_buffer_into_result(
    packed_t* candidate_buffer,
    packed_t* scratch_buffer,
    int* candidate_count,
    packed_t (&result_pairs)[K / kWarpThreads],
    float& max_distance)
{
    static_assert((K & (K - 1)) == 0, "K must be a power of two");
    constexpr int ITEMS_PER_LANE = K / kWarpThreads;
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

    const int lane = threadIdx.x & (kWarpThreads - 1);
    const packed_t inf_pair = pack_pair(CUDART_INF_F, -1);

    __syncwarp();

    int count = 0;
    if (lane == 0) {
        count = *candidate_count;
    }
    count = __shfl_sync(FULL_MASK, count, 0);

    // Pad the unused tail with +inf so the prescribed sort/merge path can be
    // used uniformly for both full and partial buffers.
#pragma unroll
    for (int item = 0; item < ITEMS_PER_LANE; ++item) {
        const int pos = item * kWarpThreads + lane;
        if (pos >= count) {
            candidate_buffer[pos] = inf_pair;
        }
    }
    __syncwarp();

    // Step 1: sort the candidate buffer in ascending order.
    bitonic_sort_shared<K>(candidate_buffer);

    // Copy the current sorted intermediate result to shared scratch so the
    // reverse-indexed accesses needed by the bitonic merge are cheap.
#pragma unroll
    for (int item = 0; item < ITEMS_PER_LANE; ++item) {
        const int pos = item * kWarpThreads + lane;
        scratch_buffer[pos] = result_pairs[item];
    }
    __syncwarp();

    // Step 2: min(buffer[i], result[k-i-1]) -> bitonic sequence containing the
    // k smallest elements of the union.
#pragma unroll
    for (int item = 0; item < ITEMS_PER_LANE; ++item) {
        const int pos = item * kWarpThreads + lane;
        const packed_t a = candidate_buffer[pos];
        const packed_t b = scratch_buffer[K - 1 - pos];
        candidate_buffer[pos] = (a < b) ? a : b;
    }
    __syncwarp();

    // Step 3: sort the merged bitonic sequence ascending.
    bitonic_sort_shared<K>(candidate_buffer);

    // Copy the updated top-k back to the warp-private register file.
#pragma unroll
    for (int item = 0; item < ITEMS_PER_LANE; ++item) {
        const int pos = item * kWarpThreads + lane;
        result_pairs[item] = candidate_buffer[pos];
    }

    // The k-th nearest neighbor lives in global position K-1, which maps to
    // lane 31 and slot ITEMS_PER_LANE-1 because K is always a multiple of 32.
    float tail = 0.0f;
    if (lane == kWarpThreads - 1) {
        tail = unpack_dist(result_pairs[ITEMS_PER_LANE - 1]);
    }
    max_distance = __shfl_sync(FULL_MASK, tail, kWarpThreads - 1);

    if (lane == 0) {
        *candidate_count = 0;
    }
    __syncwarp();
}

// Kernel overview:
//   * one warp owns one query,
//   * the whole block cooperatively loads data tiles into shared memory,
//   * each active warp computes distances from its own query to the cached tile,
//   * accepted candidates are appended to a per-warp shared buffer,
//   * whenever the buffer becomes full, it is merged into the warp-private top-k.
template <int K, int WARPS_PER_BLOCK, int DATA_TILE_POINTS>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpThreads)
void knn_2d_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    output_pair_t* __restrict__ result)
{
    static_assert((K & (K - 1)) == 0, "K must be a power of two");
    static_assert(K >= kWarpThreads && K <= 1024, "K out of supported range");
    static_assert(K % kWarpThreads == 0, "K must be divisible by warp size");
    static_assert(DATA_TILE_POINTS % kWarpThreads == 0, "Tile size must be a multiple of warp size");
    static_assert(WARPS_PER_BLOCK >= 1, "At least one warp per block is required");
    static_assert(WARPS_PER_BLOCK * kWarpThreads <= 1024, "Block size exceeds CUDA limit");

    constexpr int ITEMS_PER_LANE = K / kWarpThreads;
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    const bool active = (query_idx < query_count);

    // Shared-memory layout:
    //   [tile of float2 data points]
    //   [per-warp candidate buffer, packed distance/index pairs]
    //   [per-warp scratch buffer, same format]
    //   [per-warp candidate counts]
    extern __shared__ unsigned char shared_raw[];
    auto* s_tile = reinterpret_cast<float2*>(shared_raw);
    auto* s_candidate = reinterpret_cast<packed_t*>(s_tile + DATA_TILE_POINTS);
    auto* s_scratch = s_candidate + static_cast<size_t>(WARPS_PER_BLOCK) * K;
    auto* s_counts = reinterpret_cast<int*>(s_scratch + static_cast<size_t>(WARPS_PER_BLOCK) * K);

    packed_t* candidate_buffer = s_candidate + static_cast<size_t>(warp_in_block) * K;
    packed_t* scratch_buffer = s_scratch + static_cast<size_t>(warp_in_block) * K;
    int* candidate_count = s_counts + warp_in_block;

    // Warp-private top-k, striped across lanes:
    //   result position p is stored in lane (p & 31), slot (p >> 5).
    packed_t topk[ITEMS_PER_LANE];
    float max_distance = CUDART_INF_F;
    float qx = 0.0f;
    float qy = 0.0f;

    if (lane == 0) {
        *candidate_count = 0;
    }
    __syncwarp();

    if (active) {
        // Load the query point once per warp and broadcast to all lanes.
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);

        // Initialize the intermediate top-k from the first K data points.
        // This immediately establishes a useful max_distance and prevents the
        // candidate buffer from filling with every early point.
#pragma unroll
        for (int item = 0; item < ITEMS_PER_LANE; ++item) {
            const int data_idx = item * kWarpThreads + lane;
            const float2 p = data[data_idx];
            const float dist = squared_l2_2d(qx, qy, p);
            scratch_buffer[data_idx] = pack_pair(dist, data_idx);
        }
        __syncwarp();

        bitonic_sort_shared<K>(scratch_buffer);

#pragma unroll
        for (int item = 0; item < ITEMS_PER_LANE; ++item) {
            const int pos = item * kWarpThreads + lane;
            topk[item] = scratch_buffer[pos];
        }

        float tail = 0.0f;
        if (lane == kWarpThreads - 1) {
            tail = unpack_dist(topk[ITEMS_PER_LANE - 1]);
        }
        max_distance = __shfl_sync(FULL_MASK, tail, kWarpThreads - 1);
    }

    // Process the remaining data points in shared-memory tiles.
    for (int tile_begin = K; tile_begin < data_count; tile_begin += DATA_TILE_POINTS) {
        const int remaining = data_count - tile_begin;
        const int tile_count = (remaining < DATA_TILE_POINTS) ? remaining : DATA_TILE_POINTS;

        // Cooperative block-wide load of the next tile.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_tile[i] = data[tile_begin + i];
        }
        __syncthreads();

        if (active) {
            const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

            // One chunk = 32 cached points, one candidate point per lane.
#pragma unroll 1
            for (int chunk = 0; chunk < tile_count; chunk += kWarpThreads) {
                const int local_idx = chunk + lane;

                float dist = 0.0f;
                int data_idx = tile_begin + local_idx;
                bool accept = false;

                if (local_idx < tile_count) {
                    const float2 p = s_tile[local_idx];
                    dist = squared_l2_2d(qx, qy, p);
                    accept = (dist < max_distance);
                }

                // Within a 32-point chunk, use one warp-aggregated atomicAdd plus
                // per-lane prefix ranks. This still uses atomicAdd exactly for the
                // shared candidate count / position reservation, but avoids 32
                // independent atomics when many lanes accept simultaneously.
                unsigned pending_mask = __ballot_sync(FULL_MASK, accept);

                while (pending_mask != 0u) {
                    const int rank = __popc(pending_mask & lane_mask_lt);
                    const int nvalid = __popc(pending_mask);

                    int base = 0;
                    if (lane == 0) {
                        base = atomicAdd(candidate_count, nvalid);
                    }
                    base = __shfl_sync(FULL_MASK, base, 0);

                    const int available = (base < K) ? (K - base) : 0;

                    if (available > 0 && accept && rank < available) {
                        candidate_buffer[base + rank] = pack_pair(dist, data_idx);
                    }
                    __syncwarp();

                    // The requested behavior is to merge immediately whenever the
                    // buffer becomes full, including the exact-fill case.
                    const bool buffer_full = (base + nvalid) >= K;

                    if (!buffer_full) {
                        accept = false;
                        pending_mask = 0u;
                    } else {
                        const bool leftover = accept && (rank >= available);

                        if (lane == 0) {
                            *candidate_count = K;
                        }
                        __syncwarp();

                        merge_candidate_buffer_into_result<K>(
                            candidate_buffer, scratch_buffer, candidate_count, topk, max_distance);

                        // Any candidates that did not fit are re-evaluated against
                        // the tightened threshold; those that are no longer useful
                        // are discarded immediately.
                        accept = leftover && (dist < max_distance);
                        pending_mask = __ballot_sync(FULL_MASK, accept);
                    }
                }
            }
        }

        // All warps must finish consuming the cached tile before it is reused.
        __syncthreads();
    }

    if (active) {
        int pending = 0;
        if (lane == 0) {
            pending = *candidate_count;
        }
        pending = __shfl_sync(FULL_MASK, pending, 0);

        if (pending > 0) {
            merge_candidate_buffer_into_result<K>(
                candidate_buffer, scratch_buffer, candidate_count, topk, max_distance);
        }

        // Write the final sorted top-k list back to global memory in row-major
        // query-major order: result[query_idx * K + j].
        output_pair_t* out_row = result + static_cast<size_t>(query_idx) * static_cast<size_t>(K);

#pragma unroll
        for (int item = 0; item < ITEMS_PER_LANE; ++item) {
            const int pos = item * kWarpThreads + lane;
            const packed_t v = topk[item];

            // Direct member stores avoid relying on any device-side std::pair
            // constructors; the destination memory already exists in cudaMalloc-
            // allocated storage.
            out_row[pos].first = unpack_idx(v);
            out_row[pos].second = unpack_dist(v);
        }
    }
}

template <int K, int WARPS_PER_BLOCK, int DATA_TILE_POINTS>
constexpr size_t shared_bytes_for_variant() {
    return static_cast<size_t>(DATA_TILE_POINTS) * sizeof(float2) +
           static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(K) * sizeof(packed_t) * 2 +
           static_cast<size_t>(WARPS_PER_BLOCK) * sizeof(int);
}

template <int WARPS_PER_BLOCK>
inline int blocks_for_queries(const int query_count) {
    return (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
}

template <int K, int WARPS_PER_BLOCK, int DATA_TILE_POINTS>
inline bool fits_shared_budget(const int max_optin_shared_bytes) {
    return shared_bytes_for_variant<K, WARPS_PER_BLOCK, DATA_TILE_POINTS>() <=
           static_cast<size_t>(max_optin_shared_bytes);
}

template <int K, int WARPS_PER_BLOCK, int DATA_TILE_POINTS>
inline void launch_variant(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    output_pair_t* result)
{
    constexpr size_t SHARED_BYTES = shared_bytes_for_variant<K, WARPS_PER_BLOCK, DATA_TILE_POINTS>();

    // The interface requested a void return type, so this launcher intentionally
    // keeps the call path minimal and leaves CUDA error handling to the caller's
    // surrounding workflow.
    cudaFuncSetAttribute(
        knn_2d_kernel<K, WARPS_PER_BLOCK, DATA_TILE_POINTS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SHARED_BYTES));

    cudaFuncSetAttribute(
        knn_2d_kernel<K, WARPS_PER_BLOCK, DATA_TILE_POINTS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const dim3 block(WARPS_PER_BLOCK * kWarpThreads);
    const dim3 grid(blocks_for_queries<WARPS_PER_BLOCK>(query_count));

    knn_2d_kernel<K, WARPS_PER_BLOCK, DATA_TILE_POINTS>
        <<<grid, block, SHARED_BYTES>>>(query, query_count, data, data_count, result);
}

template <int K, int WARPS_PER_BLOCK, int DATA_TILE_POINTS>
inline bool try_launch_variant(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    output_pair_t* result,
    int min_grid_blocks,
    int max_optin_shared_bytes)
{
    if (fits_shared_budget<K, WARPS_PER_BLOCK, DATA_TILE_POINTS>(max_optin_shared_bytes) &&
        blocks_for_queries<WARPS_PER_BLOCK>(query_count) >= min_grid_blocks)
    {
        launch_variant<K, WARPS_PER_BLOCK, DATA_TILE_POINTS>(
            query, query_count, data, data_count, result);
        return true;
    }
    return false;
}

template <int K>
inline void dispatch_k(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    output_pair_t* result,
    int sm_count,
    int max_optin_shared_bytes)
{
    constexpr int TILE = kDataTilePoints;

    const int min_grid_blocks =
        (sm_count * kTargetGridCoverageNum + (kTargetGridCoverageDen - 1)) / kTargetGridCoverageDen;

    if constexpr (K <= 256) {
        // Small/medium K can profit from fairly wide blocks because the
        // per-query register/shared footprint is modest.
        if (try_launch_variant<K, 32, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K, 24, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K, 16, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K, 12, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K,  8, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;

        // Fallback for smaller query batches and/or smaller shared-memory budgets.
        launch_variant<K, 4, TILE>(query, query_count, data, data_count, result);
    } else if constexpr (K == 512) {
        // K=512 still benefits from large tile reuse, but the larger top-k state
        // makes the very wide variants expensive in shared memory. W=24 is H100-
        // friendly; W=18 stays within the A100 envelope.
        if (try_launch_variant<K, 24, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K, 18, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K, 12, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K,  8, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;

        launch_variant<K, 4, TILE>(query, query_count, data, data_count, result);
    } else {
        // K=1024 is the heaviest case. W=12 fits only on Hopper-class large
        // shared-memory budgets; W=9 fits on A100; the narrower variants keep
        // enough grid-level parallelism when query_count is smaller.
        if (try_launch_variant<K, 12, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K,  9, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K,  6, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;
        if (try_launch_variant<K,  4, TILE>(query, query_count, data, data_count, result, min_grid_blocks, max_optin_shared_bytes)) return;

        launch_variant<K, 2, TILE>(query, query_count, data, data_count, result);
    }
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

    int device = 0;
    int sm_count = 0;
    int max_optin_shared_bytes = 0;

    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&max_optin_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    if (sm_count <= 0) {
        sm_count = 1;
    }
    if (max_optin_shared_bytes <= 0) {
        // Conservative fallback that still allows the narrowest variants.
        max_optin_shared_bytes = 49152;
    }

    switch (k) {
        case 32:
            dispatch_k<32>(query, query_count, data, data_count, result, sm_count, max_optin_shared_bytes);
            break;
        case 64:
            dispatch_k<64>(query, query_count, data, data_count, result, sm_count, max_optin_shared_bytes);
            break;
        case 128:
            dispatch_k<128>(query, query_count, data, data_count, result, sm_count, max_optin_shared_bytes);
            break;
        case 256:
            dispatch_k<256>(query, query_count, data, data_count, result, sm_count, max_optin_shared_bytes);
            break;
        case 512:
            dispatch_k<512>(query, query_count, data, data_count, result, sm_count, max_optin_shared_bytes);
            break;
        case 1024:
            dispatch_k<1024>(query, query_count, data, data_count, result, sm_count, max_optin_shared_bytes);
            break;
        default:
            // The problem statement guarantees valid k values, so this path is
            // unreachable for conforming callers.
            break;
    }
}