#include <cuda_runtime.h>
#include <algorithm>
#include <utility>

// Warp-centric exact k-NN for 2D points in squared Euclidean distance.
// - One warp handles one query.
// - The whole block stages the current data batch in shared memory.
// - Each warp owns two private shared-memory slices:
//     1) a sorted intermediate top-K result,
//     2) a candidate buffer with a shared count.
// - Candidate insertions use warp-aggregated atomicAdd on the per-warp shared counter.
// - Buffer flush / merge follows the requested procedure exactly:
//     sort buffer, half-cleaner against the reversed intermediate result, sort again.
// - No extra device memory is allocated.
//
// The tile size below is chosen for modern data-center GPUs (A100/H100 class).
// 2016 = 63 * 32 is the largest warp-aligned tile that still preserves favorable
// shared-memory fit on A100 for the hard cases K=512 and K=1024.
namespace {

constexpr int kWarpSize            = 32;
constexpr int kMaxWarpsPerBlock    = 32;
constexpr int kTilePoints          = 2016;
constexpr size_t kTileBytes        = static_cast<size_t>(kTilePoints) * sizeof(float2);
using Entry = unsigned long long;

static_assert(kTilePoints % kWarpSize == 0, "Tile size must be warp-aligned.");
static_assert(kTilePoints >= 1024, "The first tile must be large enough to bootstrap K=1024.");
static_assert(sizeof(std::pair<int, float>) == 8, "Unexpected std::pair<int,float> layout.");

// Distances are non-negative, so IEEE-754 bit patterns are monotonically ordered when
// reinterpreted as unsigned integers. Packing {distance_bits, index_bits} into one 64-bit
// word lets the compare/swap network move one payload instead of separate index+distance arrays.
__device__ __forceinline__ Entry pack_entry(const int index, const float distance) {
    return (static_cast<Entry>(__float_as_uint(distance)) << 32) |
           static_cast<unsigned int>(index);
}

__device__ __forceinline__ int unpack_index(const Entry e) {
    return static_cast<int>(static_cast<unsigned int>(e));
}

__device__ __forceinline__ float unpack_distance(const Entry e) {
    return __uint_as_float(static_cast<unsigned int>(e >> 32));
}

__device__ __forceinline__ Entry sentinel_entry() {
    return pack_entry(-1, CUDART_INF_F);
}

__device__ __forceinline__ float squared_l2(const float qx, const float qy, const float2& p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

template <int K>
__device__ __forceinline__ void bitonic_sort_shared(Entry* values) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32,1024].");
    static_assert((K % kWarpSize) == 0, "K must be divisible by warp size.");

    constexpr int kItemsPerLane = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);

    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int t = 0; t < kItemsPerLane; ++t) {
                const int i = (t << 5) + lane;
                const int l = i ^ stride;
                if (l > i) {
                    const bool up = ((i & size) == 0);
                    const Entry a = values[i];
                    const Entry b = values[l];
                    const bool swap = up ? (a > b) : (a < b);
                    if (swap) {
                        values[i] = b;
                        values[l] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Flush the candidate buffer into the sorted intermediate top-K.
// If current_count < K, the remainder is padded with +inf sentinels so the same full-K merge path
// can be used for both regular flushes and the final partial flush.
template <int K>
__device__ __forceinline__ float merge_candidate_buffer_into_topk(
    Entry* topk_entries,
    Entry* candidate_entries,
    int* candidate_count,
    const int current_count)
{
    constexpr int kItemsPerLane = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);

    if (current_count < K) {
        const Entry inf = sentinel_entry();
        for (int i = current_count + lane; i < K; i += kWarpSize) {
            candidate_entries[i] = inf;
        }
    }

    if (lane == 0) {
        *candidate_count = 0;
    }
    __syncwarp();

    // Step 1: sort the buffer ascending.
    bitonic_sort_shared<K>(candidate_entries);

    // Step 2: half-cleaner against the reversed sorted intermediate result.
    // The minima form a bitonic sequence that contains the true top-K of the union.
#pragma unroll
    for (int t = 0; t < kItemsPerLane; ++t) {
        const int i = (t << 5) + lane;
        const Entry a = candidate_entries[i];
        const Entry b = topk_entries[K - 1 - i];
        candidate_entries[i] = (a < b) ? a : b;
    }
    __syncwarp();

    // Step 3: sort the bitonic sequence ascending to restore the invariant.
    bitonic_sort_shared<K>(candidate_entries);

#pragma unroll
    for (int t = 0; t < kItemsPerLane; ++t) {
        const int i = (t << 5) + lane;
        topk_entries[i] = candidate_entries[i];
    }
    __syncwarp();

    return unpack_distance(topk_entries[K - 1]);
}

// Process one shared-memory tile of data points.
// The warp examines one 32-point packet at a time so it can:
// - ballot the points that beat max_distance,
// - reserve a contiguous segment in the candidate buffer with one atomicAdd,
// - keep the candidate buffer exact without ever dropping a valid point.
//
// If a packet would overflow the fixed-size candidate buffer, the current partial buffer is flushed
// first, max_distance is tightened, and the same packet is re-tested against the new threshold.
template <int K>
__device__ __forceinline__ void process_tile_packets(
    const float qx,
    const float qy,
    const float2* tile_points,
    const int start,
    const int tile_count,
    const int tile_base,
    Entry* topk_entries,
    Entry* candidate_entries,
    int* candidate_count,
    float& max_distance,
    int& buffer_count)
{
    constexpr unsigned kFullMask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (kWarpSize - 1);

    for (int packet_base = start; packet_base < tile_count; packet_base += kWarpSize) {
        const int local_idx = packet_base + lane;
        const bool valid = (local_idx < tile_count);

        float distance = 0.0f;
        Entry candidate = 0ull;

        if (valid) {
            const float2 p = tile_points[local_idx];
            distance = squared_l2(qx, qy, p);
            candidate = pack_entry(tile_base + local_idx, distance);
        }

        bool keep = valid && (distance < max_distance);
        unsigned keep_mask = __ballot_sync(kFullMask, keep);
        int keep_count = __popc(keep_mask);

        if (keep_count == 0) {
            continue;
        }

        // Flush early if this warp packet would overflow the fixed-size buffer.
        if (buffer_count + keep_count > K) {
            max_distance = merge_candidate_buffer_into_topk<K>(
                topk_entries, candidate_entries, candidate_count, buffer_count);
            buffer_count = 0;

            keep = valid && (distance < max_distance);
            keep_mask = __ballot_sync(kFullMask, keep);
            keep_count = __popc(keep_mask);

            if (keep_count == 0) {
                continue;
            }
        }

        const unsigned lower_lanes = (lane == 0) ? 0u : ((1u << lane) - 1u);
        const int prefix = __popc(keep_mask & lower_lanes);

        int base_pos = 0;
        if (lane == 0) {
            base_pos = atomicAdd(candidate_count, keep_count);
        }
        base_pos = __shfl_sync(kFullMask, base_pos, 0);

        if (keep) {
            candidate_entries[base_pos + prefix] = candidate;
        }
        __syncwarp();

        buffer_count = base_pos + keep_count;

        // The prompt asks for an immediate merge whenever the buffer becomes full.
        if (buffer_count == K) {
            max_distance = merge_candidate_buffer_into_topk<K>(
                topk_entries, candidate_entries, candidate_count, K);
            buffer_count = 0;
        }
    }
}

template <int K>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    const int query_count,
    const float2* __restrict__ data,
    const int data_count,
    std::pair<int, float>* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32,1024].");

    extern __shared__ __align__(16) unsigned char smem[];

    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int query_id = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;
    const bool active_query = (query_id < query_count);

    float2* const tile_points = reinterpret_cast<float2*>(smem);
    Entry* const topk_storage = reinterpret_cast<Entry*>(tile_points + kTilePoints);
    Entry* const candidate_storage = topk_storage + static_cast<size_t>(warps_per_block) * K;
    int* const candidate_counts = reinterpret_cast<int*>(candidate_storage + static_cast<size_t>(warps_per_block) * K);

    Entry* const topk_entries = topk_storage + static_cast<size_t>(warp_id) * K;
    Entry* const candidate_entries = candidate_storage + static_cast<size_t>(warp_id) * K;
    int* const candidate_count = candidate_counts + warp_id;

    // Load the query point once per warp, then broadcast it.
    float2 q = make_float2(0.0f, 0.0f);
    if (lane == 0 && active_query) {
        q = query[query_id];
    }
    const float qx = __shfl_sync(0xFFFFFFFFu, q.x, 0);
    const float qy = __shfl_sync(0xFFFFFFFFu, q.y, 0);

    float max_distance = CUDART_INF_F;
    int buffer_count = 0;

    // Iterate over the data set in shared-memory tiles.
    for (int tile_base = 0; tile_base < data_count; tile_base += kTilePoints) {
        int tile_count = data_count - tile_base;
        if (tile_count > kTilePoints) {
            tile_count = kTilePoints;
        }

        // Cooperative tile load.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            tile_points[i] = data[tile_base + i];
        }
        __syncthreads();

        if (active_query) {
            if (tile_base == 0) {
                // Bootstrap the sorted intermediate top-K from the first K data points of the first tile.
                // This avoids starting with max_distance = +inf for the whole data set.
                constexpr int kItemsPerLane = K / kWarpSize;
#pragma unroll
                for (int t = 0; t < kItemsPerLane; ++t) {
                    const int i = (t << 5) + lane;
                    const float2 p = tile_points[i];
                    topk_entries[i] = pack_entry(i, squared_l2(qx, qy, p));
                }

                if (lane == 0) {
                    *candidate_count = 0;
                }
                __syncwarp();

                bitonic_sort_shared<K>(topk_entries);
                max_distance = unpack_distance(topk_entries[K - 1]);
                buffer_count = 0;

                // The remainder of the first tile is handled through the candidate buffer path.
                process_tile_packets<K>(
                    qx, qy, tile_points, K, tile_count, tile_base,
                    topk_entries, candidate_entries, candidate_count,
                    max_distance, buffer_count);
            } else {
                process_tile_packets<K>(
                    qx, qy, tile_points, 0, tile_count, tile_base,
                    topk_entries, candidate_entries, candidate_count,
                    max_distance, buffer_count);
            }
        }

        __syncthreads();
    }

    if (active_query) {
        // Final partial flush, if any candidates remain buffered.
        if (buffer_count > 0) {
            (void)merge_candidate_buffer_into_topk<K>(
                topk_entries, candidate_entries, candidate_count, buffer_count);
        }

        const size_t out_base = static_cast<size_t>(query_id) * K;
        constexpr int kItemsPerLane = K / kWarpSize;
#pragma unroll
        for (int t = 0; t < kItemsPerLane; ++t) {
            const int i = (t << 5) + lane;
            const Entry e = topk_entries[i];
            result[out_base + i].first = unpack_index(e);
            result[out_base + i].second = unpack_distance(e);
        }
    }
}

template <int K>
__host__ __forceinline__ size_t shared_bytes_for_launch(const int warps_per_block) {
    return kTileBytes +
           static_cast<size_t>(warps_per_block) *
               (static_cast<size_t>(2) * K * sizeof(Entry) + sizeof(int));
}

template <int K>
void launch_knn_specialized(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    std::pair<int, float>* result)
{
    int device = 0;
    (void)cudaGetDevice(&device);

    int sm_count = 1;
    int max_dynamic_smem = 0;
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    (void)cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    if (max_dynamic_smem <= 0) {
        (void)cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    }

    const long long per_warp_bytes =
        static_cast<long long>(2LL * K * static_cast<long long>(sizeof(Entry)) + sizeof(int));
    const long long available_shared =
        static_cast<long long>(max_dynamic_smem) - static_cast<long long>(kTileBytes);

    int max_warps_shared = static_cast<int>(available_shared / per_warp_bytes);
    max_warps_shared = std::max(1, std::min(kMaxWarpsPerBlock, max_warps_shared));

    // Heuristic: choose the largest warp count that still leaves roughly one query-group block per SM.
    // This balances:
    //   - better shared-tile reuse with more warps/block,
    //   - enough blocks to keep the whole GPU busy when query_count is only a few thousand.
    const int warps_for_parallelism =
        std::max(1, (query_count + sm_count / 2) / sm_count);

    int warps_per_block = std::min(max_warps_shared, warps_for_parallelism);
    warps_per_block = std::min(warps_per_block, query_count);
    warps_per_block = std::max(1, warps_per_block);

    const int threads_per_block = warps_per_block * kWarpSize;
    const size_t smem_bytes = shared_bytes_for_launch<K>(warps_per_block);
    const int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    knn_kernel<K><<<blocks, threads_per_block, smem_bytes>>>(
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

    switch (k) {
        case 32:   launch_knn_specialized<32>(query, query_count, data, data_count, result); break;
        case 64:   launch_knn_specialized<64>(query, query_count, data, data_count, result); break;
        case 128:  launch_knn_specialized<128>(query, query_count, data, data_count, result); break;
        case 256:  launch_knn_specialized<256>(query, query_count, data, data_count, result); break;
        case 512:  launch_knn_specialized<512>(query, query_count, data, data_count, result); break;
        case 1024: launch_knn_specialized<1024>(query, query_count, data, data_count, result); break;
        default:   break; // The problem statement guarantees valid K.
    }
}