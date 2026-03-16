#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace knn_detail {

// Warp-per-query, shared-memory tiled k-NN for 2D points.
//
// Key design choices:
//  - One warp owns one query for the full scan.
//  - The database is streamed in 2048-point tiles staged in shared memory.
//    The tile is stored as SoA (x[] and y[]) to avoid the shared-memory bank conflict pattern
//    that contiguous float2 loads would create.
//  - Each query keeps a private top-K max-heap in shared memory. Only lane 0 mutates that heap.
//    This is intentional: after the first K points, only a small fraction of candidates beat the
//    current K-th best threshold, so the expensive heap maintenance is sparse while all 32 lanes
//    still stay busy on distance evaluation and candidate voting.
//  - K only takes six legal values, so the kernel is specialized at compile time for K.
//  - The host picks 4/8/16/32 warps per block based on query_count and SM count, selecting the
//    largest value that still leaves roughly >= 1 block/SM. This balances tile reuse against
//    grid-level parallelism on A100/H100-class GPUs.

constexpr int kWarpSize   = 32;
constexpr int kTilePoints = 2048;  // 16 KiB shared tile (x[] + y[]).
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kA100MaxOptinSharedPerBlock = 163840;  // Strictest target in the requested class.

static_assert((kTilePoints % kWarpSize) == 0, "Tile size must be warp aligned.");
static_assert(kTilePoints >= 1024, "The first tile must always contain all legal K values.");

using ResultPair = std::pair<int, float>;

struct alignas(8) Neighbor {
    float dist;
    int   idx;
};
static_assert(sizeof(Neighbor) == 8, "Neighbor must stay compact.");
static_assert((2 * kTilePoints * static_cast<int>(sizeof(float))) % alignof(Neighbor) == 0,
              "Heap storage must stay aligned behind the shared tile.");

__device__ __forceinline__ float squared_l2_2d(float qx, float qy, float x, float y) {
    const float dx = qx - x;
    const float dy = qy - y;
    return fmaf(dx, dx, dy * dy);
}

// Sift a single node down inside a max-heap. Implemented with the standard "hole" method to
// minimize shared-memory traffic.
__device__ __forceinline__ void sift_down_max_heap(Neighbor* heap, int size, int root) {
    Neighbor value = heap[root];
    int child = (root << 1) + 1;

    while (child < size) {
        int right = child + 1;
        if (right < size && heap[right].dist > heap[child].dist) {
            child = right;
        }
        if (heap[child].dist <= value.dist) {
            break;
        }
        heap[root] = heap[child];
        root = child;
        child = (root << 1) + 1;
    }
    heap[root] = value;
}

template <int K>
__device__ __forceinline__ void build_max_heap(Neighbor* heap) {
    for (int i = (K >> 1) - 1; i >= 0; --i) {
        sift_down_max_heap(heap, K, i);
    }
}

// Heap sort over the already-built max-heap. Final order is ascending distance, which matches
// the output contract result[i * k + j] = j-th nearest neighbor.
template <int K>
__device__ __forceinline__ void heap_sort_ascending(Neighbor* heap) {
    for (int end = K - 1; end > 0; --end) {
        const Neighbor max_value = heap[0];
        const Neighbor tail      = heap[end];
        heap[end] = max_value;

        int root  = 0;
        int child = 1;
        while (child < end) {
            int right = child + 1;
            if (right < end && heap[right].dist > heap[child].dist) {
                child = right;
            }
            if (heap[child].dist <= tail.dist) {
                break;
            }
            heap[root] = heap[child];
            root = child;
            child = (root << 1) + 1;
        }
        heap[root] = tail;
    }
}

// Full-tile path: the iteration count is compile-time known, so the compiler can unroll it.
template <int BLOCK_THREADS>
__device__ __forceinline__ void load_full_tile_to_shared(
    const float2* __restrict__ data_src,
    float* __restrict__ tile_x,
    float* __restrict__ tile_y)
{
#pragma unroll
    for (int i = static_cast<int>(threadIdx.x); i < kTilePoints; i += BLOCK_THREADS) {
        const float2 p = data_src[i];
        tile_x[i] = p.x;
        tile_y[i] = p.y;
    }
}

// Initialize the per-query heap from the first K points of the first tile.
// Warp shuffles with the *_sync variants are the warp-scope synchronization primitive here.
template <int K>
__device__ __forceinline__ void init_heap_from_first_k(
    const float* __restrict__ tile_x,
    const float* __restrict__ tile_y,
    float qx, float qy,
    Neighbor* __restrict__ heap,
    int lane)
{
    int write_base = 0;

    for (int off = 0; off < K; off += kWarpSize) {
        const float x    = tile_x[off + lane];
        const float y    = tile_y[off + lane];
        const float dist = squared_l2_2d(qx, qy, x, y);
        const int   idx  = off + lane;  // First tile starts at global index 0.

#pragma unroll
        for (int src = 0; src < kWarpSize; ++src) {
            const float dist_src = __shfl_sync(kFullMask, dist, src);
            const int   idx_src  = __shfl_sync(kFullMask, idx,  src);
            if (lane == 0) {
                heap[write_base + src].dist = dist_src;
                heap[write_base + src].idx  = idx_src;
            }
        }
        write_base += kWarpSize;
    }

    if (lane == 0) {
        build_max_heap<K>(heap);
    }
}

// Scan one [start, end) range inside the currently cached tile.
template <int K>
__device__ __forceinline__ void process_tile_range(
    const float* __restrict__ tile_x,
    const float* __restrict__ tile_y,
    int tile_base,
    int start,
    int end,
    float qx, float qy,
    Neighbor* __restrict__ heap,
    int lane)
{
    // Only lane 0 owns the heap root in a live register; other lanes just receive it by shuffle.
    float worst_lane = CUDART_INF_F;
    if (lane == 0) {
        worst_lane = heap[0].dist;
    }

    for (int off = start; off < end; off += kWarpSize) {
        float dist = CUDART_INF_F;
        int   idx  = -1;

        const int local = off + lane;
        if (local < end) {
            const float x = tile_x[local];
            const float y = tile_y[local];
            dist = squared_l2_2d(qx, qy, x, y);
            idx  = tile_base + local;
        }

        const float worst = __shfl_sync(kFullMask, worst_lane, 0);
        unsigned candidate_mask = __ballot_sync(kFullMask, dist < worst);

        // Every lane executes the same sequence of shuffles; only lane 0 performs the actual
        // heap update. Candidates are rechecked against the updated root before replacement.
        while (candidate_mask) {
            const int src_lane = __ffs(candidate_mask) - 1;
            const float cand_dist = __shfl_sync(kFullMask, dist, src_lane);
            const int   cand_idx  = __shfl_sync(kFullMask, idx,  src_lane);

            if (lane == 0 && cand_dist < worst_lane) {
                heap[0].dist = cand_dist;
                heap[0].idx  = cand_idx;
                sift_down_max_heap(heap, K, 0);
                worst_lane = heap[0].dist;
            }
            candidate_mask &= candidate_mask - 1;
        }
    }
}

template <int K, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    ResultPair* __restrict__ result)
{
    static_assert(K == 32 || K == 64 || K == 128 || K == 256 || K == 512 || K == 1024,
                  "Illegal K specialization.");
    static_assert(WARPS_PER_BLOCK == 4 || WARPS_PER_BLOCK == 8 ||
                  WARPS_PER_BLOCK == 16 || WARPS_PER_BLOCK == 32,
                  "Unsupported warp count.");
    static_assert(K <= kTilePoints, "The first tile must contain the initial heap population.");

    constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * kWarpSize;

    const int lane             = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp_id_in_block = static_cast<int>(threadIdx.x) >> 5;
    const int block_query_base = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK;

    // Entire block can exit before any barrier if it covers no queries.
    if (block_query_base >= query_count) {
        return;
    }

    const int  query_idx    = block_query_base + warp_id_in_block;
    const bool active_query = (query_idx < query_count);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    float*   tile_x = reinterpret_cast<float*>(smem_raw);
    float*   tile_y = tile_x + kTilePoints;
    Neighbor* heaps = reinterpret_cast<Neighbor*>(tile_y + kTilePoints);

    // Each warp gets a private K-entry heap in shared memory.
    Neighbor* heap = heaps + warp_id_in_block * K;

    // Load the query once per warp and broadcast it.
    float qx_lane = 0.0f;
    float qy_lane = 0.0f;
    if (lane == 0 && active_query) {
        const float2 q = query[query_idx];
        qx_lane = q.x;
        qy_lane = q.y;
    }
    const float qx = __shfl_sync(kFullMask, qx_lane, 0);
    const float qy = __shfl_sync(kFullMask, qy_lane, 0);

    // ----- First tile: it always contains at least K points because K <= 1024 < 2048.
    int tile_count = (data_count < kTilePoints) ? data_count : kTilePoints;
    if (tile_count == kTilePoints) {
        load_full_tile_to_shared<BLOCK_THREADS>(data, tile_x, tile_y);
    } else {
        for (int i = static_cast<int>(threadIdx.x); i < tile_count; i += BLOCK_THREADS) {
            const float2 p = data[i];
            tile_x[i] = p.x;
            tile_y[i] = p.y;
        }
    }

    // Shared tile is read by all warps, so a block-wide barrier is required.
    __syncthreads();

    if (active_query) {
        init_heap_from_first_k<K>(tile_x, tile_y, qx, qy, heap, lane);
        if (tile_count > K) {
            process_tile_range<K>(tile_x, tile_y, 0, K, tile_count, qx, qy, heap, lane);
        }
    }

    // Another block-wide barrier is required before reusing the shared tile storage.
    __syncthreads();

    // ----- Remaining tiles.
    for (int tile_base = kTilePoints; tile_base < data_count; tile_base += kTilePoints) {
        tile_count = data_count - tile_base;
        if (tile_count > kTilePoints) {
            tile_count = kTilePoints;
        }

        const float2* tile_src = data + tile_base;
        if (tile_count == kTilePoints) {
            load_full_tile_to_shared<BLOCK_THREADS>(tile_src, tile_x, tile_y);
        } else {
            for (int i = static_cast<int>(threadIdx.x); i < tile_count; i += BLOCK_THREADS) {
                const float2 p = tile_src[i];
                tile_x[i] = p.x;
                tile_y[i] = p.y;
            }
        }

        __syncthreads();

        if (active_query) {
            process_tile_range<K>(tile_x, tile_y, tile_base, 0, tile_count, qx, qy, heap, lane);
        }

        __syncthreads();
    }

    // Final per-query writeback.
    if (active_query && lane == 0) {
        heap_sort_ascending<K>(heap);

        ResultPair* out = result + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);
        for (int i = 0; i < K; ++i) {
            out[i].first  = heap[i].idx;
            out[i].second = heap[i].dist;
        }
    }
}

template <int K, int WARPS_PER_BLOCK>
inline void launch_specialized(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    ResultPair* result)
{
    constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * kWarpSize;
    constexpr std::size_t smem_bytes =
        (2ull * kTilePoints * sizeof(float)) +
        (static_cast<std::size_t>(WARPS_PER_BLOCK) * static_cast<std::size_t>(K) * sizeof(Neighbor));

    static_assert(smem_bytes <= static_cast<std::size_t>(kA100MaxOptinSharedPerBlock),
                  "Chosen launch configuration exceeds A100 shared-memory limits.");

    // The target GPUs support opt-in shared memory above 48 KiB.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int grid_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    knn_kernel<K, WARPS_PER_BLOCK><<<grid_blocks, BLOCK_THREADS, smem_bytes>>>(
        query, query_count, data, data_count, result);
}

// Dispatch helper for K. K=1024 cannot use 32 warps/block because its per-warp heap is 8 KiB.
template <int K>
inline void launch_k(
    int warps_per_block,
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    ResultPair* result)
{
    if constexpr (K == 1024) {
        switch (warps_per_block) {
            case 16: launch_specialized<1024, 16>(query, query_count, data, data_count, result); break;
            case 8:  launch_specialized<1024,  8>(query, query_count, data, data_count, result); break;
            default: launch_specialized<1024,  4>(query, query_count, data, data_count, result); break;
        }
    } else {
        switch (warps_per_block) {
            case 32: launch_specialized<K, 32>(query, query_count, data, data_count, result); break;
            case 16: launch_specialized<K, 16>(query, query_count, data, data_count, result); break;
            case 8:  launch_specialized<K,  8>(query, query_count, data, data_count, result); break;
            default: launch_specialized<K,  4>(query, query_count, data, data_count, result); break;
        }
    }
}

// Choose the largest power-of-two number of query-warps per block that still leaves at least
// about one block per SM. This preserves enough grid parallelism for "thousands of queries"
// while maximizing tile reuse inside each block.
inline int select_warps_per_block(int max_warps_per_block, int query_count, int sm_count) {
    const auto enough_blocks = [query_count, sm_count](int warps_per_block) {
        return ((query_count + warps_per_block - 1) / warps_per_block) >= sm_count;
    };

    if (max_warps_per_block >= 32 && enough_blocks(32)) return 32;
    if (max_warps_per_block >= 16 && enough_blocks(16)) return 16;
    if (max_warps_per_block >= 8  && enough_blocks(8))  return 8;
    return 4;
}

}  // namespace knn_detail

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

    // Query the active device so the block-size heuristic scales from A100 to H100.
    int sm_count = 108;  // Conservative default if the query fails.
    int device   = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
        int value = 0;
        if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
            value > 0) {
            sm_count = value;
        }
    }

    const int max_warps_per_block = (k == 1024) ? 16 : 32;
    const int warps_per_block =
        knn_detail::select_warps_per_block(max_warps_per_block, query_count, sm_count);

    switch (k) {
        case 32:   knn_detail::launch_k<32>  (warps_per_block, query, query_count, data, data_count, result); break;
        case 64:   knn_detail::launch_k<64>  (warps_per_block, query, query_count, data, data_count, result); break;
        case 128:  knn_detail::launch_k<128> (warps_per_block, query, query_count, data, data_count, result); break;
        case 256:  knn_detail::launch_k<256> (warps_per_block, query, query_count, data, data_count, result); break;
        case 512:  knn_detail::launch_k<512> (warps_per_block, query, query_count, data, data_count, result); break;
        case 1024: knn_detail::launch_k<1024>(warps_per_block, query, query_count, data, data_count, result); break;
        default:
            // Per problem statement, k is always valid, so this path is unreachable.
            break;
    }
}