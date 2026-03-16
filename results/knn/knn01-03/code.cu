#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace {

// Output type alias to keep signatures readable.
using ResultPair = std::pair<int, float>;

// Compact POD used for the per-query top-K heap in shared memory.
// Distance is placed first because heap operations compare on distance every time.
struct alignas(8) Neighbor {
    float dist;
    int idx;
};

static_assert(sizeof(Neighbor) == 8, "Neighbor must remain a compact 8-byte POD.");
static_assert(alignof(Neighbor) == 8, "Neighbor must remain 8-byte aligned.");

// Kernel configuration.
// Design notes:
//
// 1) One warp owns one query and maintains an exact max-heap of its current K best neighbors.
// 2) A block handles multiple queries so that each data tile loaded from HBM is reused across
//    many queries before being discarded.
// 3) The data tile is stored in shared memory in structure-of-arrays form (x[] and y[])
//    rather than as float2[] to avoid 64-bit shared-memory access patterns that can introduce
//    avoidable bank conflicts.
// 4) Launch bounds are derived conservatively from A100 limits so the same binary also fits H100.
template<int K, int WARPS_PER_BLOCK>
struct KernelConfig {
    static constexpr int kWarpSize = 32;
    static constexpr int kThreads = WARPS_PER_BLOCK * kWarpSize;
    static constexpr int kTilePoints = 1024; // 8 KiB tile (x[] + y[]), a good balance of reuse and occupancy.

    static constexpr int kSharedBytes =
        WARPS_PER_BLOCK * K * static_cast<int>(sizeof(Neighbor)) +
        2 * kTilePoints * static_cast<int>(sizeof(float)); // x[] + y[]

    static constexpr int kA100ThreadsPerSM = 2048;
    static constexpr int kA100SharedPerSM = 163840; // opt-in per-SM shared budget on A100

    static constexpr int kResidentByThreads = kA100ThreadsPerSM / kThreads;
    static constexpr int kResidentByShared  = kA100SharedPerSM / kSharedBytes;
    static constexpr int kResidentBlocks =
        (kResidentByThreads < kResidentByShared) ? kResidentByThreads : kResidentByShared;

    // Cap the launch-bounds hint so we do not over-constrain register allocation on small-K cases.
    static constexpr int kLaunchBoundsMinBlocks =
        (kResidentBlocks > 4) ? 4 : kResidentBlocks;

    static_assert((K & (K - 1)) == 0 && K >= 32 && K <= 1024, "Unsupported K.");
    static_assert(kThreads <= 1024, "Block size exceeds the hardware limit.");
    static_assert(kSharedBytes <= kA100SharedPerSM, "Shared-memory footprint exceeds A100 limits.");
    static_assert(kResidentBlocks >= 1, "Kernel configuration does not fit on A100.");
    static_assert((kTilePoints % kWarpSize) == 0, "Tile size must be a whole number of warps.");
    static_assert((kSharedBytes % 256) == 0, "Shared-memory size should align with allocator granularity.");
};

__device__ __forceinline__ float sq_l2_2d(float qx, float qy, float px, float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return fmaf(dx, dx, dy * dy);
}

__device__ __forceinline__ void sift_down(Neighbor* heap, int root, int count) {
    Neighbor val = heap[root];

    while (true) {
        int child = (root << 1) + 1;
        if (child >= count) break;

        Neighbor child_val = heap[child];
        const int right = child + 1;
        if (right < count) {
            const Neighbor right_val = heap[right];
            if (right_val.dist > child_val.dist) {
                child = right;
                child_val = right_val;
            }
        }

        if (child_val.dist <= val.dist) break;

        heap[root] = child_val;
        root = child;
    }

    heap[root] = val;
}

template<int K>
__device__ __forceinline__ void build_max_heap(Neighbor* heap) {
    for (int i = (K >> 1); i > 0;) {
        --i;
        sift_down(heap, i, K);
    }
}

template<int K>
__device__ __forceinline__ void replace_root_if_better(Neighbor* heap, float cand_dist, int cand_idx) {
    if (cand_dist < heap[0].dist) {
        heap[0].dist = cand_dist;
        heap[0].idx  = cand_idx;
        sift_down(heap, 0, K);
    }
}

template<int K>
__device__ __forceinline__ void sort_heap_ascending(Neighbor* heap) {
    // Standard heapsort on the max-heap. After this, heap[0..K-1] is ascending by distance.
    for (int end = K - 1; end > 0; --end) {
        const Neighbor tmp = heap[0];
        heap[0] = heap[end];
        heap[end] = tmp;
        sift_down(heap, 0, end);
    }
}

template<int THREADS, int TILE_POINTS>
__device__ __forceinline__ void load_data_tile_soa(
    float* __restrict__ tile_x,
    float* __restrict__ tile_y,
    const float2* __restrict__ data,
    int base,
    int count) {
#pragma unroll
    for (int i = threadIdx.x; i < TILE_POINTS; i += THREADS) {
        if (i < count) {
            const float2 p = data[base + i];
            tile_x[i] = p.x;
            tile_y[i] = p.y;
        }
    }
}

template<int K, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(KernelConfig<K, WARPS_PER_BLOCK>::kThreads,
                             KernelConfig<K, WARPS_PER_BLOCK>::kLaunchBoundsMinBlocks)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    ResultPair* __restrict__ result) {
    using Config = KernelConfig<K, WARPS_PER_BLOCK>;
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

    // Dynamic shared-memory layout:
    //   [ WARPS_PER_BLOCK * K Neighbor heap entries ][ tile_x[TilePoints] ][ tile_y[TilePoints] ]
    extern __shared__ unsigned char smem_raw[];
    Neighbor* const all_heaps = reinterpret_cast<Neighbor*>(smem_raw);
    float* const tile_x = reinterpret_cast<float*>(all_heaps + WARPS_PER_BLOCK * K);
    float* const tile_y = tile_x + Config::kTilePoints;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool active_query = (query_idx < query_count);

    Neighbor* const heap = all_heaps + warp * K;

    // Load one query point per warp and broadcast to the full warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active_query && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Seed the exact streaming top-K structure with the first K database points.
    if (active_query) {
        for (int i = lane; i < K; i += 32) {
            const float2 p = data[i];
            heap[i].dist = sq_l2_2d(qx, qy, p.x, p.y);
            heap[i].idx  = i;
        }
    }
    __syncwarp(FULL_MASK);
    if (active_query && lane == 0) {
        build_max_heap<K>(heap);
    }
    __syncwarp(FULL_MASK);

    // Stream the remainder of the database.
    //
    // Exactness comes from the standard online top-K invariant:
    // if a candidate's distance is not better than the current kth-best distance,
    // it can be discarded permanently.
    for (int base = K; base < data_count; base += Config::kTilePoints) {
        const int remaining  = data_count - base;
        const int tile_count = (remaining < Config::kTilePoints) ? remaining : Config::kTilePoints;

        load_data_tile_soa<Config::kThreads, Config::kTilePoints>(tile_x, tile_y, data, base, tile_count);
        __syncthreads();

        if (active_query) {
            float threshold = (lane == 0) ? heap[0].dist : 0.0f;
            threshold = __shfl_sync(FULL_MASK, threshold, 0);

            for (int off = 0; off < tile_count; off += 32) {
                const int local_idx = off + lane;
                const bool valid = (local_idx < tile_count);

                float dist = 0.0f;
                int data_idx = base + local_idx;
                if (valid) {
                    dist = sq_l2_2d(qx, qy, tile_x[local_idx], tile_y[local_idx]);
                }

                // All lanes that beat the current threshold form a candidate mask.
                // Candidates are processed one by one inside the warp; no inter-warp locking
                // is needed because each warp owns a disjoint heap.
                unsigned mask = __ballot_sync(FULL_MASK, valid && (dist < threshold));
                while (mask) {
                    const int src_lane = __ffs(mask) - 1;
                    const float cand_dist = __shfl_sync(FULL_MASK, dist, src_lane);
                    const int cand_idx    = __shfl_sync(FULL_MASK, data_idx, src_lane);

                    if (lane == 0) {
                        // threshold is the current root distance. If earlier insertions in this same
                        // batch already shrank the threshold enough, many stale candidates are skipped
                        // here without touching the heap.
                        if (cand_dist < threshold) {
                            replace_root_if_better<K>(heap, cand_dist, cand_idx);
                            threshold = heap[0].dist;
                        }
                    }
                    threshold = __shfl_sync(FULL_MASK, threshold, 0);

                    mask &= (mask - 1);
                }
            }
        }

        __syncthreads();
    }

    // Convert the heap to ascending order so result[q * K + j] is the j-th nearest neighbor.
    // Direct member writes are used on std::pair to avoid relying on any host-only constructors
    // or assignment operators in device code.
    if (active_query) {
        if (lane == 0) {
            sort_heap_ascending<K>(heap);
        }
        __syncwarp(FULL_MASK);

        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);
        for (int i = lane; i < K; i += 32) {
            result[out_base + static_cast<std::size_t>(i)].first  = heap[i].idx;
            result[out_base + static_cast<std::size_t>(i)].second = heap[i].dist;
        }
    }
}

inline int blocks_for_queries(int query_count, int warps_per_block) {
    return (query_count + warps_per_block - 1) / warps_per_block;
}

template<int K, int WARPS_PER_BLOCK>
inline void launch_knn_kernel(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    ResultPair* result) {
    using Config = KernelConfig<K, WARPS_PER_BLOCK>;
    constexpr int threads = Config::kThreads;
    constexpr int shared_bytes = Config::kSharedBytes;

    // Opt in to the larger dynamic shared-memory budget and bias the unified L1/shared pool
    // toward shared memory, which is the dominant on-chip resource used by this kernel.
    cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);
    cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int grid = blocks_for_queries(query_count, WARPS_PER_BLOCK);
    knn_kernel<K, WARPS_PER_BLOCK><<<grid, threads, shared_bytes>>>(
        query, query_count, data, data_count, result);
}

// Choose the largest legal queries-per-block value that still leaves at least roughly one block
// per SM. Larger blocks reuse each data tile across more queries and therefore reduce HBM traffic.
template<int K>
inline void launch_best_up_to_32(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    ResultPair* result,
    int sm_count) {
    if (blocks_for_queries(query_count, 32) >= sm_count) {
        launch_knn_kernel<K, 32>(query, query_count, data, data_count, result);
    } else if (blocks_for_queries(query_count, 16) >= sm_count) {
        launch_knn_kernel<K, 16>(query, query_count, data, data_count, result);
    } else if (blocks_for_queries(query_count, 8) >= sm_count) {
        launch_knn_kernel<K, 8>(query, query_count, data, data_count, result);
    } else {
        launch_knn_kernel<K, 4>(query, query_count, data, data_count, result);
    }
}

// K = 1024 cannot use 32 warps/block on A100 because the per-block shared-memory footprint
// would exceed the opt-in limit, so 16 warps/block is the largest legal choice.
template<int K>
inline void launch_best_up_to_16(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    ResultPair* result,
    int sm_count) {
    if (blocks_for_queries(query_count, 16) >= sm_count) {
        launch_knn_kernel<K, 16>(query, query_count, data, data_count, result);
    } else if (blocks_for_queries(query_count, 8) >= sm_count) {
        launch_knn_kernel<K, 8>(query, query_count, data, data_count, result);
    } else {
        launch_knn_kernel<K, 4>(query, query_count, data, data_count, result);
    }
}

} // namespace

void run_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k) {
    if (query_count <= 0) return;

    int device = 0;
    int sm_count = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    // Launch is intentionally asynchronous, matching normal CUDA kernel-launch semantics.
    switch (k) {
        case 32:
            launch_best_up_to_32<32>(query, query_count, data, data_count, result, sm_count);
            break;
        case 64:
            launch_best_up_to_32<64>(query, query_count, data, data_count, result, sm_count);
            break;
        case 128:
            launch_best_up_to_32<128>(query, query_count, data, data_count, result, sm_count);
            break;
        case 256:
            launch_best_up_to_32<256>(query, query_count, data, data_count, result, sm_count);
            break;
        case 512:
            launch_best_up_to_32<512>(query, query_count, data, data_count, result, sm_count);
            break;
        case 1024:
            launch_best_up_to_16<1024>(query, query_count, data, data_count, result, sm_count);
            break;
        default:
            // Per problem statement, k is always a supported power of two.
            break;
    }
}