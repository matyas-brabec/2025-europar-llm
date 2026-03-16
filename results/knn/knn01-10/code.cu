#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

// Exact batched 2D k-NN specialized for very large `data_count` and
// K in {32, 64, 128, 256, 512, 1024}.
//
// Design choices tuned for A100/H100-class GPUs:
// - 256 threads/CTA = 8 warps/CTA, and each warp owns one query.
// - The CTA streams `data` through a 1024-point shared-memory tile reused by all 8 warps,
//   cutting global traffic by 8x relative to one-query-per-block streaming.
// - Each warp keeps its current top-K in a shared-memory max-heap. The heap root is the
//   current worst retained neighbor, so most points are rejected by a single comparison.
// - The first 1024-point tile also seeds the heap for every supported K, so no extra device
//   memory is needed beyond per-CTA shared memory.
//
// The implementation returns squared L2 distances exactly as requested; no square root is taken.

namespace {

constexpr int kWarpSize          = 32;
constexpr int kBlockThreads      = 256;   // 8 warps = 8 queries/CTA.
constexpr int kQueriesPerBlock   = kBlockThreads / kWarpSize;
constexpr int kTilePoints        = 1024;  // 8 KB tile (x,y), chosen to preserve occupancy at K=1024.
constexpr int kTileLoadsPerThread = kTilePoints / kBlockThreads;
constexpr unsigned kFullMask     = 0xFFFFFFFFu;

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a whole number of warps.");
static_assert(kTilePoints % kWarpSize == 0, "Tile size must be warp-aligned.");
static_assert(kTileLoadsPerThread * kBlockThreads == kTilePoints, "Tile load shape must be exact.");
static_assert(kTilePoints >= 1024, "The first tile must cover the largest supported K.");

struct alignas(8) Candidate {
    int   index;
    float dist;
};

static_assert(sizeof(Candidate) == 8, "Candidate must remain compact.");

// Squared Euclidean distance in 2D; matches the problem statement exactly.
__device__ __forceinline__ float squared_l2(const float qx, const float qy,
                                            const float px, const float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return __fmaf_rn(dx, dx, dy * dy);
}

// Cooperative global->shared load for one tile.
// Shared storage is SoA (x[], y[]) so warps later issue simple 32-bit shared loads.
__device__ __forceinline__ void load_data_tile(const float2* __restrict__ data,
                                               const int base,
                                               const int data_count,
                                               float* __restrict__ tile_x,
                                               float* __restrict__ tile_y) {
#pragma unroll
    for (int it = 0; it < kTileLoadsPerThread; ++it) {
        const int local = threadIdx.x + it * kBlockThreads;
        const int idx   = base + local;
        if (idx < data_count) {
            const float2 p = data[idx];
            tile_x[local] = p.x;
            tile_y[local] = p.y;
        }
    }
}

// Standard max-heap sift-down on distance only.
// Ties are intentionally left unresolved because the problem statement imposes no tie policy.
__device__ __forceinline__ void heap_sift_down(Candidate* heap, const int size, const int root) {
    Candidate value = heap[root];
    int idx = root;

    while (true) {
        const int left = (idx << 1) + 1;
        if (left >= size) break;

        int child = left;
        const int right = left + 1;
        if (right < size && heap[right].dist > heap[left].dist) {
            child = right;
        }

        if (!(heap[child].dist > value.dist)) break;

        heap[idx] = heap[child];
        idx = child;
    }

    heap[idx] = value;
}

template <int K>
__device__ __forceinline__ void build_max_heap(Candidate* heap) {
#pragma unroll 1
    for (int i = (K >> 1) - 1; i >= 0; --i) {
        heap_sift_down(heap, K, i);
    }
}

template <int K>
__device__ __forceinline__ void heap_sort_ascending(Candidate* heap) {
#pragma unroll 1
    for (int end = K - 1; end > 0; --end) {
        const Candidate tmp = heap[0];
        heap[0] = heap[end];
        heap[end] = tmp;
        heap_sift_down(heap, end, 0);
    }
}

// Process one shared-memory tile against one query/warp.
//
// START_OFFSET is compile-time because the first tile starts at K (the first K points already
// seeded the heap), while all later tiles start at 0.
// PARTIAL is compile-time so the hot full-tile path has no bounds checks; only the final tile
// (or the small-data case) pays for range guarding.
//
// All lanes participate in the ballot/shuffle collectives. Only lane 0 mutates the shared heap.
template <int K, int START_OFFSET, bool PARTIAL>
__device__ __forceinline__ void process_tile_candidates(Candidate* __restrict__ heap,
                                                        const float qx,
                                                        const float qy,
                                                        const float* __restrict__ tile_x,
                                                        const float* __restrict__ tile_y,
                                                        const int base,
                                                        const int valid,
                                                        const int lane) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a supported power of two.");
    static_assert(K % kWarpSize == 0, "K must be warp-aligned.");
    static_assert(START_OFFSET % kWarpSize == 0, "Tile start must be warp-aligned.");
    static_assert(START_OFFSET <= kTilePoints, "Tile start must lie inside the tile.");

    float threshold = (lane == 0) ? heap[0].dist : 0.0f;
    threshold = __shfl_sync(kFullMask, threshold, 0);

#pragma unroll
    for (int off = START_OFFSET; off < kTilePoints; off += kWarpSize) {
        const int local = off + lane;
        const bool in_range = PARTIAL ? (local < valid) : true;

        float dist = 0.0f;
        const int point_index = base + local;

        if (in_range) {
            dist = squared_l2(qx, qy, tile_x[local], tile_y[local]);
        }

        unsigned mask = __ballot_sync(kFullMask, in_range && (dist < threshold));

        while (mask) {
            const int src_lane = __ffs(mask) - 1;
            const float cand_dist = __shfl_sync(kFullMask, dist, src_lane);
            const int cand_index  = __shfl_sync(kFullMask, point_index, src_lane);

            if (lane == 0 && cand_dist < heap[0].dist) {
                heap[0].index = cand_index;
                heap[0].dist  = cand_dist;
                heap_sift_down(heap, K, 0);
            }

            mask &= (mask - 1);
        }

        if (lane == 0) {
            threshold = heap[0].dist;
        }
        threshold = __shfl_sync(kFullMask, threshold, 0);
    }
}

template <int K>
__global__ __launch_bounds__(kBlockThreads, 2)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                std::pair<int, float>* __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a supported power of two.");
    static_assert(K <= kTilePoints, "The seed tile must contain the entire initial top-K set.");

    // Shared layout:
    //   [ per-warp heaps ][ tile_x[1024] ][ tile_y[1024] ]
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Candidate* const heaps = reinterpret_cast<Candidate*>(smem_raw);
    float* const tile_x = reinterpret_cast<float*>(heaps + kQueriesPerBlock * K);
    float* const tile_y = tile_x + kTilePoints;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;

    const int query_index = blockIdx.x * kQueriesPerBlock + warp;
    const bool active = query_index < query_count;

    Candidate* const heap = heaps + warp * K;

    // One query per warp; lane 0 loads and broadcasts to the rest of the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_index];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Largest multiple of the tile size not exceeding data_count.
    // All tiles before this boundary are full and use the bounds-check-free hot path.
    const int full_limit = (data_count / kTilePoints) * kTilePoints;

    // Load the first tile. Since kTilePoints == 1024 and K <= 1024, the entire initial seed set
    // lives inside this tile for every supported K.
    load_data_tile(data, 0, data_count, tile_x, tile_y);
    __syncthreads();

    // Seed each warp-local heap with the first K points from the first tile.
    if (active) {
#pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            heap[i].index = i;
            heap[i].dist  = squared_l2(qx, qy, tile_x[i], tile_y[i]);
        }
    }
    __syncwarp(kFullMask);

    if (active && lane == 0) {
        build_max_heap<K>(heap);
    }
    __syncwarp(kFullMask);

    // Process the remainder of the first tile. If data_count < 1024, this is the only tile.
    if (active) {
        if (full_limit > 0) {
            process_tile_candidates<K, K, false>(heap, qx, qy, tile_x, tile_y, 0, 0, lane);
        } else {
            process_tile_candidates<K, K, true>(heap, qx, qy, tile_x, tile_y, 0, data_count, lane);
        }
    }
    __syncthreads();

    // Process remaining full tiles.
    for (int base = kTilePoints; base < full_limit; base += kTilePoints) {
        load_data_tile(data, base, data_count, tile_x, tile_y);
        __syncthreads();

        if (active) {
            process_tile_candidates<K, 0, false>(heap, qx, qy, tile_x, tile_y, base, 0, lane);
        }

        __syncthreads();
    }

    // Process the final partial tile, if any.
    if (full_limit > 0 && full_limit < data_count) {
        load_data_tile(data, full_limit, data_count, tile_x, tile_y);
        __syncthreads();

        if (active) {
            process_tile_candidates<K, 0, true>(heap, qx, qy, tile_x, tile_y,
                                                full_limit, data_count - full_limit, lane);
        }

        __syncthreads();
    }

    // Sort the final heap in ascending distance order so result[i * K + j] is the j-th neighbor.
    if (active && lane == 0) {
        heap_sort_ascending<K>(heap);
    }
    __syncwarp(kFullMask);

    // Write out the sorted results. Fields are assigned individually to avoid relying on any
    // device-side std::pair constructor/assignment support.
    if (active) {
        const size_t out_base = static_cast<size_t>(query_index) * static_cast<size_t>(K);
#pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            result[out_base + static_cast<size_t>(i)].first  = heap[i].index;
            result[out_base + static_cast<size_t>(i)].second = heap[i].dist;
        }
    }
}

template <int K>
inline void launch_knn_specialized(const float2* query,
                                   const int query_count,
                                   const float2* data,
                                   const int data_count,
                                   std::pair<int, float>* result) {
    const size_t shared_bytes =
        static_cast<size_t>(kQueriesPerBlock) * static_cast<size_t>(K) * sizeof(Candidate) +
        static_cast<size_t>(2 * kTilePoints) * sizeof(float);

    // K=1024 exceeds the default 48 KiB dynamic shared-memory limit, so opt in as needed.
    if (shared_bytes > (48u * 1024u)) {
        (void)cudaFuncSetAttribute(
            knn_kernel<K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
    }

    // This kernel is shared-memory dominated and benefits from maximizing the shared carveout.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int blocks = (query_count + kQueriesPerBlock - 1) / kQueriesPerBlock;

    // Launch is intentionally asynchronous on the current default stream.
    knn_kernel<K><<<blocks, kBlockThreads, shared_bytes>>>(
        query, query_count, data, data_count, result);
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

    // Runtime dispatch keeps the public interface simple while still giving the kernel compile-time
    // knowledge of K for heap sizing and loop specialization.
    switch (k) {
        case 32:   launch_knn_specialized<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_specialized<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_specialized<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_specialized<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_specialized<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_specialized<1024>(query, query_count, data, data_count, result); break;
        default:
            // The problem statement guarantees valid K, so this path is unreachable for valid input.
            break;
    }
}