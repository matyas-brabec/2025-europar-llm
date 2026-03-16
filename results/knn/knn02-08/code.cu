#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

// Exact 2D k-NN for large data sets on Ampere/Hopper-class GPUs.
//
// Design choices:
//
// 1) One warp computes one query exactly, as requested.
// 2) A block processes several queries at once so that one shared-memory tile of `data`
//    is reused by multiple warps/queries before the next global-memory load.
// 3) Each query owns a private top-k buffer stored as a warp-private max-heap in shared
//    memory.  The heap root is the current admission threshold.
// 4) All 32 lanes evaluate candidate distances in parallel.  Heap updates are performed
//    by lane 0 only.  This is intentional and efficient for exact top-k selection:
//    after the first few batches, accepted replacements become rare, so serializing only
//    the rare heap updates is substantially cheaper than forcing a full warp-wide sort or
//    merge on every candidate.
// 5) Tiles are single-buffered on purpose.  For K=512/1024 the per-warp heaps already
//    dominate shared-memory usage, and keeping occupancy high is more valuable than
//    double-buffering the data tile.
//
// Distances are squared Euclidean distances as required; no square root is taken.

namespace {

using ResultPair = std::pair<int, float>;

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

struct __align__(8) HeapEntry {
    float dist;
    int   idx;
};

static_assert(sizeof(HeapEntry) == 8, "HeapEntry must remain compact.");

// Squared L2 distance in 2D.  Using FMA keeps the instruction count low.
__device__ __forceinline__ float sq_l2_2d(const float2 p, const float qx, const float qy) {
    const float dx = p.x - qx;
    const float dy = p.y - qy;
    return fmaf(dx, dx, dy * dy);
}

// Standard max-heap sift-down.  Only lane 0 calls this, so it is optimized for
// low shared-memory traffic rather than wide parallelism.
__device__ __forceinline__ void sift_down(HeapEntry* heap, int root, const int size) {
    HeapEntry entry = heap[root];

    while (true) {
        int child = (root << 1) + 1;
        if (child >= size) {
            break;
        }

        const int right = child + 1;
        if (right < size && heap[right].dist > heap[child].dist) {
            child = right;
        }

        if (heap[child].dist <= entry.dist) {
            break;
        }

        heap[root] = heap[child];
        root = child;
    }

    heap[root] = entry;
}

template <int K>
__device__ __forceinline__ void build_max_heap(HeapEntry* heap) {
    for (int i = (K >> 1) - 1; i >= 0; --i) {
        sift_down(heap, i, K);
    }
}

template <int K>
__device__ __forceinline__ void replace_heap_root(HeapEntry* heap, const float dist, const int idx) {
    heap[0].dist = dist;
    heap[0].idx  = idx;
    sift_down(heap, 0, K);
}

template <int K>
__device__ __forceinline__ void sort_heap_ascending(HeapEntry* heap) {
    // Standard in-place heapsort on a max-heap.  The final order is ascending by distance,
    // which matches result[q * k + j] = j-th nearest neighbor.
    for (int end = K - 1; end > 0; --end) {
        const HeapEntry tmp = heap[0];
        heap[0] = heap[end];
        heap[end] = tmp;
        sift_down(heap, 0, end);
    }
}

// Process one shared-memory tile of candidates for one query/warp.
//
// `heap_top` is broadcast to all lanes and is the current worst element of the retained
// top-k set.  Lanes first filter candidates locally against this threshold.  Any lane
// whose candidate might enter the heap is collected via ballot; those candidates are then
// replayed one-by-one to lane 0, which rechecks against the updated heap root before doing
// the exact heap replacement.
//
// This keeps the hot path (distance evaluation) fully parallel and makes the exact
// top-k update cost proportional to the number of actual replacements, not the number
// of scanned points.
template <int K>
__device__ __forceinline__ void process_tile_for_query(
    const float2* __restrict__ sh_data,
    const int batch_begin,
    const int local_begin,
    const int batch_count,
    const float qx,
    const float qy,
    HeapEntry* __restrict__ heap,
    const int lane,
    float& heap_top)
{
    #pragma unroll 1
    for (int base = local_begin; base < batch_count; base += kWarpSize) {
        const int local_idx = base + lane;

        float dist = CUDART_INF_F;
        int idx = -1;
        const bool valid = (local_idx < batch_count);

        if (valid) {
            const float2 p = sh_data[local_idx];
            dist = sq_l2_2d(p, qx, qy);
            idx = batch_begin + local_idx;
        }

        unsigned replace_mask = __ballot_sync(kFullMask, valid && (dist < heap_top));

        while (replace_mask != 0u) {
            const int src_lane = __ffs(replace_mask) - 1;
            const float cand_dist = __shfl_sync(kFullMask, dist, src_lane);
            const int cand_idx = __shfl_sync(kFullMask, idx, src_lane);

            if (lane == 0) {
                // Recheck against the *current* root because earlier insertions from the same
                // 32-point chunk may already have tightened the admission threshold.
                if (cand_dist < heap[0].dist) {
                    replace_heap_root<K>(heap, cand_dist, cand_idx);
                }
                heap_top = heap[0].dist;
            }

            heap_top = __shfl_sync(kFullMask, heap_top, 0);
            replace_mask &= (replace_mask - 1u);
        }
    }
}

template <int K, int WARPS_PER_BLOCK, int BATCH_POINTS>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
void knn_kernel(
    const float2* __restrict__ query,
    const int query_count,
    const float2* __restrict__ data,
    const int data_count,
    ResultPair* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(BATCH_POINTS >= K, "The first tile must be large enough to seed the heap.");
    static_assert((BATCH_POINTS % kWarpSize) == 0, "Tile size must be warp-aligned.");
    static_assert(WARPS_PER_BLOCK == 4 || WARPS_PER_BLOCK == 8 || WARPS_PER_BLOCK == 16,
                  "Only the tuned warp counts are supported.");

    extern __shared__ __align__(16) unsigned char smem_raw[];

    float2* const sh_data = reinterpret_cast<float2*>(smem_raw);
    HeapEntry* const sh_heaps =
        reinterpret_cast<HeapEntry*>(sh_data + BATCH_POINTS);

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int query_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active_query = (query_id < query_count);

    // Load the query once per warp and broadcast it.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active_query && lane == 0) {
        const float2 q = query[query_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    HeapEntry* const heap = sh_heaps + static_cast<std::size_t>(warp_id) * K;
    float heap_top = CUDART_INF_F;

    // Scan the entire data set tile by tile.  Every tile is loaded once per block and
    // then reused by all WARPS_PER_BLOCK queries in the block.
    for (int batch_begin = 0; batch_begin < data_count; batch_begin += BATCH_POINTS) {
        int batch_count = data_count - batch_begin;
        if (batch_count > BATCH_POINTS) {
            batch_count = BATCH_POINTS;
        }

        // Cooperative global->shared load of the current data tile.
        #pragma unroll 1
        for (int i = threadIdx.x; i < batch_count; i += blockDim.x) {
            sh_data[i] = data[batch_begin + i];
        }

        // Shared tile must be visible to all warps before any warp starts consuming it.
        __syncthreads();

        if (active_query) {
            if (batch_begin == 0) {
                // Seed the exact top-k heap from the first K points.  We require BATCH_POINTS >= K,
                // so the entire seed lives in the first shared tile.
                #pragma unroll
                for (int pos = lane; pos < K; pos += kWarpSize) {
                    const float2 p = sh_data[pos];
                    heap[pos].dist = sq_l2_2d(p, qx, qy);
                    heap[pos].idx  = pos;
                }

                // Lane 0 heapifies the warp-private shared-memory buffer.  Because the heap
                // was written cooperatively by the whole warp, a warp barrier is required.
                __syncwarp(kFullMask);

                if (lane == 0) {
                    build_max_heap<K>(heap);
                    heap_top = heap[0].dist;
                }

                heap_top = __shfl_sync(kFullMask, heap_top, 0);
            }

            const int local_begin = (batch_begin == 0) ? K : 0;

            process_tile_for_query<K>(
                sh_data,
                batch_begin,
                local_begin,
                batch_count,
                qx,
                qy,
                heap,
                lane,
                heap_top);
        }

        // All warps in the block must be done with the current shared tile before it is
        // overwritten by the next global-memory batch.
        __syncthreads();
    }

    if (active_query) {
        // Convert the max-heap into ascending order for the required result layout.
        if (lane == 0) {
            sort_heap_ascending<K>(heap);
        }

        // The sorted heap is read by the whole warp, so we need a warp barrier after
        // lane 0 finishes the in-place heapsort.
        __syncwarp(kFullMask);

        const std::size_t out_base = static_cast<std::size_t>(query_id) * K;

        // Write std::pair members directly instead of constructing std::pair on device.
        #pragma unroll
        for (int pos = lane; pos < K; pos += kWarpSize) {
            result[out_base + pos].first  = heap[pos].idx;
            result[out_base + pos].second = heap[pos].dist;
        }
    }
}

static inline int ceil_div(const int n, const int d) {
    return (n + d - 1) / d;
}

template <int K, int WARPS_PER_BLOCK, int BATCH_POINTS>
constexpr std::size_t shared_bytes() {
    return static_cast<std::size_t>(BATCH_POINTS) * sizeof(float2) +
           static_cast<std::size_t>(WARPS_PER_BLOCK) * static_cast<std::size_t>(K) * sizeof(HeapEntry);
}

// Heuristic for block width selection:
//
// Larger WARPS_PER_BLOCK improves reuse of each shared-memory data tile and therefore
// reduces global-memory traffic, but too large a block count can underfill the GPU when
// query_count is only in the low-thousands.  We therefore pick the largest block width
// whose grid still gives at least ~75% as many blocks as SMs.  This keeps almost all SMs
// occupied while preserving substantial cross-query tile reuse.
template <int MAX_WARPS_PER_BLOCK>
static inline int choose_warps_per_block(const int query_count, const int sm_count) {
    static_assert(MAX_WARPS_PER_BLOCK == 8 || MAX_WARPS_PER_BLOCK == 16,
                  "MAX_WARPS_PER_BLOCK must be 8 or 16.");

    if constexpr (MAX_WARPS_PER_BLOCK >= 16) {
        if (4LL * ceil_div(query_count, 16) >= 3LL * sm_count) {
            return 16;
        }
    }

    if constexpr (MAX_WARPS_PER_BLOCK >= 8) {
        if (4LL * ceil_div(query_count, 8) >= 3LL * sm_count) {
            return 8;
        }
    }

    return 4;
}

template <int K, int WARPS_PER_BLOCK, int BATCH_POINTS>
static inline void launch_variant(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    ResultPair* result)
{
    constexpr std::size_t kSharedBytes = shared_bytes<K, WARPS_PER_BLOCK, BATCH_POINTS>();

    // This kernel is explicitly shared-memory heavy.  Ask the runtime for the largest
    // shared-memory carveout and opt in to the required dynamic shared-memory size.
    (void)cudaFuncSetCacheConfig(knn_kernel<K, WARPS_PER_BLOCK, BATCH_POINTS>, cudaFuncCachePreferShared);
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK, BATCH_POINTS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK, BATCH_POINTS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes));

    const dim3 block(WARPS_PER_BLOCK * kWarpSize);
    const dim3 grid(ceil_div(query_count, WARPS_PER_BLOCK));

    // Normal CUDA semantics: the launch is asynchronous with respect to the host.
    knn_kernel<K, WARPS_PER_BLOCK, BATCH_POINTS>
        <<<grid, block, kSharedBytes>>>(query, query_count, data, data_count, result);
}

template <int K>
static inline void dispatch_k(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    ResultPair* result,
    const int sm_count)
{
    // Tile sizes tuned to keep occupancy healthy while amortizing synchronization:
    // - K <= 256: 4096-point tiles work well and still leave ample shared memory.
    // - K >= 512: the per-warp heap becomes large enough that 2048-point tiles are the
    //   better occupancy / synchronization trade-off.
    constexpr int kBatchPoints = (K <= 256) ? 4096 : 2048;
    constexpr int kMaxWarpsPerBlock = (K == 1024) ? 8 : 16;

    static_assert(kBatchPoints >= K, "Tile must hold the initial heap seed.");
    static_assert(kMaxWarpsPerBlock == 8 || kMaxWarpsPerBlock == 16, "Unexpected launch policy.");

    const int warps_per_block = choose_warps_per_block<kMaxWarpsPerBlock>(query_count, sm_count);

    if constexpr (kMaxWarpsPerBlock == 16) {
        switch (warps_per_block) {
            case 16:
                launch_variant<K, 16, kBatchPoints>(query, query_count, data, data_count, result);
                return;
            case 8:
                launch_variant<K, 8, kBatchPoints>(query, query_count, data, data_count, result);
                return;
            default:
                launch_variant<K, 4, kBatchPoints>(query, query_count, data, data_count, result);
                return;
        }
    } else {
        switch (warps_per_block) {
            case 8:
                launch_variant<K, 8, kBatchPoints>(query, query_count, data, data_count, result);
                return;
            default:
                launch_variant<K, 4, kBatchPoints>(query, query_count, data, data_count, result);
                return;
        }
    }
}

} // anonymous namespace

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
    int sm_count = 1;
    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count < 1) {
        sm_count = 1;
    }

    switch (k) {
        case 32:
            dispatch_k<32>(query, query_count, data, data_count, result, sm_count);
            return;
        case 64:
            dispatch_k<64>(query, query_count, data, data_count, result, sm_count);
            return;
        case 128:
            dispatch_k<128>(query, query_count, data, data_count, result, sm_count);
            return;
        case 256:
            dispatch_k<256>(query, query_count, data, data_count, result, sm_count);
            return;
        case 512:
            dispatch_k<512>(query, query_count, data, data_count, result, sm_count);
            return;
        case 1024:
            dispatch_k<1024>(query, query_count, data, data_count, result, sm_count);
            return;
        default:
            // Per the problem statement this path should never be taken.
            return;
    }
}