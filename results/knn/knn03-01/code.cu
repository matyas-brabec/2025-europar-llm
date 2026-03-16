#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace {

// Exact 2D k-NN for large query/data sets on A100/H100-class GPUs.
//
// Design choices:
// - One warp computes one query, exactly as requested.
// - 16 warps/block (512 threads) maximizes data reuse: each shared-memory data batch is
//   consumed by 16 independent queries before the block advances.
// - A 2048-point batch keeps the worst-case shared-memory footprint (k=1024) below
//   Ampere's 163,840 B/block dynamic shared-memory limit while still amortizing
//   block-wide synchronization well.
// - The shared cache is stored as SoA (x[], y[]) instead of float2 AoS to avoid the
//   repeated shared-memory bank conflicts that an AoS layout would create when every
//   warp streams the cached points.
// - Each warp owns one private max-heap of K neighbors in shared memory. Only lane 0
//   mutates that heap; all 32 lanes stay busy on distance evaluation and use shuffles
//   to feed only potentially useful candidates to lane 0. This is exact and efficient:
//   after the initial fill, streaming top-k replacements are rare compared to the total
//   number of distance computations.
// - Distances are squared Euclidean distances as required; no sqrt is taken.

constexpr int WARP_SIZE            = 32;
constexpr int THREADS_PER_BLOCK    = 512;
constexpr int WARPS_PER_BLOCK      = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int BATCH_POINTS         = 2048;
constexpr int FULL_BATCH_FLOAT4S   = BATCH_POINTS / 2;
constexpr unsigned FULL_MASK       = 0xffffffffu;
constexpr std::size_t AMPERE_MAX_DYNAMIC_SHARED_BYTES = 163840;

struct alignas(8) Neighbor {
    float dist;
    int   idx;
};

static_assert(THREADS_PER_BLOCK % WARP_SIZE == 0, "Block size must be a multiple of warp size.");
static_assert(BATCH_POINTS % WARP_SIZE == 0, "Batch size must be a multiple of warp size.");
static_assert(BATCH_POINTS % 2 == 0, "Batch size must be even for float4 vectorized loads.");
static_assert(FULL_BATCH_FLOAT4S == 2 * THREADS_PER_BLOCK,
              "The full-batch loader assumes exactly two float4 loads per thread.");
static_assert(sizeof(Neighbor) == 8, "Neighbor must stay compact.");
static_assert((2 * BATCH_POINTS * sizeof(float)) % alignof(Neighbor) == 0,
              "Shared-memory heap base must stay naturally aligned.");

template <int K>
constexpr std::size_t shared_bytes_for_k() {
    return 2ull * BATCH_POINTS * sizeof(float) +
           static_cast<std::size_t>(WARPS_PER_BLOCK) * K * sizeof(Neighbor);
}

static_assert(shared_bytes_for_k<1024>() <= AMPERE_MAX_DYNAMIC_SHARED_BYTES,
              "Worst-case shared-memory footprint exceeds Ampere's per-block limit.");

__device__ __forceinline__ bool neighbor_worse(const Neighbor& a, const Neighbor& b) {
    return (a.dist > b.dist) || ((a.dist == b.dist) && (a.idx > b.idx));
}

__device__ __forceinline__ bool neighbor_better(const Neighbor& a, const Neighbor& b) {
    return neighbor_worse(b, a);
}

__device__ __forceinline__ void swap_neighbors(Neighbor& a, Neighbor& b) {
    const Neighbor tmp = a;
    a = b;
    b = tmp;
}

// Sift-down for a max-heap: the root stores the current worst kept neighbor.
template <int K>
__device__ __forceinline__ void sift_down_max(Neighbor* heap, int size, int root) {
    Neighbor value = heap[root];
    int child = (root << 1) + 1;

    while (child < size) {
        const int right = child + 1;
        if (right < size && neighbor_worse(heap[right], heap[child])) {
            child = right;
        }
        if (!neighbor_worse(heap[child], value)) {
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
        sift_down_max<K>(heap, K, i);
    }
}

template <int K>
__device__ __forceinline__ void sort_heap_ascending(Neighbor* heap) {
    // In-place heapsort of the max-heap. Result becomes ascending by distance
    // (and by index for the tie-breaker used inside the heap).
    for (int end = K - 1; end > 0; --end) {
        swap_neighbors(heap[0], heap[end]);
        sift_down_max<K>(heap, end, 0);
    }
}

template <int K>
__device__ __forceinline__ void insert_candidate(Neighbor* heap, int& heap_fill, const Neighbor& cand) {
    if (heap_fill < K) {
        heap[heap_fill++] = cand;
        if (heap_fill == K) {
            build_max_heap<K>(heap);
        }
    } else if (neighbor_better(cand, heap[0])) {
        heap[0] = cand;
        sift_down_max<K>(heap, K, 0);
    }
}

// Block-cooperative batch load into shared-memory SoA cache.
// The common full-batch path is manually unrolled: with the chosen constants each thread
// performs exactly two coalesced float4 loads (4 points/thread).
__device__ __forceinline__
void load_batch_soa(float* __restrict__ batch_x,
                    float* __restrict__ batch_y,
                    const float2* __restrict__ data,
                    int batch_start,
                    int batch_size) {
    const float4* __restrict__ g4 =
        reinterpret_cast<const float4*>(data + batch_start);

    if (batch_size == BATCH_POINTS) {
        const int chunk0 = threadIdx.x;
        const float4 v0 = g4[chunk0];
        const int j0 = chunk0 << 1;
        batch_x[j0]     = v0.x;
        batch_y[j0]     = v0.y;
        batch_x[j0 + 1] = v0.z;
        batch_y[j0 + 1] = v0.w;

        const int chunk1 = chunk0 + THREADS_PER_BLOCK;
        const float4 v1 = g4[chunk1];
        const int j1 = chunk1 << 1;
        batch_x[j1]     = v1.x;
        batch_y[j1]     = v1.y;
        batch_x[j1 + 1] = v1.z;
        batch_y[j1 + 1] = v1.w;
    } else {
        const int pair_chunks = batch_size >> 1;
        for (int chunk = threadIdx.x; chunk < pair_chunks; chunk += THREADS_PER_BLOCK) {
            const float4 v = g4[chunk];
            const int j = chunk << 1;
            batch_x[j]     = v.x;
            batch_y[j]     = v.y;
            batch_x[j + 1] = v.z;
            batch_y[j + 1] = v.w;
        }

        if ((batch_size & 1) && threadIdx.x == 0) {
            const float2 last = data[batch_start + batch_size - 1];
            batch_x[batch_size - 1] = last.x;
            batch_y[batch_size - 1] = last.y;
        }
    }
}

// Warp-local processing of one cached batch.
//
// Important correctness detail:
// - We build a warp-wide mask of lanes that may matter.
// - Then we replay those lanes in ascending lane order via shuffles.
// - Lane 0 rechecks each replayed candidate against the *current* heap root before
//   insertion. This preserves exact streaming semantics even though the threshold can
//   tighten within the 32-point tile.
template <int K, bool FULL_BATCH>
__device__ __forceinline__
void process_cached_batch(float qx,
                          float qy,
                          const float* __restrict__ batch_x,
                          const float* __restrict__ batch_y,
                          int batch_start,
                          int batch_size,
                          Neighbor* __restrict__ heap,
                          int& heap_fill,
                          int lane) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0),
                  "K must be a power of two in [32, 1024].");

    const int limit = FULL_BATCH ? BATCH_POINTS : batch_size;

    for (int local = 0; local < limit; local += WARP_SIZE) {
        const int point = local + lane;
        const bool valid = FULL_BATCH || (point < batch_size);

        float px = 0.0f;
        float py = 0.0f;
        if (valid) {
            px = batch_x[point];
            py = batch_y[point];
        }

        const float dx = px - qx;
        const float dy = py - qy;
        const float dist = __fmaf_rn(dx, dx, dy * dy);  // squared L2
        const int data_idx = batch_start + point;

        // Broadcast the current fill state from lane 0. While the heap is not full yet,
        // every valid lane participates. Once the heap is full, a strict '< root' filter
        // is enough: ties are explicitly left unspecified by the interface, and avoiding
        // '<=' removes useless heap traffic.
        const int filled = __shfl_sync(FULL_MASK, (lane == 0) ? heap_fill : 0, 0);

        unsigned selected_mask;
        if (filled < K) {
            selected_mask = __ballot_sync(FULL_MASK, valid);
        } else {
            const float threshold =
                __shfl_sync(FULL_MASK, (lane == 0) ? heap[0].dist : 0.0f, 0);
            selected_mask = __ballot_sync(FULL_MASK, valid && (dist < threshold));
        }

        while (selected_mask) {
            const int src_lane = __ffs(static_cast<int>(selected_mask)) - 1;

            // All lanes must execute the shuffles; only lane 0 consumes the values.
            const float cand_dist = __shfl_sync(FULL_MASK, dist, src_lane);
            const int cand_idx    = __shfl_sync(FULL_MASK, data_idx, src_lane);

            if (lane == 0) {
                const Neighbor cand{cand_dist, cand_idx};
                insert_candidate<K>(heap, heap_fill, cand);
            }

            selected_mask &= (selected_mask - 1);
        }

        // Required on Volta+ due to independent thread scheduling: lane 0 mutates the
        // per-warp shared-memory heap while the other lanes are idle.
        __syncwarp(FULL_MASK);
    }
}

template <int K>
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                std::pair<int, float>* __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0),
                  "K must be a power of two in [32, 1024].");

    // int4 guarantees a 16-byte-aligned base for the dynamic shared-memory region.
    extern __shared__ int4 shared_base[];
    unsigned char* smem = reinterpret_cast<unsigned char*>(shared_base);

    float* batch_x = reinterpret_cast<float*>(smem);
    float* batch_y = batch_x + BATCH_POINTS;
    Neighbor* heap_base = reinterpret_cast<Neighbor*>(batch_y + BATCH_POINTS);

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & (WARP_SIZE - 1);
    const int qid     = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool valid_query = (qid < query_count);

    Neighbor* const heap = heap_base + static_cast<std::size_t>(warp_id) * K;

    // Lane 0 owns the heap metadata for the warp.
    int heap_fill = 0;

    float qx = 0.0f;
    float qy = 0.0f;
    if (valid_query && lane == 0) {
        const float2 q = query[qid];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_POINTS) {
        const int remaining  = data_count - batch_start;
        const int batch_size = (remaining > BATCH_POINTS) ? BATCH_POINTS : remaining;

        load_batch_soa(batch_x, batch_y, data, batch_start, batch_size);
        __syncthreads();

        if (valid_query) {
            if (batch_size == BATCH_POINTS) {
                process_cached_batch<K, true>(
                    qx, qy, batch_x, batch_y, batch_start, batch_size, heap, heap_fill, lane);
            } else {
                process_cached_batch<K, false>(
                    qx, qy, batch_x, batch_y, batch_start, batch_size, heap, heap_fill, lane);
            }
        }

        // Protect the shared batch buffer before it is overwritten by the next iteration.
        __syncthreads();
    }

    if (valid_query && lane == 0) {
        sort_heap_ascending<K>(heap);

        const std::size_t out_base =
            static_cast<std::size_t>(qid) * static_cast<std::size_t>(K);

        // Only lane 0 writes results. Output volume is tiny relative to the full scan,
        // so keeping the critical path simple is preferable here.
        for (int i = 0; i < K; ++i) {
            result[out_base + static_cast<std::size_t>(i)].first  = heap[i].idx;
            result[out_base + static_cast<std::size_t>(i)].second = heap[i].dist;
        }
    }
}

template <int K>
inline void launch_knn_impl(const float2* query,
                            int query_count,
                            const float2* data,
                            int data_count,
                            std::pair<int, float>* result) {
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const std::size_t smem_bytes = shared_bytes_for_k<K>();

    // Request the large dynamic shared-memory allocation and bias the SM carveout
    // toward shared memory. This is required for the k=1024 case on Ampere.
    cudaFuncSetAttribute(knn_kernel<K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
    cudaFuncSetAttribute(knn_kernel<K>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    // Intentionally asynchronous, like a normal CUDA launcher.
    knn_kernel<K><<<blocks, THREADS_PER_BLOCK, smem_bytes>>>(
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

    switch (k) {
        case 32:   launch_knn_impl<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_impl<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_impl<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_impl<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_impl<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_impl<1024>(query, query_count, data, data_count, result); break;
        default:
            // Per problem statement this path is unreachable (k is always valid).
            return;
    }
}