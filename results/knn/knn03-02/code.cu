#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace knn_detail {

// Kernel design summary:
// - One warp computes one query.
// - 8 warps/block lets every shared-memory tile of data be reused by 8 queries.
// - The data tile is cached in shared memory in SoA form (x[] / y[]) so the hot
//   distance loop uses conflict-free 32-bit shared loads.
// - The current top-k set for each query is stored as a warp-private max-heap in
//   shared memory. The heap stores packed (distance_bits, index) pairs in 64 bits.
//   Squared Euclidean distances are non-negative, so IEEE-754 bit order is monotone
//   as an unsigned integer; raw 64-bit comparisons therefore implement
//   "distance first, index second" ordering. The prompt allows arbitrary tie handling.
// - Only lane 0 mutates the heap. After the warm-up phase, streaming top-k updates
//   become sparse, so a single-lane heap update is cheaper than carrying a large,
//   dynamically indexed per-thread structure that would otherwise risk local-memory
//   spills at k = 1024.
// - k is specialized at compile time (32, 64, 128, 256, 512, 1024) so heap depth,
//   shared-memory footprint, and initialization work are all fully known to the compiler.

constexpr unsigned FULL_MASK      = 0xffffffffu;
constexpr int      WARP_SIZE      = 32;
constexpr int      WARPS_PER_BLOCK = 8;
constexpr int      BLOCK_THREADS   = WARPS_PER_BLOCK * WARP_SIZE;

// 2048 is the largest power-of-two tile that keeps the worst-case kernel
// (k = 1024, 8 warps/block) at 80 KiB/block dynamic shared memory:
//   tile cache: 2 * 2048 * sizeof(float) = 16 KiB
//   warp heaps: 8 * 1024 * sizeof(uint64) = 64 KiB
// Total = 80 KiB, so two blocks still fit comfortably on A100/H100-class SMs.
constexpr int BATCH_POINTS = 2048;

using packed_neighbor_t = unsigned long long;
using output_pair_t     = std::pair<int, float>;

static_assert(BLOCK_THREADS == 256, "Expected 256-thread blocks.");
static_assert(BATCH_POINTS % BLOCK_THREADS == 0, "Batch size must be divisible by block size.");
static_assert(BATCH_POINTS % WARP_SIZE == 0, "Batch size must be divisible by warp size.");
static_assert((2 * BATCH_POINTS * static_cast<int>(sizeof(float))) % static_cast<int>(alignof(packed_neighbor_t)) == 0,
              "Heap region in dynamic shared memory must stay naturally aligned.");

constexpr int ilog2_constexpr(int x) {
    return (x <= 1) ? 0 : 1 + ilog2_constexpr(x >> 1);
}

template <int K>
constexpr int shared_bytes_for() {
    return 2 * BATCH_POINTS * static_cast<int>(sizeof(float)) +
           WARPS_PER_BLOCK * K * static_cast<int>(sizeof(packed_neighbor_t));
}

static_assert(shared_bytes_for<1024>() <= 96 * 1024,
              "Worst-case dynamic shared memory footprint must remain under the common opt-in limit.");

// Pack distance and index into one 64-bit word.
// High 32 bits: raw float bits of non-negative distance.
// Low  32 bits: index.
// Since all distances are squared L2 norms, they are non-negative, so unsigned
// integer comparison on the high 32 bits preserves the distance ordering.
__device__ __forceinline__ packed_neighbor_t pack_neighbor(float dist, int idx) {
    return (static_cast<packed_neighbor_t>(__float_as_uint(dist)) << 32) |
           static_cast<unsigned int>(idx);
}

__device__ __forceinline__ float unpack_distance(packed_neighbor_t v) {
    return __uint_as_float(static_cast<unsigned int>(v >> 32));
}

__device__ __forceinline__ int unpack_index(packed_neighbor_t v) {
    return static_cast<int>(static_cast<unsigned int>(v));
}

// Replace the root of a max-heap and sift the replacement down.
// Only lane 0 calls this; the heap lives in warp-private shared memory.
template <int K>
__device__ __forceinline__ void heap_replace_root_serial(
    packed_neighbor_t* __restrict__ heap,
    int heap_size,
    packed_neighbor_t value)
{
    constexpr int HEAP_LEVELS = ilog2_constexpr(K);

    int pos = 0;

#pragma unroll
    for (int level = 0; level < HEAP_LEVELS; ++level) {
        const int left = (pos << 1) + 1;
        if (left >= heap_size) {
            break;
        }

        int max_child_pos = left;
        packed_neighbor_t max_child = heap[left];

        const int right = left + 1;
        if (right < heap_size) {
            const packed_neighbor_t right_value = heap[right];
            if (right_value > max_child) {
                max_child = right_value;
                max_child_pos = right;
            }
        }

        if (max_child > value) {
            heap[pos] = max_child;
            pos = max_child_pos;
        } else {
            break;
        }
    }

    heap[pos] = value;
}

// Process one cached data batch for a single query/warp.
// FULL_BATCH is specialized so the common full-tile case has no bounds checks
// in the hot distance loop. The monotone root property lets us prefilter
// candidates with a ballot against the current root: if a candidate is not
// better than the root now, it can never become better later in the same step,
// because the root only decreases as inserts happen.
template <int K, bool FULL_BATCH>
__device__ __forceinline__ void process_cached_batch(
    float qx,
    float qy,
    const float* __restrict__ sm_x,
    const float* __restrict__ sm_y,
    int batch_base,
    int valid,
    packed_neighbor_t* __restrict__ warp_heap,
    packed_neighbor_t& root_pack,
    int lane,
    packed_neighbor_t inf_pack)
{
    constexpr int BATCH_STEPS = BATCH_POINTS / WARP_SIZE;
    const int steps = FULL_BATCH ? BATCH_STEPS : ((valid + WARP_SIZE - 1) / WARP_SIZE);

#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        const int sm_index = step * WARP_SIZE + lane;

        packed_neighbor_t cand_local = inf_pack;

        if constexpr (FULL_BATCH) {
            const float dx = qx - sm_x[sm_index];
            const float dy = qy - sm_y[sm_index];
            const float dist = __fmaf_rn(dx, dx, dy * dy);
            cand_local = pack_neighbor(dist, batch_base + sm_index);
        } else {
            if (sm_index < valid) {
                const float dx = qx - sm_x[sm_index];
                const float dy = qy - sm_y[sm_index];
                const float dist = __fmaf_rn(dx, dx, dy * dy);
                cand_local = pack_neighbor(dist, batch_base + sm_index);
            }
        }

        packed_neighbor_t root = __shfl_sync(FULL_MASK, root_pack, 0);
        unsigned int improve_mask = __ballot_sync(FULL_MASK, cand_local < root);

        while (improve_mask != 0u) {
            const int src_lane = __ffs(improve_mask) - 1;
            const packed_neighbor_t cand = __shfl_sync(FULL_MASK, cand_local, src_lane);

            root = __shfl_sync(FULL_MASK, root_pack, 0);
            if (cand < root) {
                if (lane == 0) {
                    heap_replace_root_serial<K>(warp_heap, K, cand);
                    root_pack = warp_heap[0];
                }
            }

            improve_mask &= (improve_mask - 1);
        }
    }
}

template <int K>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    output_pair_t* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    static_assert((K % WARP_SIZE) == 0, "K must be divisible by warp size.");

    constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;
    constexpr int LOADS_PER_THREAD = BATCH_POINTS / BLOCK_THREADS;

    extern __shared__ unsigned char shared_raw[];

    // Shared-memory layout:
    // [ sm_x[BATCH_POINTS] | sm_y[BATCH_POINTS] | heaps[WARPS_PER_BLOCK][K] ]
    float* sm_x = reinterpret_cast<float*>(shared_raw);
    float* sm_y = sm_x + BATCH_POINTS;
    packed_neighbor_t* heaps = reinterpret_cast<packed_neighbor_t*>(sm_y + BATCH_POINTS);

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane    = tid & (WARP_SIZE - 1);

    const int query_idx = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active   = query_idx < query_count;

    packed_neighbor_t* warp_heap = heaps + warp_id * K;

    float qx = 0.0f;
    float qy = 0.0f;

    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }

    // Shuffle itself is the warp synchronization primitive here.
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    const packed_neighbor_t inf_pack = pack_neighbor(CUDART_INF_F, -1);
    packed_neighbor_t root_pack = inf_pack;

    if (active) {
        // Cooperative heap initialization: each lane writes K/32 heap slots.
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            warp_heap[i * WARP_SIZE + lane] = inf_pack;
        }

        // Shared-memory communication within the warp requires a warp barrier.
        __syncwarp(FULL_MASK);
    }

    for (int batch_base = 0; batch_base < data_count; batch_base += BATCH_POINTS) {
        const int remaining = data_count - batch_base;
        const bool full_batch = (remaining >= BATCH_POINTS);
        const int valid = full_batch ? BATCH_POINTS : remaining;

        // Cooperative block-wide load of the next data tile into shared memory.
        if (full_batch) {
#pragma unroll
            for (int load = 0; load < LOADS_PER_THREAD; ++load) {
                const int sm_index = load * BLOCK_THREADS + tid;
                const float2 p = data[batch_base + sm_index];
                sm_x[sm_index] = p.x;
                sm_y[sm_index] = p.y;
            }
        } else {
#pragma unroll
            for (int load = 0; load < LOADS_PER_THREAD; ++load) {
                const int sm_index = load * BLOCK_THREADS + tid;
                if (sm_index < valid) {
                    const float2 p = data[batch_base + sm_index];
                    sm_x[sm_index] = p.x;
                    sm_y[sm_index] = p.y;
                }
            }
        }

        // Block-wide barrier for the shared tile.
        __syncthreads();

        if (active) {
            if (full_batch) {
                process_cached_batch<K, true>(qx, qy, sm_x, sm_y, batch_base, valid, warp_heap, root_pack, lane, inf_pack);
            } else {
                process_cached_batch<K, false>(qx, qy, sm_x, sm_y, batch_base, valid, warp_heap, root_pack, lane, inf_pack);
            }
        }

        // Ensure no warp starts overwriting the shared tile before all warps are done with it.
        __syncthreads();
    }

    // Final heap sort: repeatedly pop the current maximum and write backwards so
    // the output is ascending by distance. Only lane 0 writes std::pair members,
    // which avoids relying on any device-side std::pair constructors/operators.
    if (active && lane == 0) {
        output_pair_t* out = result + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        for (int remaining = K; remaining > 0; --remaining) {
            const packed_neighbor_t best = warp_heap[0];
            out[remaining - 1].first  = unpack_index(best);
            out[remaining - 1].second = unpack_distance(best);

            if (remaining > 1) {
                heap_replace_root_serial<K>(warp_heap, remaining - 1, warp_heap[remaining - 1]);
            }
        }
    }
}

template <int K>
inline void launch_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    output_pair_t* result)
{
    constexpr int shared_bytes = shared_bytes_for<K>();

    // k = 1024 needs 80 KiB dynamic shared memory, so opt in explicitly.
    // Prefer shared memory over L1: this kernel is built around shared tiling and
    // a warp-private shared heap, while the data stream is essentially one-pass.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shared_bytes);
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn_kernel<K><<<grid, block, shared_bytes>>>(query, query_count, data, data_count, result);
}

}  // namespace knn_detail

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k)
{
    // Asynchronous launch on the current/default stream.
    if (query_count <= 0) {
        return;
    }

#define KNN_LAUNCH_CASE(KVAL) \
    case KVAL:                \
        knn_detail::launch_knn<KVAL>(query, query_count, data, data_count, result); \
        break

    switch (k) {
        KNN_LAUNCH_CASE(32);
        KNN_LAUNCH_CASE(64);
        KNN_LAUNCH_CASE(128);
        KNN_LAUNCH_CASE(256);
        KNN_LAUNCH_CASE(512);
        KNN_LAUNCH_CASE(1024);
        default:
            // Per problem statement, k is always valid.
            break;
    }

#undef KNN_LAUNCH_CASE
}