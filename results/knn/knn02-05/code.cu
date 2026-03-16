#include <cuda_runtime.h>
#include <utility>
#include <type_traits>
#include <cstddef>

namespace knn_detail {

// Brute-force 2D k-NN specialized for the constraints in the prompt.
//
// Design choices:
//  - One warp computes one query.
//  - A block processes several queries at once, so each global-memory tile of `data` is
//    loaded once and reused by multiple query warps.
//  - Each warp owns a private top-k state stored as a max-heap in shared memory.
//    The heap root is the current worst accepted neighbor / rejection threshold.
//  - Each 32-point batch is processed warp-wide. Distances that do not beat the threshold
//    are rejected with only a compare. Distances that do beat the threshold are serialized
//    through lane 0 using ballots/shuffles; this keeps the common reject path extremely cheap.
//  - The shared-memory tile is stored as SoA (x[] and y[]) instead of float2[] to avoid the
//    64-bit shared-memory access pattern that would otherwise introduce unnecessary bank pressure.
//  - k is specialized at compile time for the six allowed values, so heap loops and per-warp
//    array addressing are constant-folded.

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// Device code only needs a POD layout-compatible view of std::pair<int,float>.
// The public interface still uses std::pair<int,float>.
struct KnnPair {
    int   first;
    float second;
};

static_assert(std::is_standard_layout<KnnPair>::value, "KnnPair must be standard layout.");
static_assert(std::is_standard_layout<std::pair<int, float>>::value,
              "std::pair<int,float> must be standard layout for raw device storage use.");
static_assert(sizeof(KnnPair) == sizeof(std::pair<int, float>),
              "KnnPair must match std::pair<int,float> size.");
static_assert(alignof(KnnPair) == alignof(std::pair<int, float>),
              "KnnPair must match std::pair<int,float> alignment.");
static_assert(offsetof(KnnPair, first) == offsetof(std::pair<int, float>, first),
              "KnnPair.first must match std::pair<int,float>::first offset.");
static_assert(offsetof(KnnPair, second) == offsetof(std::pair<int, float>, second),
              "KnnPair.second must match std::pair<int,float>::second offset.");

inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

// The total number of active query warps is fixed by `query_count`.
// Therefore we maximize warps/block (to maximize tile reuse and cut global traffic)
// subject to still having at least one block available per SM, so that no SM goes idle
// only because the grid is too small.
inline int choose_warps_per_block(int query_count, int sm_count, int max_warps) {
    if (max_warps >= 32 && ceil_div_int(query_count, 32) >= sm_count) return 32;
    if (max_warps >= 16 && ceil_div_int(query_count, 16) >= sm_count) return 16;
    if (max_warps >= 8  && ceil_div_int(query_count, 8)  >= sm_count) return 8;
    return 4;  // Smallest useful block under the stated workload regime.
}

__device__ __forceinline__ float sq_l2_2d(float qx, float qy, float px, float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return fmaf(dx, dx, dy * dy);
}

// Standard max-heap sift-down on shared-memory arrays.
template <int K>
__device__ __forceinline__ void heap_sift_down(float* dist, int* idx, int root, int size) {
    float v_dist = dist[root];
    int   v_idx  = idx[root];
    int child = (root << 1) + 1;

    #pragma unroll 1
    while (child < size) {
        int max_child = child;
        const int right = child + 1;
        if (right < size && dist[right] > dist[child]) {
            max_child = right;
        }
        if (dist[max_child] <= v_dist) {
            break;
        }
        dist[root] = dist[max_child];
        idx[root]  = idx[max_child];
        root = max_child;
        child = (root << 1) + 1;
    }

    dist[root] = v_dist;
    idx[root]  = v_idx;
}

template <int K>
__device__ __forceinline__ void heap_build(float* dist, int* idx) {
    #pragma unroll 1
    for (int root = (K >> 1) - 1; root >= 0; --root) {
        heap_sift_down<K>(dist, idx, root, K);
    }
}

// Replace heap root with a better candidate and restore the max-heap property.
// This is the hot path for accepted candidates, so it uses the "hole" form rather
// than a store-followed-by-sift to reduce shared-memory traffic.
template <int K>
__device__ __forceinline__ void heap_replace_root(float* dist, int* idx, float new_dist, int new_idx) {
    int root = 0;
    int child = 1;

    #pragma unroll 1
    while (child < K) {
        int max_child = child;
        const int right = child + 1;
        if (right < K && dist[right] > dist[child]) {
            max_child = right;
        }
        if (dist[max_child] <= new_dist) {
            break;
        }
        dist[root] = dist[max_child];
        idx[root]  = idx[max_child];
        root = max_child;
        child = (root << 1) + 1;
    }

    dist[root] = new_dist;
    idx[root]  = new_idx;
}

// Final sort: max-heap -> ascending order, as required by result[i*k + j].
template <int K>
__device__ __forceinline__ void heap_sort_ascending(float* dist, int* idx) {
    #pragma unroll 1
    for (int end = K - 1; end > 0; --end) {
        const float top_dist = dist[0];
        const int   top_idx  = idx[0];
        dist[0]   = dist[end];
        idx[0]    = idx[end];
        dist[end] = top_dist;
        idx[end]  = top_idx;
        heap_sift_down<K>(dist, idx, 0, end);
    }
}

// Warp-wide scan of one shared-memory tile segment.
// Each iteration handles up to 32 candidates (one per lane).
// Only lanes whose candidate beats the current threshold participate in the
// serialized update loop; all other lanes pay only the compare.
template <int K>
__device__ __forceinline__ float scan_tile_segment(
    float qx, float qy,
    const float* tile_x, const float* tile_y,
    int global_base, int local_begin, int local_end,
    float* heap_dist, int* heap_idx,
    int lane, float worst)
{
    #pragma unroll 1
    for (int base = local_begin; base < local_end; base += WARP_SIZE) {
        const int pos = base + lane;

        float dist = 0.0f;
        int   index = -1;
        bool  pending = false;

        if (pos < local_end) {
            dist = sq_l2_2d(qx, qy, tile_x[pos], tile_y[pos]);
            index = global_base + pos;
            pending = (dist < worst);
        }

        while (true) {
            const unsigned pending_mask = __ballot_sync(FULL_MASK, pending);
            if (pending_mask == 0u) {
                break;
            }

            const int src_lane = __ffs(pending_mask) - 1;
            const float cand_dist = __shfl_sync(FULL_MASK, dist, src_lane);
            const int   cand_idx  = __shfl_sync(FULL_MASK, index, src_lane);

            // Only lane 0 mutates the heap; all lanes receive the new threshold by shuffle.
            if (lane == 0) {
                if (cand_dist < worst) {
                    heap_replace_root<K>(heap_dist, heap_idx, cand_dist, cand_idx);
                    worst = heap_dist[0];
                }
            }
            worst = __shfl_sync(FULL_MASK, worst, 0);

            if (lane == src_lane) {
                pending = false;
            }
            if (pending && !(dist < worst)) {
                pending = false;
            }
        }
    }

    return worst;
}

template <int K, int TILE_POINTS>
__global__ __launch_bounds__(1024, 2)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    KnnPair* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    static_assert((K % WARP_SIZE) == 0, "K must be a multiple of warp size.");
    static_assert(TILE_POINTS >= K, "The first tile must be large enough to seed the heap.");
    static_assert((TILE_POINTS % WARP_SIZE) == 0, "Tile size should be warp-aligned.");

    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & (WARP_SIZE - 1);

    const int block_query_base = blockIdx.x * warps_per_block;
    if (block_query_base >= query_count) {
        return;  // Entire block inactive.
    }

    extern __shared__ __align__(16) unsigned char smem_raw[];

    // Shared layout:
    //   tile_x[TILE_POINTS]
    //   tile_y[TILE_POINTS]
    //   heap_dist[warps_per_block][K]
    //   heap_idx [warps_per_block][K]
    float* tile_x = reinterpret_cast<float*>(smem_raw);
    float* tile_y = tile_x + TILE_POINTS;
    float* all_heap_dist = tile_y + TILE_POINTS;
    int*   all_heap_idx  = reinterpret_cast<int*>(all_heap_dist + warps_per_block * K);

    const int q = block_query_base + warp_id;
    const bool active_query = (q < query_count);

    float qx = 0.0f;
    float qy = 0.0f;
    float worst = CUDART_INF_F;
    float* heap_dist = nullptr;
    int*   heap_idx  = nullptr;

    if (active_query) {
        const float2 qp = query[q];
        qx = qp.x;
        qy = qp.y;
        heap_dist = all_heap_dist + warp_id * K;
        heap_idx  = all_heap_idx  + warp_id * K;
    }

    // Process `data` in shared-memory tiles. Two block-wide barriers per tile:
    // one after the load, one before the next overwrite of the tile buffers.
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS) {
        int tile_size = data_count - tile_base;
        if (tile_size > TILE_POINTS) {
            tile_size = TILE_POINTS;
        }

        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            const float2 p = data[tile_base + i];
            tile_x[i] = p.x;
            tile_y[i] = p.y;
        }
        __syncthreads();

        if (active_query) {
            if (tile_base == 0) {
                // Seed the private top-k heap from the first K points.
                #pragma unroll
                for (int i = lane; i < K; i += WARP_SIZE) {
                    heap_dist[i] = sq_l2_2d(qx, qy, tile_x[i], tile_y[i]);
                    heap_idx[i]  = i;
                }
                __syncwarp();

                if (lane == 0) {
                    heap_build<K>(heap_dist, heap_idx);
                    worst = heap_dist[0];
                }
                worst = __shfl_sync(FULL_MASK, worst, 0);

                // Continue scanning the rest of the first tile.
                worst = scan_tile_segment<K>(
                    qx, qy, tile_x, tile_y,
                    tile_base, K, tile_size,
                    heap_dist, heap_idx, lane, worst);
            } else {
                worst = scan_tile_segment<K>(
                    qx, qy, tile_x, tile_y,
                    tile_base, 0, tile_size,
                    heap_dist, heap_idx, lane, worst);
            }
        }

        __syncthreads();
    }

    if (active_query) {
        // Convert the max-heap to ascending order so j=0 is the nearest neighbor.
        if (lane == 0) {
            heap_sort_ascending<K>(heap_dist, heap_idx);
        }
        __syncwarp();

        const int out_base = q * K;
        #pragma unroll
        for (int i = lane; i < K; i += WARP_SIZE) {
            result[out_base + i].first  = heap_idx[i];
            result[out_base + i].second = heap_dist[i];
        }
    }
}

template <int K, int TILE_POINTS>
inline void launch_knn_impl(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int warps_per_block)
{
    const int block_threads = warps_per_block * WARP_SIZE;
    const int grid_blocks   = ceil_div_int(query_count, warps_per_block);

    const size_t shared_bytes =
        (size_t(2) * TILE_POINTS + size_t(warps_per_block) * K) * sizeof(float) +
        (size_t(warps_per_block) * K) * sizeof(int);

    // The selected tile sizes are deliberately chosen so this stays within the
    // dynamic shared-memory budgets of A100/H100-class parts while still allowing
    // strong occupancy. Ask the runtime for the required carve-out explicitly.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, TILE_POINTS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));

    (void)cudaFuncSetAttribute(
        knn_kernel<K, TILE_POINTS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    knn_kernel<K, TILE_POINTS><<<grid_blocks, block_threads, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<KnnPair*>(result));
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

    int device = 0;
    int sm_count = 1;
    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count < 1) {
        sm_count = 1;
    }

    // Tile-size policy:
    //   k =  32,  64 : 8192 points/tile  (tiny heaps; large tiles reduce synchronization)
    //   k = 128      : 4096 points/tile
    //   k >= 256     : 2048 points/tile  (keeps large per-warp heaps within the shared budget
    //                                     even for many warps/block)
    using namespace knn_detail;

    switch (k) {
        case 32: {
            const int wpb = choose_warps_per_block(query_count, sm_count, 32);
            launch_knn_impl<32, 8192>(query, query_count, data, data_count, result, wpb);
            return;
        }
        case 64: {
            const int wpb = choose_warps_per_block(query_count, sm_count, 32);
            launch_knn_impl<64, 8192>(query, query_count, data, data_count, result, wpb);
            return;
        }
        case 128: {
            const int wpb = choose_warps_per_block(query_count, sm_count, 32);
            launch_knn_impl<128, 4096>(query, query_count, data, data_count, result, wpb);
            return;
        }
        case 256: {
            const int wpb = choose_warps_per_block(query_count, sm_count, 32);
            launch_knn_impl<256, 2048>(query, query_count, data, data_count, result, wpb);
            return;
        }
        case 512: {
            const int wpb = choose_warps_per_block(query_count, sm_count, 16);
            launch_knn_impl<512, 2048>(query, query_count, data, data_count, result, wpb);
            return;
        }
        case 1024: {
            const int wpb = choose_warps_per_block(query_count, sm_count, 8);
            launch_knn_impl<1024, 2048>(query, query_count, data, data_count, result, wpb);
            return;
        }
        default:
            // Per prompt, k is always valid; this path is only a defensive no-op.
            return;
    }
}