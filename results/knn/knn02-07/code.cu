#include <cuda_runtime.h>
#include <utility>

namespace {

using ResultPair = std::pair<int, float>;
using U64 = unsigned long long;

// Tuned for modern data-center GPUs.
// - One warp computes one query, exactly as requested.
// - Data points are staged in shared memory in 2D tiles.
// - The tile is stored as SoA (x[] / y[]) instead of float2[] to avoid the shared-memory
//   bank pattern that contiguous 64-bit float2 accesses would create.
// - Each warp owns a private top-k structure: a max-heap of packed (distance, index) pairs.
//   The heap is kept in shared memory, but each warp uses a disjoint slice, so the copy is
//   private to that query.
//
// The tile size is chosen so that the first tile always contains the full warm-up set for the
// largest supported k (1024), which lets us seed the heap entirely from shared memory.
constexpr int kWarpSize      = 32;
constexpr int kMaxSupportedK = 1024;
constexpr int kTilePoints    = 2048;
constexpr size_t kTileSharedBytes = 2ull * kTilePoints * sizeof(float);

static_assert(kTilePoints >= kMaxSupportedK, "The first tile must contain the maximum supported k.");
static_assert((kTilePoints % kWarpSize) == 0, "Tile size must be a multiple of the warp size.");

constexpr int ceil_div_int(const int n, const int d) {
    return (n + d - 1) / d;
}

// Squared distances are always non-negative, so the IEEE-754 bit pattern of the float is
// monotonically ordered as an unsigned integer. We exploit that to compare candidates by
// integer key instead of by a {float,index} struct.
__device__ __forceinline__ unsigned distance_bits_2d(
    const float qx, const float qy,
    const float px, const float py)
{
    const float dx = qx - px;
    const float dy = qy - py;
    const float dist = __fmaf_rn(dx, dx, dy * dy);
    return __float_as_uint(dist);
}

__device__ __forceinline__ U64 pack_key(const unsigned dist_bits, const int idx) {
    return (static_cast<U64>(dist_bits) << 32) | static_cast<unsigned int>(idx);
}

__device__ __forceinline__ U64 make_key(
    const float qx, const float qy,
    const float px, const float py,
    const int idx)
{
    return pack_key(distance_bits_2d(qx, qy, px, py), idx);
}

// Standard binary-heap sift-down. Only lane 0 executes heap maintenance, which is efficient
// because once the heap is warm, only a small fraction of points survive the distance gate.
__device__ __forceinline__ void sift_down(U64* heap, int root, const int n) {
    U64 value = heap[root];

    while (true) {
        int child = (root << 1) + 1;
        if (child >= n) {
            break;
        }

        int best_child = child;
        U64 best_value = heap[child];

        const int right = child + 1;
        if (right < n) {
            const U64 right_value = heap[right];
            if (right_value > best_value) {
                best_child = right;
                best_value = right_value;
            }
        }

        if (best_value <= value) {
            break;
        }

        heap[root] = best_value;
        root = best_child;
    }

    heap[root] = value;
}

template <int K>
__device__ __forceinline__ void build_max_heap(U64* heap) {
#pragma unroll 1
    for (int i = (K >> 1) - 1; i >= 0; --i) {
        sift_down(heap, i, K);
    }
}

template <int K>
__device__ __forceinline__ void heap_sort_ascending(U64* heap) {
#pragma unroll 1
    for (int end = K - 1; end > 0; --end) {
        const U64 tmp = heap[0];
        heap[0] = heap[end];
        heap[end] = tmp;
        sift_down(heap, 0, end);
    }
}

// Process a tile that has already been loaded into shared memory.
// The common path is cheap: every lane computes one distance and only compares it against
// the current worst top-k distance. Only surviving candidates are serialized through lane 0
// for exact heap maintenance.
template <int K>
__device__ __forceinline__ U64 process_loaded_tile(
    U64* heap,
    U64 worst,
    const float qx,
    const float qy,
    const float* sh_x,
    const float* sh_y,
    const int tile_base,
    const int tile_valid,
    const int start_local,
    const int lane)
{
    unsigned worst_dist_bits = static_cast<unsigned>(worst >> 32);

#pragma unroll 1
    for (int local_base = start_local; local_base < tile_valid; local_base += kWarpSize) {
        const int local = local_base + lane;

        bool eligible = false;
        U64 key = ~0ull;

        if (local < tile_valid) {
            const unsigned dist_bits = distance_bits_2d(qx, qy, sh_x[local], sh_y[local]);
            // Tie handling is intentionally minimal: the problem states that any tie resolution
            // is acceptable, so the fast gate only checks the distance.
            eligible = (dist_bits < worst_dist_bits);
            if (eligible) {
                key = pack_key(dist_bits, tile_base + local);
            }
        }

        unsigned mask = __ballot_sync(0xFFFFFFFFu, eligible);
        while (mask != 0u) {
            const int src_lane = __ffs(mask) - 1;
            const U64 candidate = __shfl_sync(0xFFFFFFFFu, key, src_lane);

            U64 new_worst = 0ull;
            if (lane == 0) {
                if (candidate < heap[0]) {
                    heap[0] = candidate;
                    sift_down(heap, 0, K);
                }
                new_worst = heap[0];
            }

            // The shuffle is the warp synchronization point for this update path.
            worst = __shfl_sync(0xFFFFFFFFu, new_worst, 0);
            worst_dist_bits = static_cast<unsigned>(worst >> 32);
            mask &= (mask - 1);
        }
    }

    return worst;
}

// Cooperative block-wide load of one data tile into shared memory.
template <int WARPS_PER_BLOCK>
__device__ __forceinline__ void load_tile_soa(
    const float2* __restrict__ data,
    const int data_count,
    const int tile_base,
    float* sh_x,
    float* sh_y)
{
    constexpr int kThreadsPerBlock = WARPS_PER_BLOCK * kWarpSize;

#pragma unroll 1
    for (int local = threadIdx.x; local < kTilePoints; local += kThreadsPerBlock) {
        const int global_idx = tile_base + local;
        if (global_idx < data_count) {
            const float2 p = data[global_idx];
            sh_x[local] = p.x;
            sh_y[local] = p.y;
        }
    }
}

template <int K, int WARPS_PER_BLOCK>
__launch_bounds__(WARPS_PER_BLOCK * 32)
__global__ void knn2d_kernel(
    const float2* __restrict__ query,
    const int query_count,
    const float2* __restrict__ data,
    const int data_count,
    ResultPair* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32,1024].");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of the warp size.");

    __shared__ float sh_x[kTilePoints];
    __shared__ float sh_y[kTilePoints];
    extern __shared__ U64 sh_heaps[];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & (kWarpSize - 1);

    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active = (query_idx < query_count);

    // Warp-private top-k buffer.
    U64* const heap = sh_heaps + static_cast<size_t>(warp_id) * static_cast<size_t>(K);

    // Lane 0 loads the query point once; the warp receives it through shuffles.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xFFFFFFFFu, qx, 0);
    qy = __shfl_sync(0xFFFFFFFFu, qy, 0);

    // First tile: guaranteed to contain the full warm-up set because kTilePoints >= max supported K.
    load_tile_soa<WARPS_PER_BLOCK>(data, data_count, 0, sh_x, sh_y);
    __syncthreads();

    const int first_valid = (data_count < kTilePoints) ? data_count : kTilePoints;
    U64 worst = ~0ull;

    if (active) {
        // Seed the heap with the first K candidates from the first shared-memory tile.
        for (int local = lane; local < K; local += kWarpSize) {
            heap[local] = make_key(qx, qy, sh_x[local], sh_y[local], local);
        }

        // The heap contents were written through shared memory, so we need a warp barrier before
        // lane 0 starts heap construction.
        __syncwarp(0xFFFFFFFFu);
        if (lane == 0) {
            build_max_heap<K>(heap);
        }
        __syncwarp(0xFFFFFFFFu);

        worst = __shfl_sync(0xFFFFFFFFu, (lane == 0) ? heap[0] : 0ull, 0);

        // Continue with the rest of the first tile.
        worst = process_loaded_tile<K>(heap, worst, qx, qy, sh_x, sh_y, 0, first_valid, K, lane);
    }

    // Ensure all warps are done with the current shared-memory tile before reusing it.
    __syncthreads();

    // Remaining tiles.
    for (int tile_base = kTilePoints; tile_base < data_count; tile_base += kTilePoints) {
        load_tile_soa<WARPS_PER_BLOCK>(data, data_count, tile_base, sh_x, sh_y);
        __syncthreads();

        if (active) {
            int tile_valid = data_count - tile_base;
            if (tile_valid > kTilePoints) {
                tile_valid = kTilePoints;
            }
            worst = process_loaded_tile<K>(heap, worst, qx, qy, sh_x, sh_y, tile_base, tile_valid, 0, lane);
        }

        __syncthreads();
    }

    if (active) {
        // Convert the max-heap to ascending order so result[i * k + j] is the j-th nearest neighbor.
        if (lane == 0) {
            heap_sort_ascending<K>(heap);
        }

        // Shared-memory hand-off from lane 0 to the full warp.
        __syncwarp(0xFFFFFFFFu);

        ResultPair* const out = result + static_cast<size_t>(query_idx) * static_cast<size_t>(K);

        // Write std::pair members individually so device code does not depend on std::pair
        // constructors or assignment operators being device-callable.
        for (int local = lane; local < K; local += kWarpSize) {
            const U64 key = heap[local];
            out[local].first  = static_cast<int>(static_cast<unsigned int>(key));
            out[local].second = __uint_as_float(static_cast<unsigned int>(key >> 32));
        }
    }
}

template <int K, int WARPS_PER_BLOCK>
constexpr size_t dynamic_shared_bytes() {
    return static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(K) * sizeof(U64);
}

template <int K, int WARPS_PER_BLOCK>
constexpr size_t total_shared_bytes() {
    return kTileSharedBytes + dynamic_shared_bytes<K, WARPS_PER_BLOCK>();
}

template <int K, int WARPS_PER_BLOCK>
inline void launch_knn_specialized(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    ResultPair* result)
{
    constexpr int kThreadsPerBlock = WARPS_PER_BLOCK * kWarpSize;
    constexpr size_t kDynamicShared = dynamic_shared_bytes<K, WARPS_PER_BLOCK>();

    // Ask the runtime for the shared-memory-heavy configuration needed by this specialization.
    (void)cudaFuncSetAttribute(
        knn2d_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    (void)cudaFuncSetAttribute(
        knn2d_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynamicShared));

    const dim3 block(kThreadsPerBlock);
    const dim3 grid(ceil_div_int(query_count, WARPS_PER_BLOCK));

    knn2d_kernel<K, WARPS_PER_BLOCK>
        <<<grid, block, kDynamicShared>>>(query, query_count, data, data_count, result);
}

// Runtime policy for the number of warps per block.
//
// Larger blocks improve data-tile reuse because more queries consume the same shared-memory tile.
// Smaller blocks produce more grid blocks, which matters when the query batch is only in the
// low-thousands. The heuristic below keeps roughly 75% of the SM count covered by grid blocks
// and otherwise maximizes reuse.
template <int K>
inline int choose_warps_per_block(
    const int query_count,
    const int sm_count,
    const int max_optin_shared)
{
    const int min_grid_blocks = (3 * sm_count + 3) / 4;  // ceil(0.75 * sm_count)
    const size_t max_shared = static_cast<size_t>(max_optin_shared);

    if (total_shared_bytes<K, 32>() <= max_shared && ceil_div_int(query_count, 32) >= min_grid_blocks) return 32;
    if (total_shared_bytes<K, 16>() <= max_shared && ceil_div_int(query_count, 16) >= min_grid_blocks) return 16;
    if (total_shared_bytes<K,  8>() <= max_shared && ceil_div_int(query_count,  8) >= min_grid_blocks) return 8;
    if (total_shared_bytes<K,  4>() <= max_shared && ceil_div_int(query_count,  4) >= min_grid_blocks) return 4;
    if (total_shared_bytes<K,  2>() <= max_shared && ceil_div_int(query_count,  2) >= min_grid_blocks) return 2;
    return 1;
}

// k is runtime but only takes six values, so we fully specialize on K. That lets the compiler
// constant-fold heap sizes, warm-up loops, and output loops.
template <int K>
inline void launch_knn_for_k(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    ResultPair* result,
    const int sm_count,
    const int max_optin_shared)
{
    switch (choose_warps_per_block<K>(query_count, sm_count, max_optin_shared)) {
        case 32: launch_knn_specialized<K, 32>(query, query_count, data, data_count, result); break;
        case 16: launch_knn_specialized<K, 16>(query, query_count, data, data_count, result); break;
        case 8:  launch_knn_specialized<K,  8>(query, query_count, data, data_count, result); break;
        case 4:  launch_knn_specialized<K,  4>(query, query_count, data, data_count, result); break;
        case 2:  launch_knn_specialized<K,  2>(query, query_count, data, data_count, result); break;
        default: launch_knn_specialized<K,  1>(query, query_count, data, data_count, result); break;
    }
}

}  // namespace

void run_knn(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    std::pair<int, float>* result,
    const int k)
{
    if (query_count <= 0) {
        return;
    }

    // Conservative defaults are used only if the attribute queries fail; on the target A100/H100
    // class GPUs, these calls succeed and provide the actual device limits.
    int device = 0;
    int sm_count = 128;
    int max_optin_shared = 48 * 1024;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    (void)cudaDeviceGetAttribute(&max_optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    switch (k) {
        case 32:   launch_knn_for_k<32>(query, query_count, data, data_count, result, sm_count, max_optin_shared); break;
        case 64:   launch_knn_for_k<64>(query, query_count, data, data_count, result, sm_count, max_optin_shared); break;
        case 128:  launch_knn_for_k<128>(query, query_count, data, data_count, result, sm_count, max_optin_shared); break;
        case 256:  launch_knn_for_k<256>(query, query_count, data, data_count, result, sm_count, max_optin_shared); break;
        case 512:  launch_knn_for_k<512>(query, query_count, data, data_count, result, sm_count, max_optin_shared); break;
        case 1024: launch_knn_for_k<1024>(query, query_count, data, data_count, result, sm_count, max_optin_shared); break;
        default:
            // Inputs are specified to be valid; this path exists only as a defensive no-op.
            break;
    }
}