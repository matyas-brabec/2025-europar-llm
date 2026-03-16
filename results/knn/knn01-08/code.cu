#include <cuda_runtime.h>
#include <utility>
#include <cstdint>
#include <cstddef>

namespace {

// Exact k-NN for 2D points, optimized for the stated regime:
//
// - One warp owns one query.
// - One 256-thread CTA therefore processes 8 queries at once.
// - Data are streamed through a 1024-point shared-memory tile (8 KiB), so each
//   global data load is reused by 8 different queries inside the CTA.
// - Each warp keeps an exact top-K max-heap in shared memory.
// - K is specialized at compile time for all valid values {32,64,128,256,512,1024},
//   which removes runtime branches from the inner heap logic.
// - The tile size is chosen to be exactly the largest supported K. This lets us
//   initialize the heap from the first K points in O(K): lanes write the first K
//   packed keys directly, and lane 0 heapifies once.
// - Candidate filtering uses a snapshot of the current heap root. The root can
//   only decrease as the scan proceeds, so snapshot filtering may admit extra
//   candidates but cannot miss a true top-K element. Lane 0 rechecks against the
//   live heap root before replacing it, preserving exactness.
// - No additional device memory is allocated; only dynamic shared memory is used.
//
// Packed heap key:
//   high 32 bits: IEEE-754 bits of the non-negative squared distance
//   low  32 bits: point index (arbitrary tie-break, allowed by the problem)
//
// For non-negative floats, the bit pattern is monotone with the numeric value,
// so unsigned integer comparison gives the correct distance ordering.

using ResultPair = std::pair<int, float>;
using PackedKey  = unsigned long long;

constexpr int      WARP_SIZE         = 32;
constexpr int      BLOCK_THREADS     = 256;                // 8 warps / CTA
constexpr int      WARPS_PER_BLOCK   = BLOCK_THREADS / WARP_SIZE;
constexpr int      DATA_TILE_POINTS  = 1024;               // 8 KiB of float2
constexpr int      LOADS_PER_THREAD  = DATA_TILE_POINTS / BLOCK_THREADS; // 4
constexpr int      DATA_TILE_CHUNKS  = DATA_TILE_POINTS / WARP_SIZE;     // 32
constexpr unsigned FULL_MASK         = 0xFFFFFFFFu;
constexpr PackedKey PACKED_INF       = ~PackedKey{0};

static_assert(BLOCK_THREADS % WARP_SIZE == 0, "BLOCK_THREADS must be a multiple of warp size.");
static_assert(DATA_TILE_POINTS % BLOCK_THREADS == 0, "Tile size must divide evenly across the CTA.");
static_assert(DATA_TILE_POINTS % WARP_SIZE == 0, "Tile size must divide evenly across a warp.");
static_assert(sizeof(float2) == 8, "This kernel assumes float2 is 8 bytes.");

__device__ __forceinline__ PackedKey pack_key(const float dist, const int idx) {
    return (static_cast<PackedKey>(__float_as_uint(dist)) << 32) |
           static_cast<PackedKey>(static_cast<unsigned int>(idx));
}

__device__ __forceinline__ float unpack_dist(const PackedKey key) {
    return __uint_as_float(static_cast<unsigned int>(key >> 32));
}

__device__ __forceinline__ int unpack_idx(const PackedKey key) {
    return static_cast<int>(static_cast<unsigned int>(key));
}

// Explicit 64-bit shuffle helper. Using two 32-bit shuffles is portable across
// CUDA toolchains and avoids relying on a 64-bit overload being present.
__device__ __forceinline__ PackedKey shfl_packed_key(const PackedKey value, const int src_lane) {
    const unsigned lo = __shfl_sync(FULL_MASK, static_cast<unsigned>(value),       src_lane);
    const unsigned hi = __shfl_sync(FULL_MASK, static_cast<unsigned>(value >> 32), src_lane);
    return (static_cast<PackedKey>(hi) << 32) | static_cast<PackedKey>(lo);
}

__device__ __forceinline__ float squared_l2(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return __fmaf_rn(dx, dx, dy * dy);
}

// Build a fixed-size max-heap in-place. Only lane 0 of a warp calls this, so a
// simple serial heapify is best.
template <int K>
__device__ __forceinline__ void build_max_heap(PackedKey* heap) {
    for (int parent = (K >> 1) - 1; parent >= 0; --parent) {
        const PackedKey v = heap[parent];
        int root = parent;

        while (true) {
            const int left = (root << 1) + 1;
            if (left >= K) break;

            const int right = left + 1;
            int max_child = left;
            if (right < K && heap[right] > heap[left]) {
                max_child = right;
            }

            if (heap[max_child] <= v) break;
            heap[root] = heap[max_child];
            root = max_child;
        }

        heap[root] = v;
    }
}

// Replace the root if the new key is smaller. Returns whether the heap changed.
// Again, only lane 0 calls this.
template <int K>
__device__ __forceinline__ bool replace_root_if_smaller(PackedKey* heap, const PackedKey key) {
    if (key >= heap[0]) {
        return false;
    }

    int root = 0;
    while (true) {
        const int left = (root << 1) + 1;
        if (left >= K) break;

        const int right = left + 1;
        int max_child = left;
        if (right < K && heap[right] > heap[left]) {
            max_child = right;
        }

        if (heap[max_child] <= key) break;
        heap[root] = heap[max_child];
        root = max_child;
    }

    heap[root] = key;
    return true;
}

// Heapsort turns the max-heap into ascending order, which matches the required
// "j-th nearest neighbor" result layout.
template <int K>
__device__ __forceinline__ void heap_sort_ascending(PackedKey* heap) {
    for (int end = K - 1; end > 0; --end) {
        const PackedKey max_value = heap[0];
        const PackedKey tail      = heap[end];

        int root = 0;
        while (true) {
            const int left = (root << 1) + 1;
            if (left >= end) break;

            const int right = left + 1;
            int max_child = left;
            if (right < end && heap[right] > heap[left]) {
                max_child = right;
            }

            if (heap[max_child] <= tail) break;
            heap[root] = heap[max_child];
            root = max_child;
        }

        heap[root] = tail;
        heap[end]  = max_value;
    }
}

// Coalesced CTA-wide load of one data tile into shared memory.
__device__ __forceinline__ void load_data_tile(
    const float2* __restrict__ data,
    const int tile_base,
    const int tile_count,
    float2* __restrict__ tile,
    const int tid)
{
#pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; ++i) {
        const int off = tid + i * BLOCK_THREADS;
        if (off < tile_count) {
            tile[off] = data[tile_base + off];
        }
    }
}

// Process a tile for one query-warp.
//
// root_key_lane0 is meaningful only on lane 0; other lanes carry a dummy copy.
// Every chunk uses a broadcast snapshot of the root to prefilter candidates.
// Lane 0 then rechecks each candidate against the live root before replacing.
template <int K>
__device__ __forceinline__ PackedKey process_tile_for_query(
    const float qx,
    const float qy,
    const float2* __restrict__ tile,
    const int tile_base,
    const int tile_count,
    const int start_chunk,
    PackedKey* __restrict__ heap,
    const int lane,
    PackedKey root_key_lane0)
{
    for (int c = start_chunk; c < DATA_TILE_CHUNKS; ++c) {
        const int chunk_base = c * WARP_SIZE;
        if (chunk_base >= tile_count) {
            break;
        }

        const PackedKey root_snapshot = shfl_packed_key(root_key_lane0, 0);
        const int off = chunk_base + lane;

        PackedKey key = PACKED_INF;
        if (off < tile_count) {
            const float2 p = tile[off];
            key = pack_key(squared_l2(qx, qy, p), tile_base + off);
        }

        unsigned candidate_mask = __ballot_sync(FULL_MASK, key < root_snapshot);
        while (candidate_mask) {
            const int src_lane = __ffs(candidate_mask) - 1;
            const PackedKey candidate_key = shfl_packed_key(key, src_lane);

            if (lane == 0) {
                if (replace_root_if_smaller<K>(heap, candidate_key)) {
                    root_key_lane0 = heap[0];
                }
            }

            candidate_mask &= candidate_mask - 1;
        }
    }

    return root_key_lane0;
}

template <int K>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void knn_kernel(
    const float2* __restrict__ query,
    const int query_count,
    const float2* __restrict__ data,
    const int data_count,
    ResultPair* __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K % WARP_SIZE == 0, "K must be a multiple of warp size.");
    static_assert(K <= DATA_TILE_POINTS, "The fixed data tile must hold the initial K points.");

    // Shared layout:
    //   [ WARPS_PER_BLOCK * K packed heap entries ][ DATA_TILE_POINTS float2 tile ]
    extern __shared__ PackedKey shared_mem[];
    PackedKey* const heaps = shared_mem;
    float2* const tile = reinterpret_cast<float2*>(heaps + WARPS_PER_BLOCK * K);

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane    = tid & (WARP_SIZE - 1);

    const int query_idx   = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active_query = (query_idx < query_count);

    PackedKey* const my_heap = heaps + warp_id * K;

    // Load the query point once per warp and broadcast to the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active_query) {
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(FULL_MASK, qx, 0);
        qy = __shfl_sync(FULL_MASK, qy, 0);
    }

    // First tile: this always contains the first K points, because K <= 1024
    // and data_count >= K. We exploit that to initialize the heap in O(K).
    const int first_tile_count = (data_count < DATA_TILE_POINTS) ? data_count : DATA_TILE_POINTS;
    load_data_tile(data, 0, first_tile_count, tile, tid);
    __syncthreads();

    PackedKey root_key = PACKED_INF;

    if (active_query) {
        constexpr int INIT_CHUNKS = K / WARP_SIZE;

#pragma unroll
        for (int c = 0; c < INIT_CHUNKS; ++c) {
            const int off = c * WARP_SIZE + lane;
            const float2 p = tile[off];
            my_heap[off] = pack_key(squared_l2(qx, qy, p), off);
        }

        // Make the initial K keys visible to lane 0 before heapify.
        __syncwarp(FULL_MASK);

        if (lane == 0) {
            build_max_heap<K>(my_heap);
            root_key = my_heap[0];
        }

        // If K < 1024, the first tile still has trailing points that must be
        // scanned with the now-initialized heap.
        if (K < DATA_TILE_POINTS) {
            root_key = process_tile_for_query<K>(
                qx, qy, tile, 0, first_tile_count, INIT_CHUNKS, my_heap, lane, root_key);
        }
    }

    __syncthreads();

    // Remaining tiles.
    for (int tile_base = DATA_TILE_POINTS; tile_base < data_count; tile_base += DATA_TILE_POINTS) {
        int tile_count = data_count - tile_base;
        if (tile_count > DATA_TILE_POINTS) {
            tile_count = DATA_TILE_POINTS;
        }

        load_data_tile(data, tile_base, tile_count, tile, tid);
        __syncthreads();

        if (active_query) {
            root_key = process_tile_for_query<K>(
                qx, qy, tile, tile_base, tile_count, 0, my_heap, lane, root_key);
        }

        __syncthreads();
    }

    // Sort the final top-K set into ascending order and write out results.
    if (active_query) {
        if (lane == 0) {
            heap_sort_ascending<K>(my_heap);
        }

        // Shared-memory visibility for the warp before parallel stores.
        __syncwarp(FULL_MASK);

        const std::size_t out_base = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);
        for (int i = lane; i < K; i += WARP_SIZE) {
            const PackedKey key = my_heap[i];
            result[out_base + static_cast<std::size_t>(i)].first  = unpack_idx(key);
            result[out_base + static_cast<std::size_t>(i)].second = unpack_dist(key);
        }
    }
}

template <int K>
void launch_knn_kernel(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    ResultPair* result)
{
    // Dynamic shared memory:
    //   8 query-heaps * K packed keys + 1024-point data tile
    //
    // 256 threads / CTA is a good balance on A100/H100:
    // it gives 8-way data reuse per tile while still allowing 2 CTAs/SM on
    // A100 even for K=1024 (and 3 CTAs/SM on H100).
    constexpr std::size_t required_shared_bytes =
        static_cast<std::size_t>(WARPS_PER_BLOCK) * static_cast<std::size_t>(K) * sizeof(PackedKey) +
        static_cast<std::size_t>(DATA_TILE_POINTS) * sizeof(float2);

    if (required_shared_bytes > (48u * 1024u)) {
        // K=1024 needs opt-in shared memory on modern data-center GPUs.
        cudaFuncSetAttribute(
            knn_kernel<K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(required_shared_bytes));
        cudaFuncSetAttribute(
            knn_kernel<K>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100);
    }

    const int grid_x = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Standard CUDA semantics: the launch is asynchronous with respect to the host.
    knn_kernel<K><<<grid_x, BLOCK_THREADS, required_shared_bytes>>>(
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
        case 32:   launch_knn_kernel<32>(query, query_count, data, data_count, result);   return;
        case 64:   launch_knn_kernel<64>(query, query_count, data, data_count, result);   return;
        case 128:  launch_knn_kernel<128>(query, query_count, data, data_count, result);  return;
        case 256:  launch_knn_kernel<256>(query, query_count, data, data_count, result);  return;
        case 512:  launch_knn_kernel<512>(query, query_count, data, data_count, result);  return;
        case 1024: launch_knn_kernel<1024>(query, query_count, data, data_count, result); return;
        default:
            // Per the problem statement, k is always valid. Keep the default
            // branch empty so the interface remains total.
            return;
    }
}