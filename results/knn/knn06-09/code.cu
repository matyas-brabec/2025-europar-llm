#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <type_traits>

namespace {

// Warp-specialized 2D k-NN.
// Design choices tuned for A100/H100-class GPUs:
//
// * One warp computes one query.
// * The resident top-K list stays in registers, distributed across the 32 lanes.
// * Each warp owns:
//     - a K-entry candidate buffer in shared memory,
//     - a shared mirror of the current top-K used only during sort/merge.
// * The whole block streams the database through shared memory in 1024-point tiles.
// * Candidate insertion uses a warp-aggregated atomicAdd as requested.
// * Merging avoids a 2K temporary buffer:
//     1) sort the candidate buffer in shared memory with a warp-local bitonic network;
//     2) merge the sorted candidates with the already-sorted top-K using merge-path.
//
// No extra device memory is allocated.

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int BATCH_POINTS = 1024;
constexpr unsigned FULL_MASK = 0xffffffffu;

// POD mirror of the required output payload (index, squared distance).
// We use a trivial POD internally because raw cudaMalloc memory does not contain
// constructed std::pair objects. The ABI assumption is guarded by sizeof().
struct Neighbor {
    int   first;   // data-point index
    float second;  // squared Euclidean distance
};

static_assert(std::is_trivially_copyable<Neighbor>::value, "Neighbor must be trivially copyable.");
static_assert(sizeof(Neighbor) == sizeof(std::pair<int, float>),
              "std::pair<int,float> layout is not compatible with Neighbor on this toolchain.");
static_assert(BATCH_POINTS % WARP_SIZE == 0, "BATCH_POINTS must be a multiple of warp size.");

__device__ __forceinline__ Neighbor make_sentinel() {
    Neighbor n;
    n.first = INT_MAX;
    n.second = CUDART_INF_F;
    return n;
}

__device__ __forceinline__ bool neighbor_less(const Neighbor &a, const Neighbor &b) {
    return (a.second < b.second) || ((a.second == b.second) && (a.first < b.first));
}

__device__ __forceinline__ float squared_l2(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dy, dy, dx * dx);
}

// In-place bitonic sort of a K-element array in shared memory, owned by one warp.
// K is a power of two in [32, 1024].
template <int K>
__device__ __forceinline__ void warp_bitonic_sort_shared(Neighbor *arr, const int lane) {
    static_assert(K >= 32 && K <= 1024, "Unsupported K.");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");

    #pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < K; i += WARP_SIZE) {
                const int j = i ^ stride;
                if (j > i) {
                    const Neighbor a = arr[i];
                    const Neighbor b = arr[j];
                    const bool up = ((i & size) == 0);
                    const bool do_swap = up ? neighbor_less(b, a) : neighbor_less(a, b);
                    if (do_swap) {
                        arr[i] = b;
                        arr[j] = a;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

// The resident top-K is stored as contiguous chunks:
//   lane 0 owns [0, ITEMS_PER_THREAD),
//   lane 1 owns [ITEMS_PER_THREAD, 2*ITEMS_PER_THREAD), ...
// Therefore lane 31 owns the global K-th element.
template <int K>
__device__ __forceinline__ float kth_distance(const Neighbor (&best)[K / WARP_SIZE]) {
    return __shfl_sync(FULL_MASK, best[(K / WARP_SIZE) - 1].second, WARP_SIZE - 1);
}

template <int K>
__device__ __forceinline__ void sort_register_best(
    Neighbor (&best)[K / WARP_SIZE],
    Neighbor *best_sh,
    const int lane)
{
    constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        best_sh[lane * ITEMS_PER_THREAD + t] = best[t];
    }
    __syncwarp(FULL_MASK);

    warp_bitonic_sort_shared<K>(best_sh, lane);

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        best[t] = best_sh[lane * ITEMS_PER_THREAD + t];
    }
    __syncwarp(FULL_MASK);
}

// Merge-path co-rank for two sorted K-element arrays.
// Returns how many elements must be taken from A among the first p merged outputs.
template <int K>
__device__ __forceinline__ int co_rank(const int p, const Neighbor *a, const Neighbor *b) {
    int low  = 0;
    int high = p;

    while (low <= high) {
        const int i = (low + high) >> 1;
        const int j = p - i;

        if (i > 0 && j < K && neighbor_less(b[j], a[i - 1])) {
            high = i - 1;  // took too many from A
        } else if (j > 0 && i < K && !neighbor_less(b[j - 1], a[i])) {
            low = i + 1;   // took too few from A
        } else {
            return i;
        }
    }

    return low;
}

// Merge the shared candidate buffer into the resident top-K.
// The current top-K is first mirrored into shared memory because merge-path needs
// arbitrary indexed reads. The resident state after the merge stays in registers.
template <int K>
__device__ __forceinline__ void merge_candidates_into_best(
    Neighbor (&best)[K / WARP_SIZE],
    Neighbor *best_sh,
    Neighbor *cand_sh,
    int *cand_count,
    const int lane)
{
    int count = 0;
    if (lane == 0) {
        count = *cand_count;
    }
    count = __shfl_sync(FULL_MASK, count, 0);
    if (count == 0) {
        return;
    }

    const Neighbor sentinel = make_sentinel();

    // Pad to K with +inf so that both inputs to merge-path always have length K.
    for (int i = lane; i < K; i += WARP_SIZE) {
        if (i >= count) {
            cand_sh[i] = sentinel;
        }
    }
    __syncwarp(FULL_MASK);

    // Sort candidates in-place.
    warp_bitonic_sort_shared<K>(cand_sh, lane);

    // Publish the current sorted top-K to shared memory.
    constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        best_sh[lane * ITEMS_PER_THREAD + t] = best[t];
    }
    __syncwarp(FULL_MASK);

    // Each lane owns one contiguous slice of the merged output.
    int i = co_rank<K>(lane * ITEMS_PER_THREAD, best_sh, cand_sh);
    int j = lane * ITEMS_PER_THREAD - i;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        Neighbor a = sentinel;
        Neighbor b = sentinel;
        if (i < K) a = best_sh[i];
        if (j < K) b = cand_sh[j];

        const bool take_a = !neighbor_less(b, a);
        best[t] = take_a ? a : b;
        i += static_cast<int>(take_a);
        j += static_cast<int>(!take_a);
    }

    __syncwarp(FULL_MASK);
    if (lane == 0) {
        *cand_count = 0;
    }
    __syncwarp(FULL_MASK);
}

// Process a [begin, end) subrange of the current shared-memory tile.
// Each iteration handles one warp-wide chunk of up to 32 points.
template <int K>
__device__ __forceinline__ void process_batch_points(
    const float2 *batch,
    const int begin,
    const int end,
    const int global_base,
    const float qx,
    const float qy,
    Neighbor (&best)[K / WARP_SIZE],
    Neighbor *best_sh,
    Neighbor *cand_sh,
    int *cand_count,
    const int lane,
    float &max_distance)
{
    for (int tile = begin; tile < end; tile += WARP_SIZE) {
        const int valid = ((end - tile) < WARP_SIZE) ? (end - tile) : WARP_SIZE;

        Neighbor candidate = make_sentinel();
        bool pass = false;

        if (lane < valid) {
            const float2 p = batch[tile + lane];
            candidate.first = global_base + tile + lane;
            candidate.second = squared_l2(qx, qy, p);
            pass = candidate.second < max_distance;
        }

        unsigned mask = __ballot_sync(FULL_MASK, pass);
        int n = __popc(mask);

        // If this chunk would overflow the fixed-size candidate buffer,
        // flush the existing buffer first, then re-test the chunk against the
        // tighter updated threshold (max_distance can only decrease).
        int count = 0;
        if (lane == 0) {
            count = *cand_count;
        }
        count = __shfl_sync(FULL_MASK, count, 0);

        if (count + n > K) {
            merge_candidates_into_best<K>(best, best_sh, cand_sh, cand_count, lane);
            max_distance = kth_distance<K>(best);

            pass = (lane < valid) && (candidate.second < max_distance);
            mask = __ballot_sync(FULL_MASK, pass);
            n = __popc(mask);
        }

        if (n != 0) {
            const int leader = __ffs(mask) - 1;
            int base = 0;

            // Warp-aggregated atomicAdd: one atomic per warp-chunk, as requested.
            if (lane == leader) {
                base = atomicAdd(cand_count, n);
            }
            base = __shfl_sync(FULL_MASK, base, leader);

            const unsigned prior = mask & ((lane == 0) ? 0u : ((1u << lane) - 1u));
            if (pass) {
                cand_sh[base + __popc(prior)] = candidate;
            }

            __syncwarp(FULL_MASK);

            // If the buffer became exactly full, merge immediately.
            if (base + n == K) {
                merge_candidates_into_best<K>(best, best_sh, cand_sh, cand_count, lane);
                max_distance = kth_distance<K>(best);
            }
        } else {
            __syncwarp(FULL_MASK);
        }
    }
}

template <int K>
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void knn_kernel(
    const float2 * __restrict__ query,
    const int query_count,
    const float2 * __restrict__ data,
    const int data_count,
    Neighbor * __restrict__ result)
{
    static_assert(K >= 32 && K <= 1024, "Unsupported K.");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K <= BATCH_POINTS, "BATCH_POINTS must cover the initial K seed points.");

    extern __shared__ __align__(16) unsigned char smem_raw[];

    // Shared-memory layout:
    // [0, BATCH_POINTS)                          : cached data tile
    // [per warp: K entries]                     : shared mirror of current top-K
    // [per warp: K entries]                     : candidate buffer
    // [per warp: 1 int]                         : candidate count
    float2   *batch_data      = reinterpret_cast<float2*>(smem_raw);
    Neighbor *best_all        = reinterpret_cast<Neighbor*>(batch_data + BATCH_POINTS);
    Neighbor *cand_all        = best_all + WARPS_PER_BLOCK * K;
    int      *cand_count_all  = reinterpret_cast<int*>(cand_all + WARPS_PER_BLOCK * K);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int query_idx = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active   = (query_idx < query_count);

    Neighbor *best_sh   = best_all + warp_id * K;
    Neighbor *cand_sh   = cand_all + warp_id * K;
    int      *cand_count = cand_count_all + warp_id;

    if (lane == 0) {
        *cand_count = 0;
    }
    __syncthreads();

    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    Neighbor best[K / WARP_SIZE];
    float max_distance = CUDART_INF_F;
    bool initialized = false;

    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_POINTS) {
        int loaded = data_count - batch_start;
        if (loaded > BATCH_POINTS) {
            loaded = BATCH_POINTS;
        }

        // Cooperative block-wide load of the next data tile.
        for (int i = tid; i < loaded; i += THREADS_PER_BLOCK) {
            batch_data[i] = data[batch_start + i];
        }
        __syncthreads();

        if (active) {
            int begin = 0;

            // Seed the resident top-K from the first K data points.
            // Because BATCH_POINTS >= K and data_count >= K, this always fits in the first tile.
            if (!initialized) {
                constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;

                #pragma unroll
                for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
                    const int pos = lane * ITEMS_PER_THREAD + t;
                    best[t].first = pos;
                    best[t].second = squared_l2(qx, qy, batch_data[pos]);
                }

                sort_register_best<K>(best, best_sh, lane);
                max_distance = kth_distance<K>(best);
                initialized = true;
                begin = K;
            }

            process_batch_points<K>(
                batch_data,
                begin,
                loaded,
                batch_start,
                qx,
                qy,
                best,
                best_sh,
                cand_sh,
                cand_count,
                lane,
                max_distance);
        }

        __syncthreads();
    }

    if (active) {
        int remaining = 0;
        if (lane == 0) {
            remaining = *cand_count;
        }
        remaining = __shfl_sync(FULL_MASK, remaining, 0);
        if (remaining != 0) {
            merge_candidates_into_best<K>(best, best_sh, cand_sh, cand_count, lane);
        }

        Neighbor *out = result + static_cast<size_t>(query_idx) * K;
        constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;

        #pragma unroll
        for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
            out[lane * ITEMS_PER_THREAD + t] = best[t];
        }
    }
}

template <int K>
constexpr size_t shared_bytes_for_kernel() {
    return
        BATCH_POINTS * sizeof(float2) +
        2 * WARPS_PER_BLOCK * K * sizeof(Neighbor) +
        WARPS_PER_BLOCK * sizeof(int);
}

// K only takes six legal values, so dispatching to a fully specialized kernel
// gives the compiler exact register array sizes and fixed loop trip counts.
template <int K>
inline void launch_knn_template(
    const float2 *query,
    const int query_count,
    const float2 *data,
    const int data_count,
    std::pair<int, float> *result)
{
    constexpr size_t smem_bytes = shared_bytes_for_kernel<K>();

    // Opt in to the required amount of dynamic shared memory and bias the carveout
    // toward shared memory because this kernel is explicitly shared-memory-heavy.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    knn_kernel<K><<<grid, block, smem_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<Neighbor*>(result));
}

} // namespace

void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k)
{
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:   launch_knn_template<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_template<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_template<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_template<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_template<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_template<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Inputs are stated to be valid; this is just a defensive fallback.
    }
}