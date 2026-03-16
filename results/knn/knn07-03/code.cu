#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace knn_cuda_internal {

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// Largest universal tile that keeps all chosen high-reuse CTA configurations
// below the opt-in per-block shared-memory budget of modern datacenter GPUs
// such as A100/H100, while noticeably reducing block-wide barrier frequency.
constexpr int DATA_TILE_POINTS = 2048;
constexpr std::size_t MAX_TARGET_SHARED_BYTES = 163840;

struct alignas(8) PairDI {
    float dist;
    int idx;
};

static_assert(sizeof(PairDI) == 8, "PairDI must remain compact.");
static_assert((DATA_TILE_POINTS % WARP_SIZE) == 0, "Tile size must be a multiple of the warp size.");

__device__ __forceinline__ PairDI make_pairdi(const float dist, const int idx) {
    PairDI p;
    p.dist = dist;
    p.idx = idx;
    return p;
}

__device__ __forceinline__ PairDI sentinel_pair() {
    return make_pairdi(CUDART_INF_F, INT_MAX);
}

// Tie handling is arbitrary per the prompt; we use index as a deterministic
// secondary key so the sorting network sees a total order.
__device__ __forceinline__ bool pair_less(const PairDI a, const PairDI b) {
    return (a.dist < b.dist) || ((a.dist == b.dist) && (a.idx < b.idx));
}

__device__ __forceinline__ PairDI pair_min(const PairDI a, const PairDI b) {
    return pair_less(b, a) ? b : a;
}

// In-place bitonic sort over a warp-private shared-memory array of length K.
// Lane t owns logical positions t, t + 32, t + 64, ...
template <int K>
__device__ __forceinline__ void bitonic_sort_shared(PairDI* const arr, const int lane) {
    for (int stage = 2; stage <= K; stage <<= 1) {
        for (int stride = stage >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int i = lane; i < K; i += WARP_SIZE) {
                const int j = i ^ stride;
                if (j > i) {
                    const PairDI a = arr[i];
                    const PairDI b = arr[j];
                    const bool up = ((i & stage) == 0);
                    const bool do_swap = up ? pair_less(b, a) : pair_less(a, b);
                    if (do_swap) {
                        arr[i] = b;
                        arr[j] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Flush the shared candidate buffer into the warp-private top-k.
//
// This follows the requested merge procedure exactly:
// 1. sort the shared candidate buffer ascending with bitonic sort;
// 2. build the bitonic "small half" using
//      merged[i] = min(buffer[i], result[k - i - 1]);
// 3. bitonic-sort that merged sequence ascending.
template <int K>
__device__ __forceinline__ void flush_buffer(
    PairDI* const cand,
    PairDI* const scratch,
    int* const count,
    float (&result_dist)[K / WARP_SIZE],
    int (&result_idx)[K / WARP_SIZE],
    const int lane,
    float& max_distance)
{
    constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;

    __syncwarp();

    int c = 0;
    if (lane == 0) {
        c = *count;
    }
    c = __shfl_sync(FULL_MASK, c, 0);

    if (c == 0) {
        return;
    }

    const PairDI inf = sentinel_pair();

    // Pad the inactive tail with +inf sentinels so the full K-element buffer can
    // always be sorted by the fixed-size bitonic network.
#pragma unroll
    for (int i = lane; i < K; i += WARP_SIZE) {
        if (i >= c) {
            cand[i] = inf;
        }
    }
    __syncwarp();

    bitonic_sort_shared<K>(cand, lane);

    // Materialize the warp-private result into shared scratch so the merge can
    // read it in reverse order while still keeping the live top-k private between flushes.
#pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        const int pos = t * WARP_SIZE + lane;
        scratch[pos] = make_pairdi(result_dist[t], result_idx[t]);
    }
    __syncwarp();

    // Build the bitonic "small half".
#pragma unroll
    for (int i = lane; i < K; i += WARP_SIZE) {
        cand[i] = pair_min(cand[i], scratch[K - 1 - i]);
    }
    __syncwarp();

    bitonic_sort_shared<K>(cand, lane);

    // Reload the updated top-k back into the warp-private copy.
#pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        const int pos = t * WARP_SIZE + lane;
        const PairDI v = cand[pos];
        result_dist[t] = v.dist;
        result_idx[t] = v.idx;
    }

    float last = 0.0f;
    if (lane == (WARP_SIZE - 1)) {
        last = result_dist[ITEMS_PER_THREAD - 1];
    }
    max_distance = __shfl_sync(FULL_MASK, last, WARP_SIZE - 1);

    if (lane == 0) {
        *count = 0;
    }
    __syncwarp();
}

// One warp handles one query. The block stages a tile of `data` into shared memory
// so all query warps in the block reuse the same global loads. No extra device memory
// is allocated: the live top-k is private to the warp, while the candidate buffer and
// merge scratch are warp-private slices of shared memory.
template <int K, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS)
void knn_kernel(
    const float2* __restrict__ query,
    const int query_count,
    const float2* __restrict__ data,
    const int data_count,
    std::pair<int, float>* __restrict__ result_out)
{
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");
    static_assert((K % WARP_SIZE) == 0, "K must be divisible by warp size.");
    static_assert((BLOCK_THREADS % WARP_SIZE) == 0, "Block size must be a multiple of warp size.");

    constexpr int ITEMS_PER_THREAD = K / WARP_SIZE;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;

    extern __shared__ unsigned char smem_raw[];

    // Shared layout:
    //   [cached data tile][candidate buffers][merge scratch][candidate counts]
    float2* const s_data = reinterpret_cast<float2*>(smem_raw);
    PairDI* const s_cand = reinterpret_cast<PairDI*>(s_data + DATA_TILE_POINTS);
    PairDI* const s_scratch = s_cand + WARPS_PER_BLOCK * K;
    int* const s_count = reinterpret_cast<int*>(s_scratch + WARPS_PER_BLOCK * K);

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x >> 5;
    const int query_idx = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool valid_query = (query_idx < query_count);

    PairDI* const cand = s_cand + warp_id * K;
    PairDI* const scratch = s_scratch + warp_id * K;
    int* const count = s_count + warp_id;

    // Warp-private top-k distributed across the 32 lanes.
    float result_dist[ITEMS_PER_THREAD];
    int result_idx[ITEMS_PER_THREAD];
#pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        result_dist[t] = CUDART_INF_F;
        result_idx[t] = INT_MAX;
    }

    if (lane == 0) {
        *count = 0;
    }

    // Broadcast the query point once and keep it in registers across the whole scan.
    float qx = 0.0f;
    float qy = 0.0f;
    if (valid_query && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Replicated across the warp; updated whenever the private top-k changes.
    float max_distance = CUDART_INF_F;

    __syncthreads();

    for (int tile_base = 0; tile_base < data_count; tile_base += DATA_TILE_POINTS) {
        int tile_count = data_count - tile_base;
        if (tile_count > DATA_TILE_POINTS) {
            tile_count = DATA_TILE_POINTS;
        }

        for (int i = threadIdx.x; i < tile_count; i += BLOCK_THREADS) {
            s_data[i] = data[tile_base + i];
        }
        __syncthreads();

        if (valid_query) {
            // Process the tile in warp-sized micro-batches. This lets the warp:
            //  - compute 32 distances in parallel,
            //  - ballot surviving candidates,
            //  - compact them with one warp-aggregated atomicAdd.
            for (int seg = 0; seg < tile_count; seg += WARP_SIZE) {
                const int local_i = seg + lane;
                const bool valid_point = (local_i < tile_count);

                float dist = 0.0f;
                const int idx = tile_base + local_i;

                if (valid_point) {
                    const float2 p = s_data[local_i];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);
                }

                unsigned mask = __ballot_sync(FULL_MASK, valid_point && (dist < max_distance));
                int num = __popc(mask);

                if (num != 0) {
                    // If this whole micro-batch could overflow the shared candidate buffer,
                    // flush first, then re-evaluate against the tighter threshold.
                    int current_count = 0;
                    if (lane == 0) {
                        current_count = *count;
                    }
                    current_count = __shfl_sync(FULL_MASK, current_count, 0);

                    if (current_count + num > K) {
                        flush_buffer<K>(cand, scratch, count, result_dist, result_idx, lane, max_distance);
                        mask = __ballot_sync(FULL_MASK, valid_point && (dist < max_distance));
                        num = __popc(mask);
                    }

                    if (num != 0) {
                        // The prompt requires atomicAdd for the candidate count update.
                        // We do it once per micro-batch and derive each lane's slot by
                        // adding its prefix within the ballot mask.
                        int base = 0;
                        if (lane == 0) {
                            base = atomicAdd(count, num);
                        }
                        base = __shfl_sync(FULL_MASK, base, 0);

                        const unsigned lane_bit = 1u << lane;
                        if (valid_point && (mask & lane_bit)) {
                            const unsigned prior = mask & (lane_bit - 1u);
                            const int pos = base + __popc(prior);
                            cand[pos] = make_pairdi(dist, idx);
                        }

                        __syncwarp();

                        // Flush immediately on a full candidate buffer.
                        if (base + num == K) {
                            flush_buffer<K>(cand, scratch, count, result_dist, result_idx, lane, max_distance);
                        }
                    }
                }
            }
        }

        // The next tile will overwrite s_data, so all warps must be done with it first.
        __syncthreads();
    }

    if (valid_query) {
        flush_buffer<K>(cand, scratch, count, result_dist, result_idx, lane, max_distance);

        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);
#pragma unroll
        for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
            const std::size_t out_pos = out_base + static_cast<std::size_t>(t * WARP_SIZE + lane);
            result_out[out_pos].first = result_idx[t];
            result_out[out_pos].second = result_dist[t];
        }
    }
}

template <int K, int BLOCK_THREADS>
constexpr std::size_t shared_bytes_for_kernel() {
    return static_cast<std::size_t>(DATA_TILE_POINTS) * sizeof(float2) +
           static_cast<std::size_t>(2) * static_cast<std::size_t>(BLOCK_THREADS / WARP_SIZE) *
               static_cast<std::size_t>(K) * sizeof(PairDI) +
           static_cast<std::size_t>(BLOCK_THREADS / WARP_SIZE) * sizeof(int);
}

template <int K, int BLOCK_THREADS>
inline void launch_knn_impl(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    std::pair<int, float>* result)
{
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;
    constexpr std::size_t SHMEM_BYTES = shared_bytes_for_kernel<K, BLOCK_THREADS>();
    static_assert(SHMEM_BYTES <= MAX_TARGET_SHARED_BYTES,
                  "Selected specialization exceeds the per-block shared-memory budget.");

    const dim3 block(BLOCK_THREADS);
    const unsigned int grid_x = static_cast<unsigned int>(
        (static_cast<std::size_t>(query_count) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const dim3 grid(grid_x);

    // Opt into the large dynamic shared-memory footprint needed by:
    //   - the cached data tile,
    //   - one candidate buffer per warp/query,
    //   - one merge scratch array per warp/query.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, BLOCK_THREADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SHMEM_BYTES));
    (void)cudaFuncSetAttribute(
        knn_kernel<K, BLOCK_THREADS>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    knn_kernel<K, BLOCK_THREADS><<<grid, block, SHMEM_BYTES>>>(
        query, query_count, data, data_count, result);
}

} // namespace knn_cuda_internal

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

    // Hyper-parameter choice:
    // The dominant cost is repeatedly streaming the large `data` array. Therefore we pick
    // the largest number of query warps per block that still fits the shared-memory budget,
    // maximizing reuse of each cached data tile:
    //   k <= 256  -> 32 warps/block  (1024 threads)
    //   k == 512  -> 16 warps/block  ( 512 threads)
    //   k == 1024 ->  8 warps/block  ( 256 threads)
    switch (k) {
        case 32:
            knn_cuda_internal::launch_knn_impl<32, 1024>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_cuda_internal::launch_knn_impl<64, 1024>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_cuda_internal::launch_knn_impl<128, 1024>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_cuda_internal::launch_knn_impl<256, 1024>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_cuda_internal::launch_knn_impl<512, 512>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_cuda_internal::launch_knn_impl<1024, 256>(query, query_count, data, data_count, result);
            break;
        default:
            // Inputs are guaranteed valid by the caller.
            break;
    }
}