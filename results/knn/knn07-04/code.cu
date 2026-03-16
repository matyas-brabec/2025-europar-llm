#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>

namespace {

// High-level design:
// - One warp owns one query for the whole kernel lifetime.
// - A whole thread block streams the data set through shared memory in batches.
// - Each warp keeps two private K-element shared-memory arrays:
//     1) result_* : current sorted top-K for its query.
//     2) buffer_* : candidate buffer filled via a warp-aggregated shared-memory atomicAdd.
// - Whenever the candidate buffer fills (or at the very end), the warp performs exactly the
//   merge requested by the prompt:
//     1) bitonic-sort the buffer ascending,
//     2) build the bitonic "min(buffer[i], result[K-1-i])" sequence,
//     3) bitonic-sort that sequence ascending to obtain the updated top-K.
//
// Implementation notes tuned for A100/H100:
// - 256 threads / 8 warps per block balance:
//     * shared-memory reuse of each loaded data batch across 8 queries,
//     * enough block-level parallelism for the "thousands of queries" regime,
//     * feasibility for K up to 1024 under A100's 160 KiB opt-in shared-memory limit.
// - Shared memory uses a structure-of-arrays layout (separate x/y, distance/index arrays) to
//   avoid the extra shared-memory bank conflicts that 8-byte AoS/float2 accesses would create.
// - Batch size is specialized per K and chosen as the largest warp-aligned value that preserves
//   a good A100-compatible active-block count for that K specialization.

using ResultPair = std::pair<int, float>;

constexpr int kWarpSize        = 32;
constexpr unsigned kFullMask   = 0xFFFFFFFFu;
constexpr int kWarpsPerBlock   = 8;
constexpr int kBlockThreads    = kWarpsPerBlock * kWarpSize;
constexpr int kInvalidIndex    = INT_MAX;
constexpr int kMaxOptInSmem    = 163840; // 160 KiB, the A100-compatible opt-in per-block limit.

static_assert(kBlockThreads == 256, "This kernel is tuned for 256 threads per block.");

// Per-K tuning:
// Each batch size is a multiple of 32 so that each warp processes whole chunks and the shared
// arrays remain naturally 128-byte aligned.
// Comments show the intended A100-compatible active-block target with 8 warps/block.
template <int K> struct KernelTuning;

template <> struct KernelTuning<32>   { static constexpr int batch_points = 2016; }; // 8 blocks/SM
template <> struct KernelTuning<64>   { static constexpr int batch_points = 2368; }; // 6 blocks/SM
template <> struct KernelTuning<128>  { static constexpr int batch_points = 2016; }; // 5 blocks/SM
template <> struct KernelTuning<256>  { static constexpr int batch_points = 2720; }; // 3 blocks/SM
template <> struct KernelTuning<512>  { static constexpr int batch_points = 2016; }; // 2 blocks/SM
template <> struct KernelTuning<1024> { static constexpr int batch_points = 4064; }; // 1 block /SM

template <int K>
constexpr std::size_t shared_bytes_for_kernel() {
    constexpr int B = KernelTuning<K>::batch_points;
    return
        static_cast<std::size_t>(2) * B * sizeof(float) +                   // batch_x, batch_y
        static_cast<std::size_t>(kWarpsPerBlock) * sizeof(int) +            // per-warp candidate counts
        static_cast<std::size_t>(2) * kWarpsPerBlock * K * sizeof(float) +  // result_dist, buffer_dist
        static_cast<std::size_t>(2) * kWarpsPerBlock * K * sizeof(int);     // result_idx,  buffer_idx
}

// Warp-private state: each warp gets disjoint shared-memory slices, so the state is private
// to one query even though it physically resides in shared memory.
struct WarpTopKState {
    float *result_dist;
    int   *result_idx;
    float *buffer_dist;
    int   *buffer_idx;
    int   *cand_count;
};

__device__ __forceinline__ bool pair_less(float a_dist, int a_idx,
                                          float b_dist, int b_idx) {
    // Sorting/merging uses a total order (distance, then index) to keep the bitonic network
    // deterministic. Candidate admission still uses only "dist < max_distance", exactly as
    // requested by the prompt, so ties at the threshold remain arbitrary.
    if (a_dist < b_dist) return true;
    if (a_dist > b_dist) return false;
    return a_idx < b_idx;
}

__device__ __forceinline__ float squared_distance(float qx, float qy,
                                                  float px, float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return __fmaf_rn(dx, dx, dy * dy);
}

template <int N>
__device__ __forceinline__ void bitonic_sort_asc(float *dist, int *idx) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    // The outer loops are intentionally not forcibly unrolled to keep instruction footprint
    // reasonable for N up to 1024. The innermost per-lane strided loop is unrolled because
    // its trip count is compile-time fixed (N / 32).
    #pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int i = lane; i < N; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    const float di = dist[i];
                    const float dj = dist[j];
                    const int   ii = idx[i];
                    const int   ij = idx[j];

                    const bool up = ((i & size) == 0);
                    const bool do_swap = up
                        ? pair_less(dj, ij, di, ii)
                        : pair_less(di, ii, dj, ij);

                    if (do_swap) {
                        dist[i] = dj; idx[i] = ij;
                        dist[j] = di; idx[j] = ii;
                    }
                }
            }
            __syncwarp();
        }
    }
}

template <int K>
__device__ __forceinline__ float flush_candidate_buffer(WarpTopKState &state) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    // Make sure all candidate writes are visible before the warp starts consuming the buffer.
    __syncwarp();

    int count = 0;
    if (lane == 0) {
        count = *state.cand_count;
    }
    count = __shfl_sync(kFullMask, count, 0);

    // Pad the unused tail with +inf so the full-K bitonic network can be reused unchanged.
    #pragma unroll
    for (int i = count + lane; i < K; i += kWarpSize) {
        state.buffer_dist[i] = CUDART_INF_F;
        state.buffer_idx[i]  = kInvalidIndex;
    }
    __syncwarp();

    // Step 1: sort the candidate buffer ascending.
    bitonic_sort_asc<K>(state.buffer_dist, state.buffer_idx);

    // Step 2: build the K-element bitonic sequence that contains the K smallest elements from
    //         buffer U result by taking minima against the reversed current result.
    #pragma unroll
    for (int i = lane; i < K; i += kWarpSize) {
        const int rev = K - 1 - i;

        const float bd = state.buffer_dist[i];
        const int   bi = state.buffer_idx[i];
        const float rd = state.result_dist[rev];
        const int   ri = state.result_idx[rev];

        if (pair_less(rd, ri, bd, bi)) {
            state.buffer_dist[i] = rd;
            state.buffer_idx[i]  = ri;
        }
    }
    __syncwarp();

    // Step 3: bitonic-sort the merged bitonic sequence ascending.
    bitonic_sort_asc<K>(state.buffer_dist, state.buffer_idx);

    // The freshly sorted data now lives in buffer_*. Swap the role of the two arrays instead
    // of copying K elements back.
    float *tmp_dist = state.result_dist;
    state.result_dist = state.buffer_dist;
    state.buffer_dist = tmp_dist;

    int *tmp_idx = state.result_idx;
    state.result_idx = state.buffer_idx;
    state.buffer_idx = tmp_idx;

    if (lane == 0) {
        *state.cand_count = 0;
    }
    __syncwarp();

    float max_distance = 0.0f;
    if (lane == 0) {
        max_distance = state.result_dist[K - 1];
    }
    return __shfl_sync(kFullMask, max_distance, 0);
}

template <int K>
__device__ __forceinline__ void process_cached_batch(
    float qx, float qy,
    const float *batch_x,
    const float *batch_y,
    int batch_base,
    int start,
    int tile_count,
    WarpTopKState &state,
    float &max_distance) {

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const unsigned lane_prefix_mask =
        (lane == 0) ? 0u : ((1u << static_cast<unsigned>(lane)) - 1u);

    // One cached point per lane per iteration, so at most 32 new candidates appear in a chunk.
    // Because K >= 32, a single pre-flush is enough to make room before insertion.
    for (int local_base = start; local_base < tile_count; local_base += kWarpSize) {
        const int local_idx = local_base + lane;

        bool  valid      = (local_idx < tile_count);
        bool  qualifies  = false;
        float dist       = 0.0f;
        int   global_idx = 0;

        if (valid) {
            const float px = batch_x[local_idx];
            const float py = batch_y[local_idx];
            dist = squared_distance(qx, qy, px, py);
            global_idx = batch_base + local_idx;

            // Strict '<' on max_distance follows the prompt exactly.
            qualifies = (dist < max_distance);
        }

        unsigned mask = __ballot_sync(kFullMask, qualifies);
        int num_new = __popc(mask);
        if (num_new == 0) {
            continue;
        }

        int current_count = 0;
        if (lane == 0) {
            current_count = *state.cand_count;
        }
        current_count = __shfl_sync(kFullMask, current_count, 0);

        // If the current chunk would overflow the K-sized buffer, flush first.
        if (current_count + num_new > K) {
            max_distance = flush_candidate_buffer<K>(state);

            // The flush may tighten max_distance. Re-test the already computed distances so we
            // do not enqueue obviously dominated candidates into the now-empty buffer.
            qualifies = valid && (dist < max_distance);
            mask = __ballot_sync(kFullMask, qualifies);
            num_new = __popc(mask);
            if (num_new == 0) {
                continue;
            }
        }

        // Warp-aggregated atomicAdd: one shared atomic per warp-chunk instead of one per lane,
        // while still using atomicAdd exactly as requested to reserve buffer positions.
        int base_slot = 0;
        if (lane == 0) {
            base_slot = atomicAdd(state.cand_count, num_new);
        }
        base_slot = __shfl_sync(kFullMask, base_slot, 0);

        if (qualifies) {
            const int rank = __popc(mask & lane_prefix_mask);
            state.buffer_dist[base_slot + rank] = dist;
            state.buffer_idx[base_slot + rank]  = global_idx;
        }
        __syncwarp();

        // If the insertion filled the buffer exactly, merge immediately so max_distance stays
        // as tight as possible for subsequent pruning.
        if (base_slot + num_new == K) {
            max_distance = flush_candidate_buffer<K>(state);
        }
    }
}

template <int K>
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           ResultPair * __restrict__ output) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K >= 32 && K <= 1024, "Supported K range is [32, 1024].");

    constexpr int kBatchPoints = KernelTuning<K>::batch_points;
    static_assert((kBatchPoints % kWarpSize) == 0, "Batch size must be warp-aligned.");
    static_assert(kBatchPoints > K, "The first batch must be able to seed the initial top-K.");

    extern __shared__ unsigned char shared_raw[];

    // Shared-memory layout (SoA to minimize bank conflicts):
    //   batch_x[kBatchPoints]
    //   batch_y[kBatchPoints]
    //   cand_count[kWarpsPerBlock]
    //   result_dist[kWarpsPerBlock * K]
    //   result_idx [kWarpsPerBlock * K]
    //   buffer_dist[kWarpsPerBlock * K]
    //   buffer_idx [kWarpsPerBlock * K]
    unsigned char *ptr = shared_raw;

    float *batch_x = reinterpret_cast<float *>(ptr);
    ptr += static_cast<std::size_t>(kBatchPoints) * sizeof(float);

    float *batch_y = reinterpret_cast<float *>(ptr);
    ptr += static_cast<std::size_t>(kBatchPoints) * sizeof(float);

    int *cand_count = reinterpret_cast<int *>(ptr);
    ptr += static_cast<std::size_t>(kWarpsPerBlock) * sizeof(int);

    float *result_dist_storage = reinterpret_cast<float *>(ptr);
    ptr += static_cast<std::size_t>(kWarpsPerBlock) * K * sizeof(float);

    int *result_idx_storage = reinterpret_cast<int *>(ptr);
    ptr += static_cast<std::size_t>(kWarpsPerBlock) * K * sizeof(int);

    float *buffer_dist_storage = reinterpret_cast<float *>(ptr);
    ptr += static_cast<std::size_t>(kWarpsPerBlock) * K * sizeof(float);

    int *buffer_idx_storage = reinterpret_cast<int *>(ptr);

    const int lane    = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp_id = static_cast<int>(threadIdx.x) >> 5;
    const int query_idx = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_id;
    const bool active = (query_idx < query_count);

    // Every warp owns one query and one private top-K state.
    const int warp_offset = warp_id * K;
    WarpTopKState state;
    state.result_dist = result_dist_storage + warp_offset;
    state.result_idx  = result_idx_storage  + warp_offset;
    state.buffer_dist = buffer_dist_storage + warp_offset;
    state.buffer_idx  = buffer_idx_storage  + warp_offset;
    state.cand_count  = cand_count + warp_id;

    if (lane == 0) {
        *state.cand_count = 0;
    }
    __syncwarp();

    float qx = 0.0f;
    float qy = 0.0f;
    float max_distance = CUDART_INF_F;

    if (active) {
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);
    }

    for (int batch_base = 0; batch_base < data_count; batch_base += kBatchPoints) {
        int tile_count = data_count - batch_base;
        if (tile_count > kBatchPoints) {
            tile_count = kBatchPoints;
        }

        // Whole-block cooperative load of the next data batch into shared memory.
        for (int i = static_cast<int>(threadIdx.x); i < tile_count; i += kBlockThreads) {
            const float2 p = data[batch_base + i];
            batch_x[i] = p.x;
            batch_y[i] = p.y;
        }
        __syncthreads();

        if (active) {
            if (batch_base == 0) {
                // Seed the intermediate result with the first K data points.
                #pragma unroll
                for (int i = lane; i < K; i += kWarpSize) {
                    state.result_dist[i] = squared_distance(qx, qy, batch_x[i], batch_y[i]);
                    state.result_idx[i]  = i;
                }
                __syncwarp();

                // Invariant after this sort: result_* is ascending.
                bitonic_sort_asc<K>(state.result_dist, state.result_idx);

                if (lane == 0) {
                    max_distance = state.result_dist[K - 1];
                }
                max_distance = __shfl_sync(kFullMask, max_distance, 0);

                // Process the remainder of the first cached batch.
                process_cached_batch<K>(
                    qx, qy,
                    batch_x, batch_y,
                    batch_base,
                    K,
                    tile_count,
                    state,
                    max_distance);
            } else {
                // Process a regular cached batch.
                process_cached_batch<K>(
                    qx, qy,
                    batch_x, batch_y,
                    batch_base,
                    0,
                    tile_count,
                    state,
                    max_distance);
            }
        }

        // No barrier is needed after the final batch because the shared cache will not be reused.
        if (batch_base + tile_count < data_count) {
            __syncthreads();
        }
    }

    if (active) {
        int pending = 0;
        if (lane == 0) {
            pending = *state.cand_count;
        }
        pending = __shfl_sync(kFullMask, pending, 0);

        if (pending > 0) {
            max_distance = flush_candidate_buffer<K>(state);
            (void)max_distance;
        }

        // Write the final sorted top-K (indices + squared distances) back to global memory.
        ResultPair *out = output + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        #pragma unroll
        for (int i = lane; i < K; i += kWarpSize) {
            out[i].first  = state.result_idx[i];
            out[i].second = state.result_dist[i];
        }
    }
}

template <int K>
void launch_knn_specialized(const float2 *query,
                            int query_count,
                            const float2 *data,
                            int data_count,
                            ResultPair *output) {
    constexpr std::size_t smem_bytes = shared_bytes_for_kernel<K>();
    static_assert(smem_bytes <= kMaxOptInSmem,
                  "Shared-memory footprint exceeds the A100-compatible 160 KiB opt-in limit.");

    // Several specializations exceed the default 48 KiB dynamic shared-memory limit, so opt in.
    // Prefer shared-memory carveout because this kernel is explicitly shared-memory heavy.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int grid = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    // Intentionally asynchronous: the caller owns synchronization and error handling policy.
    knn_kernel<K><<<grid, kBlockThreads, smem_bytes>>>(
        query, query_count, data, data_count, output);
}

} // anonymous namespace

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
        case 32:   launch_knn_specialized<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_specialized<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_specialized<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_specialized<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_specialized<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_specialized<1024>(query, query_count, data, data_count, result); break;
        default:
            // Per the prompt, inputs are always valid, so this path is unreachable in normal use.
            break;
    }
}