#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

// Exact brute-force 2D k-NN on the GPU.
// Design choices tuned for modern data-center GPUs (A100/H100):
//   * One warp computes one query.
//   * One 256-thread block = 8 warps = 8 queries processed concurrently.
//   * Each block streams the data set in shared-memory tiles that are reused by all 8 warps.
//   * For each warp/query, two K-sized shared-memory arrays are reserved:
//       - current top-K (sorted ascending by distance)
//       - candidate buffer (unsorted while filling, then sorted and merged)
//     These two arrays ping-pong, so no third K-sized scratch buffer is needed.
//   * The shared-memory tile is stored as SoA (x[], y[]) instead of float2[] to avoid
//     shared-memory bank conflicts on repeated warp reads.
//
// The interface is fixed to std::pair<int, float>.  Device code writes through a POD
// mirror with matching size/alignment.  This is the ABI used by mainstream CUDA host
// toolchains for pair<int,float>; the static_asserts catch obvious mismatches.
struct alignas(alignof(std::pair<int, float>)) PairIF {
    int first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>),
              "std::pair<int,float> must match PairIF size");
static_assert(alignof(PairIF) == alignof(std::pair<int, float>),
              "std::pair<int,float> must match PairIF alignment");

constexpr int KNN_WARP_SIZE = 32;
constexpr int KNN_WARPS_PER_BLOCK = 8;
constexpr int KNN_BLOCK_THREADS = KNN_WARP_SIZE * KNN_WARPS_PER_BLOCK;

// A100 exposes 160 KiB (= 163840 B) opt-in shared memory per block.
// For K <= 512, we keep the whole block at 80 KiB so that two blocks can reside per SM.
// For K == 1024, two blocks are impossible anyway, so we use almost the full 160 KiB
// and keep a small 4 KiB guard band.
constexpr int KNN_A100_MAX_OPTIN_SHARED_BYTES = 160 * 1024;
constexpr int KNN_TWO_BLOCKS_SHARED_BUDGET = KNN_A100_MAX_OPTIN_SHARED_BYTES / 2;
constexpr int KNN_ONE_BLOCK_GUARD_BYTES = 4 * 1024;
constexpr unsigned KNN_FULL_MASK = 0xffffffffu;

template <int K>
struct KNNLaunchConfig {
    static_assert(K == 32 || K == 64 || K == 128 || K == 256 || K == 512 || K == 1024,
                  "Unsupported K");

    static constexpr int top_and_buffer_bytes =
        2 * KNN_WARPS_PER_BLOCK * K * static_cast<int>(sizeof(float) + sizeof(int));

    static constexpr int block_shared_budget_bytes =
        (K <= 512) ? KNN_TWO_BLOCKS_SHARED_BUDGET
                   : (KNN_A100_MAX_OPTIN_SHARED_BYTES - KNN_ONE_BLOCK_GUARD_BYTES);

    static constexpr int tile_points =
        (block_shared_budget_bytes - top_and_buffer_bytes) / static_cast<int>(sizeof(float2));

    static constexpr int shared_bytes =
        top_and_buffer_bytes + tile_points * static_cast<int>(sizeof(float2));

    static_assert(tile_points > 0, "Tile must be positive");
    static_assert((tile_points % KNN_BLOCK_THREADS) == 0,
                  "Tile size must be a multiple of blockDim for balanced cooperative loads");
    static_assert(shared_bytes <= KNN_A100_MAX_OPTIN_SHARED_BYTES,
                  "Shared-memory footprint exceeds A100 opt-in limit");
};

template <typename T>
__device__ __forceinline__ void knn_swap_ptrs(T*& a, T*& b) {
    T* tmp = a;
    a = b;
    b = tmp;
}

// Full bitonic sort of K key/value pairs in warp-private shared memory.
// The array is sorted ascending by distance.
template <int K>
__device__ __forceinline__ void knn_sort_shared_pairs(float* dist, int* idx) {
    const int lane = threadIdx.x & (KNN_WARP_SIZE - 1);

    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int i = lane; i < K; i += KNN_WARP_SIZE) {
                const int j = i ^ stride;
                if (j > i) {
                    const float di = dist[i];
                    const float dj = dist[j];
                    const int ii = idx[i];
                    const int ij = idx[j];

                    const bool ascending = ((i & size) == 0);
                    const bool do_swap = ascending ? (dj < di) : (di < dj);

                    if (do_swap) {
                        dist[i] = dj;
                        idx[i] = ij;
                        dist[j] = di;
                        idx[j] = ii;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge-path partition on the first `diag` outputs of two sorted arrays A and B,
// both of length K, both sorted ascending.  Returns how many elements come from A.
// This variant is stable with respect to A on equal keys.
template <int K>
__device__ __forceinline__ int knn_merge_path_partition(const float* a, const float* b, int diag) {
    int low = 0;
    int high = diag;  // diag is only ever in [0, K] here.

    while (low < high) {
        const int mid = (low + high) >> 1;
        const int b_prev = diag - mid - 1;

        if (b_prev >= 0 && a[mid] <= b[b_prev]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low;
}

// Flushes the warp-private candidate buffer into the running top-K:
//   1) pad tail with +inf / -1 if buffer is only partially full
//   2) sort the K candidates in shared memory
//   3) if this is the first flush, the sorted buffer simply becomes the top-K
//   4) otherwise, warp-parallel merge-path computes the first K outputs of
//      merge(top, buffer), writes them into the buffer array, and then swaps pointers
template <int K>
__device__ __forceinline__ void knn_flush_buffer(
    float*& top_d,
    int*& top_i,
    float*& buf_d,
    int*& buf_i,
    int& buf_count,
    bool& top_empty,
    float& kth_dist)
{
    constexpr int ITEMS_PER_LANE = K / KNN_WARP_SIZE;
    const int lane = threadIdx.x & (KNN_WARP_SIZE - 1);

    if (buf_count < K) {
        for (int pos = buf_count + lane; pos < K; pos += KNN_WARP_SIZE) {
            buf_d[pos] = CUDART_INF_F;
            buf_i[pos] = -1;
        }
    }

    // Makes all earlier candidate writes visible before sorting.
    __syncwarp();

    knn_sort_shared_pairs<K>(buf_d, buf_i);

    if (top_empty) {
        // First full/partial sorted buffer directly becomes the running top-K.
        knn_swap_ptrs(top_d, buf_d);
        knn_swap_ptrs(top_i, buf_i);
        top_empty = false;
    } else {
        const int out_begin = lane * ITEMS_PER_LANE;
        const int out_end = out_begin + ITEMS_PER_LANE;

        const int a_begin = knn_merge_path_partition<K>(top_d, buf_d, out_begin);
        const int a_end = knn_merge_path_partition<K>(top_d, buf_d, out_end);
        const int b_begin = out_begin - a_begin;
        const int b_end = out_end - a_end;

        float merged_d[ITEMS_PER_LANE];
        int merged_i[ITEMS_PER_LANE];

        int a_pos = a_begin;
        int b_pos = b_begin;

#pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const float da = (a_pos < a_end) ? top_d[a_pos] : CUDART_INF_F;
            const float db = (b_pos < b_end) ? buf_d[b_pos] : CUDART_INF_F;

            if (da <= db) {
                merged_d[t] = da;
                merged_i[t] = top_i[a_pos];
                ++a_pos;
            } else {
                merged_d[t] = db;
                merged_i[t] = buf_i[b_pos];
                ++b_pos;
            }
        }

        // Keep reads from top/buf completely separated from the output writes.
        __syncwarp();

#pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            buf_d[out_begin + t] = merged_d[t];
            buf_i[out_begin + t] = merged_i[t];
        }

        __syncwarp();

        // The merged output was written into the buffer arrays; make them the new top-K.
        knn_swap_ptrs(top_d, buf_d);
        knn_swap_ptrs(top_i, buf_i);
    }

    float local_kth = (lane == KNN_WARP_SIZE - 1) ? top_d[K - 1] : 0.0f;
    kth_dist = __shfl_sync(KNN_FULL_MASK, local_kth, KNN_WARP_SIZE - 1);
    buf_count = 0;
}

template <int K>
__launch_bounds__(KNN_BLOCK_THREADS)
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    PairIF* __restrict__ result)
{
    using Cfg = KNNLaunchConfig<K>;
    constexpr int TILE_POINTS = Cfg::tile_points;
    constexpr int ITEMS_PER_LANE = K / KNN_WARP_SIZE;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & (KNN_WARP_SIZE - 1);
    const int query_id = blockIdx.x * KNN_WARPS_PER_BLOCK + warp_id;
    const bool active_query = (query_id < query_count);

    // Shared-memory layout:
    //   tile_x[TILE_POINTS], tile_y[TILE_POINTS],
    //   top/buf dist+index arrays for each warp
    extern __shared__ unsigned char smem_raw[];
    float* tile_x = reinterpret_cast<float*>(smem_raw);
    float* tile_y = tile_x + TILE_POINTS;
    float* shared_a_d = tile_y + TILE_POINTS;
    int* shared_a_i = reinterpret_cast<int*>(shared_a_d + KNN_WARPS_PER_BLOCK * K);
    float* shared_b_d = reinterpret_cast<float*>(shared_a_i + KNN_WARPS_PER_BLOCK * K);
    int* shared_b_i = reinterpret_cast<int*>(shared_b_d + KNN_WARPS_PER_BLOCK * K);

    const int warp_base = warp_id * K;

    float* top_d = shared_a_d + warp_base;
    int* top_i = shared_a_i + warp_base;
    float* buf_d = shared_b_d + warp_base;
    int* buf_i = shared_b_i + warp_base;

    // Query point lives in registers for the whole scan.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active_query && lane == 0) {
        const float2 q = query[query_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(KNN_FULL_MASK, qx, 0);
    qy = __shfl_sync(KNN_FULL_MASK, qy, 0);

    int buf_count = 0;
    bool top_empty = true;
    float kth_dist = CUDART_INF_F;

    const unsigned lower_lane_mask =
        (lane == 0) ? 0u : ((1u << lane) - 1u);

    // Stream the data set through shared memory in tiles.
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        int tile_count = data_count - base;
        if (tile_count > TILE_POINTS) {
            tile_count = TILE_POINTS;
        }

        // Whole block cooperatively loads the next tile to shared memory.
        for (int t = tid; t < tile_count; t += KNN_BLOCK_THREADS) {
            const float2 p = data[base + t];
            tile_x[t] = p.x;
            tile_y[t] = p.y;
        }

        __syncthreads();

        if (active_query) {
            // Process the cached tile in warp-sized chunks so all warp collectives
            // operate on a uniform control flow.
            for (int tile_off = 0; tile_off < tile_count; tile_off += KNN_WARP_SIZE) {
                const int local = tile_off + lane;

                float dist = 0.0f;
                int idx = -1;
                bool active = false;

                if (local < tile_count) {
                    const float dx = qx - tile_x[local];
                    const float dy = qy - tile_y[local];
                    dist = __fmaf_rn(dx, dx, dy * dy);  // squared L2 distance
                    idx = base + local;
                    active = (dist < kth_dist);         // strictly closer than current K-th
                }

                // Append all qualifying lanes into the warp-private candidate buffer.
                // If the append would overflow the buffer, the surviving lanes are
                // re-evaluated after the flush because the K-th threshold tightens.
                unsigned mask = __ballot_sync(KNN_FULL_MASK, active);
                while (mask) {
                    const int rank = __popc(mask & lower_lane_mask);
                    const int count = __popc(mask);
                    const int space = K - buf_count;
                    const int inserted = (count < space) ? count : space;

                    if (active && rank < space) {
                        const int pos = buf_count + rank;
                        buf_d[pos] = dist;
                        buf_i[pos] = idx;
                    }

                    const bool remain = active && (rank >= space);
                    buf_count += inserted;

                    if (buf_count == K) {
                        knn_flush_buffer<K>(top_d, top_i, buf_d, buf_i,
                                            buf_count, top_empty, kth_dist);
                    }

                    active = remain && (dist < kth_dist);
                    mask = __ballot_sync(KNN_FULL_MASK, active);
                }
            }
        }

        // The tile storage is reused by the next batch.
        __syncthreads();
    }

    if (active_query && buf_count > 0) {
        knn_flush_buffer<K>(top_d, top_i, buf_d, buf_i,
                            buf_count, top_empty, kth_dist);
    }

    if (active_query) {
        const std::size_t out_base =
            static_cast<std::size_t>(query_id) * static_cast<std::size_t>(K);

        // Strided writeback avoids shared-memory bank conflicts on the final readout
        // and keeps global stores naturally coalesced.
#pragma unroll
        for (int pos = lane; pos < K; pos += KNN_WARP_SIZE) {
            PairIF out;
            out.first = top_i[pos];
            out.second = top_d[pos];
            result[out_base + static_cast<std::size_t>(pos)] = out;
        }
    }
}

template <int K>
inline void knn_launch(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    PairIF* result)
{
    using Cfg = KNNLaunchConfig<K>;
    constexpr int SHARED_BYTES = Cfg::shared_bytes;

    // Opt in to the required dynamic shared-memory footprint and strongly prefer
    // the shared-memory carve-out because this kernel is intentionally shared-heavy.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SHARED_BYTES);
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const dim3 block(KNN_BLOCK_THREADS);
    const dim3 grid((query_count + KNN_WARPS_PER_BLOCK - 1) / KNN_WARPS_PER_BLOCK);

    knn_kernel<K><<<grid, block, SHARED_BYTES>>>(query, query_count, data, data_count, result);
}

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

    PairIF* out = reinterpret_cast<PairIF*>(result);

    // K is guaranteed to be a supported power of two.  Compile-time specialization
    // lets the compiler fully know K, shared-memory footprint, and per-lane work.
    switch (k) {
        case 32:
            knn_launch<32>(query, query_count, data, data_count, out);
            break;
        case 64:
            knn_launch<64>(query, query_count, data, data_count, out);
            break;
        case 128:
            knn_launch<128>(query, query_count, data, data_count, out);
            break;
        case 256:
            knn_launch<256>(query, query_count, data, data_count, out);
            break;
        case 512:
            knn_launch<512>(query, query_count, data, data_count, out);
            break;
        case 1024:
            knn_launch<1024>(query, query_count, data, data_count, out);
            break;
        default:
            // The problem statement guarantees validity; this is only a defensive no-op.
            break;
    }
}