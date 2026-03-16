#include <cuda_runtime.h>
#include <utility>
#include <climits>

namespace knn_detail {

/*
  High-level design:

  - One query is owned by one warp, as requested.
  - One thread block contains 8 warps (256 threads), so a block processes 8 queries at once.
    This is a good compromise on A100/H100: it gives substantial reuse of each shared-memory
    data tile across multiple queries while still keeping enough blocks in flight.

  - For each warp / query:
      * The current top-k ("intermediate result") is kept as a private warp-distributed copy
        in registers: lane L owns ranks {L, L+32, L+64, ...}.
      * The candidate buffer of size k lives in shared memory.
      * An additional shared-memory scratch region of size k is used only while merging; this
        does not allocate any extra device memory outside the kernel.

  - The block walks over the dataset in tiles cached in shared memory.
    All threads cooperatively load one tile of data points.
    Each warp computes distances from its own query to the cached tile and appends only
    candidates with distance < current k-th neighbor into its shared candidate buffer.

  - When the candidate buffer becomes full (or would overflow), the warp merges it with the
    register-resident top-k:
      * pad unused candidate slots with +inf
      * spill the current top-k to the second half of the shared workspace
      * sort the 2k (candidate + top-k) pairs in shared memory with a warp-cooperative bitonic sort
      * reload the first k pairs back to registers

    This shared-memory merge avoids random indexed access into register arrays, which would
    otherwise risk local-memory spills and hurt performance.

  - Distances returned are squared Euclidean distances, exactly as required.
*/

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE;
constexpr unsigned FULL_MASK = 0xffffffffu;

struct alignas(std::pair<int, float>) PairIF {
    int first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "std::pair<int,float> must match PairIF size");
static_assert(alignof(PairIF) == alignof(std::pair<int, float>), "std::pair<int,float> must match PairIF alignment");

template <int K>
struct KernelConfig {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "k must be a power of two in [32,1024]");
    enum {
        kItemsPerThread = K / WARP_SIZE,
        // Per warp: k candidate pairs + k scratch pairs used during merge.
        kWorksetBytes = WARPS_PER_BLOCK * (2 * K) * (int(sizeof(float)) + int(sizeof(int))),
        // k=1024 forces 1 block/SM from shared memory on A100/H100 with the chosen 8-warps/block design.
        // Smaller k can sustain 2 blocks/SM, so the host launcher sizes the tile accordingly.
        kPreferredBlocksPerSM = (K == 1024 ? 1 : 2)
    };
};

__device__ __forceinline__ unsigned lane_mask_lt() {
    unsigned m;
    asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

__device__ __forceinline__ bool pair_less(float ad, int ai, float bd, int bi) {
    return (ad < bd) || (ad == bd && ai < bi);
}

__device__ __forceinline__ bool pair_greater(float ad, int ai, float bd, int bi) {
    return pair_less(bd, bi, ad, ai);
}

template <int SLOT, int ITEMS>
struct InitTop {
    __device__ __forceinline__ static void run(float (&top_dist)[ITEMS], int (&top_idx)[ITEMS]) {
        top_dist[SLOT] = CUDART_INF_F;
        top_idx[SLOT] = INT_MAX;
        InitTop<SLOT + 1, ITEMS>::run(top_dist, top_idx);
    }
};

template <int ITEMS>
struct InitTop<ITEMS, ITEMS> {
    __device__ __forceinline__ static void run(float (&)[ITEMS], int (&)[ITEMS]) {}
};

template <int SLOT, int ITEMS, int K>
struct SpillTopToShared {
    __device__ __forceinline__ static void run(
        const float (&top_dist)[ITEMS],
        const int (&top_idx)[ITEMS],
        float *work_dist,
        int *work_idx,
        int lane) {
        const int pos = K + (SLOT << 5) + lane;
        work_dist[pos] = top_dist[SLOT];
        work_idx[pos] = top_idx[SLOT];
        SpillTopToShared<SLOT + 1, ITEMS, K>::run(top_dist, top_idx, work_dist, work_idx, lane);
    }
};

template <int ITEMS, int K>
struct SpillTopToShared<ITEMS, ITEMS, K> {
    __device__ __forceinline__ static void run(
        const float (&)[ITEMS],
        const int (&)[ITEMS],
        float *,
        int *,
        int) {}
};

template <int SLOT, int ITEMS>
struct LoadTopFromShared {
    __device__ __forceinline__ static void run(
        float (&top_dist)[ITEMS],
        int (&top_idx)[ITEMS],
        const float *work_dist,
        const int *work_idx,
        int lane) {
        const int pos = (SLOT << 5) + lane;
        top_dist[SLOT] = work_dist[pos];
        top_idx[SLOT] = work_idx[pos];
        LoadTopFromShared<SLOT + 1, ITEMS>::run(top_dist, top_idx, work_dist, work_idx, lane);
    }
};

template <int ITEMS>
struct LoadTopFromShared<ITEMS, ITEMS> {
    __device__ __forceinline__ static void run(
        float (&)[ITEMS],
        int (&)[ITEMS],
        const float *,
        const int *,
        int) {}
};

template <int SLOT, int ITEMS>
struct StoreResults {
    __device__ __forceinline__ static void run(
        const float (&top_dist)[ITEMS],
        const int (&top_idx)[ITEMS],
        PairIF *result,
        int base_out,
        int lane) {
        const int pos = base_out + (SLOT << 5) + lane;
        result[pos].first = top_idx[SLOT];
        result[pos].second = top_dist[SLOT];
        StoreResults<SLOT + 1, ITEMS>::run(top_dist, top_idx, result, base_out, lane);
    }
};

template <int ITEMS>
struct StoreResults<ITEMS, ITEMS> {
    __device__ __forceinline__ static void run(
        const float (&)[ITEMS],
        const int (&)[ITEMS],
        PairIF *,
        int,
        int) {}
};

template <int N>
__device__ __forceinline__ void sort_shared_bitonic(float *dist, int *idx) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Keep the fixed-size network as loops instead of fully unrolling it;
    // this controls code size while remaining efficient because merges are
    // relatively rare after the distance threshold becomes selective.
    #pragma unroll 1
    for (int size = 2; size <= N; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll 1
            for (int i = lane; i < N; i += WARP_SIZE) {
                const int j = i ^ stride;
                if (j > i) {
                    const float di = dist[i];
                    const int ii = idx[i];
                    const float dj = dist[j];
                    const int ij = idx[j];

                    const bool ascending = ((i & size) == 0);
                    const bool do_swap = ascending ? pair_greater(di, ii, dj, ij)
                                                   : pair_less(di, ii, dj, ij);
                    if (do_swap) {
                        dist[i] = dj;
                        idx[i]  = ij;
                        dist[j] = di;
                        idx[j]  = ii;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

template <int K>
__device__ __forceinline__ void merge_candidate_buffer(
    float (&top_dist)[KernelConfig<K>::kItemsPerThread],
    int (&top_idx)[KernelConfig<K>::kItemsPerThread],
    float *work_dist,
    int *work_idx,
    int candidate_count) {
    enum { ITEMS = KernelConfig<K>::kItemsPerThread };
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Ensure all previous candidate writes are visible to the warp.
    __syncwarp(FULL_MASK);

    // Pad the unused candidate slots with sentinels so the fixed-size sort remains valid.
    for (int i = lane; i < K; i += WARP_SIZE) {
        if (i >= candidate_count) {
            work_dist[i] = CUDART_INF_F;
            work_idx[i]  = INT_MAX;
        }
    }

    // Copy the private register-resident top-k into the scratch half of shared memory.
    SpillTopToShared<0, ITEMS, K>::run(top_dist, top_idx, work_dist, work_idx, lane);
    __syncwarp(FULL_MASK);

    // Sort candidate[0..K) + top[K..2K), then keep the smallest K.
    sort_shared_bitonic<2 * K>(work_dist, work_idx);

    // Reload the new top-k back into the private register copy.
    LoadTopFromShared<0, ITEMS>::run(top_dist, top_idx, work_dist, work_idx, lane);
    __syncwarp(FULL_MASK);
}

template <int K>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void knn_kernel(
    const float2 *__restrict__ query,
    int query_count,
    const float2 *__restrict__ data,
    int data_count,
    PairIF *__restrict__ result,
    int tile_points) {
    enum { ITEMS = KernelConfig<K>::kItemsPerThread };

    extern __shared__ unsigned char smem_raw[];

    // Shared-memory layout:
    //   [tile_points x float2]                                : cached input tile
    //   [WARPS_PER_BLOCK x (2K) x float]                      : per-warp candidate buffer + merge scratch distances
    //   [WARPS_PER_BLOCK x (2K) x int]                        : per-warp candidate buffer + merge scratch indices
    float2 *tile = reinterpret_cast<float2 *>(smem_raw);
    float *work_dist_all = reinterpret_cast<float *>(tile + tile_points);
    int *work_idx_all = reinterpret_cast<int *>(work_dist_all + WARPS_PER_BLOCK * (2 * K));

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid >> 5;

    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool active_query = (query_idx < query_count);

    float *my_work_dist = work_dist_all + warp * (2 * K);
    int *my_work_idx = work_idx_all + warp * (2 * K);

    float top_dist[ITEMS];
    int top_idx[ITEMS];
    InitTop<0, ITEMS>::run(top_dist, top_idx);

    float qx = 0.0f;
    float qy = 0.0f;
    if (active_query && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    float kth = CUDART_INF_F;
    int buf_count = 0;

    for (int base = 0; base < data_count; base += tile_points) {
        const int remaining = data_count - base;
        const int valid = (remaining < tile_points) ? remaining : tile_points;

        // Cooperative tile load by the whole block.
        for (int i = tid; i < valid; i += BLOCK_THREADS) {
            tile[i] = data[base + i];
        }
        __syncthreads();

        if (active_query) {
            // Process the tile in warp-sized chunks so warp ballots compact the accepted candidates.
            for (int j = 0; j < valid; j += WARP_SIZE) {
                const int local = j + lane;

                bool pred = false;
                float dist = 0.0f;
                int gidx = base + local;

                if (local < valid) {
                    const float2 p = tile[local];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = __fmaf_rn(dx, dx, dy * dy);
                    pred = (dist < kth);  // strict, as requested ("not closer than" gets skipped)
                }

                unsigned pass = __ballot_sync(FULL_MASK, pred);
                int pass_count = __popc(pass);

                // If the current warp-sized batch would overflow the buffer, merge the buffer first.
                if (buf_count + pass_count > K) {
                    if (buf_count > 0) {
                        merge_candidate_buffer<K>(top_dist, top_idx, my_work_dist, my_work_idx, buf_count);
                        buf_count = 0;
                        kth = __shfl_sync(FULL_MASK, top_dist[ITEMS - 1], WARP_SIZE - 1);
                    }

                    // Re-evaluate the current 32 candidates against the tighter threshold.
                    pred = false;
                    if (local < valid) {
                        pred = (dist < kth);
                    }
                    pass = __ballot_sync(FULL_MASK, pred);
                    pass_count = __popc(pass);
                }

                if (pred) {
                    const int pos = buf_count + __popc(pass & lane_mask_lt());
                    my_work_dist[pos] = dist;
                    my_work_idx[pos] = gidx;
                }
                buf_count += pass_count;

                if (buf_count == K) {
                    merge_candidate_buffer<K>(top_dist, top_idx, my_work_dist, my_work_idx, buf_count);
                    buf_count = 0;
                    kth = __shfl_sync(FULL_MASK, top_dist[ITEMS - 1], WARP_SIZE - 1);
                }
            }
        }

        // All warps in the block must finish with the tile before it is overwritten.
        __syncthreads();
    }

    if (active_query) {
        if (buf_count > 0) {
            merge_candidate_buffer<K>(top_dist, top_idx, my_work_dist, my_work_idx, buf_count);
        }

        // Store the final top-k in ascending order of squared distance.
        StoreResults<0, ITEMS>::run(top_dist, top_idx, result, query_idx * K, lane);
    }
}

inline int shared_budget_bytes(int preferred_blocks_per_sm) {
    // Reasonable fallbacks if attribute queries are unavailable for some reason.
    int per_sm = (preferred_blocks_per_sm == 1) ? (160 * 1024) : (80 * 1024);
    int per_block_optin = per_sm;

    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
        int value = 0;
        if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device) == cudaSuccess && value > 0) {
            per_sm = value;
        }
        value = 0;
        if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess && value > 0) {
            per_block_optin = value;
        }
    }

    int budget = per_sm / preferred_blocks_per_sm;
    if (budget > per_block_optin) {
        budget = per_block_optin;
    }

    // Shared-memory allocation granularity is naturally coarse; rounding down to 256 B keeps
    // the launch arithmetic predictable.
    budget &= ~255;
    return budget;
}

template <int K>
inline void launch_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result) {
    typedef KernelConfig<K> Cfg;

    int preferred_blocks = Cfg::kPreferredBlocksPerSM;
    int budget = shared_budget_bytes(preferred_blocks);

    // If the requested residency leaves no room for even one warp-sized tile, fall back to 1 block/SM.
    const int min_tile_bytes = int(sizeof(float2) * WARP_SIZE);
    if (budget < Cfg::kWorksetBytes + min_tile_bytes) {
        preferred_blocks = 1;
        budget = shared_budget_bytes(1);
    }

    int available_tile_bytes = budget - Cfg::kWorksetBytes;
    if (available_tile_bytes < min_tile_bytes) {
        available_tile_bytes = min_tile_bytes;
    }

    int tile_points = available_tile_bytes / int(sizeof(float2));
    tile_points = (tile_points / WARP_SIZE) * WARP_SIZE;
    if (tile_points < WARP_SIZE) {
        tile_points = WARP_SIZE;
    }

    // Avoid allocating a larger tile than the dataset needs.
    const int max_needed_points = ((data_count - 1) / WARP_SIZE + 1) * WARP_SIZE;
    if (tile_points > max_needed_points) {
        tile_points = max_needed_points;
    }

    const int shared_bytes = Cfg::kWorksetBytes + tile_points * int(sizeof(float2));
    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    (void)cudaFuncSetAttribute(knn_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    (void)cudaFuncSetAttribute(knn_kernel<K>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    knn_kernel<K><<<blocks, BLOCK_THREADS, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<PairIF *>(result),
        tile_points);
}

}  // namespace knn_detail

void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k) {
    if (query_count <= 0) {
        return;
    }

    switch (k) {
        case 32:   knn_detail::launch_knn<32>(query, query_count, data, data_count, result); break;
        case 64:   knn_detail::launch_knn<64>(query, query_count, data, data_count, result); break;
        case 128:  knn_detail::launch_knn<128>(query, query_count, data, data_count, result); break;
        case 256:  knn_detail::launch_knn<256>(query, query_count, data, data_count, result); break;
        case 512:  knn_detail::launch_knn<512>(query, query_count, data, data_count, result); break;
        case 1024: knn_detail::launch_knn<1024>(query, query_count, data, data_count, result); break;
        default:   break;  // inputs are guaranteed valid; this is a defensive no-op
    }
}