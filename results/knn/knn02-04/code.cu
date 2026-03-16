#include <cuda_runtime.h>
#include <utility>

namespace {

// Tuned for modern datacenter GPUs (A100/H100 class):
// - 8 warps/block => 256 threads/block, a good balance between occupancy and
//   per-tile reuse of the streamed data set across queries.
// - 4096-point shared-memory tile => 32 KiB as SoA (x[] + y[]), which fits
//   comfortably while still being large enough to amortize synchronization.
//   It is also larger than the maximum legal K (1024), so the first tile always
//   contains the entire bootstrap set used to seed the top-K frontier.
constexpr int kWarpSize       = 32;
constexpr int kWarpsPerBlock  = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;
constexpr int kTilePoints     = 4096;
constexpr unsigned kFullMask  = 0xFFFFFFFFu;

using ResultPair = std::pair<int, float>;

__device__ __forceinline__ float sq_l2(const float qx, const float qy,
                                       const float px, const float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return fmaf(dx, dx, dy * dy);
}

__device__ __forceinline__ bool pair_less(const float da, const int ia,
                                          const float db, const int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

// Cooperative tile load:
// the global input is AoS (float2), but the shared tile is stored as SoA
// (x[] and y[] separately). This avoids the shared-memory bank behavior of
// repeated 64-bit float2 accesses in the hot inner loop.
template <int THREADS>
__device__ __forceinline__
void load_tile_soa(const float2* __restrict__ data,
                   const int base,
                   const int count,
                   float* __restrict__ sh_x,
                   float* __restrict__ sh_y) {
#pragma unroll 4
    for (int i = threadIdx.x; i < count; i += THREADS) {
        const float2 p = data[base + i];
        sh_x[i] = p.x;
        sh_y[i] = p.y;
    }
}

// Each lane owns K/32 entries of the current top-K frontier.
// Dynamic indexing into local arrays is intentionally avoided; all accesses are
// done through fully-unrolled constant-index loops so the compiler can scalarize
// the frontier into registers.
template <int ITEMS_PER_LANE>
__device__ __forceinline__
void refresh_lane_max(const float (&top_dist)[ITEMS_PER_LANE],
                      float& lane_max_val,
                      int& lane_max_slot) {
    lane_max_val  = top_dist[0];
    lane_max_slot = 0;
#pragma unroll
    for (int slot = 1; slot < ITEMS_PER_LANE; ++slot) {
        if (top_dist[slot] > lane_max_val) {
            lane_max_val  = top_dist[slot];
            lane_max_slot = slot;
        }
    }
}

__device__ __forceinline__
void reduce_warp_max(const float lane_max_val,
                     const int lane_max_slot,
                     const int lane,
                     float& warp_max_val,
                     int& warp_max_pos) {
    float best_val = lane_max_val;
    int   best_pos = lane + (lane_max_slot << 5);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_val = __shfl_down_sync(kFullMask, best_val, offset);
        const int   other_pos = __shfl_down_sync(kFullMask, best_pos, offset);
        if ((other_val > best_val) || ((other_val == best_val) && (other_pos > best_pos))) {
            best_val = other_val;
            best_pos = other_pos;
        }
    }

    warp_max_val = __shfl_sync(kFullMask, best_val, 0);
    warp_max_pos = __shfl_sync(kFullMask, best_pos, 0);
}

template <int ITEMS_PER_LANE>
__device__ __forceinline__
void replace_owner_slot_and_refresh_lane_max(float (&top_dist)[ITEMS_PER_LANE],
                                             int (&top_idx)[ITEMS_PER_LANE],
                                             const int owner_slot,
                                             const float new_dist,
                                             const int new_idx,
                                             float& lane_max_val,
                                             int& lane_max_slot) {
#pragma unroll
    for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
        if (slot == owner_slot) {
            top_dist[slot] = new_dist;
            top_idx[slot]  = new_idx;
        }
    }
    refresh_lane_max(top_dist, lane_max_val, lane_max_slot);
}

// Process one shared-memory tile, in 32-point batches.
// One lane computes one candidate distance per batch; a ballot identifies which
// candidates beat the current threshold (the worst element currently in top-K).
// Qualified candidates are inserted one-by-one. Only the lane owning the current
// worst slot performs the replacement and local re-scan; the warp then reduces
// the per-lane maxima to refresh the acceptance threshold.
template <int ITEMS_PER_LANE>
__device__ __forceinline__
void process_tile_segment(const float* __restrict__ sh_x,
                          const float* __restrict__ sh_y,
                          const int base,
                          const int start,
                          const int count,
                          const float qx,
                          const float qy,
                          const int lane,
                          float (&top_dist)[ITEMS_PER_LANE],
                          int (&top_idx)[ITEMS_PER_LANE],
                          float& lane_max_val,
                          int& lane_max_slot,
                          float& warp_max_val,
                          int& warp_max_pos) {
    for (int pos_base = start; pos_base < count; pos_base += kWarpSize) {
        const int  pos   = pos_base + lane;
        const bool valid = (pos < count);

        float cand_dist = CUDART_INF_F;
        int   cand_idx  = -1;

        if (valid) {
            cand_dist = sq_l2(qx, qy, sh_x[pos], sh_y[pos]);
            cand_idx  = base + pos;
        }

        unsigned pending = __ballot_sync(kFullMask, valid && (cand_dist < warp_max_val));

        while (pending) {
            const int src_lane = __ffs(pending) - 1;
            const float sel_dist = __shfl_sync(kFullMask, cand_dist, src_lane);
            const int   sel_idx  = __shfl_sync(kFullMask, cand_idx,  src_lane);

            // Because the threshold may tighten after earlier insertions from the
            // same 32-point batch, re-check against the current warp threshold.
            if (sel_dist < warp_max_val) {
                const int owner_lane = warp_max_pos & 31;
                const int owner_slot = warp_max_pos >> 5;

                if (lane == owner_lane) {
                    replace_owner_slot_and_refresh_lane_max(
                        top_dist, top_idx, owner_slot, sel_dist, sel_idx,
                        lane_max_val, lane_max_slot
                    );
                }

                // Explicit warp barrier after intra-warp state mutation, per the
                // problem statement's requirement for synchronized communication.
                __syncwarp(kFullMask);
                reduce_warp_max(lane_max_val, lane_max_slot, lane, warp_max_val, warp_max_pos);
            }

            pending &= (pending - 1);
        }
    }
}

// Final ordering: bitonic sort of the K retained candidates.
// The scan stage keeps only an unsorted top-K set, because that minimizes the
// per-point update cost; sorting once at the end is cheaper than maintaining
// global order while streaming millions of points.
template <int K>
__device__ __forceinline__
void warp_bitonic_sort_shared(float* __restrict__ dist,
                              int* __restrict__ idx,
                              const int lane) {
#pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
#pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < K; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = ((i & size) == 0);

                    const float di = dist[i];
                    const float dj = dist[j];
                    const int   ii = idx[i];
                    const int   ij = idx[j];

                    const bool do_swap = ascending
                        ? pair_less(dj, ij, di, ii)   // ascending: swap if second < first
                        : pair_less(di, ii, dj, ij);  // descending: swap if first < second

                    if (do_swap) {
                        dist[i] = dj;
                        dist[j] = di;
                        idx[i]  = ij;
                        idx[j]  = ii;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

template <int K, int WARPS_PER_BLOCK, int TILE_POINTS>
__launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
__global__ void knn_kernel(const float2* __restrict__ query,
                           const int query_count,
                           const float2* __restrict__ data,
                           const int data_count,
                           ResultPair* __restrict__ result) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");
    static_assert((K % kWarpSize) == 0, "K must be divisible by warp size.");
    static_assert((TILE_POINTS % kWarpSize) == 0, "Tile size must be warp-aligned.");
    static_assert(K <= TILE_POINTS, "First tile must contain the whole bootstrap set.");

    constexpr int ITEMS_PER_LANE = K / kWarpSize;

    const int lane    = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int query_id = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active = (query_id < query_count);

    // Dynamic shared memory is reused for two different phases:
    // 1) during the streaming scan: shared SoA tile (x[] + y[])
    // 2) during final ordering: per-warp scratch buffers for the bitonic sort
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* const sh_x = reinterpret_cast<float*>(smem_raw);
    float* const sh_y = sh_x + TILE_POINTS;

    // Query coordinates are loaded once by lane 0 and broadcast to the warp.
    float qx_lane0 = 0.0f;
    float qy_lane0 = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_id];
        qx_lane0 = q.x;
        qy_lane0 = q.y;
    }
    const float qx = __shfl_sync(kFullMask, qx_lane0, 0);
    const float qy = __shfl_sync(kFullMask, qy_lane0, 0);

    // Private top-K frontier for this query, striped across the warp lanes.
    float top_dist[ITEMS_PER_LANE];
    int   top_idx[ITEMS_PER_LANE];

    float lane_max_val = 0.0f;
    int   lane_max_slot = 0;
    float warp_max_val = 0.0f;
    int   warp_max_pos = 0;

    // ---------------------------
    // First tile: bootstrap top-K
    // ---------------------------
    const int first_tile_count = (data_count < TILE_POINTS) ? data_count : TILE_POINTS;
    load_tile_soa<WARPS_PER_BLOCK * kWarpSize>(data, 0, first_tile_count, sh_x, sh_y);
    __syncthreads();

    if (active) {
        // Seed the frontier directly with the first K points.
#pragma unroll
        for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
            const int pos = lane + (slot << 5);
            top_dist[slot] = sq_l2(qx, qy, sh_x[pos], sh_y[pos]);
            top_idx[slot]  = pos;
        }

        refresh_lane_max(top_dist, lane_max_val, lane_max_slot);
        reduce_warp_max(lane_max_val, lane_max_slot, lane, warp_max_val, warp_max_pos);

        // Process the remainder of the first tile with the normal streaming update.
        process_tile_segment(
            sh_x, sh_y,
            /*base=*/0,
            /*start=*/K,
            /*count=*/first_tile_count,
            qx, qy, lane,
            top_dist, top_idx,
            lane_max_val, lane_max_slot,
            warp_max_val, warp_max_pos
        );
    }

    __syncthreads();

    // ------------------------------------------
    // Remaining tiles: full streaming scan phase
    // ------------------------------------------
    for (int base = TILE_POINTS; base < data_count; base += TILE_POINTS) {
        const int tile_count = ((data_count - base) < TILE_POINTS) ? (data_count - base) : TILE_POINTS;

        load_tile_soa<WARPS_PER_BLOCK * kWarpSize>(data, base, tile_count, sh_x, sh_y);
        __syncthreads();

        if (active) {
            process_tile_segment(
                sh_x, sh_y,
                /*base=*/base,
                /*start=*/0,
                /*count=*/tile_count,
                qx, qy, lane,
                top_dist, top_idx,
                lane_max_val, lane_max_slot,
                warp_max_val, warp_max_pos
            );
        }

        __syncthreads();
    }

    // ---------------------------
    // Final sort and result write
    // ---------------------------
    if (active) {
        // Reuse the same dynamic shared-memory allocation as per-warp scratch.
        float* const sort_dist_all = reinterpret_cast<float*>(smem_raw);
        int* const sort_idx_all =
            reinterpret_cast<int*>(sort_dist_all + static_cast<size_t>(WARPS_PER_BLOCK) * K);

        float* const warp_sort_dist = sort_dist_all + static_cast<size_t>(warp_id) * K;
        int* const warp_sort_idx    = sort_idx_all  + static_cast<size_t>(warp_id) * K;

#pragma unroll
        for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
            const int pos = lane + (slot << 5);
            warp_sort_dist[pos] = top_dist[slot];
            warp_sort_idx[pos]  = top_idx[slot];
        }
        __syncwarp(kFullMask);

        warp_bitonic_sort_shared<K>(warp_sort_dist, warp_sort_idx, lane);

        const size_t out_base = static_cast<size_t>(query_id) * static_cast<size_t>(K);
#pragma unroll
        for (int pos = lane; pos < K; pos += kWarpSize) {
            result[out_base + pos].first  = warp_sort_idx[pos];
            result[out_base + pos].second = warp_sort_dist[pos];
        }
    }
}

template <int K>
void launch_knn_impl(const float2* query,
                     const int query_count,
                     const float2* data,
                     const int data_count,
                     ResultPair* result) {
    constexpr size_t kTileBytes =
        static_cast<size_t>(2) * kTilePoints * sizeof(float); // x[] + y[]
    constexpr size_t kSortBytes =
        static_cast<size_t>(kWarpsPerBlock) * K * (sizeof(float) + sizeof(int));
    constexpr size_t kSharedBytes = (kTileBytes > kSortBytes) ? kTileBytes : kSortBytes;

    // Opt in to the dynamic shared-memory footprint required by the chosen tile size
    // and by the worst-case final sort buffer (K=1024 => 64 KiB).
    cudaFuncSetAttribute(
        knn_kernel<K, kWarpsPerBlock, kTilePoints>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSharedBytes)
    );

    // This kernel is explicitly shared-memory centric.
    cudaFuncSetAttribute(
        knn_kernel<K, kWarpsPerBlock, kTilePoints>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100
    );
    cudaFuncSetCacheConfig(
        knn_kernel<K, kWarpsPerBlock, kTilePoints>,
        cudaFuncCachePreferShared
    );

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

    // Deliberately asynchronous: no device-wide synchronization here.
    knn_kernel<K, kWarpsPerBlock, kTilePoints>
        <<<grid, block, kSharedBytes>>>(query, query_count, data, data_count, result);
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

#define KNN_LAUNCH_CASE(KVAL) \
    case KVAL:                \
        launch_knn_impl<KVAL>(query, query_count, data, data_count, result); \
        return

    switch (k) {
        KNN_LAUNCH_CASE(32);
        KNN_LAUNCH_CASE(64);
        KNN_LAUNCH_CASE(128);
        KNN_LAUNCH_CASE(256);
        KNN_LAUNCH_CASE(512);
        KNN_LAUNCH_CASE(1024);
        default:
            // Per the problem statement, k is always valid.
            return;
    }

#undef KNN_LAUNCH_CASE
}