#include <cuda_runtime.h>
#include <utility>

namespace {

// One warp owns one query, exactly as requested.
constexpr int WARP_SIZE = 32;

// Tuned for A100/H100-class GPUs:
// - 16 warps/block lets one data tile be reused by 16 concurrent queries.
// - 1024 cached float2 points/tile means exactly two global loads/thread/tile.
constexpr int BLOCK_THREADS = 512;
constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;
constexpr int TILE_POINTS = 1024;
constexpr int CANDIDATE_BUFFER = WARP_SIZE;
constexpr unsigned FULL_MASK = 0xffffffffu;

static_assert(BLOCK_THREADS % WARP_SIZE == 0, "Block size must be a multiple of warp size.");
static_assert(TILE_POINTS % WARP_SIZE == 0, "Tile size must be a multiple of warp size.");
static_assert(TILE_POINTS >= 1024, "The first tile must be able to hold the largest supported k.");

// Per-thread constant lane mask used for warp-local prefix sums.
__device__ __forceinline__ unsigned lane_mask_lt() {
    unsigned m;
    asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

// Squared Euclidean distance in 2D; no sqrt, as required.
__device__ __forceinline__ float squared_l2(const float qx, const float qy, const float2 p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

// In-register bitonic sort of exactly 32 (distance, index) pairs distributed one per lane.
// Ascending by distance; ties are intentionally left unspecified.
__device__ __forceinline__ void warp_sort32(float& dist, int& idx, const int lane) {
#pragma unroll
    for (int k = 2; k <= WARP_SIZE; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const float peer_dist = __shfl_xor_sync(FULL_MASK, dist, j);
            const int   peer_idx  = __shfl_xor_sync(FULL_MASK, idx,  j);

            const bool ascending = ((lane & k) == 0);
            const bool swap = ascending ? (dist > peer_dist) : (dist < peer_dist);
            if (swap) {
                dist = peer_dist;
                idx  = peer_idx;
            }
        }
    }
}

// One-time initialization sort of the first K distances kept in warp-private shared memory.
// This is deliberately not aggressively unrolled to avoid code size blow-up for K=1024.
template <int K>
__device__ __forceinline__ void warp_sort_shared_topk(float* dist, int* idx, const int lane) {
#pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
#pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll 1
            for (int i = lane; i < K; i += WARP_SIZE) {
                const int j = i ^ stride;
                if (j > i) {
                    const float di = dist[i];
                    const float dj = dist[j];
                    const int   ii = idx[i];
                    const int   ij = idx[j];

                    const bool ascending = ((i & size) == 0);
                    const bool swap = ascending ? (di > dj) : (di < dj);
                    if (swap) {
                        dist[i] = dj;
                        idx[i]  = ij;
                        dist[j] = di;
                        idx[j]  = ii;
                    }
                }
            }
            // Only the owning warp touches this segment, so __syncwarp is sufficient.
            __syncwarp();
        }
    }
}

// Parallel merge of:
//   A = current sorted top-K in shared memory (length K)
//   B = 32 sorted candidate pairs distributed one per lane (length 32)
// producing the smallest K elements of A union B back into A.
//
// Each lane computes exactly K/32 output elements in registers.
// That avoids a second K-sized shared-memory buffer while still updating K values
// cooperatively across the whole warp.
template <int K>
__device__ __forceinline__ void warp_merge_topk(
    float* top_dist,
    int* top_idx,
    float cand_dist_lane,
    int cand_idx_lane,
    const int lane)
{
    constexpr int K_PER_LANE = K / WARP_SIZE;

    float out_dist[K_PER_LANE];
    int   out_idx[K_PER_LANE];

    // Merge-path partition for the beginning of this lane's output interval.
    const int diag = lane * K_PER_LANE;
    int low = 0;
    int high = (diag < CANDIDATE_BUFFER) ? diag : CANDIDATE_BUFFER;
    int part_b = 0;
    bool found = false;

    while (low <= high) {
        const int b = (low + high) >> 1;
        const int a = diag - b;

        const float a_right = (a < K) ? top_dist[a] : CUDART_INF_F;
        const float a_left  = (a > 0) ? top_dist[a - 1] : -CUDART_INF_F;
        const float b_right = (b < CANDIDATE_BUFFER) ? __shfl_sync(FULL_MASK, cand_dist_lane, b) : CUDART_INF_F;
        const float b_left  = (b > 0) ? __shfl_sync(FULL_MASK, cand_dist_lane, b - 1) : -CUDART_INF_F;

        if (b > 0 && a < K && b_left > a_right) {
            high = b - 1;  // took too many from B
        } else if (a > 0 && b < CANDIDATE_BUFFER && a_left > b_right) {
            low = b + 1;   // took too few from B
        } else {
            part_b = b;
            found = true;
            break;
        }
    }
    if (!found) {
        part_b = low;
    }

    int a = diag - part_b;
    int b = part_b;

#pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        const float av = (a < K) ? top_dist[a] : CUDART_INF_F;
        const int   ai = (a < K) ? top_idx[a]  : -1;

        float bv = CUDART_INF_F;
        int   bi = -1;
        if (b < CANDIDATE_BUFFER) {
            bv = __shfl_sync(FULL_MASK, cand_dist_lane, b);
            bi = __shfl_sync(FULL_MASK, cand_idx_lane,  b);
        }

        // Strict '<' is intentional; tie handling is unspecified by the API.
        const bool take_b = (bv < av);
        out_dist[i] = take_b ? bv : av;
        out_idx[i]  = take_b ? bi : ai;

        b += static_cast<int>(take_b);
        a += static_cast<int>(!take_b);
    }

    // Make sure nobody overwrites the old shared-memory top-K before every lane
    // has finished reading from it.
    __syncwarp();

    const int out_base = diag;
#pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        top_dist[out_base + i] = out_dist[i];
        top_idx[out_base + i]  = out_idx[i];
    }

    __syncwarp();
}

// Merge a sorted 32-candidate warp vector into the current top-K if it can improve it.
template <int K>
__device__ __forceinline__ float warp_merge_sorted_candidates(
    float* top_dist,
    int* top_idx,
    float cand_dist_lane,
    int cand_idx_lane,
    const int lane,
    float worst_dist)
{
    const float best_candidate = __shfl_sync(FULL_MASK, cand_dist_lane, 0);
    if (best_candidate < worst_dist) {
        warp_merge_topk<K>(top_dist, top_idx, cand_dist_lane, cand_idx_lane, lane);

        float w = 0.0f;
        if (lane == 0) {
            w = top_dist[K - 1];
        }
        worst_dist = __shfl_sync(FULL_MASK, w, 0);
    }
    return worst_dist;
}

// Flush a partially filled candidate buffer (<= 32 elements) stored in warp-private shared memory.
template <int K>
__device__ __forceinline__ float warp_flush_candidates(
    float* top_dist,
    int* top_idx,
    const float* cand_dist_buf,
    const int* cand_idx_buf,
    const int buf_count,
    const int lane,
    float worst_dist)
{
    // Required because candidate buffer entries were produced by different lanes.
    __syncwarp();

    float cand_dist = (lane < buf_count) ? cand_dist_buf[lane] : CUDART_INF_F;
    int   cand_idx  = (lane < buf_count) ? cand_idx_buf[lane]  : -1;

    warp_sort32(cand_dist, cand_idx, lane);
    return warp_merge_sorted_candidates<K>(top_dist, top_idx, cand_dist, cand_idx, lane, worst_dist);
}

// Consume one shared-memory tile for one query warp.
// The exact current top-K stays in shared memory; improving points are buffered in batches of up to 32.
// This drastically reduces expensive K-way merges after the initial K fill, while remaining exact.
template <int K>
__device__ __forceinline__ void process_tile_for_query(
    const float2* tile,
    const int tile_base,
    const int start_offset,
    const int tile_count,
    const float qx,
    const float qy,
    float* top_dist,
    int* top_idx,
    float* cand_dist_buf,
    int* cand_idx_buf,
    const int lane,
    const unsigned lane_lt_mask,
    float& worst_dist,
    int& buf_count)
{
#pragma unroll 1
    for (int chunk = start_offset; chunk < tile_count; chunk += WARP_SIZE) {
        const int local_idx = chunk + lane;

        float dist = CUDART_INF_F;
        int data_idx = -1;
        if (local_idx < tile_count) {
            const float2 p = tile[local_idx];
            dist = squared_l2(qx, qy, p);
            data_idx = tile_base + local_idx;
        }

        // Strict '<' is enough because tie resolution is unspecified.
        bool better = (dist < worst_dist);
        unsigned mask = __ballot_sync(FULL_MASK, better);
        int count = __popc(mask);

        if (count == 0) {
            continue;
        }

        // Fast path: if the buffer is empty and all 32 lanes produced improving candidates,
        // sort and merge the chunk directly without touching the shared candidate buffer.
        if (buf_count == 0 && count == CANDIDATE_BUFFER) {
            warp_sort32(dist, data_idx, lane);
            worst_dist = warp_merge_sorted_candidates<K>(top_dist, top_idx, dist, data_idx, lane, worst_dist);
            continue;
        }

        // If the current chunk would overflow the candidate buffer, flush the buffer first,
        // then re-test the current chunk against the tighter threshold.
        if (buf_count + count > CANDIDATE_BUFFER) {
            worst_dist = warp_flush_candidates<K>(
                top_dist, top_idx, cand_dist_buf, cand_idx_buf, buf_count, lane, worst_dist);
            buf_count = 0;

            better = (dist < worst_dist);
            mask = __ballot_sync(FULL_MASK, better);
            count = __popc(mask);

            if (count == 0) {
                continue;
            }

            if (count == CANDIDATE_BUFFER) {
                warp_sort32(dist, data_idx, lane);
                worst_dist = warp_merge_sorted_candidates<K>(top_dist, top_idx, dist, data_idx, lane, worst_dist);
                continue;
            }
        }

        const int rank = __popc(mask & lane_lt_mask);
        if (better) {
            cand_dist_buf[buf_count + rank] = dist;
            cand_idx_buf[buf_count + rank] = data_idx;
        }
        buf_count += count;

        if (buf_count == CANDIDATE_BUFFER) {
            worst_dist = warp_flush_candidates<K>(
                top_dist, top_idx, cand_dist_buf, cand_idx_buf, buf_count, lane, worst_dist);
            buf_count = 0;
        }
    }
}

template <int K>
__global__ __launch_bounds__(BLOCK_THREADS, 1)
void knn_kernel(
    const float2* __restrict__ query,
    const int query_count,
    const float2* __restrict__ data,
    const int data_count,
    std::pair<int, float>* __restrict__ result)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];

    // Shared-memory layout:
    //   [data tile][top-K distances][top-K indices][candidate distances][candidate indices]
    float2* tile = reinterpret_cast<float2*>(smem_raw);
    unsigned char* ptr = reinterpret_cast<unsigned char*>(tile + TILE_POINTS);

    float* top_dist = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * K * sizeof(float);

    int* top_idx = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * K * sizeof(int);

    float* cand_dist = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * CANDIDATE_BUFFER * sizeof(float);

    int* cand_idx = reinterpret_cast<int*>(ptr);

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned lane_lt_mask = lane_mask_lt();

    const int qidx = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
    const bool active = (qidx < query_count);

    float qx = 0.0f;
    float qy = 0.0f;
    if (active) {
        const float2 q = query[qidx];
        qx = q.x;
        qy = q.y;
    }

    float* my_top_dist = top_dist + warp_id * K;
    int*   my_top_idx  = top_idx  + warp_id * K;
    float* my_cand_dist = cand_dist + warp_id * CANDIDATE_BUFFER;
    int*   my_cand_idx  = cand_idx  + warp_id * CANDIDATE_BUFFER;

    float worst_dist = CUDART_INF_F;
    int buf_count = 0;

    // First tile:
    // - everyone cooperatively loads it
    // - each active warp initializes its private top-K from the first K data points
    // - the remainder of the tile is processed with the normal buffered-update path
    const int first_tile_count = (data_count < TILE_POINTS) ? data_count : TILE_POINTS;
    for (int i = threadIdx.x; i < first_tile_count; i += BLOCK_THREADS) {
        tile[i] = data[i];
    }
    __syncthreads();

    if (active) {
#pragma unroll
        for (int i = lane; i < K; i += WARP_SIZE) {
            const float2 p = tile[i];
            my_top_dist[i] = squared_l2(qx, qy, p);
            my_top_idx[i] = i;
        }
        __syncwarp();

        warp_sort_shared_topk<K>(my_top_dist, my_top_idx, lane);

        float w = 0.0f;
        if (lane == 0) {
            w = my_top_dist[K - 1];
        }
        worst_dist = __shfl_sync(FULL_MASK, w, 0);

        process_tile_for_query<K>(
            tile, 0, K, first_tile_count,
            qx, qy,
            my_top_dist, my_top_idx,
            my_cand_dist, my_cand_idx,
            lane, lane_lt_mask,
            worst_dist, buf_count);
    }

    __syncthreads();

    // Remaining tiles.
    for (int tile_base = TILE_POINTS; tile_base < data_count; tile_base += TILE_POINTS) {
        const int remaining = data_count - tile_base;
        const int tile_count = (remaining < TILE_POINTS) ? remaining : TILE_POINTS;

        for (int i = threadIdx.x; i < tile_count; i += BLOCK_THREADS) {
            tile[i] = data[tile_base + i];
        }
        __syncthreads();

        if (active) {
            process_tile_for_query<K>(
                tile, tile_base, 0, tile_count,
                qx, qy,
                my_top_dist, my_top_idx,
                my_cand_dist, my_cand_idx,
                lane, lane_lt_mask,
                worst_dist, buf_count);
        }

        __syncthreads();
    }

    if (active) {
        if (buf_count > 0) {
            worst_dist = warp_flush_candidates<K>(
                my_top_dist, my_top_idx, my_cand_dist, my_cand_idx, buf_count, lane, worst_dist);
        }

        const size_t out_base = static_cast<size_t>(qidx) * K;
#pragma unroll
        for (int i = lane; i < K; i += WARP_SIZE) {
            result[out_base + i].first = my_top_idx[i];
            result[out_base + i].second = my_top_dist[i];
        }
    }
}

template <int K>
constexpr size_t shared_bytes_for_kernel() {
    return static_cast<size_t>(TILE_POINTS) * sizeof(float2) +
           static_cast<size_t>(WARPS_PER_BLOCK) * (
               static_cast<size_t>(K) * sizeof(float) +
               static_cast<size_t>(K) * sizeof(int) +
               static_cast<size_t>(CANDIDATE_BUFFER) * sizeof(float) +
               static_cast<size_t>(CANDIDATE_BUFFER) * sizeof(int));
}

template <int K>
inline void launch_knn_variant(
    const float2* query,
    const int query_count,
    const float2* data,
    const int data_count,
    std::pair<int, float>* result)
{
    constexpr size_t SHMEM_BYTES = shared_bytes_for_kernel<K>();

    // Worst case (K=1024) needs opt-in shared memory on A100/H100-class GPUs.
    cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SHMEM_BYTES));
    cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int grid_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    knn_kernel<K><<<grid_blocks, BLOCK_THREADS, SHMEM_BYTES>>>(
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

    // Dispatch to compile-time specializations:
    // k is guaranteed to be one of {32, 64, 128, 256, 512, 1024}.
    switch (k) {
        case 32:   launch_knn_variant<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_variant<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_variant<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_variant<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_variant<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_variant<1024>(query, query_count, data, data_count, result); break;
        default:
            // Unreachable for valid inputs.
            break;
    }

    // Intentionally no synchronization here:
    // the API keeps normal CUDA asynchronous launch semantics.
}