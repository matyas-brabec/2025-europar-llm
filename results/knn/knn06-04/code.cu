#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <utility>

/*
  Optimized warp-centric k-NN for 2D float points.

  Key design choices:
  - One warp computes one query.
  - The intermediate top-k for a query is warp-private and distributed across the 32 lanes:
      each lane owns K/32 (distance, index) pairs in registers.
  - Each warp also owns a K-entry candidate buffer in shared memory plus a shared counter.
  - The whole block streams the dataset through a shared-memory tile so that every loaded
    data point is reused by all warps in the block.
  - Candidate insertion uses a warp-aggregated atomicAdd on the per-warp shared counter,
    exactly as requested, to reserve contiguous slots in the candidate buffer.
  - Whenever the candidate buffer reaches K entries, it is merged with the warp-private
    intermediate result and max_distance is refreshed from the current k-th neighbor.
  - K is specialized at compile time for the only legal values {32,64,128,256,512,1024}
    so that the warp-private top-k stays in registers and merge/search loop bounds are known.
*/

namespace {

constexpr int WARP_THREADS     = 32;
constexpr unsigned FULL_MASK   = 0xFFFFFFFFu;

/*
  1024-point tiles keep the shared-memory footprint under the default 48 KiB dynamic
  shared-memory budget for all launch configurations used below:

    K <= 512 : 8 warps/block
    K = 1024 : 4 warps/block

  That avoids any need for dynamic shared-memory attribute tuning while still amortizing
  block-wide tile loads over multiple queries.
*/
constexpr int DATA_TILE_POINTS = 1024;

/*
  The public interface uses std::pair<int,float>. Device code writes a layout-compatible
  POD instead of depending on device availability of std::pair operations.
*/
struct PairIF {
    int   first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>),
              "std::pair<int,float> is not 8 bytes on this platform.");

template <int N>
struct StaticLog2 {
    static constexpr int value = 1 + StaticLog2<(N >> 1)>::value;
};

template <>
struct StaticLog2<1> {
    static constexpr int value = 0;
};

struct DistIndex {
    float dist;
    int   idx;
};

__device__ __forceinline__ DistIndex make_pos_inf()
{
    DistIndex v;
    v.dist = CUDART_INF_F;
    v.idx  = INT_MAX;
    return v;
}

__device__ __forceinline__ DistIndex make_neg_inf()
{
    DistIndex v;
    v.dist = -CUDART_INF_F;
    v.idx  = INT_MIN;
    return v;
}

/*
  Deterministic total order used only inside sort/merge.
  The user only requires max_distance to be the distance of the k-th neighbor, so the
  filter still uses strict "dist < max_distance" as requested; ties do not need special
  handling there.
*/
__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib)
{
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ bool pair_less(const DistIndex &a, const DistIndex &b)
{
    return pair_less(a.dist, a.idx, b.dist, b.idx);
}

/*
  Fetch an arbitrary element from the warp-distributed top-k array.
  Position p is owned by lane (p mod 32), slot (p / 32).
*/
__device__ __forceinline__ DistIndex warp_get_top(const float *top_dist, const int *top_idx, int pos)
{
    const int lane  = threadIdx.x & (WARP_THREADS - 1);
    const int owner = pos & (WARP_THREADS - 1);
    const int slot  = pos >> 5;

    float d = (lane == owner) ? top_dist[slot] : 0.0f;
    int   i = (lane == owner) ? top_idx[slot]  : 0;

    DistIndex v;
    v.dist = __shfl_sync(FULL_MASK, d, owner);
    v.idx  = __shfl_sync(FULL_MASK, i, owner);
    return v;
}

__device__ __forceinline__ DistIndex cand_get(const float *cand_dist, const int *cand_idx, int pos)
{
    DistIndex v;
    v.dist = cand_dist[pos];
    v.idx  = cand_idx[pos];
    return v;
}

/*
  Bitonic sort of the per-warp candidate buffer in shared memory.

  Only one warp touches each buffer, so warp-wide barriers are sufficient.
  The buffer is padded with +inf sentinels up to K so the whole sort/merge path can
  operate on fixed-size power-of-two arrays.
*/
template <int K>
__device__ __forceinline__ void sort_candidate_buffer(float *cand_dist, int *cand_idx, int count)
{
    const int lane = threadIdx.x & (WARP_THREADS - 1);

    for (int pos = lane; pos < K; pos += WARP_THREADS) {
        if (pos >= count) {
            cand_dist[pos] = CUDART_INF_F;
            cand_idx[pos]  = INT_MAX;
        }
    }
    __syncwarp(FULL_MASK);

#pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
#pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int pos = lane; pos < K; pos += WARP_THREADS) {
                const int partner = pos ^ stride;

                if (pos < partner) {
                    const float a_d = cand_dist[pos];
                    const int   a_i = cand_idx[pos];
                    const float b_d = cand_dist[partner];
                    const int   b_i = cand_idx[partner];

                    const bool up   = ((pos & size) == 0);
                    const bool swap = up ? pair_less(b_d, b_i, a_d, a_i)
                                         : pair_less(a_d, a_i, b_d, b_i);

                    if (swap) {
                        cand_dist[pos]     = b_d;
                        cand_idx[pos]      = b_i;
                        cand_dist[partner] = a_d;
                        cand_idx[partner]  = a_i;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

/*
  Return merged rank p from:
    A = current warp-private sorted top-k
    B = shared-memory sorted candidate buffer (padded to K with +inf)

  This uses a fixed-iteration binary search so that every lane executes the same shuffle
  sequence; that keeps warp shuffles well-defined even though the search decisions are lane-
  dependent.
*/
template <int K>
__device__ __forceinline__ DistIndex merged_rank(const float *top_dist,
                                                 const int   *top_idx,
                                                 const float *cand_dist,
                                                 const int   *cand_idx,
                                                 int p)
{
    constexpr int LOG2_K = StaticLog2<K>::value;

    int low  = 0;
    int high = p;

#pragma unroll
    for (int iter = 0; iter < LOG2_K + 1; ++iter) {
        const int i = (low + high) >> 1;
        const int j = p - i;

        const DistIndex a_im1 = (i > 0) ? warp_get_top(top_dist, top_idx, i - 1) : make_neg_inf();
        const DistIndex a_i   = warp_get_top(top_dist, top_idx, i);
        const DistIndex b_j   = cand_get(cand_dist, cand_idx, j);
        const DistIndex b_jm1 = (j > 0) ? cand_get(cand_dist, cand_idx, j - 1) : make_neg_inf();

        const bool go_left  = pair_less(b_j, a_im1);  // a[i-1] > b[j]
        const bool go_right = (!go_left) && pair_less(a_i, b_jm1); // b[j-1] > a[i]
        const bool found    = !(go_left || go_right);

        low  = found ? i : (go_right ? (i + 1) : low);
        high = found ? i : (go_left  ? (i - 1) : high);
    }

    const int i = low;
    const int j = p - i;

    const DistIndex a_i = warp_get_top(top_dist, top_idx, i);
    const DistIndex b_j = cand_get(cand_dist, cand_idx, j);

    return pair_less(b_j, a_i) ? b_j : a_i;
}

/*
  Merge the shared candidate buffer with the warp-private top-k.

  Important optimization:
  - The merge writes output ranks from high to low.
  - Output rank p only depends on input ranks <= p.
  - Therefore, higher slots can be overwritten in place without needing a second K-sized
    warp-private array.

  A warp barrier is placed before each slot-group write to ensure every lane has finished
  reading the old values from the current slot-group.
*/
template <int K>
__device__ __forceinline__ void merge_candidate_buffer(float *top_dist,
                                                       int   *top_idx,
                                                       float *cand_dist,
                                                       int   *cand_idx,
                                                       int   *cand_count,
                                                       float &max_distance)
{
    constexpr int ITEMS_PER_LANE = K / WARP_THREADS;

    const int lane = threadIdx.x & (WARP_THREADS - 1);

    int count = (lane == 0) ? *cand_count : 0;
    count = __shfl_sync(FULL_MASK, count, 0);

    if (count == 0) {
        return;
    }

    sort_candidate_buffer<K>(cand_dist, cand_idx, count);

    for (int slot = ITEMS_PER_LANE - 1; slot >= 0; --slot) {
        const int p = (slot << 5) + lane;
        const DistIndex out = merged_rank<K>(top_dist, top_idx, cand_dist, cand_idx, p);

        __syncwarp(FULL_MASK);
        top_dist[slot] = out.dist;
        top_idx[slot]  = out.idx;
    }

    /*
      The k-th neighbor is always position K-1, which belongs to lane 31 and the last slot
      because K is a multiple of 32.
    */
    max_distance = __shfl_sync(FULL_MASK, top_dist[ITEMS_PER_LANE - 1], WARP_THREADS - 1);

    if (lane == 0) {
        *cand_count = 0;
    }
    __syncwarp(FULL_MASK);
}

/*
  Insert at most one candidate per lane for the current 32-point chunk.

  The caller already computed the initial pending mask with __ballot_sync.
  We reserve buffer space with a single warp-aggregated atomicAdd on the per-warp shared
  counter, then use per-lane ranks inside that reservation to write candidates contiguously.

  If the reservation fills or overflows the buffer, the buffer is merged immediately so
  max_distance stays current, exactly as requested.
*/
template <int K>
__device__ __forceinline__ void insert_candidate_group(unsigned pending_mask,
                                                       bool pending,
                                                       int candidate_idx,
                                                       float candidate_dist,
                                                       float *top_dist,
                                                       int   *top_idx,
                                                       float *cand_dist,
                                                       int   *cand_idx,
                                                       int   *cand_count,
                                                       float &max_distance)
{
    const int lane = threadIdx.x & (WARP_THREADS - 1);
    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    while (pending_mask != 0u) {
        const int pending_count = __popc(pending_mask);

        int base = 0;
        if (lane == 0) {
            base = atomicAdd(cand_count, pending_count);
        }
        base = __shfl_sync(FULL_MASK, base, 0);

        const int rank = __popc(pending_mask & lane_mask_lt);

        int take = 0;
        if (base < K) {
            const int space = K - base;
            take = (pending_count < space) ? pending_count : space;
        }

        if (pending && rank < take) {
            const int pos = base + rank;
            cand_dist[pos] = candidate_dist;
            cand_idx[pos]  = candidate_idx;
        }

        __syncwarp(FULL_MASK);

        /*
          If the buffer is not yet full, we are done for this chunk.
          If it became exactly full or overflowed, we force a merge now.
        */
        if (base + pending_count < K) {
            break;
        }

        if (lane == 0) {
            *cand_count = K;
        }
        __syncwarp(FULL_MASK);

        merge_candidate_buffer<K>(top_dist, top_idx, cand_dist, cand_idx, cand_count, max_distance);

        /*
          Candidates that overflowed the full buffer are reconsidered against the updated
          max_distance. This can discard many of them immediately after the merge.
        */
        if (base + pending_count == K) {
            pending = false;
        } else {
            pending = pending && (rank >= take) && (candidate_dist < max_distance);
        }

        pending_mask = __ballot_sync(FULL_MASK, pending);
    }
}

template <int K, int BLOCK_WARPS>
constexpr std::size_t shared_bytes_for_kernel()
{
    return std::size_t(DATA_TILE_POINTS) * sizeof(float2) +
           std::size_t(BLOCK_WARPS) * sizeof(int) +
           std::size_t(BLOCK_WARPS) * std::size_t(K) * (sizeof(float) + sizeof(int));
}

template <int K, int BLOCK_WARPS>
__launch_bounds__(BLOCK_WARPS * WARP_THREADS)
__global__ void knn_kernel(const float2 *__restrict__ query,
                           int query_count,
                           const float2 *__restrict__ data,
                           int data_count,
                           PairIF *__restrict__ result)
{
    static_assert((K % WARP_THREADS) == 0, "K must be a multiple of 32.");
    static_assert((BLOCK_WARPS * WARP_THREADS) <= 1024, "Block size exceeds CUDA limit.");

    constexpr int ITEMS_PER_LANE = K / WARP_THREADS;
    constexpr int BLOCK_THREADS  = BLOCK_WARPS * WARP_THREADS;

    const int lane = threadIdx.x & (WARP_THREADS - 1);
    const int warp = threadIdx.x >> 5;

    const int query_idx = static_cast<int>(blockIdx.x) * BLOCK_WARPS + warp;
    const bool active   = (query_idx < query_count);

    /*
      Shared-memory layout:
        [DATA_TILE_POINTS * float2]
        [BLOCK_WARPS * int counters]
        [BLOCK_WARPS * (K floats + K ints) candidate buffers]
    */
    extern __shared__ unsigned char smem[];

    float2 *s_data = reinterpret_cast<float2 *>(smem);

    unsigned char *ptr = reinterpret_cast<unsigned char *>(s_data + DATA_TILE_POINTS);

    int *s_counts = reinterpret_cast<int *>(ptr);
    ptr += std::size_t(BLOCK_WARPS) * sizeof(int);

    constexpr std::size_t PER_WARP_BUFFER_BYTES =
        std::size_t(K) * sizeof(float) + std::size_t(K) * sizeof(int);

    unsigned char *warp_buffer = ptr + std::size_t(warp) * PER_WARP_BUFFER_BYTES;
    float *warp_cand_dist = reinterpret_cast<float *>(warp_buffer);
    int   *warp_cand_idx  = reinterpret_cast<int *>(warp_buffer + std::size_t(K) * sizeof(float));
    int   *warp_count     = s_counts + warp;

    /*
      Warp-private intermediate top-k.
      Each lane owns K/32 entries; together, the warp holds the full sorted top-k.
    */
    float top_dist[ITEMS_PER_LANE];
    int   top_idx[ITEMS_PER_LANE];

#pragma unroll
    for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
        top_dist[slot] = CUDART_INF_F;
        top_idx[slot]  = INT_MAX;
    }

    if (lane == 0) {
        *warp_count = 0;
    }

    /*
      One thread loads the query point and broadcasts it to the whole warp.
    */
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    float max_distance = CUDART_INF_F;

    /*
      Stream the full dataset through a block-shared tile. Every warp reuses the same tile
      for its own query before the block moves to the next batch.
    */
    for (int tile_base = 0; tile_base < data_count; tile_base += DATA_TILE_POINTS) {
        int tile_size = data_count - tile_base;
        if (tile_size > DATA_TILE_POINTS) {
            tile_size = DATA_TILE_POINTS;
        }

#pragma unroll
        for (int load = threadIdx.x; load < DATA_TILE_POINTS; load += BLOCK_THREADS) {
            if (load < tile_size) {
                s_data[load] = data[tile_base + load];
            }
        }

        __syncthreads();

        if (active) {
            /*
              Process the tile in 32-point chunks: lane l handles tile element chunk+l.
              A warp ballot identifies the candidates that beat max_distance, then a single
              atomicAdd reserves contiguous slots in the warp-local shared candidate buffer.
            */
            for (int chunk = 0; chunk < tile_size; chunk += WARP_THREADS) {
                const int local_idx = chunk + lane;

                float dist = 0.0f;
                int data_idx = tile_base + local_idx;
                bool pass = false;

                if (local_idx < tile_size) {
                    const float2 p = s_data[local_idx];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);  // squared L2 distance
                    pass = (dist < max_distance);
                }

                const unsigned pass_mask = __ballot_sync(FULL_MASK, pass);
                if (pass_mask != 0u) {
                    insert_candidate_group<K>(pass_mask,
                                              pass,
                                              data_idx,
                                              dist,
                                              top_dist,
                                              top_idx,
                                              warp_cand_dist,
                                              warp_cand_idx,
                                              warp_count,
                                              max_distance);
                }
            }
        }

        __syncthreads();
    }

    /*
      Flush any partially filled candidate buffer after the last data tile.
    */
    if (active) {
        int count = (lane == 0) ? *warp_count : 0;
        count = __shfl_sync(FULL_MASK, count, 0);

        if (count > 0) {
            merge_candidate_buffer<K>(top_dist,
                                      top_idx,
                                      warp_cand_dist,
                                      warp_cand_idx,
                                      warp_count,
                                      max_distance);
        }

        /*
          The final warp-private top-k is already sorted in ascending distance order.
          Distances are squared Euclidean distances, per the interface contract.
        */
        PairIF *out = result + std::size_t(query_idx) * std::size_t(K);

#pragma unroll
        for (int slot = 0; slot < ITEMS_PER_LANE; ++slot) {
            const int pos = (slot << 5) + lane;
            out[pos].first  = top_idx[slot];
            out[pos].second = top_dist[slot];
        }
    }
}

template <int K, int BLOCK_WARPS>
void launch_knn(const float2 *query,
                int query_count,
                const float2 *data,
                int data_count,
                PairIF *result)
{
    static_assert(shared_bytes_for_kernel<K, BLOCK_WARPS>() <= (48u * 1024u),
                  "Chosen tile/block configuration exceeds the default 48 KiB shared-memory budget.");

    const dim3 block(BLOCK_WARPS * WARP_THREADS);
    const dim3 grid((query_count + BLOCK_WARPS - 1) / BLOCK_WARPS);
    constexpr std::size_t shared_bytes = shared_bytes_for_kernel<K, BLOCK_WARPS>();

    knn_kernel<K, BLOCK_WARPS><<<grid, block, shared_bytes>>>(query,
                                                              query_count,
                                                              data,
                                                              data_count,
                                                              result);
}

} // anonymous namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0) {
        return;
    }

    PairIF *device_result = reinterpret_cast<PairIF *>(result);

    /*
      Dispatch over the only legal K values. The selected block sizes are chosen so that:
      - K <= 512 uses 8 warps/block for good tile reuse and enough blocks for typical
        query counts in the low-thousands on A100/H100-class GPUs.
      - K = 1024 uses 4 warps/block to keep per-thread state and shared-memory usage balanced.
    */
    switch (k) {
        case 32:
            launch_knn<32, 8>(query, query_count, data, data_count, device_result);
            break;
        case 64:
            launch_knn<64, 8>(query, query_count, data, data_count, device_result);
            break;
        case 128:
            launch_knn<128, 8>(query, query_count, data, data_count, device_result);
            break;
        case 256:
            launch_knn<256, 8>(query, query_count, data, data_count, device_result);
            break;
        case 512:
            launch_knn<512, 8>(query, query_count, data, data_count, device_result);
            break;
        case 1024:
            launch_knn<1024, 4>(query, query_count, data, data_count, device_result);
            break;
        default:
            /*
              Per the problem statement this path is unreachable, but keep a silent return
              so the interface remains well-defined even if called incorrectly.
            */
            break;
    }
}