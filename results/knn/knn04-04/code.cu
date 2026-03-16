#include <cuda_runtime.h>
#include <utility>
#include <climits>

namespace {

// Exact warp-per-query k-NN for 2D points.
//
// Design choices tuned for modern data-center GPUs (A100/H100 class):
// - One warp computes one query.
// - A 256-thread block = 8 warps = 8 queries per block.
// - Data points are streamed in 1024-point tiles. 1024 is the maximum supported k,
//   so the whole initial top-k seed always comes from a single tile.
// - Each warp keeps its private exact top-k in shared memory (not local memory),
//   which avoids massive register pressure / local-memory spills for k up to 1024.
// - Candidate updates are buffered: lanes compact all points that beat the current
//   worst kept distance, then the warp sorts that small buffer and merges it into
//   the exact top-k cooperatively.
// - The top-k / candidate buffers are private per warp, but stored in shared memory;
//   warp-local synchronization uses __syncwarp(), while tile loading uses __syncthreads().
//
// Exactness note:
// The current threshold (worst distance in top-k) only decreases over time. Delaying
// a merge therefore can only admit extra candidates, never reject a true neighbor.
// This lets us amortize expensive exact merges without losing correctness.

constexpr int WARP_SIZE               = 32;
constexpr int BLOCK_THREADS           = 256;
constexpr int WARPS_PER_BLOCK         = BLOCK_THREADS / WARP_SIZE;
constexpr int BATCH_POINTS            = 1024;  // 256 threads * 4 loads/thread
constexpr int LOADS_PER_THREAD        = BATCH_POINTS / BLOCK_THREADS;
constexpr int PARTIAL_FLUSH_THRESHOLD = WARP_SIZE;  // flush a partially filled buffer once it reaches one warp's worth

static_assert(BLOCK_THREADS % WARP_SIZE == 0, "BLOCK_THREADS must be a multiple of warp size");
static_assert(BATCH_POINTS % BLOCK_THREADS == 0, "BATCH_POINTS must be divisible by BLOCK_THREADS");

struct alignas(8) Neighbor {
    float dist;
    int   idx;
};

template <int K>
struct KnnTraits {
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024]");
    static_assert((K & (K - 1)) == 0, "K must be a power of two");
    static_assert(K % WARP_SIZE == 0, "K must be divisible by warp size");

    static constexpr int items_per_lane = K / WARP_SIZE;

    // Candidate buffer cap:
    // - For small K, match K exactly.
    // - For large K, cap at 128 so that for K=1024 the whole block stays at 80 KiB:
    //     tile (8 KiB) + 8 warps * topk (8 KiB each) + 8 warps * cand (1 KiB each)
    //   which allows 2 resident 256-thread blocks / SM on A100 (164 KiB shared mem).
    static constexpr int buffer_cap = (K < 128 ? K : 128);

    static constexpr size_t shared_bytes =
        BATCH_POINTS * sizeof(float2) +
        WARPS_PER_BLOCK * K * sizeof(Neighbor) +
        WARPS_PER_BLOCK * buffer_cap * sizeof(Neighbor);
};

__device__ __forceinline__ float sq_l2_2d(float qx, float qy, const float2& p) {
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return __fmaf_rn(dx, dx, dy * dy);
}

__device__ __forceinline__ Neighbor pos_inf_neighbor() {
    Neighbor n;
    n.dist = CUDART_INF_F;
    n.idx  = INT_MAX;
    return n;
}

// In-place bitonic sort over a warp-private shared-memory array of power-of-two length N.
// Only one warp accesses the array, so __syncwarp() is sufficient.
template <int N>
__device__ __forceinline__ void bitonic_sort_shared(Neighbor* arr) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += WARP_SIZE) {
                const int ixj = i ^ stride;
                if (ixj > i) {
                    const bool ascending = ((i & size) == 0);

                    Neighbor a = arr[i];
                    Neighbor b = arr[ixj];

                    const bool do_swap = ascending ? (a.dist > b.dist) : (a.dist < b.dist);
                    if (do_swap) {
                        arr[i]   = b;
                        arr[ixj] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge-path / co-rank partitioning for merging A[0:K) and B[0:n), both sorted ascending by distance.
// Ties are resolved stably in favor of A; tie behavior is otherwise unconstrained by the API.
template <int K>
__device__ __forceinline__ int co_rank(int diag, const Neighbor* A, const Neighbor* B, int n) {
    int low  = diag - n;
    if (low < 0) low = 0;

    int high = (diag < K) ? diag : K;

    while (low <= high) {
        const int i = (low + high) >> 1;
        const int j = diag - i;

        const float a_im1 = (i > 0) ? A[i - 1].dist : -CUDART_INF_F;
        const float a_i   = (i < K) ? A[i].dist     :  CUDART_INF_F;
        const float b_jm1 = (j > 0) ? B[j - 1].dist : -CUDART_INF_F;
        const float b_j   = (j < n) ? B[j].dist     :  CUDART_INF_F;

        if (a_im1 > b_j) {
            high = i - 1;
        } else if (b_jm1 >= a_i) {
            low = i + 1;
        } else {
            return i;
        }
    }

    return low;
}

// Exact merge of the current sorted top-k with a sorted candidate list.
// Each lane produces a contiguous output segment of length K/32 entirely in registers,
// then writes it back in-place to topk. This avoids a second shared-memory top-k buffer.
template <int K>
__device__ __forceinline__ void merge_topk_with_candidates(Neighbor* topk, const Neighbor* cand, int cand_count) {
    constexpr int ITEMS_PER_LANE = KnnTraits<K>::items_per_lane;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const int out_begin = lane * ITEMS_PER_LANE;
    const int out_end   = out_begin + ITEMS_PER_LANE;

    const int a_begin = co_rank<K>(out_begin, topk, cand, cand_count);
    const int a_end   = co_rank<K>(out_end,   topk, cand, cand_count);
    int b_begin       = out_begin - a_begin;
    int b_end         = out_end   - a_end;

    int ai = a_begin;
    int bi = b_begin;

    // Primitive arrays are easier for the compiler to scalarize into registers than an array of structs.
    float out_dist[ITEMS_PER_LANE];
    int   out_idx [ITEMS_PER_LANE];

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        const bool a_valid = (ai < a_end);
        const bool b_valid = (bi < b_end);

        const float a_dist = a_valid ? topk[ai].dist : CUDART_INF_F;
        const int   a_idx  = a_valid ? topk[ai].idx  : INT_MAX;

        const float b_dist = b_valid ? cand[bi].dist : CUDART_INF_F;
        const int   b_idx  = b_valid ? cand[bi].idx  : INT_MAX;

        // Stable A-before-B on ties.
        const bool take_b = b_valid && (!a_valid || (b_dist < a_dist));

        out_dist[t] = take_b ? b_dist : a_dist;
        out_idx [t] = take_b ? b_idx  : a_idx;

        if (take_b) {
            ++bi;
        } else {
            ++ai;
        }
    }

    // Nobody writes topk until every lane has finished reading from it.
    __syncwarp();

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_LANE; ++t) {
        topk[out_begin + t].dist = out_dist[t];
        topk[out_begin + t].idx  = out_idx[t];
    }

    __syncwarp();
}

// Sort the current candidate buffer and merge it into the exact top-k.
template <int K>
__device__ __forceinline__ void flush_candidates(Neighbor* topk, Neighbor* cand, int& buf_count, float& threshold) {
    if (buf_count == 0) {
        return;
    }

    constexpr int BUFFER_CAP = KnnTraits<K>::buffer_cap;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Sort only the smallest power-of-two region that covers the current number of candidates.
    int sort_n = 32;
    if constexpr (BUFFER_CAP > 32) {
        if (buf_count > 32) {
            sort_n = 64;
        }
        if constexpr (BUFFER_CAP > 64) {
            if (buf_count > 64) {
                sort_n = 128;
            }
        }
    }

    // Pad the tail with +inf so bitonic sort keeps all real candidates at the front.
    for (int i = lane; i < sort_n; i += WARP_SIZE) {
        if (i >= buf_count) {
            cand[i] = pos_inf_neighbor();
        }
    }
    __syncwarp();

    if (sort_n == 32) {
        bitonic_sort_shared<32>(cand);
    } else if constexpr (BUFFER_CAP >= 64) {
        if (sort_n == 64) {
            bitonic_sort_shared<64>(cand);
        } else if constexpr (BUFFER_CAP >= 128) {
            bitonic_sort_shared<128>(cand);
        }
    }

    merge_topk_with_candidates<K>(topk, cand, buf_count);

    threshold = topk[K - 1].dist;
    buf_count = 0;
}

// Process tile[start:end) for one warp/query.
// Lanes compute one candidate each per 32-point sub-iteration, compact all points that pass the
// current threshold into the warp-private candidate buffer, and flush that buffer as needed.
template <int K>
__device__ __forceinline__ void process_tile_range(
    const float2* tile,
    int start,
    int end,
    int global_base,
    float qx,
    float qy,
    Neighbor* topk,
    Neighbor* cand,
    int& buf_count,
    float& threshold)
{
    constexpr int BUFFER_CAP = KnnTraits<K>::buffer_cap;
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

    const int lane = threadIdx.x & (WARP_SIZE - 1);

    for (int offset = start; offset < end; offset += WARP_SIZE) {
        const int local_idx = offset + lane;
        const bool valid_lane = (local_idx < end);

        float candidate_dist = CUDART_INF_F;
        int   candidate_idx  = -1;

        if (valid_lane) {
            const float2 p = tile[local_idx];
            candidate_dist = sq_l2_2d(qx, qy, p);
            candidate_idx  = global_base + local_idx;
        }

        bool keep = valid_lane && (candidate_dist < threshold);
        unsigned mask = __ballot_sync(FULL_MASK, keep);
        int n = __popc(mask);

        if (n == 0) {
            continue;
        }

        // If the current buffered candidates do not fit together with this warp-wide batch,
        // flush the buffer first, then re-test against the tighter threshold.
        if (buf_count + n > BUFFER_CAP) {
            flush_candidates<K>(topk, cand, buf_count, threshold);

            keep = valid_lane && (candidate_dist < threshold);
            mask = __ballot_sync(FULL_MASK, keep);
            n    = __popc(mask);

            if (n == 0) {
                continue;
            }
        }

        if (keep) {
            const unsigned lower_lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
            const int rank = __popc(mask & lower_lane_mask);

            cand[buf_count + rank].dist = candidate_dist;
            cand[buf_count + rank].idx  = candidate_idx;
        }

        // Shared-memory writes above must be visible before a possible flush reads the buffer.
        __syncwarp();

        buf_count += n;

        if (buf_count == BUFFER_CAP) {
            flush_candidates<K>(topk, cand, buf_count, threshold);
        }
    }
}

template <int K>
__global__ __launch_bounds__(BLOCK_THREADS, 2)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result)
{
    constexpr int ITEMS_PER_LANE = KnnTraits<K>::items_per_lane;
    constexpr int BUFFER_CAP     = KnnTraits<K>::buffer_cap;
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

    extern __shared__ unsigned char smem_raw[];

    // Dynamic shared-memory layout:
    // [data tile | topk for all warps | candidate buffers for all warps]
    float2*  tile     = reinterpret_cast<float2*>(smem_raw);
    Neighbor* topk_all = reinterpret_cast<Neighbor*>(tile + BATCH_POINTS);
    Neighbor* cand_all = topk_all + WARPS_PER_BLOCK * K;

    const int tid      = threadIdx.x;
    const int lane     = tid & (WARP_SIZE - 1);
    const int warp_id  = tid >> 5;
    const int query_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active  = (query_id < query_count);

    Neighbor* topk = topk_all + warp_id * K;
    Neighbor* cand = cand_all + warp_id * BUFFER_CAP;

    // Load the query point once per warp and broadcast it with shuffles.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_id];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    int   buf_count = 0;
    float threshold = CUDART_INF_F;

    for (int batch_base = 0; batch_base < data_count; batch_base += BATCH_POINTS) {
        int valid = data_count - batch_base;
        if (valid > BATCH_POINTS) {
            valid = BATCH_POINTS;
        }

        // Whole block cooperatively loads the current data tile into shared memory.
        #pragma unroll
        for (int i = 0; i < LOADS_PER_THREAD; ++i) {
            const int local = i * BLOCK_THREADS + tid;
            if (local < valid) {
                tile[local] = data[batch_base + local];
            }
        }

        __syncthreads();

        if (active) {
            if (batch_base == 0) {
                // Seed the exact top-k from the first K points of the first tile.
                // K is always <= 1024, so this seed is entirely contained in this tile.
                #pragma unroll
                for (int t = 0; t < ITEMS_PER_LANE; ++t) {
                    const int pos = t * WARP_SIZE + lane;
                    const float2 p = tile[pos];
                    topk[pos].dist = sq_l2_2d(qx, qy, p);
                    topk[pos].idx  = pos;  // batch_base == 0 here
                }

                __syncwarp();
                bitonic_sort_shared<K>(topk);
                threshold = topk[K - 1].dist;

                // Process the remainder of the first tile and then flush it unconditionally.
                // This bootstrap makes the threshold much tighter before scanning the long tail.
                if (valid > K) {
                    process_tile_range<K>(tile, K, valid, batch_base, qx, qy, topk, cand, buf_count, threshold);
                }
                flush_candidates<K>(topk, cand, buf_count, threshold);
            } else {
                process_tile_range<K>(tile, 0, valid, batch_base, qx, qy, topk, cand, buf_count, threshold);

                // Heuristic partial flush:
                // if at least one warp's worth of candidates has accumulated by the end of a tile,
                // pay the merge cost now to keep the threshold reasonably current.
                if (buf_count >= PARTIAL_FLUSH_THRESHOLD) {
                    flush_candidates<K>(topk, cand, buf_count, threshold);
                }
            }
        }

        // All warps must finish consuming this tile before the block overwrites shared memory.
        __syncthreads();
    }

    if (active) {
        // Final exact merge for any buffered tail candidates.
        flush_candidates<K>(topk, cand, buf_count, threshold);

        // topk is kept sorted ascending by distance, so writing it out directly yields
        // result[query_id * K + j] = j-th nearest neighbor (index, squared distance).
        const int out_base = query_id * K;

        #pragma unroll
        for (int t = 0; t < ITEMS_PER_LANE; ++t) {
            const int pos = t * WARP_SIZE + lane;
            result[out_base + pos].first  = topk[pos].idx;
            result[out_base + pos].second = topk[pos].dist;
        }
    }
}

template <int K>
void launch_knn_impl(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result)
{
    constexpr size_t SHARED_BYTES = KnnTraits<K>::shared_bytes;

    // Opt in to the required dynamic shared-memory footprint.
    (void)cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(SHARED_BYTES));

    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Asynchronous launch on the default stream, matching normal CUDA runtime semantics.
    knn_kernel<K><<<blocks, BLOCK_THREADS, SHARED_BYTES>>>(
        query, query_count, data, data_count, result);
}

}  // namespace

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
        case 32:   launch_knn_impl<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_impl<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_impl<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_impl<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_impl<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_impl<1024>(query, query_count, data, data_count, result); break;
        default:
            // Inputs are guaranteed valid by contract; keep the default branch for safety.
            break;
    }
}