#include <cuda_runtime.h>
#include <utility>

// Tuned warp-per-query 2D k-NN for modern NVIDIA data-center GPUs (A100/H100 class).
//
// Core design:
//   * One warp computes one query.
//   * The current top-K is kept as a warp-private sorted array distributed across the 32 lanes.
//     Each lane owns K/32 (indices, distances) pairs in private per-thread storage.
//   * Each warp also owns a K-entry candidate buffer in shared memory.
//   * The whole block stages a batch of data points in shared memory so every loaded point
//     is reused by all query-warps in the block.
//   * Shared memory uses SoA for the data tile (x[] and y[]) to avoid the 64-bit shared-memory
//     bank conflicts that a float2 AoS layout would create.
//   * When the candidate buffer fills, it is sorted in shared memory with a warp-wide bitonic
//     network and then merged with the current top-K via merge-path. This keeps the top-K
//     globally sorted, so the K-th distance is always cheap to read and can be used for pruning.
//
// No extra device allocations are performed; all scratch space is per-thread private state or
// per-block shared memory.
//
// The host entry point intentionally only launches the kernel on the default stream; it does not
// insert a device-wide synchronization so callers can pipeline it with other GPU work.

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

// Runtime-tunable tile sizes. The launcher picks among these based on the selected
// warps-per-block and the device's opt-in shared-memory capacity.
//
// 2048 points is the conservative "keep at least 2 resident blocks if possible" tile.
// 4096 points is the preferred large tile when it does not hurt residency.
// 8192 points is only used when the kernel is already forced into a 1-block-per-SM regime;
// on H100-class parts it reduces refill/barrier overhead for the heaviest K values.
constexpr int kMinBatchPoints                  = 32;
constexpr int kPreferredBatchTwoBlockSmall     = 2048;
constexpr int kPreferredBatchTwoBlockLarge     = 4096;
constexpr int kPreferredBatchOneBlockHeavy     = 8192;

// Total-order comparator used by the sort/merge code.
// Distances are the primary key; index is a deterministic tiebreaker so merge-path partitions
// are unambiguous. The public API does not require any specific tie resolution.
__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ float squared_l2(float qx, float qy, float px, float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return fmaf(dx, dx, dy * dy);
}

// In-place bitonic sort of K (distance, index) pairs stored in shared memory.
// K is a power of two in [32, 1024]. One warp cooperatively sorts one buffer.
// The loops are intentionally kept rolled to avoid massive code growth for K=1024;
// merge events are relatively infrequent compared to the streaming distance phase.
template <int K>
__device__ __forceinline__ void warp_bitonic_sort_shared(float* dist, int* idx, int lane) {
    #pragma unroll 1
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll 1
            for (int i = lane; i < K; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool ascending = ((i & size) == 0);

                    const float di = dist[i];
                    const float dj = dist[j];
                    const int   ii = idx[i];
                    const int   ij = idx[j];

                    const bool do_swap =
                        ascending ? pair_less(dj, ij, di, ii)
                                  : pair_less(di, ii, dj, ij);

                    if (do_swap) {
                        dist[i] = dj;
                        idx[i]  = ij;
                        dist[j] = di;
                        idx[j]  = ii;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

// Merge-path partition for the first `diag` outputs of merge(A, B),
// where A and B are both length-K sorted arrays.
template <int K>
__device__ __forceinline__ int merge_path_partition(
    int diag,
    const float* a_dist, const int* a_idx,
    const float* b_dist, const int* b_idx)
{
    int lo = (diag > K) ? (diag - K) : 0;
    int hi = (diag < K) ? diag : K;

    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        const int j   = diag - mid;

        const bool move_right =
            (mid < K) && (j > 0) &&
            pair_less(a_dist[mid], a_idx[mid], b_dist[j - 1], b_idx[j - 1]);

        if (move_right) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Merge the warp's candidate buffer with the current top-K.
//
// Inputs:
//   * top_dist/top_idx : warp-private current top-K, globally sorted across the warp
//   * s_cand_*         : shared-memory candidate buffer for this warp
//   * s_top_*          : shared-memory temporary copy of the current top-K
//
// Output:
//   * top_dist/top_idx updated in place to the new top-K, still globally sorted
//   * return value is the new K-th distance, broadcast from lane 31
template <int K>
__device__ __forceinline__ float merge_buffer_warp(
    float (&top_dist)[K / kWarpSize],
    int   (&top_idx )[K / kWarpSize],
    float* s_cand_dist, int* s_cand_idx,
    float* s_top_dist,  int* s_top_idx,
    int candidate_count,
    int lane)
{
    constexpr int kItemsPerThread = K / kWarpSize;

    // Ensure all previous candidate writes are visible within the warp.
    __syncwarp(kFullMask);

    // Copy the current top-K from warp-private storage to shared memory so every lane can
    // access it randomly during merge-path partitioning and merging.
    #pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const int pos = lane * kItemsPerThread + i;
        s_top_dist[pos] = top_dist[i];
        s_top_idx [pos] = top_idx [i];
    }

    // Pad the inactive tail of the candidate buffer with sentinels so the sort can always
    // run on a full power-of-two array.
    for (int pos = candidate_count + lane; pos < K; pos += kWarpSize) {
        s_cand_dist[pos] = CUDART_INF_F;
        s_cand_idx [pos] = -1;
    }

    __syncwarp(kFullMask);

    // Sort candidates ascending by (distance, index).
    warp_bitonic_sort_shared<K>(s_cand_dist, s_cand_idx, lane);

    // Each lane merges a contiguous stripe of size K/32 from the first K outputs.
    const int diag = lane * kItemsPerThread;
    int a = merge_path_partition<K>(diag, s_top_dist, s_top_idx, s_cand_dist, s_cand_idx);
    int b = diag - a;

    #pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        const bool take_a =
            (a < K) &&
            ((b >= K) || !pair_less(s_cand_dist[b], s_cand_idx[b], s_top_dist[a], s_top_idx[a]));

        if (take_a) {
            top_dist[i] = s_top_dist[a];
            top_idx [i] = s_top_idx [a];
            ++a;
        } else {
            top_dist[i] = s_cand_dist[b];
            top_idx [i] = s_cand_idx [b];
            ++b;
        }
    }

    // Lane 31 owns the last global slot of the distributed top-K.
    return __shfl_sync(kFullMask, top_dist[kItemsPerThread - 1], 31);
}

template <int K, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * kWarpSize)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result,
    int batch_points)
{
    static_assert((K & (K - 1)) == 0, "K must be a power of two");
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024]");
    static_assert((K % kWarpSize) == 0, "K must be divisible by warp size");
    static_assert(WARPS_PER_BLOCK >= 1 && WARPS_PER_BLOCK <= 32, "Invalid warps-per-block");
    static_assert(WARPS_PER_BLOCK * kWarpSize <= 1024, "Block is too large");

    constexpr int kItemsPerThread = K / kWarpSize;

    const int tid  = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    const int block_query_base = blockIdx.x * WARPS_PER_BLOCK;
    if (block_query_base >= query_count) {
        return;
    }

    const int query_idx = block_query_base + warp;
    const bool active   = (query_idx < query_count);

    // Shared-memory layout:
    //   [cand_dist][cand_idx][top_dist_tmp][top_idx_tmp][tile_x][tile_y]
    extern __shared__ int shared_i32[];
    unsigned char* shared_ptr = reinterpret_cast<unsigned char*>(shared_i32);

    float* s_cand_dist_all = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(float);

    int* s_cand_idx_all = reinterpret_cast<int*>(shared_ptr);
    shared_ptr += static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(int);

    float* s_top_dist_all = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(float);

    int* s_top_idx_all = reinterpret_cast<int*>(shared_ptr);
    shared_ptr += static_cast<size_t>(WARPS_PER_BLOCK) * K * sizeof(int);

    float* s_tile_x = reinterpret_cast<float*>(shared_ptr);
    shared_ptr += static_cast<size_t>(batch_points) * sizeof(float);

    float* s_tile_y = reinterpret_cast<float*>(shared_ptr);

    const int warp_offset = warp * K;

    float* s_cand_dist = s_cand_dist_all + warp_offset;
    int*   s_cand_idx  = s_cand_idx_all  + warp_offset;
    float* s_top_dist  = s_top_dist_all  + warp_offset;
    int*   s_top_idx   = s_top_idx_all   + warp_offset;

    // One query is loaded once per warp and broadcast via shuffles.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Warp-private current top-K. These arrays are the "private copy" requested by the prompt.
    // The global order is:
    //   lane 0 owns ranks [0, K/32),
    //   lane 1 owns ranks [K/32, 2*K/32), ...
    float top_dist[kItemsPerThread];
    int   top_idx [kItemsPerThread];

    #pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        top_dist[i] = CUDART_INF_F;
        top_idx [i] = -1;
    }

    float kth = CUDART_INF_F;
    int buffer_count = 0;

    // Stream over the data set in shared-memory tiles.
    for (int batch_start = 0; batch_start < data_count; batch_start += batch_points) {
        int batch_count = data_count - batch_start;
        if (batch_count > batch_points) {
            batch_count = batch_points;
        }

        // Cooperative block load into shared memory.
        for (int i = tid; i < batch_count; i += blockDim.x) {
            const float2 p = data[batch_start + i];
            s_tile_x[i] = p.x;
            s_tile_y[i] = p.y;
        }

        __syncthreads();

        if (active) {
            // All 32 lanes must participate in every ballot. Therefore the loop is organized
            // in warp-sized rounds with explicit bounds checks for the tail.
            for (int base = 0; base < batch_count; base += kWarpSize) {
                const int off = base + lane;

                float dist = CUDART_INF_F;
                int   idx  = -1;
                bool  keep = false;

                if (off < batch_count) {
                    const float px = s_tile_x[off];
                    const float py = s_tile_y[off];
                    dist = squared_l2(qx, qy, px, py);
                    idx  = batch_start + off;

                    // Strictly closer than the current K-th neighbor; ties can be dropped.
                    keep = (dist < kth);
                }

                unsigned mask = __ballot_sync(kFullMask, keep);
                int num = __popc(mask);

                if (num) {
                    // If this round would overflow the shared candidate buffer, first merge
                    // the existing buffered candidates into the current top-K.
                    if (buffer_count + num > K) {
                        kth = merge_buffer_warp<K>(
                            top_dist, top_idx,
                            s_cand_dist, s_cand_idx,
                            s_top_dist, s_top_idx,
                            buffer_count, lane);
                        buffer_count = 0;

                        // Re-evaluate the current round under the tightened threshold.
                        keep = (off < batch_count) && (dist < kth);
                        mask = __ballot_sync(kFullMask, keep);
                        num  = __popc(mask);
                    }

                    if (num) {
                        if (keep) {
                            const unsigned lower_mask = (1u << lane) - 1u;
                            const int rank = __popc(mask & lower_mask);
                            s_cand_dist[buffer_count + rank] = dist;
                            s_cand_idx [buffer_count + rank] = idx;
                        }

                        buffer_count += num;

                        // Merge immediately when the buffer reaches K candidates.
                        if (buffer_count == K) {
                            kth = merge_buffer_warp<K>(
                                top_dist, top_idx,
                                s_cand_dist, s_cand_idx,
                                s_top_dist, s_top_idx,
                                buffer_count, lane);
                            buffer_count = 0;
                        }
                    }
                }
            }
        }

        // The whole block must finish reading the staged tile before it is overwritten.
        __syncthreads();
    }

    if (active) {
        if (buffer_count > 0) {
            kth = merge_buffer_warp<K>(
                top_dist, top_idx,
                s_cand_dist, s_cand_idx,
                s_top_dist, s_top_idx,
                buffer_count, lane);
            (void)kth;
        }

        // Write the final globally sorted top-K for this query.
        const size_t out_base =
            static_cast<size_t>(query_idx) * static_cast<size_t>(K) +
            static_cast<size_t>(lane) * static_cast<size_t>(kItemsPerThread);

        #pragma unroll
        for (int i = 0; i < kItemsPerThread; ++i) {
            result[out_base + i].first  = top_idx[i];
            result[out_base + i].second = top_dist[i];
        }
    }
}

inline int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

// Shared bytes for one warp/query:
//   candidate buffer: K * (float + int)
//   merge temp copy : K * (float + int)
// Total = 2 * K * (4 + 4) = 16 * K bytes.
inline size_t fixed_shared_bytes_runtime(int k, int warps_per_block) {
    return static_cast<size_t>(k) * static_cast<size_t>(warps_per_block) * 16u;
}

template <int K, int WARPS_PER_BLOCK>
inline size_t dynamic_shared_bytes(int batch_points) {
    return static_cast<size_t>(K) * static_cast<size_t>(WARPS_PER_BLOCK) * 16u +
           static_cast<size_t>(batch_points) * 2u * sizeof(float);
}

// Warps-per-block selection heuristic:
//
// We want as many query-warps in a block as possible because staged data is reused by every warp,
// but we also do not want the grid to become so small that it under-fills the SMs.
// Because each block scans the entire data set, blocks are long-lived, so "at least ~1 block per SM
// in the grid" is a good target; going far beyond that is less important than reuse.
//
// Candidate warp counts are powers of two. The heavy K values are capped by shared-memory usage.
inline int choose_warps_per_block(int k, int query_count, int sm_count, int max_dynamic_smem) {
    const int candidates[] = {32, 16, 8, 4};
    const int max_w =
        (k <= 256) ? 32 :
        (k == 512) ? 16 :
                     8;

    int chosen = 0;

    for (int i = 0; i < 4; ++i) {
        const int w = candidates[i];
        if (w > max_w) {
            continue;
        }

        // Require enough space for at least a minimal data tile.
        const size_t fixed = fixed_shared_bytes_runtime(k, w);
        const size_t min_total =
            fixed + static_cast<size_t>(kMinBatchPoints) * 2u * sizeof(float);

        if (min_total > static_cast<size_t>(max_dynamic_smem)) {
            continue;
        }

        chosen = w;

        // Prefer the largest warp count that still yields at least ~one block per SM.
        if (ceil_div_int(query_count, w) >= sm_count) {
            return w;
        }
    }

    return chosen;
}

// Batch-size selection heuristic:
//
// 1) If a 4096-point tile still allows >= 2 resident blocks by shared memory, use 4096.
// 2) Else if 2048 still allows >= 2 resident blocks, use 2048.
// 3) Else the kernel is already effectively in a 1-block-per-SM regime, so use a deeper tile
//    (up to 8192, clamped by device shared memory). This mainly helps the large-K cases on H100.
//
// This strikes a balance between fewer barriers/refills and not over-allocating shared memory
// on lighter kernels where extra occupancy still matters.
inline int choose_batch_points(int k, int warps_per_block, int max_dynamic_smem) {
    const size_t fixed    = fixed_shared_bytes_runtime(k, warps_per_block);
    const size_t max_smem = static_cast<size_t>(max_dynamic_smem);

    if (fixed + static_cast<size_t>(kMinBatchPoints) * 2u * sizeof(float) > max_smem) {
        return 0;
    }

    const int max_fit =
        static_cast<int>((max_smem - fixed) / (2u * sizeof(float)));

    const size_t bytes_2048 =
        fixed + static_cast<size_t>(kPreferredBatchTwoBlockSmall) * 2u * sizeof(float);
    const size_t bytes_4096 =
        fixed + static_cast<size_t>(kPreferredBatchTwoBlockLarge) * 2u * sizeof(float);

    int preferred = 0;
    if (bytes_4096 <= max_smem && (max_smem / bytes_4096) >= 2u) {
        preferred = kPreferredBatchTwoBlockLarge;
    } else if (bytes_2048 <= max_smem && (max_smem / bytes_2048) >= 2u) {
        preferred = kPreferredBatchTwoBlockSmall;
    } else {
        preferred = kPreferredBatchOneBlockHeavy;
    }

    int batch_points = preferred;
    if (batch_points > max_fit) {
        batch_points = max_fit;
    }

    // Keep the warp-round inner loop aligned.
    batch_points &= ~31;

    if (batch_points < kMinBatchPoints) {
        batch_points = kMinBatchPoints;
    }

    return batch_points;
}

template <int K, int WARPS_PER_BLOCK>
inline void launch_knn_specialized(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int batch_points)
{
    const size_t smem_bytes = dynamic_shared_bytes<K, WARPS_PER_BLOCK>(batch_points);

    // Opt in to large dynamic shared-memory allocations on Ampere/Hopper.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    // Heavily shared-memory-centric kernel: bias the carveout toward shared memory.
    (void)cudaFuncSetAttribute(
        knn_kernel<K, WARPS_PER_BLOCK>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    const int blocks  = ceil_div_int(query_count, WARPS_PER_BLOCK);
    const int threads = WARPS_PER_BLOCK * kWarpSize;

    knn_kernel<K, WARPS_PER_BLOCK><<<blocks, threads, smem_bytes>>>(
        query, query_count, data, data_count, result, batch_points);

    // Preserve CUDA's usual async execution model while still surfacing launch failures.
    (void)cudaPeekAtLastError();
}

template <int K>
inline void dispatch_warps_up_to_32(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int warps_per_block,
    int batch_points)
{
    switch (warps_per_block) {
        case 32:
            launch_knn_specialized<K, 32>(query, query_count, data, data_count, result, batch_points);
            break;
        case 16:
            launch_knn_specialized<K, 16>(query, query_count, data, data_count, result, batch_points);
            break;
        case 8:
            launch_knn_specialized<K, 8>(query, query_count, data, data_count, result, batch_points);
            break;
        default:
            launch_knn_specialized<K, 4>(query, query_count, data, data_count, result, batch_points);
            break;
    }
}

template <int K>
inline void dispatch_warps_up_to_16(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int warps_per_block,
    int batch_points)
{
    switch (warps_per_block) {
        case 16:
            launch_knn_specialized<K, 16>(query, query_count, data, data_count, result, batch_points);
            break;
        case 8:
            launch_knn_specialized<K, 8>(query, query_count, data, data_count, result, batch_points);
            break;
        default:
            launch_knn_specialized<K, 4>(query, query_count, data, data_count, result, batch_points);
            break;
    }
}

template <int K>
inline void dispatch_warps_up_to_8(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int warps_per_block,
    int batch_points)
{
    switch (warps_per_block) {
        case 8:
            launch_knn_specialized<K, 8>(query, query_count, data, data_count, result, batch_points);
            break;
        default:
            launch_knn_specialized<K, 4>(query, query_count, data, data_count, result, batch_points);
            break;
    }
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

    int device = 0;
    (void)cudaGetDevice(&device);

    int sm_count = 0;
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count <= 0) {
        sm_count = 1;
    }

    int max_dynamic_smem = 0;
    (void)cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_dynamic_smem <= 0) {
        (void)cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    }
    if (max_dynamic_smem <= 0) {
        max_dynamic_smem = 48 * 1024;
    }

    const int warps_per_block = choose_warps_per_block(k, query_count, sm_count, max_dynamic_smem);
    if (warps_per_block == 0) {
        return;
    }

    const int batch_points = choose_batch_points(k, warps_per_block, max_dynamic_smem);
    if (batch_points == 0) {
        return;
    }

    switch (k) {
        case 32:
            dispatch_warps_up_to_32<32>(query, query_count, data, data_count, result, warps_per_block, batch_points);
            break;
        case 64:
            dispatch_warps_up_to_32<64>(query, query_count, data, data_count, result, warps_per_block, batch_points);
            break;
        case 128:
            dispatch_warps_up_to_32<128>(query, query_count, data, data_count, result, warps_per_block, batch_points);
            break;
        case 256:
            dispatch_warps_up_to_32<256>(query, query_count, data, data_count, result, warps_per_block, batch_points);
            break;
        case 512:
            dispatch_warps_up_to_16<512>(query, query_count, data, data_count, result, warps_per_block, batch_points);
            break;
        case 1024:
            dispatch_warps_up_to_8<1024>(query, query_count, data, data_count, result, warps_per_block, batch_points);
            break;
        default:
            // The prompt guarantees valid K, so this path is only a safety net.
            break;
    }
}