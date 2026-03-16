#include <cuda_runtime.h>
#include <utility>

namespace {

// This implementation is written for NVIDIA GPUs with 32-thread warps (A100/H100 class GPUs).
constexpr int kWarpSize     = 32;
constexpr int kWarpSizeLog2 = 5;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kInvalidIndex  = 0x7fffffff;

// POD mirror of std::pair<int, float>.
// The public interface uses std::pair<int, float>*, but relying on device-side std::pair
// support is unnecessary. We only need binary-compatible storage.
struct alignas(std::pair<int, float>) ResultPairDevice {
    int   first;
    float second;
};

static_assert(sizeof(ResultPairDevice) == sizeof(std::pair<int, float>),
              "std::pair<int, float> must match ResultPairDevice size.");
static_assert(alignof(ResultPairDevice) == alignof(std::pair<int, float>),
              "std::pair<int, float> must match ResultPairDevice alignment.");

template <int K>
struct KTraits {
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert((K % kWarpSize) == 0, "K must be divisible by the warp size.");
    static constexpr int LOCAL  = K / kWarpSize;     // register slots per thread
    // One dummy slot per 32 logical elements. The padding removes bank conflicts for:
    //   * ballot-compacted contiguous appends
    //   * stride-LOCAL accesses during swap/merge
    //   * reversed stride-LOCAL accesses during the bitonic merge step
    static constexpr int PADDED = K + K / kWarpSize;
};

__device__ __forceinline__ int logical_to_physical(int logical_idx) {
    // One extra hole after every 32 logical elements.
    return logical_idx + (logical_idx >> kWarpSizeLog2);
}

__device__ __forceinline__ bool pair_less(float ad, int ai, float bd, int bi) {
    // Distance is the primary key; index is only a deterministic tie-breaker for the sorting network.
    return (ad < bd) || ((ad == bd) && (ai < bi));
}

__device__ __forceinline__ void pair_swap(float &ad, int &ai, float &bd, int &bi) {
    const float td = ad;
    ad = bd;
    bd = td;

    const int ti = ai;
    ai = bi;
    bi = ti;
}

__device__ __forceinline__ float sq_l2_2d(float qx, float qy, float px, float py) {
    const float dx = qx - px;
    const float dy = qy - py;
    return __fmaf_rn(dx, dx, dy * dy);
}

// Distributed bitonic sort over K values.
// Layout:
//   - one warp owns K values
//   - each thread owns LOCAL = K / 32 consecutive logical positions
//   - cross-thread exchanges always happen between the same local register index;
//     those exchanges use warp shuffles
//   - intra-thread exchanges stay in registers
template <int K>
__device__ __forceinline__
void bitonic_sort(float (&dist)[KTraits<K>::LOCAL], int (&idx)[KTraits<K>::LOCAL], int lane) {
    constexpr int LOCAL = KTraits<K>::LOCAL;
    const int lane_base = lane * LOCAL;

    #pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
        // Cross-thread stages: stride >= LOCAL.
        // Since each thread stores consecutive logical positions, the compared element in the
        // partner thread always has the same local register index.
        #pragma unroll
        for (int stride = size >> 1; stride >= LOCAL; stride >>= 1) {
            const int lane_xor = stride / LOCAL;
            const bool ascending   = ((lane_base & size) == 0);
            const bool lane_lower  = ((lane & lane_xor) == 0);
            const bool take_min    = ascending ? lane_lower : !lane_lower;

            #pragma unroll
            for (int r = 0; r < LOCAL; ++r) {
                const float other_d = __shfl_xor_sync(kFullMask, dist[r], lane_xor);
                const int   other_i = __shfl_xor_sync(kFullMask, idx[r],  lane_xor);

                if (take_min) {
                    if (pair_less(other_d, other_i, dist[r], idx[r])) {
                        dist[r] = other_d;
                        idx[r]  = other_i;
                    }
                } else {
                    if (pair_less(dist[r], idx[r], other_d, other_i)) {
                        dist[r] = other_d;
                        idx[r]  = other_i;
                    }
                }
            }
        }

        // Intra-thread stages: stride < LOCAL.
        #pragma unroll
        for (int stride = ((size >> 1) < LOCAL ? (size >> 1) : (LOCAL >> 1)); stride > 0; stride >>= 1) {
            #pragma unroll
            for (int r = 0; r < LOCAL; ++r) {
                const int partner = r ^ stride;
                if (partner > r) {
                    const bool ascending = (((lane_base + r) & size) == 0);
                    if (ascending) {
                        if (pair_less(dist[partner], idx[partner], dist[r], idx[r])) {
                            pair_swap(dist[r], idx[r], dist[partner], idx[partner]);
                        }
                    } else {
                        if (pair_less(dist[r], idx[r], dist[partner], idx[partner])) {
                            pair_swap(dist[r], idx[r], dist[partner], idx[partner]);
                        }
                    }
                }
            }
        }
    }
}

// Merge the warp-local candidate buffer into the warp-local intermediate result.
// The implementation follows the requested procedure exactly:
//
// 1. Swap the shared-memory candidate buffer with the register-resident intermediate result,
//    so the buffer ends up in registers and the old intermediate result ends up in shared memory.
// 2. Bitonic-sort the candidate buffer in ascending order.
// 3. Form the requested bitonic merge:
//      merged[i] = min(sorted_buffer[i], old_result[K - i - 1])
// 4. Bitonic-sort the merged bitonic sequence in ascending order to obtain the new result.
//
// The candidate buffer may be partially full on the final flush; missing entries are treated as +inf.
template <int K>
__device__ __forceinline__
void merge_candidate_buffer(float (&dist)[KTraits<K>::LOCAL],
                            int   (&idx)[KTraits<K>::LOCAL],
                            float *warp_buf_dist,
                            int   *warp_buf_idx,
                            int lane,
                            int &buffer_count,
                            float &max_distance) {
    constexpr int LOCAL = KTraits<K>::LOCAL;
    const int lane_base = lane * LOCAL;

    // Candidate writes into shared memory happened earlier and came from arbitrary lanes.
    // A warp barrier is sufficient because the buffer is warp-private.
    __syncwarp(kFullMask);

    // Step 1: swap buffer <-> intermediate result.
    #pragma unroll
    for (int r = 0; r < LOCAL; ++r) {
        const int logical = lane_base + r;
        const int phys    = logical_to_physical(logical);

        const float cand_d = (logical < buffer_count) ? warp_buf_dist[phys] : CUDART_INF_F;
        const int   cand_i = (logical < buffer_count) ? warp_buf_idx[phys]  : kInvalidIndex;

        warp_buf_dist[phys] = dist[r];
        warp_buf_idx[phys]  = idx[r];

        dist[r] = cand_d;
        idx[r]  = cand_i;
    }

    // Make the spilled old intermediate result visible before Step 3 reads it back.
    __syncwarp(kFullMask);

    // Step 2: sort the candidate buffer in registers.
    bitonic_sort<K>(dist, idx, lane);

    // Step 3: build the requested bitonic merge in registers.
    #pragma unroll
    for (int r = 0; r < LOCAL; ++r) {
        const int logical     = lane_base + r;
        const int reverse_pos = K - 1 - logical;
        const int reverse_phys = logical_to_physical(reverse_pos);

        const float old_d = warp_buf_dist[reverse_phys];
        const int   old_i = warp_buf_idx[reverse_phys];

        if (pair_less(old_d, old_i, dist[r], idx[r])) {
            dist[r] = old_d;
            idx[r]  = old_i;
        }
    }

    // Step 4: sort the bitonic sequence in ascending order.
    bitonic_sort<K>(dist, idx, lane);

    // Update the pruning threshold with the K-th smallest distance.
    max_distance = __shfl_sync(kFullMask, dist[LOCAL - 1], 31);
    buffer_count = 0;
}

// Brute-force k-NN kernel for 2D points.
//
// Hyper-parameters chosen for A100/H100:
//   * 256 threads/block = 8 warps/block = 8 queries/block
//     This keeps enough blocks in flight for "thousands of queries" while reusing every
//     shared-memory data tile across eight independent queries.
//   * DATA_TILE = 4096 for K <= 512 and DATA_TILE = 2048 for K = 1024
//     This keeps the worst-case shared-memory footprint inside the opt-in shared memory
//     budget while preserving good tile reuse.
template <int K, int BLOCK_THREADS, int DATA_TILE>
__global__ __launch_bounds__(BLOCK_THREADS)
void knn_kernel(const float2 *__restrict__ query,
                int query_count,
                const float2 *__restrict__ data,
                int data_count,
                ResultPairDevice *__restrict__ result) {
    static_assert((BLOCK_THREADS % kWarpSize) == 0, "BLOCK_THREADS must be a multiple of 32.");

    constexpr int LOCAL           = KTraits<K>::LOCAL;
    constexpr int PADDED_K        = KTraits<K>::PADDED;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpSize;

    // Shared-memory layout:
    //   [data_x[DATA_TILE] | data_y[DATA_TILE] | cand_dist[WARPS_PER_BLOCK][PADDED_K] | cand_idx[WARPS_PER_BLOCK][PADDED_K]]
    //
    // The data tile is stored in SoA form (x[], y[]) to keep warp reads conflict-free when
    // lanes access consecutive cached points.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float *s_data_x = reinterpret_cast<float *>(smem_raw);
    float *s_data_y = s_data_x + DATA_TILE;
    float *s_buf_dist = s_data_y + DATA_TILE;
    int   *s_buf_idx  = reinterpret_cast<int *>(s_buf_dist + WARPS_PER_BLOCK * PADDED_K);

    const int tid     = threadIdx.x;
    const int warp_id = tid >> kWarpSizeLog2;
    const int lane    = tid & (kWarpSize - 1);

    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active   = (query_idx < query_count);

    // Warp-private shared candidate buffer.
    float *warp_buf_dist = s_buf_dist + warp_id * PADDED_K;
    int   *warp_buf_idx  = s_buf_idx  + warp_id * PADDED_K;

    // Register-resident intermediate result; each thread keeps LOCAL consecutive logical positions.
    float reg_dist[LOCAL];
    int   reg_idx[LOCAL];

    #pragma unroll
    for (int r = 0; r < LOCAL; ++r) {
        reg_dist[r] = CUDART_INF_F;
        reg_idx[r]  = kInvalidIndex;
    }

    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Warp-uniform replicated registers:
    //   * buffer_count = number of valid logical entries currently stored in the warp-local shared candidate buffer
    //   * max_distance = current K-th smallest distance (pruning threshold)
    int   buffer_count = 0;
    float max_distance = CUDART_INF_F;

    for (int tile_start = 0; tile_start < data_count; tile_start += DATA_TILE) {
        const int remaining = data_count - tile_start;
        const int tile_count = (remaining < DATA_TILE) ? remaining : DATA_TILE;

        // Cooperative block-wide load of a data tile from global memory into shared memory.
        for (int i = tid; i < tile_count; i += BLOCK_THREADS) {
            const float2 p = data[tile_start + i];
            s_data_x[i] = p.x;
            s_data_y[i] = p.y;
        }

        __syncthreads();

        if (active) {
            // One warp-round processes 32 cached data points: lane l handles cached point base + l.
            #pragma unroll 1
            for (int base = 0; base < tile_count; base += kWarpSize) {
                const int local_idx  = base + lane;
                const bool valid     = (local_idx < tile_count);

                float d = CUDART_INF_F;
                int data_idx = 0;

                if (valid) {
                    d = sq_l2_2d(qx, qy, s_data_x[local_idx], s_data_y[local_idx]);
                    data_idx = tile_start + local_idx;
                }

                // If adding this round would overflow the candidate buffer, merge first,
                // tighten max_distance, and retry the same round.
                while (true) {
                    const bool keep = valid && (d < max_distance);
                    const unsigned keep_mask = __ballot_sync(kFullMask, keep);
                    const int keep_count = __popc(keep_mask);

                    if ((buffer_count + keep_count > K) && (buffer_count > 0)) {
                        merge_candidate_buffer<K>(reg_dist, reg_idx,
                                                  warp_buf_dist, warp_buf_idx,
                                                  lane, buffer_count, max_distance);
                        continue;
                    }

                    // Ballot compaction: selected lanes store their candidates contiguously.
                    if (keep) {
                        const unsigned lower_lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
                        const int offset  = __popc(keep_mask & lower_lane_mask);
                        const int logical = buffer_count + offset;
                        const int phys    = logical_to_physical(logical);

                        warp_buf_dist[phys] = d;
                        warp_buf_idx[phys]  = data_idx;
                    }

                    buffer_count += keep_count;

                    // Merge immediately when the buffer becomes full so the pruning threshold stays tight.
                    if (buffer_count == K) {
                        merge_candidate_buffer<K>(reg_dist, reg_idx,
                                                  warp_buf_dist, warp_buf_idx,
                                                  lane, buffer_count, max_distance);
                    }
                    break;
                }
            }
        }

        // Reuse the shared data tile for the next batch.
        __syncthreads();
    }

    // Final flush of a partially filled candidate buffer.
    if (active && buffer_count > 0) {
        merge_candidate_buffer<K>(reg_dist, reg_idx,
                                  warp_buf_dist, warp_buf_idx,
                                  lane, buffer_count, max_distance);
    }

    // Store the final K nearest neighbors for this query.
    if (active) {
        const int out_base = query_idx * K;
        const int lane_base = lane * LOCAL;

        #pragma unroll
        for (int r = 0; r < LOCAL; ++r) {
            result[out_base + lane_base + r].first  = reg_idx[r];
            result[out_base + lane_base + r].second = reg_dist[r];
        }
    }
}

template <int K, int DATA_TILE>
inline void launch_knn_impl(const float2 *query,
                            int query_count,
                            const float2 *data,
                            int data_count,
                            std::pair<int, float> *result) {
    constexpr int BLOCK_THREADS    = 256;
    constexpr int WARPS_PER_BLOCK  = BLOCK_THREADS / kWarpSize;
    constexpr int PADDED_K         = KTraits<K>::PADDED;

    const size_t shared_bytes =
        size_t(DATA_TILE) * sizeof(float) * 2 +
        size_t(WARPS_PER_BLOCK) * size_t(PADDED_K) * (sizeof(float) + sizeof(int));

    // Opt in to the large dynamic shared-memory footprint required by the warp-local candidate buffers.
    (void)cudaFuncSetAttribute(knn_kernel<K, BLOCK_THREADS, DATA_TILE>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shared_bytes));
    (void)cudaFuncSetAttribute(knn_kernel<K, BLOCK_THREADS, DATA_TILE>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const int grid = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    knn_kernel<K, BLOCK_THREADS, DATA_TILE>
        <<<grid, BLOCK_THREADS, shared_bytes>>>(
            query,
            query_count,
            data,
            data_count,
            reinterpret_cast<ResultPairDevice *>(result));

    // Surface invalid launch configurations immediately while preserving asynchronous execution.
    (void)cudaPeekAtLastError();
}

} // namespace

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
        case 32:
            launch_knn_impl<32, 4096>(query, query_count, data, data_count, result);
            break;
        case 64:
            launch_knn_impl<64, 4096>(query, query_count, data, data_count, result);
            break;
        case 128:
            launch_knn_impl<128, 4096>(query, query_count, data, data_count, result);
            break;
        case 256:
            launch_knn_impl<256, 4096>(query, query_count, data, data_count, result);
            break;
        case 512:
            launch_knn_impl<512, 4096>(query, query_count, data, data_count, result);
            break;
        case 1024:
            // For K = 1024, a smaller tile keeps the block shared-memory footprint low enough
            // to preserve strong occupancy on A100/H100 while still reusing the tile across 8 queries.
            launch_knn_impl<1024, 2048>(query, query_count, data, data_count, result);
            break;
        default:
            // The API contract guarantees that k is valid.
            break;
    }
}