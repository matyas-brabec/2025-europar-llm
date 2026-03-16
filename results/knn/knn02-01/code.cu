#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace knn2d_detail {

// Device code should not depend on std::pair constructors/operators from the host standard library.
// The interface, however, is fixed to std::pair<int, float>. We therefore write through an ABI-compatible
// POD view and validate the expected host layout at compile time.
struct alignas(alignof(std::pair<int, float>)) device_result_pair {
    int   first;
    float second;
};

static_assert(sizeof(device_result_pair) == sizeof(std::pair<int, float>),
              "run_knn assumes std::pair<int,float> has the plain {int,float} ABI layout.");
static_assert(alignof(device_result_pair) == alignof(std::pair<int, float>),
              "run_knn assumes std::pair<int,float> has the plain {int,float} ABI alignment.");

// Shared-memory tile and candidate-chunk sizes.
// A 256-point tile is large enough to amortize the block-wide synchronization, while a smaller 128-point
// candidate chunk keeps the per-warp transient buffer small enough that k=1024 still fits in shared memory
// with a useful number of query warps per block on A100/H100-class GPUs.
constexpr int kWarpSize              = 32;
constexpr int kLoadTilePoints        = 256;
constexpr int kCandidateChunkPoints  = 128;
constexpr unsigned kFullMask         = 0xFFFFFFFFu;

// Launch heuristic:
// - Exact 2D distance scans have very low arithmetic intensity, so increasing the number of query warps per block
//   improves reuse of the staged data tile.
// - Past a point, too many warps per block reduce grid-level parallelism and start to overtrade occupancy for reuse.
// - Ampere is best capped around 16 query warps/block; Hopper can afford a slightly higher cap.
// - We then choose the largest value up to that cap that still gives enough blocks to occupy most SMs.
constexpr int kAmperePreferredWarpCap = 16;
constexpr int kHopperPreferredWarpCap = 20;
constexpr int kMinSMCoveragePercent   = 85;

static_assert(kLoadTilePoints % kCandidateChunkPoints == 0, "Tile must be an integer number of candidate chunks.");
static_assert(kCandidateChunkPoints % kWarpSize == 0, "Candidate chunk must be an integer number of warp rounds.");

// Scalar max-heap helpers.
// Only lane 0 of each query warp mutates the heap. This is intentional: for k up to 1024,
// a fully parallel top-k structure would consume much more shared/register state and require
// substantially more synchronization. After the first few chunks the acceptance threshold tightens
// quickly, so heap updates become sparse and the serialized lane-0 maintenance is typically cheap.
__device__ __forceinline__ void sift_down_max(float *dist, int *idx, int root, int size) {
    float root_dist = dist[root];
    int   root_idx  = idx[root];

    int child = (root << 1) + 1;
    while (child < size) {
        int   larger_child = child;
        float larger_dist  = dist[child];

        const int right = child + 1;
        if (right < size) {
            const float right_dist = dist[right];
            if (right_dist > larger_dist) {
                larger_child = right;
                larger_dist  = right_dist;
            }
        }

        if (root_dist >= larger_dist) {
            break;
        }

        dist[root] = larger_dist;
        idx[root]  = idx[larger_child];

        root  = larger_child;
        child = (root << 1) + 1;
    }

    dist[root] = root_dist;
    idx[root]  = root_idx;
}

template <int K>
__device__ __forceinline__ void build_max_heap(float *dist, int *idx) {
    for (int root = (K >> 1) - 1; root >= 0; --root) {
        sift_down_max(dist, idx, root, K);
    }
}

template <int K>
__device__ __forceinline__ void heap_sort_ascending(float *dist, int *idx) {
    for (int end = K - 1; end > 0; --end) {
        const float tmp_dist = dist[end];
        dist[end] = dist[0];
        dist[0]   = tmp_dist;

        const int tmp_idx = idx[end];
        idx[end] = idx[0];
        idx[0]   = tmp_idx;

        sift_down_max(dist, idx, 0, end);
    }
}

// Kernel organization:
// - one warp owns one query,
// - multiple such warps live in the same block,
// - the block stages a tile of database points in shared memory,
// - each warp scans that tile against its query,
// - per-warp intermediate top-k state lives entirely in shared memory (no extra device allocations),
// - the transient candidate buffer is private to each warp and reused chunk by chunk.
template <int K, int LOAD_TILE_POINTS, int CANDIDATE_CHUNK_POINTS>
__global__ __launch_bounds__(1024, 1)
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                device_result_pair * __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    static_assert(LOAD_TILE_POINTS % CANDIDATE_CHUNK_POINTS == 0, "Invalid tile/chunk split.");
    static_assert(CANDIDATE_CHUNK_POINTS % kWarpSize == 0, "Chunk size must be a multiple of warp size.");

    const int warps_per_block = blockDim.x / kWarpSize;
    const int warp_id         = threadIdx.x / kWarpSize;
    const int lane            = threadIdx.x & (kWarpSize - 1);

    const int block_query_base = static_cast<int>(blockIdx.x) * warps_per_block;
    if (block_query_base >= query_count) {
        return;
    }

    const int query_idx = block_query_base + warp_id;
    const bool active   = (query_idx < query_count);

    extern __shared__ unsigned char smem_raw[];
    float2 *tile = reinterpret_cast<float2 *>(smem_raw);

    float *heap_dist = reinterpret_cast<float *>(tile + LOAD_TILE_POINTS);
    int   *heap_idx  = reinterpret_cast<int *>(heap_dist + warps_per_block * K);
    float *cand_dist = reinterpret_cast<float *>(heap_idx  + warps_per_block * K);
    int   *cand_idx  = reinterpret_cast<int *>(cand_dist + warps_per_block * CANDIDATE_CHUNK_POINTS);

    float *my_heap_dist = heap_dist + warp_id * K;
    int   *my_heap_idx  = heap_idx  + warp_id * K;
    float *my_cand_dist = cand_dist + warp_id * CANDIDATE_CHUNK_POINTS;
    int   *my_cand_idx  = cand_idx  + warp_id * CANDIDATE_CHUNK_POINTS;

    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    int   heap_size = 0;
    float kth       = CUDART_INF_F;

    const unsigned lane_prefix_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
    constexpr int rounds_per_chunk  = CANDIDATE_CHUNK_POINTS / kWarpSize;

    for (int tile_base = 0; tile_base < data_count; tile_base += LOAD_TILE_POINTS) {
        int tile_count = data_count - tile_base;
        if (tile_count > LOAD_TILE_POINTS) {
            tile_count = LOAD_TILE_POINTS;
        }

        // Cooperative global->shared staging of the database tile.
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            tile[t] = data[tile_base + t];
        }
        __syncthreads();

        // Consume the staged tile in smaller candidate chunks to keep transient per-warp state compact.
        for (int chunk_offset = 0; chunk_offset < tile_count; chunk_offset += CANDIDATE_CHUNK_POINTS) {
            int chunk_count = tile_count - chunk_offset;
            if (chunk_count > CANDIDATE_CHUNK_POINTS) {
                chunk_count = CANDIDATE_CHUNK_POINTS;
            }

            float threshold_lane0 = 0.0f;
            int   heap_full_lane0 = 0;
            if (lane == 0 && active) {
                heap_full_lane0 = (heap_size == K);
                threshold_lane0 = heap_full_lane0 ? kth : CUDART_INF_F;
            }

            const float threshold = __shfl_sync(kFullMask, threshold_lane0, 0);
            const bool  heap_full = (__shfl_sync(kFullMask, heap_full_lane0, 0) != 0);

            int buf_count = 0;

            const float2 *tile_chunk       = tile + chunk_offset;
            const int     global_chunk_base = tile_base + chunk_offset;

#pragma unroll
            for (int round = 0; round < rounds_per_chunk; ++round) {
                const int local = (round << 5) + lane;
                const bool valid = active && (local < chunk_count);

                float dist = 0.0f;
                if (valid) {
                    const float2 p = tile_chunk[local];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;
                    dist = fmaf(dx, dx, dy * dy);
                }

                const bool qualify = valid && (!heap_full || (dist < threshold));
                const unsigned qualify_mask = __ballot_sync(kFullMask, qualify);
                const int nqual = __popc(qualify_mask);

                int base = 0;
                if (lane == 0) {
                    base = buf_count;
                    buf_count += nqual;
                }
                base = __shfl_sync(kFullMask, base, 0);

                if (qualify) {
                    const int pos = base + __popc(qualify_mask & lane_prefix_mask);
                    my_cand_dist[pos] = dist;
                    my_cand_idx[pos]  = global_chunk_base + local;
                }
            }

            // All candidate writes for this warp must be visible before lane 0 starts consuming them.
            __syncwarp();

            if (lane == 0 && active && buf_count > 0) {
                int i = 0;

                // Until the first K elements have been collected, simply append.
                // Once K is reached, build the max-heap once and switch to thresholded replacement.
                if (heap_size < K) {
                    int to_copy = K - heap_size;
                    if (to_copy > buf_count) {
                        to_copy = buf_count;
                    }

                    for (; i < to_copy; ++i) {
                        my_heap_dist[heap_size] = my_cand_dist[i];
                        my_heap_idx[heap_size]  = my_cand_idx[i];
                        ++heap_size;
                    }

                    if (heap_size == K) {
                        build_max_heap<K>(my_heap_dist, my_heap_idx);
                    }
                }

                if (heap_size == K) {
                    float current_kth = my_heap_dist[0];

                    for (; i < buf_count; ++i) {
                        const float cand_d = my_cand_dist[i];
                        if (cand_d < current_kth) {
                            my_heap_dist[0] = cand_d;
                            my_heap_idx[0]  = my_cand_idx[i];
                            sift_down_max(my_heap_dist, my_heap_idx, 0, K);
                            current_kth = my_heap_dist[0];
                        }
                    }

                    kth = current_kth;
                } else {
                    kth = CUDART_INF_F;
                }
            }
        }

        // Ensure all warps are finished with the current tile before it is overwritten by the next one.
        __syncthreads();
    }

    if (active) {
        // Final output must be ordered by ascending distance so that result[i*k + j] is the j-th nearest neighbor.
        if (lane == 0) {
            heap_sort_ascending<K>(my_heap_dist, my_heap_idx);
        }

        // The sorted shared-memory heap is now read by all lanes in the warp for coalesced global writes.
        __syncwarp();

        const std::size_t out_base = static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);
        for (int i = lane; i < K; i += kWarpSize) {
            result[out_base + static_cast<std::size_t>(i)].first  = my_heap_idx[i];
            result[out_base + static_cast<std::size_t>(i)].second = my_heap_dist[i];
        }
    }
}

template <int K>
inline void launch_knn_impl(const float2 *query,
                            int query_count,
                            const float2 *data,
                            int data_count,
                            device_result_pair *result) {
    if (query_count <= 0) {
        return;
    }

    int device = 0;
    int max_optin_smem = 0;
    int sm_count = 1;
    int cc_major = 8;

    (void)cudaGetDevice(&device);
    (void)cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    (void)cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);

    if (max_optin_smem <= 0) {
        // Safe fallback for modern devices if the query above unexpectedly fails.
        max_optin_smem = 98304;
    }
    if (sm_count <= 0) {
        sm_count = 1;
    }

    const int arch_preferred_cap = (cc_major >= 9) ? kHopperPreferredWarpCap : kAmperePreferredWarpCap;

    const int tile_bytes      = kLoadTilePoints * static_cast<int>(sizeof(float2));
    const int bytes_per_warp  = (K + kCandidateChunkPoints) * static_cast<int>(sizeof(float) + sizeof(int));

    int shared_cap = (max_optin_smem - tile_bytes) / bytes_per_warp;
    if (shared_cap < 1) {
        shared_cap = 1;
    }
    if (shared_cap > 32) {
        shared_cap = 32;
    }

    int preferred_cap = arch_preferred_cap;
    if (preferred_cap > shared_cap) {
        preferred_cap = shared_cap;
    }
    if (preferred_cap > query_count) {
        preferred_cap = query_count;
    }

    const int min_grid_blocks = (sm_count * kMinSMCoveragePercent + 99) / 100;

    // Pick the largest warps/block that still gives enough blocks to occupy most SMs.
    // This intentionally balances shared-memory data reuse against chip-level concurrency.
    int warps_per_block = preferred_cap;
    while (warps_per_block > 1) {
        const int grid_blocks = (query_count + warps_per_block - 1) / warps_per_block;
        if (grid_blocks >= min_grid_blocks) {
            break;
        }
        --warps_per_block;
    }

    const std::size_t smem_bytes =
        static_cast<std::size_t>(tile_bytes) +
        static_cast<std::size_t>(warps_per_block) * static_cast<std::size_t>(bytes_per_warp);

    (void)cudaFuncSetAttribute(
        knn_kernel<K, kLoadTilePoints, kCandidateChunkPoints>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    (void)cudaFuncSetAttribute(
        knn_kernel<K, kLoadTilePoints, kCandidateChunkPoints>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);

    (void)cudaFuncSetCacheConfig(
        knn_kernel<K, kLoadTilePoints, kCandidateChunkPoints>,
        cudaFuncCachePreferShared);

    const dim3 block(static_cast<unsigned>(warps_per_block * kWarpSize));
    const dim3 grid(static_cast<unsigned>((query_count + warps_per_block - 1) / warps_per_block));

    knn_kernel<K, kLoadTilePoints, kCandidateChunkPoints>
        <<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result);
}

} // namespace knn2d_detail

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    auto *result_view = reinterpret_cast<knn2d_detail::device_result_pair *>(result);

    // k is guaranteed by the interface contract to be a power of two in [32, 1024].
    // Specializing on K lets the compiler see fixed heap sizes and shared-memory layouts.
    switch (k) {
        case 32:
            knn2d_detail::launch_knn_impl<32>(query, query_count, data, data_count, result_view);
            break;
        case 64:
            knn2d_detail::launch_knn_impl<64>(query, query_count, data, data_count, result_view);
            break;
        case 128:
            knn2d_detail::launch_knn_impl<128>(query, query_count, data, data_count, result_view);
            break;
        case 256:
            knn2d_detail::launch_knn_impl<256>(query, query_count, data, data_count, result_view);
            break;
        case 512:
            knn2d_detail::launch_knn_impl<512>(query, query_count, data, data_count, result_view);
            break;
        case 1024:
            knn2d_detail::launch_knn_impl<1024>(query, query_count, data, data_count, result_view);
            break;
        default:
            // Input contract says this does not happen.
            break;
    }
}