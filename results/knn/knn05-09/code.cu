#include <cuda_runtime.h>
#include <utility>

namespace {

// Exact warp-per-query 2D k-NN.
//
// High-level structure:
// - One warp owns one query.
// - The block cooperatively loads a 2048-point tile from global memory into shared memory.
//   The tile is reused by several queries at once, reducing global traffic.
// - The current top-k for each query is kept privately by the owning warp in registers.
//   The list is distributed in a striped layout:
//       logical_position = local_slot * 32 + lane
//   This is important because it turns every bitonic-network stride into one of two cheap cases:
//     * stride < 32  -> partner is in another lane, same local slot  -> use warp shuffles
//     * stride >= 32 -> partner is in the same lane, another slot    -> pure local exchange
// - Candidates that beat the current k-th distance are appended into a per-warp shared-memory buffer.
// - Whenever that buffer fills, it is merged with the private top-k:
//     1) sort the candidate buffer in shared memory with a warp-local bitonic sort
//     2) apply a "half-cleaner" against the current sorted top-k by comparing top[p] with cand[k-1-p]
//     3) bitonic-merge the resulting bitonic sequence back into a sorted top-k
//   This avoids materializing a 2k temporary array.
// - Distances are squared Euclidean distances, exactly as requested.
//
// Tuning notes for A100/H100-class GPUs:
// - 2048 points/tile = 16 KiB shared memory for the input tile (stored as SoA x[] and y[]).
// - With this tile size:
//     * K=512  and 16 warps/block -> ~80 KiB/block
//     * K=1024 and  8 warps/block -> ~80 KiB/block
//   Both fit two resident blocks/SM on modern data-center GPUs when opt-in shared memory is enabled.
// - The launch policy below picks the largest queries-per-block that still leaves enough blocks to
//   keep SMs busy for the given query count.

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kDataTilePoints = 2048;

static_assert((kDataTilePoints % kWarpSize) == 0, "The shared-memory tile must be a multiple of warp size.");
static_assert(kDataTilePoints >= 1024, "The first tile must be able to seed the largest supported K.");

// Total order for the sorting network.
// The prompt does not constrain tie handling, but a deterministic total order is convenient.
__device__ __forceinline__ bool pair_less(const float da, const int ia, const float db, const int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ void swap_pair(float &ad, int &ai, float &bd, int &bi) {
    const float td = ad; ad = bd; bd = td;
    const int   ti = ai; ai = bi; bi = ti;
}

__device__ __forceinline__ float squared_l2(const float qx, const float qy,
                                            const float px, const float py) {
    const float dx = px - qx;
    const float dy = py - qy;
    return fmaf(dx, dx, dy * dy);
}

// Full bitonic sort of K register-resident elements distributed across the warp in striped layout.
template <int K, int ITEMS_PER_THREAD>
__device__ __forceinline__
void bitonic_sort_striped_regs(float (&dist)[ITEMS_PER_THREAD],
                               int   (&idx )[ITEMS_PER_THREAD],
                               const int lane) {
#pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
#pragma unroll
        for (int stride = (size >> 1); stride > 0; stride >>= 1) {
            if (stride < kWarpSize) {
                // Inter-lane exchange, same local slot.
#pragma unroll
                for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
                    const int p = (t << 5) + lane;

                    const float other_d = __shfl_xor_sync(kFullMask, dist[t], stride);
                    const int   other_i = __shfl_xor_sync(kFullMask, idx [t], stride);

                    const bool low      = ((lane & stride) == 0);
                    const bool ascending = ((p & size) == 0);
                    const bool keep_min = (ascending == low);

                    if (keep_min) {
                        if (pair_less(other_d, other_i, dist[t], idx[t])) {
                            dist[t] = other_d;
                            idx [t] = other_i;
                        }
                    } else {
                        if (pair_less(dist[t], idx[t], other_d, other_i)) {
                            dist[t] = other_d;
                            idx [t] = other_i;
                        }
                    }
                }
            } else {
                // Intra-lane exchange between local slots.
                const int slot_xor = (stride >> 5);
#pragma unroll
                for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
                    const int u = t ^ slot_xor;
                    if (t < u) {
                        const int p_low = (t << 5) + lane;
                        const bool ascending = ((p_low & size) == 0);

                        if (ascending) {
                            if (pair_less(dist[u], idx[u], dist[t], idx[t])) {
                                swap_pair(dist[t], idx[t], dist[u], idx[u]);
                            }
                        } else {
                            if (pair_less(dist[t], idx[t], dist[u], idx[u])) {
                                swap_pair(dist[t], idx[t], dist[u], idx[u]);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Bitonic merge of a K-element bitonic sequence held in registers in striped layout.
// This is cheaper than a full sort and is used after the half-cleaner merge step.
template <int K, int ITEMS_PER_THREAD>
__device__ __forceinline__
void bitonic_merge_striped_regs(float (&dist)[ITEMS_PER_THREAD],
                                int   (&idx )[ITEMS_PER_THREAD],
                                const int lane) {
#pragma unroll
    for (int stride = (K >> 1); stride > 0; stride >>= 1) {
        if (stride < kWarpSize) {
            // Inter-lane exchange, same local slot.
#pragma unroll
            for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
                const float other_d = __shfl_xor_sync(kFullMask, dist[t], stride);
                const int   other_i = __shfl_xor_sync(kFullMask, idx [t], stride);
                const bool low = ((lane & stride) == 0);

                if (low) {
                    if (pair_less(other_d, other_i, dist[t], idx[t])) {
                        dist[t] = other_d;
                        idx [t] = other_i;
                    }
                } else {
                    if (pair_less(dist[t], idx[t], other_d, other_i)) {
                        dist[t] = other_d;
                        idx [t] = other_i;
                    }
                }
            }
        } else {
            // Intra-lane exchange between local slots.
            const int slot_xor = (stride >> 5);
#pragma unroll
            for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
                const int u = t ^ slot_xor;
                if (t < u) {
                    if (pair_less(dist[u], idx[u], dist[t], idx[t])) {
                        swap_pair(dist[t], idx[t], dist[u], idx[u]);
                    }
                }
            }
        }
    }
}

// Full bitonic sort of a K-element candidate buffer stored in shared memory in striped layout.
// Warp-local __syncwarp() barriers are sufficient because each warp owns a disjoint buffer.
template <int K>
__device__ __forceinline__
void bitonic_sort_striped_shared(float *dist,
                                 int   *idx,
                                 const int lane) {
    constexpr int ITEMS_PER_THREAD = K / kWarpSize;

#pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
#pragma unroll
        for (int stride = (size >> 1); stride > 0; stride >>= 1) {
            if (stride < kWarpSize) {
                // Inter-lane exchange, same local slot.
#pragma unroll
                for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
                    const int p = (t << 5) + lane;

                    float self_d = dist[p];
                    int   self_i = idx [p];

                    const float other_d = __shfl_xor_sync(kFullMask, self_d, stride);
                    const int   other_i = __shfl_xor_sync(kFullMask, self_i, stride);

                    const bool low       = ((lane & stride) == 0);
                    const bool ascending = ((p & size) == 0);
                    const bool keep_min  = (ascending == low);

                    if (keep_min) {
                        if (pair_less(other_d, other_i, self_d, self_i)) {
                            self_d = other_d;
                            self_i = other_i;
                        }
                    } else {
                        if (pair_less(self_d, self_i, other_d, other_i)) {
                            self_d = other_d;
                            self_i = other_i;
                        }
                    }

                    dist[p] = self_d;
                    idx [p] = self_i;
                }
            } else {
                // Intra-lane exchange between local slots.
                const int slot_xor = (stride >> 5);
#pragma unroll
                for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
                    const int u = t ^ slot_xor;
                    if (t < u) {
                        const int p_low = (t << 5) + lane;
                        const int p_high = (u << 5) + lane;

                        float low_d  = dist[p_low];
                        int   low_i  = idx [p_low];
                        float high_d = dist[p_high];
                        int   high_i = idx [p_high];

                        const bool ascending = ((p_low & size) == 0);

                        if (ascending) {
                            if (pair_less(high_d, high_i, low_d, low_i)) {
                                dist[p_low ] = high_d;
                                idx [p_low ] = high_i;
                                dist[p_high] = low_d;
                                idx [p_high] = low_i;
                            }
                        } else {
                            if (pair_less(low_d, low_i, high_d, high_i)) {
                                dist[p_low ] = high_d;
                                idx [p_low ] = high_i;
                                dist[p_high] = low_d;
                                idx [p_high] = low_i;
                            }
                        }
                    }
                }
            }

            __syncwarp(kFullMask);
        }
    }
}

// Merge the shared-memory candidate buffer into the register-resident top-k.
//
// The candidate buffer is first padded with +inf to length K and sorted.
// Then the "half-cleaner" compares top[p] with cand[K-1-p] (candidate list reversed).
// Keeping the smaller element in top[p] yields a K-element bitonic sequence that already
// contains the K smallest elements of the 2K union. A final bitonic merge sorts it.
template <int K, int ITEMS_PER_THREAD>
__device__ __forceinline__
float merge_candidate_buffer(float (&top_dist)[ITEMS_PER_THREAD],
                             int   (&top_idx )[ITEMS_PER_THREAD],
                             float *cand_dist,
                             int   *cand_idx,
                             const int cand_count,
                             const int lane) {
    __syncwarp(kFullMask);

    // Pad the unused tail with +inf so that a partial final buffer can reuse the same full sort.
#pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        const int p = (t << 5) + lane;
        if (p >= cand_count) {
            cand_dist[p] = CUDART_INF_F;
            cand_idx [p] = -1;
        }
    }

    __syncwarp(kFullMask);

    bitonic_sort_striped_shared<K>(cand_dist, cand_idx, lane);

    // Half-cleaner against the candidate sequence reversed.
#pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
        const int reversed_pos = ((ITEMS_PER_THREAD - 1 - t) << 5) + (31 - lane);
        const float d = cand_dist[reversed_pos];
        const int   i = cand_idx [reversed_pos];

        if (pair_less(d, i, top_dist[t], top_idx[t])) {
            top_dist[t] = d;
            top_idx [t] = i;
        }
    }

    bitonic_merge_striped_regs<K, ITEMS_PER_THREAD>(top_dist, top_idx, lane);

    // The last logical position is always owned by lane 31, last local slot.
    return __shfl_sync(kFullMask, top_dist[ITEMS_PER_THREAD - 1], 31);
}

template <int K, int BLOCK_WARPS>
__global__ __launch_bounds__(BLOCK_WARPS * kWarpSize, 2)
void knn_kernel(const float2 *__restrict__ query,
                const int query_count,
                const float2 *__restrict__ data,
                const int data_count,
                std::pair<int, float> *__restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "Unsupported K.");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");
    static_assert(kDataTilePoints >= K, "The first shared-memory tile must cover the seeding set.");

    constexpr int ITEMS_PER_THREAD = K / kWarpSize;
    constexpr int BLOCK_THREADS = BLOCK_WARPS * kWarpSize;

    extern __shared__ unsigned char smem_raw[];

    // Store the input tile in SoA form to avoid 8-byte shared-memory bank conflicts on float2 reads.
    float *const sm_data_x = reinterpret_cast<float *>(smem_raw);
    float *const sm_data_y = sm_data_x + kDataTilePoints;

    // Per-warp candidate buffers in shared memory.
    float *const sm_cand_dist = sm_data_y + kDataTilePoints;
    int   *const sm_cand_idx  = reinterpret_cast<int *>(sm_cand_dist + BLOCK_WARPS * K);

    const int warp_id = (threadIdx.x >> 5);
    const int lane    = (threadIdx.x & 31);
    const int query_idx = blockIdx.x * BLOCK_WARPS + warp_id;
    const bool valid_query = (query_idx < query_count);

    const int warp_buffer_base = warp_id * K;
    float *const cand_dist = sm_cand_dist + warp_buffer_base;
    int   *const cand_idx  = sm_cand_idx  + warp_buffer_base;

    // Private per-query top-k, distributed across the warp in striped layout.
    float top_dist[ITEMS_PER_THREAD];
    int   top_idx [ITEMS_PER_THREAD];

    float qx = 0.0f;
    float qy = 0.0f;
    float kth = CUDART_INF_F;
    int cand_count = 0;
    bool initialized = false;

    // Lane-mask prefix used for warp-synchronous compaction into the candidate buffer.
    const unsigned lane_mask_lt = (1u << lane) - 1u;

    if (valid_query) {
        if (lane == 0) {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);
    }

    for (int tile_base = 0; tile_base < data_count; tile_base += kDataTilePoints) {
        const int remaining = data_count - tile_base;
        const int batch_size = (remaining < kDataTilePoints) ? remaining : kDataTilePoints;

        // Cooperative global->shared load of the current data tile.
        for (int i = threadIdx.x; i < batch_size; i += BLOCK_THREADS) {
            const float2 p = data[tile_base + i];
            sm_data_x[i] = p.x;
            sm_data_y[i] = p.y;
        }

        __syncthreads();

        if (valid_query) {
            int scan_start = 0;

            // Seed the private top-k from the first K points of the very first tile.
            // This gives a finite k-th threshold before the candidate-buffer phase begins.
            if (!initialized) {
#pragma unroll
                for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
                    const int p = (t << 5) + lane;
                    top_dist[t] = squared_l2(qx, qy, sm_data_x[p], sm_data_y[p]);
                    top_idx [t] = p;  // tile_base == 0 here
                }

                bitonic_sort_striped_regs<K, ITEMS_PER_THREAD>(top_dist, top_idx, lane);
                kth = __shfl_sync(kFullMask, top_dist[ITEMS_PER_THREAD - 1], 31);

                initialized = true;
                scan_start = K;
            }

            // Process the tile 32 points at a time; each lane handles one point in the group.
            for (int group_base = scan_start; group_base < batch_size; group_base += kWarpSize) {
                const int local_idx = group_base + lane;
                const bool in_range = (local_idx < batch_size);

                float d = 0.0f;
                int data_idx = tile_base + local_idx;
                bool take = false;

                if (in_range) {
                    d = squared_l2(qx, qy, sm_data_x[local_idx], sm_data_y[local_idx]);

                    // Strictly better only: equal-distance ties may be dropped, which is allowed.
                    take = (d < kth);
                }

                const unsigned mask = __ballot_sync(kFullMask, take);
                const int n = __popc(mask);

                if (n != 0) {
                    const int rank = __popc(mask & lane_mask_lt);

                    if (cand_count + n <= K) {
                        if (take) {
                            cand_dist[cand_count + rank] = d;
                            cand_idx [cand_count + rank] = data_idx;
                        }

                        cand_count += n;

                        if (cand_count == K) {
                            kth = merge_candidate_buffer<K, ITEMS_PER_THREAD>(
                                top_dist, top_idx, cand_dist, cand_idx, cand_count, lane);
                            cand_count = 0;
                        }
                    } else {
                        // The current 32-point group would overflow the K-entry candidate buffer.
                        // Fill the remaining slots, merge immediately, then keep any leftover
                        // candidates from this same group that are still better than the new threshold.
                        const int fill = K - cand_count;

                        if (take && rank < fill) {
                            cand_dist[cand_count + rank] = d;
                            cand_idx [cand_count + rank] = data_idx;
                        }

                        kth = merge_candidate_buffer<K, ITEMS_PER_THREAD>(
                            top_dist, top_idx, cand_dist, cand_idx, K, lane);

                        bool keep_after_merge = take && (rank >= fill) && (d < kth);
                        const unsigned rem_mask = __ballot_sync(kFullMask, keep_after_merge);
                        const int rem_n = __popc(rem_mask);
                        const int rem_rank = __popc(rem_mask & lane_mask_lt);

                        if (keep_after_merge) {
                            cand_dist[rem_rank] = d;
                            cand_idx [rem_rank] = data_idx;
                        }

                        cand_count = rem_n;
                    }
                }
            }
        }

        // All warps must finish consuming the current tile before the block overwrites it.
        __syncthreads();
    }

    if (valid_query) {
        // Final partial-buffer merge, if any candidates remain.
        if (cand_count != 0) {
            (void)merge_candidate_buffer<K, ITEMS_PER_THREAD>(
                top_dist, top_idx, cand_dist, cand_idx, cand_count, lane);
        }

        // Store the sorted top-k back in row-major order.
        const int out_base = query_idx * K;
#pragma unroll
        for (int t = 0; t < ITEMS_PER_THREAD; ++t) {
            const int p = (t << 5) + lane;
            result[out_base + p].first  = top_idx[t];
            result[out_base + p].second = top_dist[t];
        }
    }
}

// More warps/block means more reuse of each shared-memory data tile and therefore less global traffic
// per query. But if query_count is modest, overly large blocks reduce the number of blocks in flight.
// The heuristic below picks the largest of {16, 8, 4} warps/block that still gives roughly one block/SM.
inline int choose_block_warps(const int k, const int query_count) {
    int device = 0;
    (void)cudaGetDevice(&device);

    int sm_count = 1;
    (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count <= 0) sm_count = 1;

    if (k == 1024) {
        return (query_count >= 8 * sm_count) ? 8 : 4;
    }

    if (query_count >= 16 * sm_count) return 16;
    if (query_count >= 8  * sm_count) return 8;
    return 4;
}

template <int K, int BLOCK_WARPS>
inline void launch_knn(const float2 *query,
                       const int query_count,
                       const float2 *data,
                       const int data_count,
                       std::pair<int, float> *result) {
    constexpr int BLOCK_THREADS = BLOCK_WARPS * kWarpSize;
    constexpr size_t SHMEM_BYTES =
        size_t(2) * size_t(kDataTilePoints) * sizeof(float) +
        size_t(BLOCK_WARPS) * size_t(K) * (sizeof(float) + sizeof(int));

    // Opt in to the dynamic shared-memory size required by the chosen specialization.
    (void)cudaFuncSetAttribute(knn_kernel<K, BLOCK_WARPS>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(SHMEM_BYTES));
    (void)cudaFuncSetAttribute(knn_kernel<K, BLOCK_WARPS>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const dim3 block(BLOCK_THREADS);
    const dim3 grid((query_count + BLOCK_WARPS - 1) / BLOCK_WARPS);

    // Intentionally asynchronous; the caller can synchronize or inspect errors as desired.
    knn_kernel<K, BLOCK_WARPS><<<grid, block, SHMEM_BYTES>>>(
        query, query_count, data, data_count, result);
}

} // anonymous namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) return;

    const int block_warps = choose_block_warps(k, query_count);

    switch (k) {
        case 32:
            if (block_warps >= 16) launch_knn<32, 16>(query, query_count, data, data_count, result);
            else if (block_warps >= 8) launch_knn<32, 8>(query, query_count, data, data_count, result);
            else launch_knn<32, 4>(query, query_count, data, data_count, result);
            break;

        case 64:
            if (block_warps >= 16) launch_knn<64, 16>(query, query_count, data, data_count, result);
            else if (block_warps >= 8) launch_knn<64, 8>(query, query_count, data, data_count, result);
            else launch_knn<64, 4>(query, query_count, data, data_count, result);
            break;

        case 128:
            if (block_warps >= 16) launch_knn<128, 16>(query, query_count, data, data_count, result);
            else if (block_warps >= 8) launch_knn<128, 8>(query, query_count, data, data_count, result);
            else launch_knn<128, 4>(query, query_count, data, data_count, result);
            break;

        case 256:
            if (block_warps >= 16) launch_knn<256, 16>(query, query_count, data, data_count, result);
            else if (block_warps >= 8) launch_knn<256, 8>(query, query_count, data, data_count, result);
            else launch_knn<256, 4>(query, query_count, data, data_count, result);
            break;

        case 512:
            if (block_warps >= 16) launch_knn<512, 16>(query, query_count, data, data_count, result);
            else if (block_warps >= 8) launch_knn<512, 8>(query, query_count, data, data_count, result);
            else launch_knn<512, 4>(query, query_count, data, data_count, result);
            break;

        case 1024:
            if (block_warps >= 8) launch_knn<1024, 8>(query, query_count, data, data_count, result);
            else launch_knn<1024, 4>(query, query_count, data, data_count, result);
            break;

        default:
            // Inputs are guaranteed valid by the problem statement, so this path is unreachable.
            break;
    }
}