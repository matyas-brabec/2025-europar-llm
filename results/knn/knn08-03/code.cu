#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace {

// Fixed hardware-oriented tuning:
// - 256 threads/block => 8 warps/block => 8 queries/block.
//   This is a good balance between occupancy and shared-memory data reuse.
// - 1792 cached data points/block (7 float2 loads/thread) is intentionally close
//   to the largest tile that still allows two resident worst-case blocks/SM on
//   A100-class parts when K=1024 and each warp owns a K-entry shared buffer.
constexpr int kWarpSize       = 32;
constexpr int kBlockThreads   = 256;
constexpr int kWarpsPerBlock  = kBlockThreads / kWarpSize;
constexpr int kDataTilePoints = 1792;
constexpr unsigned kFullMask  = 0xffffffffu;

static_assert(kBlockThreads % kWarpSize == 0, "block size must be a multiple of warp size");
static_assert(kDataTilePoints % kBlockThreads == 0, "tile size chosen for exact cooperative loads");
static_assert(kDataTilePoints % kWarpSize == 0, "tile size chosen for exact warp-sized processing groups");

// Shared-memory candidate record.
// Distance is stored first because it is the hot comparison key.
// The record is 8 bytes and naturally maps well to shared-memory transactions.
struct alignas(8) Candidate {
    float distance;
    int   index;
};

static_assert(sizeof(Candidate) == 8, "Candidate must stay compact");

// Small helper used by the local (in-thread) stages of bitonic sort.
__device__ __forceinline__
void swap_pair(float &a_dist, int &a_idx, float &b_dist, int &b_idx) {
    const float td = a_dist;
    a_dist = b_dist;
    b_dist = td;

    const int ti = a_idx;
    a_idx = b_idx;
    b_idx = ti;
}

__device__ __forceinline__
void maybe_swap_pair(float &a_dist, int &a_idx, float &b_dist, int &b_idx, bool ascending) {
    // Ties are intentionally left unresolved in any particular way.
    const bool do_swap = ascending ? (b_dist < a_dist) : (b_dist > a_dist);
    if (do_swap) {
        swap_pair(a_dist, a_idx, b_dist, b_idx);
    }
}

// Warp-wide bitonic sort over K elements distributed as consecutive chunks:
// lane t owns positions [t * (K/32), ..., t * (K/32) + (K/32 - 1)].
//
// Because K and K/32 are powers of two, every cross-thread compare-exchange
// always targets the same register slot index in the partner lane. This is
// exactly the layout requested in the problem statement.
template <int K>
__device__ __forceinline__
void bitonic_sort_warp(float (&dist)[K / kWarpSize], int (&index)[K / kWarpSize]) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024]");
    static_assert((K % kWarpSize) == 0, "K must be divisible by warp size");

    constexpr int kItemsPerThread = K / kWarpSize;

    const int lane        = threadIdx.x & (kWarpSize - 1);
    const int thread_base = lane * kItemsPerThread;

    // Standard bitonic sort network.
    #pragma unroll
    for (int stage_size = 2; stage_size <= K; stage_size <<= 1) {
        #pragma unroll
        for (int stride = stage_size >> 1; stride > 0; stride >>= 1) {
            if (stride < kItemsPerThread) {
                // Intra-thread compare/swap: both elements are in the same lane's registers.
                #pragma unroll
                for (int r = 0; r < kItemsPerThread; ++r) {
                    const int partner_r = r ^ stride;
                    if (partner_r > r) {
                        const bool ascending = (((thread_base + r) & stage_size) == 0);
                        maybe_swap_pair(dist[r], index[r], dist[partner_r], index[partner_r], ascending);
                    }
                }
            } else {
                // Inter-thread compare/exchange: partner lives in another lane, same register slot.
                const int partner_lane_delta = stride / kItemsPerThread;

                #pragma unroll
                for (int r = 0; r < kItemsPerThread; ++r) {
                    const float other_dist = __shfl_xor_sync(kFullMask, dist[r],  partner_lane_delta);
                    const int   other_idx  = __shfl_xor_sync(kFullMask, index[r], partner_lane_delta);

                    const int  i         = thread_base + r;
                    const bool ascending = ((i & stage_size) == 0);
                    const bool lower     = ((i & stride) == 0);
                    const bool keep_min  = (lower == ascending);

                    const bool take_other = keep_min ? (other_dist < dist[r]) : (other_dist > dist[r]);
                    if (take_other) {
                        dist[r]  = other_dist;
                        index[r] = other_idx;
                    }
                }
            }

            // Keep explicit warp-stage boundaries around the network steps.
            __syncwarp();
        }
    }
}

// max_distance is the current pruning threshold, i.e. the distance of the
// current k-th nearest neighbor.
template <int K>
__device__ __forceinline__
void update_max_distance(const float (&best_dist)[K / kWarpSize], float &max_distance) {
    constexpr int kItemsPerThread = K / kWarpSize;

    float tail = CUDART_INF_F;
    if ((threadIdx.x & (kWarpSize - 1)) == (kWarpSize - 1)) {
        tail = best_dist[kItemsPerThread - 1];
    }
    max_distance = __shfl_sync(kFullMask, tail, kWarpSize - 1);
}

// Merge the shared-memory candidate buffer into the register-resident
// intermediate top-K, exactly following the requested sequence:
//
// 0. Invariant: intermediate result in registers is sorted ascending.
// 1. Swap buffer and intermediate result so the buffer is in registers.
// 2. Sort the register-resident buffer ascending with bitonic sort.
// 3. Build a bitonic sequence by taking min(buffer[i], intermediate[K-1-i]).
// 4. Bitonic-sort that sequence ascending to obtain the updated top-K.
template <int K>
__device__ __forceinline__
void merge_buffer_into_best(float (&best_dist)[K / kWarpSize],
                            int (&best_index)[K / kWarpSize],
                            Candidate *warp_buffer,
                            int buf_count,
                            float &max_distance) {
    constexpr int kItemsPerThread = K / kWarpSize;

    const int lane        = threadIdx.x & (kWarpSize - 1);
    const int thread_base = lane * kItemsPerThread;

    // Step 1: swap shared buffer <-> register-resident intermediate result.
    #pragma unroll
    for (int r = 0; r < kItemsPerThread; ++r) {
        const int pos = thread_base + r;

        Candidate buffered;
        if (pos < buf_count) {
            buffered = warp_buffer[pos];
        } else {
            buffered.distance = CUDART_INF_F;
            buffered.index    = -1;
        }

        warp_buffer[pos] = Candidate{best_dist[r], best_index[r]};

        best_dist[r]  = buffered.distance;
        best_index[r] = buffered.index;
    }
    __syncwarp();

    // Step 2: sort the buffer now living in registers.
    bitonic_sort_warp<K>(best_dist, best_index);

    // Step 3: bitonic merge construction.
    // warp_buffer currently stores the old intermediate result in ascending order.
    #pragma unroll
    for (int r = 0; r < kItemsPerThread; ++r) {
        const int pos = thread_base + r;
        const Candidate other = warp_buffer[K - 1 - pos];

        if (other.distance < best_dist[r]) {
            best_dist[r]  = other.distance;
            best_index[r] = other.index;
        }
    }
    __syncwarp();

    // Step 4: sort the resulting bitonic sequence to obtain the updated top-K.
    bitonic_sort_warp<K>(best_dist, best_index);

    // Update pruning threshold.
    update_max_distance<K>(best_dist, max_distance);
}

// Flush a warp-local shared candidate buffer into the register-resident top-K.
template <int K>
__device__ __forceinline__
void flush_buffer(float (&best_dist)[K / kWarpSize],
                  int (&best_index)[K / kWarpSize],
                  Candidate *warp_buffer,
                  int &buf_count,
                  volatile int *warp_count_ptr,
                  float &max_distance) {
    if (buf_count == 0) {
        return;
    }

    // Ensure all prior shared-memory candidate stores are visible to the warp.
    __syncwarp();

    merge_buffer_into_best<K>(best_dist, best_index, warp_buffer, buf_count, max_distance);

    buf_count = 0;
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        *warp_count_ptr = 0;
    }

    __syncwarp();
}

template <int K>
__global__ __launch_bounds__(kBlockThreads, 2)
void knn_kernel(const float2 * __restrict__ query,
                int query_count,
                const float2 * __restrict__ data,
                int data_count,
                std::pair<int, float> * __restrict__ result) {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024]");

    constexpr int kItemsPerThread = K / kWarpSize;
    constexpr int kLoadsPerThread = kDataTilePoints / kBlockThreads;

    extern __shared__ unsigned char shared_raw[];

    // Shared-memory layout:
    // [data tile][warp0 buffer][warp1 buffer]...[warpN buffer][warp candidate counts]
    float2 *data_tile = reinterpret_cast<float2 *>(shared_raw);
    Candidate *candidate_buffers =
        reinterpret_cast<Candidate *>(data_tile + kDataTilePoints);
    volatile int *candidate_counts =
        reinterpret_cast<volatile int *>(candidate_buffers + kWarpsPerBlock * K);

    const int tid     = threadIdx.x;
    const int lane    = tid & (kWarpSize - 1);
    const int warp_id = tid >> 5;

    const int query_idx    = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_id;
    const bool active_warp = (query_idx < query_count);

    Candidate *warp_buffer      = candidate_buffers + warp_id * K;
    volatile int *warp_count_ptr = candidate_counts + warp_id;

    // Each warp owns one shared counter for its candidate buffer.
    if (lane == 0) {
        *warp_count_ptr = 0;
    }
    __syncwarp();

    // Register-resident intermediate result: each lane stores K/32 consecutive items.
    float best_dist[kItemsPerThread];
    int   best_index[kItemsPerThread];

    #pragma unroll
    for (int r = 0; r < kItemsPerThread; ++r) {
        best_dist[r]  = CUDART_INF_F;
        best_index[r] = -1;
    }

    float max_distance = CUDART_INF_F;
    int   buf_count    = 0;

    // Load the query point once per warp and broadcast from lane 0.
    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && active_warp) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Prefix mask used to map ballot hits to compact positions in the shared buffer.
    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    // Process the database in shared-memory tiles.
    for (int tile_base = 0; tile_base < data_count; tile_base += kDataTilePoints) {
        int tile_count = data_count - tile_base;
        if (tile_count > kDataTilePoints) {
            tile_count = kDataTilePoints;
        }

        // Cooperative block load of the current data tile.
        #pragma unroll
        for (int i = 0; i < kLoadsPerThread; ++i) {
            const int local = tid + i * kBlockThreads;
            if (local < tile_count) {
                data_tile[local] = data[tile_base + local];
            }
        }

        __syncthreads();

        if (active_warp) {
            // Iterate over the cached tile in warp-sized groups so that every lane
            // participates in every ballot, even for the final partial group.
            for (int group_base = 0; group_base < tile_count; group_base += kWarpSize) {
                const int local = group_base + lane;

                float dist = CUDART_INF_F;
                int   idx  = -1;
                bool  pred = false;

                if (local < tile_count) {
                    const float2 p = data_tile[local];

                    const float dx = qx - p.x;
                    const float dy = qy - p.y;

                    // Squared Euclidean distance; no sqrt, exactly as requested.
                    dist = fmaf(dx, dx, dy * dy);
                    idx  = tile_base + local;
                    pred = (dist < max_distance);
                }

                const unsigned hit_mask = __ballot_sync(kFullMask, pred);

                if (hit_mask != 0u) {
                    const int hits = __popc(hit_mask);
                    const int rank = __popc(hit_mask & lane_mask_lt);
                    const int space = K - buf_count;

                    if (hits <= space) {
                        // Everything fits into the current shared candidate buffer.
                        if (pred) {
                            warp_buffer[buf_count + rank] = Candidate{dist, idx};
                        }

                        __syncwarp();

                        buf_count += hits;
                        if (lane == 0) {
                            *warp_count_ptr = buf_count;
                        }

                        // Merge immediately when the buffer becomes full.
                        if (buf_count == K) {
                            flush_buffer<K>(best_dist, best_index, warp_buffer, buf_count, warp_count_ptr, max_distance);
                        }
                    } else {
                        // The incoming hit set would overflow the buffer.
                        // Fill the remaining space, merge, then append the leftovers
                        // to the now-empty buffer.
                        if (pred && rank < space) {
                            warp_buffer[buf_count + rank] = Candidate{dist, idx};
                        }

                        __syncwarp();

                        buf_count = K;
                        if (lane == 0) {
                            *warp_count_ptr = K;
                        }

                        flush_buffer<K>(best_dist, best_index, warp_buffer, buf_count, warp_count_ptr, max_distance);

                        if (pred && rank >= space) {
                            warp_buffer[rank - space] = Candidate{dist, idx};
                        }

                        __syncwarp();

                        buf_count = hits - space;
                        if (lane == 0) {
                            *warp_count_ptr = buf_count;
                        }

                        // This second immediate flush is only needed for the degenerate
                        // K=32 case when the leftover set also fills the buffer.
                        if (buf_count == K) {
                            flush_buffer<K>(best_dist, best_index, warp_buffer, buf_count, warp_count_ptr, max_distance);
                        }
                    }
                }
            }
        }

        // The next tile overwrites data_tile, so all warps in the block must finish.
        __syncthreads();
    }

    if (active_warp) {
        // Final partial-buffer merge, if any candidates remain.
        flush_buffer<K>(best_dist, best_index, warp_buffer, buf_count, warp_count_ptr, max_distance);

        // Write the sorted top-K result. The output distance remains squared L2.
        const int out_base = query_idx * K + lane * kItemsPerThread;

        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            result[out_base + r].first  = best_index[r];
            result[out_base + r].second = best_dist[r];
        }
    }
}

template <int K>
constexpr std::size_t shared_memory_bytes() {
    return static_cast<std::size_t>(kDataTilePoints) * sizeof(float2) +
           static_cast<std::size_t>(kWarpsPerBlock)  * K * sizeof(Candidate) +
           static_cast<std::size_t>(kWarpsPerBlock)  * sizeof(int);
}

template <int K>
inline void launch_knn_kernel(const float2 *query,
                              int query_count,
                              const float2 *data,
                              int data_count,
                              std::pair<int, float> *result) {
    constexpr std::size_t smem_bytes = shared_memory_bytes<K>();

    // The interface returns void, so configuration calls are intentionally fire-and-forget.
    // The target devices in the problem statement support this dynamic shared-memory size.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem_bytes));
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const dim3 block(kBlockThreads);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

    knn_kernel<K><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result);
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

    // Launch is intentionally asynchronous; callers can synchronize externally if needed.
    switch (k) {
        case 32:   launch_knn_kernel<32>(query, query_count, data, data_count, result);   break;
        case 64:   launch_knn_kernel<64>(query, query_count, data, data_count, result);   break;
        case 128:  launch_knn_kernel<128>(query, query_count, data, data_count, result);  break;
        case 256:  launch_knn_kernel<256>(query, query_count, data, data_count, result);  break;
        case 512:  launch_knn_kernel<512>(query, query_count, data, data_count, result);  break;
        case 1024: launch_knn_kernel<1024>(query, query_count, data, data_count, result); break;
        default:   break; // Unreachable for valid inputs.
    }
}