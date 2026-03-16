#include <cuda_runtime.h>
#include <utility>
#include <climits>
#include <cstddef>
#include <cstdint>

/*
  Optimized warp-per-query 2D k-NN for squared Euclidean distance.

  Key design choices, matching the requested algorithmic structure:
  - One warp (32 threads) owns one query.
  - The intermediate top-k result is kept privately by that warp in registers:
      each lane stores K/32 sorted neighbors.
  - Each warp/query also owns:
      * a shared-memory candidate buffer of size K,
      * a shared shared-memory integer counter for the current candidate count.
  - Input data are processed in block-wide shared-memory tiles.
  - Distances are filtered against max_distance (current k-th neighbor distance).
  - Admitted candidates use atomicAdd on the per-warp shared counter to reserve slots.
  - When the candidate buffer becomes full, the warp merges it with the private top-k.
  - Final leftover candidates are merged after the last data tile.
  - No extra device/global memory is allocated.

  Merge strategy:
  - The candidate buffer is unsorted by construction.
  - On merge, the current sorted top-k is staged to shared memory.
  - The candidate buffer is sorted in shared memory with a warp-parallel bitonic sort
    (K is guaranteed to be a power of two).
  - The two sorted K-length arrays are then merged with a warp-parallel merge-path
    partitioning, keeping only the smallest K elements back in registers.

  Tuning:
  - 8 warps/block (256 threads): good balance between data-tile reuse and occupancy.
  - 2048-point shared-memory tiles: 16 KiB tile, still fits alongside the largest
    K=1024 per-warp scratch on A100/H100-class GPUs using opt-in dynamic shared memory.
*/

namespace {

constexpr int kWarpSize                 = 32;
constexpr int kWarpsPerBlock            = 8;
constexpr int kThreadsPerBlock          = kWarpsPerBlock * kWarpSize;
constexpr int kTilePoints               = 2048;
constexpr std::size_t kMaxOptInSmemB    = 163840;  // A100/H100-class opt-in per-block limit.

static_assert(kThreadsPerBlock <= 1024, "Invalid block size.");
static_assert((kTilePoints % kWarpSize) == 0, "Tile size must be a multiple of warp size.");

struct alignas(8) NeighborPair {
    float dist;
    int   idx;
};

static_assert(sizeof(NeighborPair) == 8, "Unexpected NeighborPair layout.");

template <typename T>
__host__ __device__ __forceinline__ constexpr std::size_t align_up_size(std::size_t x) {
    return (x + alignof(T) - 1) & ~(std::size_t(alignof(T)) - 1);
}

template <typename T>
__host__ __device__ __forceinline__ constexpr std::uintptr_t align_up_ptr(std::uintptr_t x) {
    return (x + alignof(T) - 1) & ~(std::uintptr_t(alignof(T)) - 1);
}

template <int K>
__host__ __device__ __forceinline__ constexpr std::size_t dynamic_shared_bytes() {
    std::size_t bytes = 0;
    bytes = align_up_size<float2>(bytes);
    bytes += std::size_t(kTilePoints) * sizeof(float2);
    bytes = align_up_size<NeighborPair>(bytes);
    bytes += std::size_t(kWarpsPerBlock) * std::size_t(K) * sizeof(NeighborPair);  // candidate buffers
    bytes = align_up_size<NeighborPair>(bytes);
    bytes += std::size_t(kWarpsPerBlock) * std::size_t(K) * sizeof(NeighborPair);  // merge scratch for top-k
    return bytes;
}

static_assert(dynamic_shared_bytes<1024>() <= kMaxOptInSmemB,
              "Worst-case shared-memory footprint exceeds the target GPU limit.");

__device__ __forceinline__ bool pair_less(const NeighborPair& a, const NeighborPair& b) {
    // Total order: distance first, index second. Ties are unconstrained by the prompt,
    // but a total order keeps the sort/merge deterministic and well-defined.
    return (a.dist < b.dist) || ((a.dist == b.dist) && (a.idx < b.idx));
}

__device__ __forceinline__ bool pair_less_equal(const NeighborPair& a, const NeighborPair& b) {
    return !pair_less(b, a);
}

template <int K>
__device__ __forceinline__ int merge_path_search(const NeighborPair* a,
                                                 const NeighborPair* b,
                                                 int diag) {
    // Standard merge-path partition on the diagonal "diag" for two sorted arrays
    // of equal length K. Returns how many items are taken from 'a' in the first
    // 'diag' output elements.
    int low  = (diag > K) ? (diag - K) : 0;
    int high = (diag < K) ? diag : K;

    while (low < high) {
        const int mid = (low + high) >> 1;
        // Compare b[diag - 1 - mid] < a[mid]
        if (pair_less(b[diag - 1 - mid], a[mid])) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

template <int K>
__device__ __forceinline__ void bitonic_sort_shared(NeighborPair* vals, int lane) {
    // Warp-parallel bitonic sort over K elements in shared memory.
    // K is a power of two in [32, 1024].
    for (int size = 2; size <= K; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int i = lane; i < K; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    NeighborPair a = vals[i];
                    NeighborPair b = vals[j];
                    const bool ascending = ((i & size) == 0);
                    const bool do_swap   = ascending ? pair_less(b, a) : pair_less(a, b);
                    if (do_swap) {
                        vals[i] = b;
                        vals[j] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

template <int K>
__device__ __forceinline__ void merge_buffer(NeighborPair* cand,
                                             NeighborPair* top_smem,
                                             int count,
                                             int lane,
                                             NeighborPair (&top)[K / kWarpSize],
                                             float& max_distance) {
    constexpr int ITEMS_PER_LANE = K / kWarpSize;

    // 1) Stage the current sorted top-k from registers to shared memory.
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        top_smem[lane * ITEMS_PER_LANE + i] = top[i];
    }

    // 2) Pad the candidate buffer to length K with +inf sentinels so that the
    //    same fixed-size sort/merge code works for both full and partial buffers.
    for (int pos = count + lane; pos < K; pos += kWarpSize) {
        cand[pos].dist = CUDART_INF_F;
        cand[pos].idx  = INT_MAX;
    }
    __syncwarp();

    // 3) Sort the shared candidate buffer.
    bitonic_sort_shared<K>(cand, lane);

    // 4) Merge the sorted current top-k and the sorted candidate buffer.
    //    Each lane owns a contiguous output segment of length ITEMS_PER_LANE.
    const int out_begin = lane * ITEMS_PER_LANE;
    const int out_end   = out_begin + ITEMS_PER_LANE;

    const int a_begin = merge_path_search<K>(top_smem, cand, out_begin);
    const int a_end   = merge_path_search<K>(top_smem, cand, out_end);

    int ai            = a_begin;
    int bi            = out_begin - a_begin;
    const int a_limit = a_end;
    const int b_limit = out_end - a_end;

#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        const bool take_a =
            (bi >= b_limit) ||
            ((ai < a_limit) && pair_less_equal(top_smem[ai], cand[bi]));
        top[i] = take_a ? top_smem[ai++] : cand[bi++];
    }

    // The k-th smallest entry is the last one, owned by lane 31.
    const float lane_last = top[ITEMS_PER_LANE - 1].dist;
    max_distance = __shfl_sync(0xFFFFFFFFu, lane_last, kWarpSize - 1);
}

template <int K>
__launch_bounds__(kThreadsPerBlock)
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two.");
    static_assert(K >= 32 && K <= 1024, "K out of supported range.");
    static_assert((K % kWarpSize) == 0, "K must be divisible by warp size.");

    constexpr int ITEMS_PER_LANE = K / kWarpSize;

    const int lane      = threadIdx.x & (kWarpSize - 1);
    const int warp      = threadIdx.x >> 5;
    const int query_idx = int(blockIdx.x) * kWarpsPerBlock + warp;
    const bool active   = (query_idx < query_count);

    // Lower-lane bit mask used to rank candidates inside a warp chunk.
    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    // Tiny shared state that is truly static: one candidate count per warp/query.
    // Large arrays remain dynamic shared memory so we can opt-in above 48 KiB.
    __shared__ int s_count[kWarpsPerBlock];

    extern __shared__ unsigned char shared_raw[];
    std::uintptr_t smem = align_up_ptr<float2>(reinterpret_cast<std::uintptr_t>(shared_raw));

    float2* s_data = reinterpret_cast<float2*>(smem);
    smem += std::size_t(kTilePoints) * sizeof(float2);

    smem = align_up_ptr<NeighborPair>(smem);
    NeighborPair* s_cand = reinterpret_cast<NeighborPair*>(smem);
    smem += std::size_t(kWarpsPerBlock) * std::size_t(K) * sizeof(NeighborPair);

    smem = align_up_ptr<NeighborPair>(smem);
    NeighborPair* s_top = reinterpret_cast<NeighborPair*>(smem);

    NeighborPair* cand     = s_cand + std::size_t(warp) * K;
    NeighborPair* top_smem = s_top  + std::size_t(warp) * K;

    // Private intermediate top-k in registers, kept globally sorted.
    NeighborPair top[ITEMS_PER_LANE];
#pragma unroll
    for (int i = 0; i < ITEMS_PER_LANE; ++i) {
        top[i].dist = CUDART_INF_F;
        top[i].idx  = INT_MAX;
    }

    float max_distance = CUDART_INF_F;

    if (lane == 0) {
        s_count[warp] = 0;
    }
    __syncthreads();

    // Load the query once per warp and broadcast it.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xFFFFFFFFu, qx, 0);
    qy = __shfl_sync(0xFFFFFFFFu, qy, 0);

    // Iterate over the dataset in shared-memory tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += kTilePoints) {
        const int remaining = data_count - tile_start;
        const int tile_count = (remaining < kTilePoints) ? remaining : kTilePoints;

        // Block-wide cooperative load of the current data tile into shared memory.
        for (int i = threadIdx.x; i < tile_count; i += kThreadsPerBlock) {
            s_data[i] = data[tile_start + i];
        }
        __syncthreads();

        if (active) {
            // Process the shared tile in warp-sized chunks of 32 points.
            //
            // This is important: because K >= 32, each lane holds at most one
            // "pending" candidate per chunk. If the shared candidate buffer is
            // nearly full, we can fill the remaining slots, merge once, and then
            // reconsider only the leftover lanes from this same 32-point chunk.
            for (int base = 0; base < tile_count; base += kWarpSize) {
                NeighborPair candidate;
                candidate.dist = 0.0f;
                candidate.idx  = 0;
                bool pending   = false;

                const int local = base + lane;
                if (local < tile_count) {
                    const float2 p = s_data[local];
                    const float dx = qx - p.x;
                    const float dy = qy - p.y;

                    // Squared Euclidean distance; no sqrt, as requested.
                    candidate.dist = fmaf(dx, dx, dy * dy);
                    candidate.idx  = tile_start + local;

                    // Strict '<' is sufficient because ties may be resolved arbitrarily.
                    pending = (candidate.dist < max_distance);
                }

                while (true) {
                    const unsigned pending_mask = __ballot_sync(0xFFFFFFFFu, pending);
                    if (pending_mask == 0u) {
                        break;
                    }

                    const int current_count = s_count[warp];
                    const int free_slots    = K - current_count;
                    const int hit_count     = __popc(pending_mask);

                    if (hit_count <= free_slots) {
                        // Everything fits into the candidate buffer.
                        // Use atomicAdd exactly as requested to reserve per-candidate slots.
                        if (pending) {
                            const int slot = atomicAdd(&s_count[warp], 1);
                            cand[slot] = candidate;
                        }
                        __syncwarp();

                        // If this insertion made the buffer full, merge immediately.
                        if (s_count[warp] == K) {
                            merge_buffer<K>(cand, top_smem, K, lane, top, max_distance);
                            if (lane == 0) {
                                s_count[warp] = 0;
                            }
                            __syncwarp();
                        }
                        break;
                    } else {
                        // Not everything fits. Keep the first 'free_slots' pending lanes
                        // (ranked by lane id), fill the buffer, merge, then re-test the
                        // leftover lanes against the new, tighter max_distance.
                        const int rank     = __popc(pending_mask & lane_mask_lt);
                        const bool take_now = pending && (rank < free_slots);

                        if (take_now) {
                            const int slot = atomicAdd(&s_count[warp], 1);
                            cand[slot] = candidate;
                        }
                        __syncwarp();

                        // By construction, the buffer is now exactly full.
                        merge_buffer<K>(cand, top_smem, K, lane, top, max_distance);
                        if (lane == 0) {
                            s_count[warp] = 0;
                        }
                        __syncwarp();

                        // Only leftover lanes survive, and only if they still beat the
                        // updated kth-distance threshold.
                        pending = pending && !take_now && (candidate.dist < max_distance);
                    }
                }
            }
        }

        // All warps must finish reading the tile before the next block-wide load.
        __syncthreads();
    }

    // Final merge of any leftover partial candidate buffer.
    if (active) {
        const int final_count = s_count[warp];
        if (final_count > 0) {
            merge_buffer<K>(cand, top_smem, final_count, lane, top, max_distance);
        }

        // Store the final sorted top-k for this query.
        // Write fields directly instead of constructing std::pair on device.
        const std::size_t out_base =
            std::size_t(query_idx) * std::size_t(K) +
            std::size_t(lane) * std::size_t(ITEMS_PER_LANE);

#pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; ++i) {
            result[out_base + std::size_t(i)].first  = top[i].idx;
            result[out_base + std::size_t(i)].second = top[i].dist;
        }
    }
}

template <int K>
void launch_knn_specialization(const float2* query,
                               int query_count,
                               const float2* data,
                               int data_count,
                               std::pair<int, float>* result) {
    constexpr std::size_t smem_bytes = dynamic_shared_bytes<K>();
    static_assert(smem_bytes <= kMaxOptInSmemB, "Shared-memory requirement too large.");

    // Required to opt in above the legacy 48 KiB dynamic shared-memory limit.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem_bytes));

    // This kernel is shared-memory heavy by design; prefer shared over L1.
    (void)cudaFuncSetAttribute(knn_kernel<K>,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const dim3 block(kThreadsPerBlock, 1, 1);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock, 1, 1);

    // Asynchronous launch on the current stream (default-stream semantics here).
    knn_kernel<K><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result);

    // Surface launch-configuration/shared-memory errors early without synchronizing.
    (void)cudaPeekAtLastError();
}

}  // namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Compile-time dispatch keeps K static, which is essential here:
    // - register storage is K/32 per lane,
    // - the candidate sort is specialized for each power-of-two K,
    // - merge scratch size is fixed and known to the compiler.
    switch (k) {
        case 32:   launch_knn_specialization<32>(query, query_count, data, data_count, result);   return;
        case 64:   launch_knn_specialization<64>(query, query_count, data, data_count, result);   return;
        case 128:  launch_knn_specialization<128>(query, query_count, data, data_count, result);  return;
        case 256:  launch_knn_specialization<256>(query, query_count, data, data_count, result);  return;
        case 512:  launch_knn_specialization<512>(query, query_count, data, data_count, result);  return;
        case 1024: launch_knn_specialization<1024>(query, query_count, data, data_count, result); return;
        default:
            // Per the problem statement, k is always valid.
            return;
    }
}