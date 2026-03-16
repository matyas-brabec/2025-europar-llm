#include <cuda_runtime.h>
#include <climits>
#include <utility>

// Exact brute-force k-NN for 2D float points.
//
// Design summary:
//   * One warp owns one query.
//   * The whole block streams the data set through shared memory in batches.
//   * Each warp keeps its current top-k in a warp-private max-heap stored in shared memory.
//     Shared memory is used instead of registers/local memory because k can be as large as 1024.
//   * Each 32-point chunk of the current batch is processed by one warp in parallel.
//   * Candidate updates are cooperative:
//       - sparse chunks: warp ballot + prefix compaction into a warp-private scratch buffer,
//       - dense chunks: cooperative 32-lane bitonic sort first, then heap updates in sorted order.
//   * No extra device memory is allocated; all temporary state lives in shared memory.
//
// Notes:
//   * Distances are squared Euclidean distances, as requested.
//   * Ties are allowed to resolve arbitrarily. Internally we use index as a deterministic
//     secondary key for heap/order stability, but the fast acceptance test compares distance only.
//   * Output is sorted ascending by distance for each query, so result[i * k + j] is the j-th
//     nearest neighbor of query[i].

namespace knn_detail {

using result_pair_t = std::pair<int, float>;

constexpr int kWarpThreads          = 32;
constexpr unsigned kFullMask        = 0xffffffffu;
constexpr int kBatchLoadsPerThread  = 4;   // Four float2 loads per thread is a good balance.
constexpr int kDenseSortThreshold   = 8;   // Sort only when many lanes survive the threshold.

__device__ __forceinline__ bool pair_less(float da, int ia, float db, int ib) {
    return (da < db) || ((da == db) && (ia < ib));
}

__device__ __forceinline__ bool pair_greater(float da, int ia, float db, int ib) {
    return (da > db) || ((da == db) && (ia > ib));
}

// Sift one heap node down in a max-heap.
// The "worst" (largest distance, then largest index) element is kept at the root.
// count is the current heap size.
__device__ __forceinline__ void heap_sift_down(float* heap_dist, int* heap_index, int start, int count) {
    float vd = heap_dist[start];
    int   vi = heap_index[start];
    int pos = start;

    #pragma unroll 1
    while (true) {
        const int left = (pos << 1) + 1;
        if (left >= count) {
            break;
        }

        const int right = left + 1;
        int child = left;
        if (right < count && pair_greater(heap_dist[right], heap_index[right], heap_dist[left], heap_index[left])) {
            child = right;
        }

        if (!pair_greater(heap_dist[child], heap_index[child], vd, vi)) {
            break;
        }

        heap_dist[pos]  = heap_dist[child];
        heap_index[pos] = heap_index[child];
        pos = child;
    }

    heap_dist[pos]  = vd;
    heap_index[pos] = vi;
}

template <int K>
__device__ __forceinline__ void build_max_heap(float* heap_dist, int* heap_index) {
    // O(K) heap construction; done once per query after the first K candidates have been loaded.
    #pragma unroll 1
    for (int parent = (K >> 1) - 1; parent >= 0; --parent) {
        heap_sift_down(heap_dist, heap_index, parent, K);
    }
}

template <int K>
__device__ __forceinline__ void heap_sort_ascending(float* heap_dist, int* heap_index) {
    // Final O(K log K) sort. This is serialized in lane 0 on purpose:
    // the scan over millions of points dominates runtime, so a simple in-place heapsort is best.
    #pragma unroll 1
    for (int end = K - 1; end > 0; --end) {
        const float td = heap_dist[0];
        heap_dist[0]   = heap_dist[end];
        heap_dist[end] = td;

        const int ti   = heap_index[0];
        heap_index[0]  = heap_index[end];
        heap_index[end]= ti;

        heap_sift_down(heap_dist, heap_index, 0, end);
    }
}

// 32-lane ascending bitonic sort on (distance, index).
// Used only for dense update chunks; inactive lanes carry (+inf, INT_MAX).
__device__ __forceinline__ void warp_bitonic_sort_32(float& d, int& i) {
    const int lane = threadIdx.x & (kWarpThreads - 1);

    #pragma unroll
    for (int k = 2; k <= kWarpThreads; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const float od = __shfl_xor_sync(kFullMask, d, j);
            const int   oi = __shfl_xor_sync(kFullMask, i, j);

            // Standard bitonic compare-exchange rule with XOR partners.
            const bool take_min = (((lane & k) == 0) == ((lane & j) == 0));
            if (take_min) {
                if (pair_greater(d, i, od, oi)) {
                    d = od;
                    i = oi;
                }
            } else {
                if (pair_less(d, i, od, oi)) {
                    d = od;
                    i = oi;
                }
            }
        }
    }
}

template <int K, int BLOCK_THREADS>
constexpr size_t knn_shared_bytes() {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0), "K must be a power of two in [32, 1024].");
    static_assert(BLOCK_THREADS == 128 || BLOCK_THREADS == 256 || BLOCK_THREADS == 512, "Unsupported block size.");
    static_assert((BLOCK_THREADS % kWarpThreads) == 0, "Block size must be a multiple of 32.");

    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpThreads;
    constexpr int BATCH_POINTS    = BLOCK_THREADS * kBatchLoadsPerThread;

    // Layout:
    //   sh_x[BATCH_POINTS]
    //   sh_y[BATCH_POINTS]
    //   heap_dist[WARPS_PER_BLOCK * K]
    //   heap_index[WARPS_PER_BLOCK * K]
    //   scratch_dist[WARPS_PER_BLOCK * 32]
    //   scratch_index[WARPS_PER_BLOCK * 32]
    return sizeof(float) * (2 * BATCH_POINTS + WARPS_PER_BLOCK * K + WARPS_PER_BLOCK * kWarpThreads) +
           sizeof(int)   * (WARPS_PER_BLOCK * K + WARPS_PER_BLOCK * kWarpThreads);
}

// Guard against accidentally exceeding A100's opt-in per-block dynamic shared-memory limit
// if tuning constants are changed later.
static_assert(knn_shared_bytes<1024, 512>() <= 163840, "Worst-case shared-memory footprint must fit A100/H100.");

template <int K, int BLOCK_THREADS>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           result_pair_t* __restrict__ result) {
    static_assert((BLOCK_THREADS % kWarpThreads) == 0, "Block size must be a multiple of 32.");

    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpThreads;
    constexpr int BATCH_POINTS    = BLOCK_THREADS * kBatchLoadsPerThread;

    extern __shared__ unsigned char smem_raw[];

    // Shared-memory batch cache.
    // We store x and y separately instead of as float2 to avoid 64-bit shared-memory bank conflicts.
    float* sh_x = reinterpret_cast<float*>(smem_raw);
    float* sh_y = sh_x + BATCH_POINTS;

    // Warp-private top-k heaps.
    float* heap_dist_all = sh_y + BATCH_POINTS;
    int*   heap_index_all = reinterpret_cast<int*>(heap_dist_all + WARPS_PER_BLOCK * K);

    // Warp-private candidate scratch space (compaction/sorted dense updates).
    float* scratch_dist_all = reinterpret_cast<float*>(heap_index_all + WARPS_PER_BLOCK * K);
    int*   scratch_index_all = reinterpret_cast<int*>(scratch_dist_all + WARPS_PER_BLOCK * kWarpThreads);

    const int tid              = threadIdx.x;
    const int lane             = tid & (kWarpThreads - 1);
    const int warp_id_in_block = tid >> 5;

    const int query_idx   = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp_id_in_block;
    const bool active_query = (query_idx < query_count);

    float* const heap_dist    = heap_dist_all + warp_id_in_block * K;
    int*   const heap_index   = heap_index_all + warp_id_in_block * K;
    float* const scratch_dist = scratch_dist_all + warp_id_in_block * kWarpThreads;
    int*   const scratch_index= scratch_index_all + warp_id_in_block * kWarpThreads;

    // One lane fetches the query point; the warp receives it via shuffles.
    float qx = 0.0f;
    float qy = 0.0f;
    if (lane == 0 && active_query) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Stream the database through shared memory in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_POINTS) {
        const int remaining   = data_count - batch_start;
        const int batch_count = (remaining < BATCH_POINTS) ? remaining : BATCH_POINTS;

        // Whole-block cooperative load into shared memory.
        // The batch size is chosen so that each thread loads exactly four float2 values when full.
        #pragma unroll
        for (int it = 0; it < kBatchLoadsPerThread; ++it) {
            const int t = tid + it * BLOCK_THREADS;
            if (t < batch_count) {
                const float2 p = data[batch_start + t];
                sh_x[t] = p.x;
                sh_y[t] = p.y;
            }
        }

        __syncthreads();

        if (active_query) {
            // Consume the shared-memory batch in 32-point chunks, one point per lane.
            #pragma unroll 1
            for (int chunk = 0; chunk < batch_count; chunk += kWarpThreads) {
                const int local_idx  = chunk + lane;
                const int global_idx = batch_start + local_idx;
                const int global_chunk_start = batch_start + chunk;

                float dist = CUDART_INF_F;
                if (local_idx < batch_count) {
                    const float dx = qx - sh_x[local_idx];
                    const float dy = qy - sh_y[local_idx];
                    dist = fmaf(dx, dx, dy * dy);
                }

                // Initialization:
                // the first K candidates are copied directly into the warp-private heap storage.
                // Because K is always a multiple of 32, initialization boundaries align with warp chunks.
                if (global_chunk_start < K) {
                    const int slot = global_idx;  // slot == data index for the first K points
                    heap_dist[slot]  = dist;
                    heap_index[slot] = slot;

                    if (global_chunk_start + kWarpThreads == K) {
                        __syncwarp(kFullMask);
                        if (lane == 0) {
                            build_max_heap<K>(heap_dist, heap_index);
                        }
                        __syncwarp(kFullMask);
                    }
                    continue;
                }

                // Fast rejection against the current worst top-k distance.
                float threshold = 0.0f;
                if (lane == 0) {
                    threshold = heap_dist[0];
                }
                threshold = __shfl_sync(kFullMask, threshold, 0);

                // Ties can be handled arbitrarily, so the fast mask compares distance only.
                const bool better = (local_idx < batch_count) && (dist < threshold);
                const unsigned better_mask = __ballot_sync(kFullMask, better);

                if (better_mask == 0u) {
                    continue;
                }

                const int better_count = __popc(better_mask);

                if (better_count > kDenseSortThreshold) {
                    // Dense-update path:
                    // many lanes survived the threshold, so it pays to sort the 32-lane candidate
                    // chunk cooperatively first. Once candidates are sorted, lane 0 can stop
                    // inserting as soon as the heap root drops below the remaining candidates.
                    float cand_dist = better ? dist : CUDART_INF_F;
                    int   cand_index = better ? global_idx : INT_MAX;

                    warp_bitonic_sort_32(cand_dist, cand_index);

                    scratch_dist[lane]  = cand_dist;
                    scratch_index[lane] = cand_index;
                    __syncwarp(kFullMask);

                    if (lane == 0) {
                        #pragma unroll 1
                        for (int c = 0; c < better_count; ++c) {
                            const float cd = scratch_dist[c];
                            if (!(cd < heap_dist[0])) {
                                break;
                            }
                            heap_dist[0]  = cd;
                            heap_index[0] = scratch_index[c];
                            heap_sift_down(heap_dist, heap_index, 0, K);
                        }
                    }

                    __syncwarp(kFullMask);
                } else {
                    // Sparse-update path:
                    // ballot + prefix compaction is cheaper than sorting the full warp.
                    if (better) {
                        const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);
                        const unsigned rank = __popc(better_mask & lane_mask_lt);
                        scratch_dist[rank]  = dist;
                        scratch_index[rank] = global_idx;
                    }
                    __syncwarp(kFullMask);

                    if (lane == 0) {
                        #pragma unroll 1
                        for (int c = 0; c < better_count; ++c) {
                            const float cd = scratch_dist[c];
                            if (cd < heap_dist[0]) {
                                heap_dist[0]  = cd;
                                heap_index[0] = scratch_index[c];
                                heap_sift_down(heap_dist, heap_index, 0, K);
                            }
                        }
                    }

                    __syncwarp(kFullMask);
                }
            }
        }

        // Make sure no warp still reads the current batch before the block overwrites shared memory.
        __syncthreads();
    }

    if (active_query) {
        if (lane == 0) {
            heap_sort_ascending<K>(heap_dist, heap_index);
        }
        __syncwarp(kFullMask);

        const size_t out_base = static_cast<size_t>(query_idx) * static_cast<size_t>(K);

        // Member-wise stores avoid depending on any device-side std::pair constructor/operator=.
        #pragma unroll 1
        for (int j = lane; j < K; j += kWarpThreads) {
            result[out_base + static_cast<size_t>(j)].first  = heap_index[j];
            result[out_base + static_cast<size_t>(j)].second = heap_dist[j];
        }
    }
}

template <int K, int BLOCK_THREADS>
inline void launch_knn_variant(const float2* query,
                               int query_count,
                               const float2* data,
                               int data_count,
                               result_pair_t* result) {
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / kWarpThreads;
    constexpr size_t SHMEM_BYTES  = knn_shared_bytes<K, BLOCK_THREADS>();

    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // The kernel is shared-memory heavy by construction; explicitly opt in to the larger
    // per-block shared-memory limit and request maximum shared-memory carveout.
    cudaFuncSetAttribute(knn_kernel<K, BLOCK_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(SHMEM_BYTES));
    cudaFuncSetAttribute(knn_kernel<K, BLOCK_THREADS>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    knn_kernel<K, BLOCK_THREADS><<<blocks, BLOCK_THREADS, SHMEM_BYTES>>>(
        query, query_count, data, data_count, result);
}

template <int BLOCK_THREADS>
inline void dispatch_k(const float2* query,
                       int query_count,
                       const float2* data,
                       int data_count,
                       result_pair_t* result,
                       int k) {
    // k is restricted to a small fixed set, so compile-time specialization is worth it.
    // This lets the compiler know the heap size and shared-memory layout exactly.
    switch (k) {
        case 32:   launch_knn_variant<32,   BLOCK_THREADS>(query, query_count, data, data_count, result); break;
        case 64:   launch_knn_variant<64,   BLOCK_THREADS>(query, query_count, data, data_count, result); break;
        case 128:  launch_knn_variant<128,  BLOCK_THREADS>(query, query_count, data, data_count, result); break;
        case 256:  launch_knn_variant<256,  BLOCK_THREADS>(query, query_count, data, data_count, result); break;
        case 512:  launch_knn_variant<512,  BLOCK_THREADS>(query, query_count, data, data_count, result); break;
        case 1024: launch_knn_variant<1024, BLOCK_THREADS>(query, query_count, data, data_count, result); break;
        default:   break;  // Inputs are guaranteed valid by the problem statement.
    }
}

inline int choose_block_threads(int query_count) {
    // Heuristic:
    // choose the largest queries-per-block tile that still gives at least one grid block per SM.
    // Larger blocks improve batch reuse and amortize synchronization better, but too-large blocks
    // reduce the number of blocks and can leave SMs idle when query_count is only in the low
    // thousands. The 128/256/512 set maps to 4/8/16 queries per block respectively.
    int sms = 108;  // Safe default matching A100-class hardware if the query fails.
    int device = 0;

    if (cudaGetDevice(&device) == cudaSuccess) {
        int queried_sms = 0;
        if (cudaDeviceGetAttribute(&queried_sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
            queried_sms > 0) {
            sms = queried_sms;
        }
    }

    if (query_count >= sms * 16) {
        return 512;
    }
    if (query_count >= sms * 8) {
        return 256;
    }
    return 128;
}

}  // namespace knn_detail

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    // Normal CUDA semantics: asynchronous launch, no extra device allocation, no implicit sync.
    if (query_count <= 0) {
        return;
    }

    const int block_threads = knn_detail::choose_block_threads(query_count);

    switch (block_threads) {
        case 512:
            knn_detail::dispatch_k<512>(query, query_count, data, data_count, result, k);
            break;
        case 256:
            knn_detail::dispatch_k<256>(query, query_count, data, data_count, result, k);
            break;
        default:
            knn_detail::dispatch_k<128>(query, query_count, data, data_count, result, k);
            break;
    }
}