#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>
#include <utility>

// This implementation maps one warp (32 threads) to one query point.
// Each warp maintains its private intermediate top-k (indices and distances) in registers,
// and a per-warp candidate buffer of size k in shared memory. Data points are processed
// in tiles cached in shared memory by the whole block. Candidates whose distances are
// smaller than the current k-th best distance (threshold) are appended to the buffer.
// Whenever the buffer becomes full, it is merged with the intermediate result using a
// warp-parallel merge-path procedure. The candidate buffer is sorted in shared memory
// by a warp-only bitonic sort. The intermediate result is kept globally sorted ascending,
// which allows efficient parallel merge and fast threshold retrieval.
//
// Constraints leveraged:
// - k is a power of two between 32 and 1024 inclusive.
// - data_count >= k
// - Arrays are device-allocated, and we allocate no additional device memory.
// - We use dynamic shared memory per block for the data tile cache and per-warp buffers.
//
// Notes:
// - To avoid large register pressure for 2k items in a merge, we copy the current best
//   (size k) into per-warp shared memory temporarily during merge. Thus, per warp we
//   need 2*k pairs (candidate + best tmp) in shared memory during merge.
// - We select WARPS_PER_BLOCK conservatively (4) so that even at k=1024 we remain well
//   within A100's per-block shared memory limit after opting-in to the maximum dynamic
//   shared memory (164KB on A100, 228KB on H100).
// - We specialize the kernel for K in {32, 64, 128, 256, 512, 1024} via templates.
//   At runtime, we dispatch to the appropriate specialization based on k.
//
// Implementation details:
// - Pair layout is compatible with std::pair<int, float> for device writes.
// - All warp collectives use a full mask (0xffffffff). Inactive warps (queries >= count)
//   still participate in ballots but never satisfy the accept condition, so they don't
//   write to buffers.
// - The bitonic sort is implemented as a warp-serial algorithm over K elements using
//   32 threads. It uses __syncwarp barriers between stages.
// - The parallel merge uses merge-path to partition the first K elements of the merge
//   of (best, sorted candidates) across 32 lanes. Each lane merges a small consecutive
//   slice (K/32 items) sequentially and writes directly into its per-lane register chunk.
//
// Tuning knobs chosen:
// - WARPS_PER_BLOCK = 4 (128 threads per block)
// - Tile size is chosen at runtime to maximally utilize the available shared memory,
//   after accounting for per-warp buffers.

#ifndef KNN_WARPS_PER_BLOCK
#define KNN_WARPS_PER_BLOCK 4
#endif

// Output pair type with the same memory layout as std::pair<int, float>.
struct __align__(8) PairIF {
    int   first;   // index
    float second;  // distance
};

// Utility: CUDA check macro (used in host function).
static inline void __checkCuda(cudaError_t err, const char* stmt, const char* file, int line) {
#if !defined(NDEBUG)
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", stmt, file, line, cudaGetErrorString(err));
        abort();
    }
#else
    (void)err; (void)stmt; (void)file; (void)line;
#endif
}
#define CUDA_CHECK(x) __checkCuda((x), #x, __FILE__, __LINE__)

template<int K>
struct KnnTraits {
    static constexpr int WARP_SIZE = 32;
    static constexpr int LANES     = 32;
    static constexpr int CHUNK     = K / LANES; // number of elements per lane
    static_assert((K % WARP_SIZE) == 0, "K must be a multiple of 32.");
};

// Warp-wide bitonic sort for K elements stored in shared memory as PairIF.
// Sort ascending by 'second' (distance).
template<int K>
__device__ __forceinline__
void warp_bitonic_sort_pairs(PairIF* arr) {
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    const unsigned lane = threadIdx.x & 31;

    // Classic bitonic sort network: size is the total sequence length of the bitonic merge,
    // stride is the distance between compare-exchange pairs at the current stage.
    for (unsigned size = 2; size <= (unsigned)K; size <<= 1) {
        bool up = true; // direction determined per index i via (i & size) == 0
        // For each bitonic merge stage, do a log(size) pass.
        for (unsigned stride = size >> 1; stride > 0; stride >>= 1) {
            // Partition the K indices across 32 lanes, each lane processes i = lane, lane+32, ...
            for (unsigned i = lane; i < (unsigned)K; i += 32u) {
                unsigned j = i ^ stride;
                if (j > i && j < (unsigned)K) {
                    bool dir = ((i & size) == 0); // true => ascending
                    PairIF a = arr[i];
                    PairIF b = arr[j];
                    // Compare distances
                    bool swap_needed = dir ? (a.second > b.second) : (a.second < b.second);
                    if (swap_needed) {
                        // Swap
                        arr[i] = b;
                        arr[j] = a;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

// Merge-path partition search for merging first diag elements from A and B.
// A and B are sorted ascending arrays of length nA and nB (here both = K).
// Returns ai, bi such that ai + bi = diag and A[ai-1] <= B[bi] and B[bi-1] < A[ai].
// A and B are in shared memory.
__device__ __forceinline__
void merge_path_search(const PairIF* __restrict__ A, const PairIF* __restrict__ B,
                       int nA, int nB, int diag, int& ai, int& bi) {
    // Clamp the search range for ai
    int lo = max(0, diag - nB);
    int hi = min(diag, nA);
    // Sentinels: we will manually handle boundaries when reading values.
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int a_idx = mid;
        int b_idx = diag - mid;

        float a_left  = (a_idx > 0)     ? A[a_idx - 1].second : -CUDART_INF_F;
        float b_left  = (b_idx > 0)     ? B[b_idx - 1].second : -CUDART_INF_F;
        float a_right = (a_idx < nA)    ? A[a_idx].second     : CUDART_INF_F;
        float b_right = (b_idx < nB)    ? B[b_idx].second     : CUDART_INF_F;

        if (a_left <= b_right && b_left < a_right) {
            ai = a_idx;
            bi = b_idx;
            return;
        }
        if (a_left > b_right) {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    // Fallback (shouldn't trigger): place all in A or B.
    ai = min(diag, nA);
    bi = diag - ai;
}

// Merge the first K elements from the sorted arrays A (best_tmp) and B (cand sorted)
// into per-lane register arrays best_idx/best_dist, using merge-path partitioning.
// Each lane writes exactly CHUNK (=K/32) outputs.
template<int K>
__device__ __forceinline__
void warp_merge_topk_from_shared(const PairIF* __restrict__ A,
                                 const PairIF* __restrict__ B,
                                 int* __restrict__ best_idx, float* __restrict__ best_dist) {
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    constexpr int CHUNK = KnnTraits<K>::CHUNK;
    const int lane = threadIdx.x & 31;

    // Each lane is responsible for output positions [d0, d1)
    int d0 = lane * CHUNK;
    int d1 = d0 + CHUNK;

    int a0, b0, a1, b1;
    merge_path_search(A, B, K, K, d0, a0, b0);
    merge_path_search(A, B, K, K, d1, a1, b1);

    // Sequentially merge the assigned segment
    int ia = a0;
    int ib = b0;
    for (int t = 0; t < CHUNK; ++t) {
        // Boundaries
        bool takeA = false;
        float va = CUDART_INF_F;
        float vb = CUDART_INF_F;
        if (ia < a1) va = A[ia].second;
        if (ib < b1) vb = B[ib].second;
        if (va <= vb) {
            takeA = true;
        }
        if (takeA) {
            best_dist[t] = va;
            best_idx[t]  = A[ia].first;
            ++ia;
        } else {
            best_dist[t] = vb;
            best_idx[t]  = B[ib].first;
            ++ib;
        }
    }
    __syncwarp(FULL_MASK);
}

// Merge utility: given per-lane register arrays best_idx/best_dist (globally sorted ascending),
// and a per-warp candidate buffer cand (length 'cand_count' <= K), perform:
// - pad cand to K with +inf
// - sort cand ascending in shared memory via warp bitonic
// - copy current best (registers) into best_tmp (shared)
// - merge-path to produce the first K elements of merge(best_tmp, cand) back into registers
template<int K>
__device__ __forceinline__
void warp_merge_buffer_with_best(PairIF* __restrict__ cand,
                                 PairIF* __restrict__ best_tmp,
                                 int    cand_count,
                                 int*   __restrict__ best_idx,
                                 float* __restrict__ best_dist) {
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    constexpr int CHUNK = KnnTraits<K>::CHUNK;
    const int lane = threadIdx.x & 31;

    // 1) Pad candidate buffer to length K with +inf distances.
    for (int i = lane; i < K; i += 32) {
        if (i >= cand_count) {
            cand[i].first  = -1;
            cand[i].second = CUDART_INF_F;
        }
    }
    __syncwarp(FULL_MASK);

    // 2) Sort candidate buffer ascending by distance.
    warp_bitonic_sort_pairs<K>(cand);

    // 3) Copy current best from per-lane registers into shared memory contiguous array.
    // Layout: lane t stores its CHUNK items at positions [t*CHUNK .. t*CHUNK + CHUNK-1]
    int base = lane * CHUNK;
    for (int j = 0; j < CHUNK; ++j) {
        best_tmp[base + j].first  = best_idx[j];
        best_tmp[base + j].second = best_dist[j];
    }
    __syncwarp(FULL_MASK);

    // 4) Merge first K elements from (best_tmp, cand) into new best arrays in registers.
    warp_merge_topk_from_shared<K>(best_tmp, cand, best_idx, best_dist);
}

// Kernel: One warp per query; WARPS_PER_BLOCK warps per block.
template<int K, int WARPS_PER_BLOCK>
__global__ void knn_kernel_2d(const float2* __restrict__ query,
                              int query_count,
                              const float2* __restrict__ data,
                              int data_count,
                              PairIF* __restrict__ result,
                              int tile_elems) {
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    constexpr int WARP_SIZE = 32;
    constexpr int CHUNK = KnnTraits<K>::CHUNK;

    extern __shared__ unsigned char smem_raw[];
    // Shared memory layout: [tile (float2) | cand[W*K] (PairIF) | best_tmp[W*K] (PairIF)]
    float2* tile = reinterpret_cast<float2*>(smem_raw);
    PairIF* base_cand = reinterpret_cast<PairIF*>(tile + tile_elems);
    PairIF* base_best = base_cand + WARPS_PER_BLOCK * K;

    const int lane      = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_b = threadIdx.x >> 5;
    const int warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_in_b; // query id handled by this warp
    const bool warp_active = (warp_id < query_count);

    // Per-warp pointers to shared memory buffers
    PairIF* cand     = base_cand + warp_in_b * K;
    PairIF* best_tmp = base_best + warp_in_b * K;

    // Load the query point once (by lane 0) and broadcast to warp
    float qx = 0.0f, qy = 0.0f;
    if (lane == 0) {
        if (warp_active) {
            float2 q = query[warp_id];
            qx = q.x; qy = q.y;
        }
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Per-warp intermediate top-k in registers (distributed across lanes)
    int   best_idx[CHUNK];
    float best_dist[CHUNK];
    for (int j = 0; j < CHUNK; ++j) {
        best_idx[j]  = -1;
        best_dist[j] = CUDART_INF_F;
    }
    // Current threshold (k-th best) for candidate filtering
    float threshold = CUDART_INF_F;

    // Candidate buffer fill count (replicated across lanes; updated by lane 0)
    int cand_count = 0;

    // Process data in tiles
    for (int tile_base = 0; tile_base < data_count; tile_base += tile_elems) {
        int tile_count = min(tile_elems, data_count - tile_base);

        // Load tile into shared memory cooperatively by the whole block
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            tile[t] = data[tile_base + t];
        }
        __syncthreads();

        // Each warp processes the cached tile: compute distances and push candidates
        for (int i = lane; i < tile_count; i += WARP_SIZE) {
            // Compute squared L2 distance to the query
            float2 p = tile[i];
            float dx = p.x - qx;
            float dy = p.y - qy;
            float dist = dx * dx + dy * dy;
            int   idx  = tile_base + i;

            // Evaluate candidate under current threshold and warp activity
            bool accept = warp_active && (dist < threshold);

            // Warp-cooperative append to the candidate buffer with overflow handling
            // If we overflow buffer capacity K, we merge buffer with best, update threshold,
            // and retry the leftover accepted candidates (if any) under the new threshold.
            unsigned mask = __ballot_sync(FULL_MASK, accept);
            int naccept = __popc(mask);

            while (naccept > 0) {
                int capacity = K - cand_count;
                if (capacity == 0) {
                    // Merge buffer with current best (buffer full)
                    warp_merge_buffer_with_best<K>(cand, best_tmp, K, best_idx, best_dist);
                    cand_count = 0;
                    // Update threshold to the new k-th best, which is the last element of global sorted best
                    float last = (lane == 31) ? best_dist[CHUNK - 1] : -CUDART_INF_F;
                    threshold = __shfl_sync(FULL_MASK, last, 31);
                    // Re-evaluate acceptance under new (smaller) threshold
                    accept = warp_active && (dist < threshold);
                    mask = __ballot_sync(FULL_MASK, accept);
                    naccept = __popc(mask);
                    continue;
                }

                int take = min(capacity, naccept);

                // Compute per-lane rank among accepted lanes
                unsigned lane_mask_lt = (1u << lane) - 1u;
                int rank_in_mask = __popc(mask & lane_mask_lt);
                bool take_this = accept && (rank_in_mask < take);

                // Broadcast base offset (previous cand_count)
                int base = __shfl_sync(FULL_MASK, cand_count, 0);

                if (take_this) {
                    int pos = base + rank_in_mask;
                    cand[pos].first  = idx;
                    cand[pos].second = dist;
                }

                // Update cand_count by lane 0
                if (lane == 0) {
                    cand_count += take;
                }
                cand_count = __shfl_sync(FULL_MASK, cand_count, 0);

                // Remove taken lanes from mask and continue if leftovers remain
                if (take == naccept) {
                    // All accepted written, we're done for this i
                    naccept = 0;
                } else {
                    // Some leftover accepted still need to be written, buffer now full -> merge
                    // Mark taken lanes as not accepted for the next round
                    accept = accept && (rank_in_mask >= take);
                    mask = __ballot_sync(FULL_MASK, accept);
                    naccept = __popc(mask);
                    // Buffer should be full here; merge to free it
                    if (cand_count == K) {
                        warp_merge_buffer_with_best<K>(cand, best_tmp, K, best_idx, best_dist);
                        cand_count = 0;
                        float last = (lane == 31) ? best_dist[CHUNK - 1] : -CUDART_INF_F;
                        threshold = __shfl_sync(FULL_MASK, last, 31);
                    }
                }
            }
        }
        __syncthreads();
    }

    // After processing all tiles, merge any remaining candidates in the buffer
    if (cand_count > 0) {
        warp_merge_buffer_with_best<K>(cand, best_tmp, cand_count, best_idx, best_dist);
        cand_count = 0;
        float last = (lane == 31) ? best_dist[CHUNK - 1] : -CUDART_INF_F;
        threshold = __shfl_sync(FULL_MASK, last, 31);
        (void)threshold;
    }

    // Write final results for this query (globally sorted ascending by distance)
    if (warp_active) {
        PairIF* out = result + warp_id * K;
        int base = lane * CHUNK;
        for (int j = 0; j < CHUNK; ++j) {
            out[base + j].first  = best_idx[j];
            out[base + j].second = best_dist[j];
        }
    }
}

// Host-side launcher: choose specialization based on k, configure shared memory and launch.
static inline int round_down_pow2(int x) {
    int p = 1;
    while ((p << 1) <= x) p <<= 1;
    return p;
}

template<int K>
void launch_knn_kernel(const float2* query, int query_count,
                       const float2* data, int data_count,
                       PairIF* result,
                       int device_id) {
    constexpr int WARPS_PER_BLOCK = KNN_WARPS_PER_BLOCK;
    constexpr int WARP_SIZE = 32;
    constexpr size_t PER_WARP_BYTES = (size_t)(2 * K) * sizeof(PairIF); // cand + best_tmp
    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    // Opt-in to the maximum dynamic shared memory per block for this kernel specialization.
    int optin_smem = 0;
#if CUDART_VERSION >= 11000
    CUDA_CHECK(cudaDeviceGetAttribute(&optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
#else
    optin_smem = prop.sharedMemPerBlock;
#endif
    // Set the function attribute to allow using up to optin_smem bytes of dynamic shared memory.
    CUDA_CHECK(cudaFuncSetAttribute(knn_kernel_2d<K, WARPS_PER_BLOCK>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    optin_smem));

    // Determine tile size to fit within shared memory limit
    // smem_total = tile_elems*sizeof(float2) + WARPS_PER_BLOCK*(2*K*sizeof(PairIF))
    size_t per_block_warp_bytes = WARPS_PER_BLOCK * PER_WARP_BYTES;
    size_t max_tile_bytes = (optin_smem > (int)per_block_warp_bytes)
                                ? (optin_smem - per_block_warp_bytes)
                                : 0;
    // Choose tile_elems as a multiple of 256 for good load balance, but at least 1024.
    int tile_elems = (int)(max_tile_bytes / sizeof(float2));
    tile_elems = max(1024, (tile_elems / 256) * 256);
    // Clamp tile_elems to data_count to avoid excessive smem request for small data.
    tile_elems = min(tile_elems, data_count > 0 ? data_count : 1);

    // Final dynamic shared memory size requested at launch
    size_t dyn_smem = (size_t)tile_elems * sizeof(float2) + per_block_warp_bytes;

    // Compute grid configuration
    int warps_needed = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(warps_needed, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    // Launch
    knn_kernel_2d<K, WARPS_PER_BLOCK><<<grid, block, dyn_smem>>>(
        query, query_count, data, data_count, result, tile_elems
    );
}

/// @FIXED
/// extern "C"

void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0) return;
    // Cast the result pointer to our layout-compatible pair type
    PairIF* result_if = reinterpret_cast<PairIF*>(result);

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));

    // Dispatch based on k (power of two between 32 and 1024 inclusive)
    switch (k) {
        case 32:   launch_knn_kernel<32 >(query, query_count, data, data_count, result_if, device_id); break;
        case 64:   launch_knn_kernel<64 >(query, query_count, data, data_count, result_if, device_id); break;
        case 128:  launch_knn_kernel<128>(query, query_count, data, data_count, result_if, device_id); break;
        case 256:  launch_knn_kernel<256>(query, query_count, data, data_count, result_if, device_id); break;
        case 512:  launch_knn_kernel<512>(query, query_count, data, data_count, result_if, device_id); break;
        case 1024: launch_knn_kernel<1024>(query, query_count, data, data_count, result_if, device_id); break;
        default:
            // Fallback: round down to nearest supported power of two (shouldn't happen per spec).
            // This ensures we never crash, but the result will be the k' nearest with k' < k.
            {
                int kk = round_down_pow2(k);
                if      (kk >= 1024) launch_knn_kernel<1024>(query, query_count, data, data_count, result_if, device_id);
                else if (kk >= 512)  launch_knn_kernel<512 >(query, query_count, data, data_count, result_if, device_id);
                else if (kk >= 256)  launch_knn_kernel<256 >(query, query_count, data, data_count, result_if, device_id);
                else if (kk >= 128)  launch_knn_kernel<128 >(query, query_count, data, data_count, result_if, device_id);
                else if (kk >= 64)   launch_knn_kernel<64  >(query, query_count, data, data_count, result_if, device_id);
                else                 launch_knn_kernel<32  >(query, query_count, data, data_count, result_if, device_id);
            }
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}
