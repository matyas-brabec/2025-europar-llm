#include <cuda_runtime.h>
#include <utility>

// This implementation computes k-NN for 2D points using one warp (32 threads) per query.
// Key design points:
// - The dataset is processed in tiles loaded into shared memory by the whole block.
// - Each warp maintains a private top-k structure in shared memory (sorted ascending by distance).
// - Candidates are collected per warp into a shared "insertion buffer" in parallel.
// - When the buffer fills (or at the end), the warp sorts the buffer in parallel (bitonic sort, warp-synchronous)
//   and then performs a parallel merge (merge-path across 32 lanes) with the current top-k to keep the best k.
// - All updates use warp-level synchronization (__syncwarp), and tiles are loaded using block synchronization (__syncthreads).
//
// Assumptions:
// - k is a power of two, 32 <= k <= 1024.
// - data_count >= k.
// - Pointers query, data, result refer to device memory allocated via cudaMalloc.
//
// Tunables chosen for modern data center GPUs (A100/H100):
// - Warps per block: 4 (i.e., 128 threads per block).
// - Tile size: 2048 points per tile.
// - Per-warp buffer size: 256 candidates.
// These balance shared memory usage (~88KB worst-case at k=1024) and parallel efficiency.
//
// Note: The algorithm uses squared Euclidean distance.
//       The output neighbors for each query are written in non-decreasing order of distance.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable parameters
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

#ifndef TILE_POINTS
#define TILE_POINTS 2048
#endif

#ifndef WARP_BUFFER_CAP
#define WARP_BUFFER_CAP 256
#endif

#ifndef ROUND_BATCH
#define ROUND_BATCH 8
#endif

// Utility: next power-of-two for positive integers
__device__ __forceinline__ int next_pow2(int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// Utility: squared L2 distance between two float2 points
__device__ __forceinline__ float dist2(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Warp-synchronous exclusive prefix sum of 'val' (32 threads).
// Returns the exclusive prefix; 'total' receives the warp-wide sum (valid in all lanes).
__device__ __forceinline__ int warp_excl_prefix_sum(int val, int &total, unsigned mask) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int inclusive = val;
    // Inclusive scan using shfl_up
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int n = __shfl_up_sync(mask, inclusive, offset);
        if (lane >= offset) inclusive += n;
    }
    total = __shfl_sync(mask, inclusive, WARP_SIZE - 1);
    int exclusive = inclusive - val;
    return exclusive;
}

// Warp-synchronous bitonic sort (ascending) on pairs (dist, idx) stored in shared memory.
// Npad must be a power of two; elements beyond 'N' should have been padded with +INF to handle non-power-of-two sizes.
// The algorithm uses the standard bitonic network with XOR-swap pattern and warp-level synchronization.
__device__ __forceinline__ void warp_bitonic_sort_pairs(float *dist, int *idx, int N, int Npad, unsigned mask) {
    // Pad remainder [N .. Npad-1] assumed already set to +INF and idx=-1 by caller.
    // Perform bitonic sort over Npad elements. Only elements < N carry meaningful finite values.
    for (int k = 2; k <= Npad; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x & (WARP_SIZE - 1); i < Npad; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    float ai = dist[i];
                    float aj = dist[ixj];
                    int ii = idx[i];
                    int ij = idx[ixj];
                    // Ascending order if (i & k) == 0; descending otherwise
                    bool ascending = ((i & k) == 0);
                    bool do_swap = ascending ? (ai > aj) : (ai < aj);
                    if (do_swap) {
                        dist[i] = aj;
                        dist[ixj] = ai;
                        idx[i] = ij;
                        idx[ixj] = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
    // After sorting, the first N entries are sorted ascending; padded +INF naturally go to the end.
    (void)N; // N is not used here beyond padding check.
}

// Merge-path search: find partition (i, j) for diagonal 'diag' in the merge of arrays A (len a_len) and B (len b_len).
// Both A and B are sorted ascending. We find i in [max(0, diag - b_len), min(diag, a_len)] s.t.
// (i == 0 || A[i-1] <= B[diag - i]) && (diag - i == 0 || B[diag - i - 1] < A[i])
__device__ __forceinline__ int merge_path_search(const float *A, int a_len, const float *B, int b_len, int diag) {
    int low = max(0, diag - b_len);
    int high = min(diag, a_len);
    while (low < high) {
        int i = (low + high) >> 1;
        int j = diag - i;
        float a_i = (i < a_len) ? A[i] : CUDART_INF_F;
        float b_jm1 = (j > 0) ? B[j - 1] : -CUDART_INF_F;
        if (a_i < b_jm1) {
            low = i + 1;
        } else {
            high = i;
        }
    }
    return low;
}

// Warp-parallel merge (merge-path) of two sorted lists A (len a_len) and B (len b_len) into output C (len out_len).
// out_len should be <= a_len + b_len. Each lane handles a disjoint segment of the output.
__device__ __forceinline__ void warp_merge_path_pairs(
    const float *A_dist, const int *A_idx, int a_len,
    const float *B_dist, const int *B_idx, int b_len,
    float *C_dist, int *C_idx, int out_len, unsigned mask) {

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int begin = (lane * out_len) / WARP_SIZE;
    int end   = ((lane + 1) * out_len) / WARP_SIZE;
    if (begin >= end) {
        return;
    }

    // Find starting partition (i, j) for 'begin', and ending partition for 'end'
    int i_begin = merge_path_search(A_dist, a_len, B_dist, b_len, begin);
    int j_begin = begin - i_begin;
    int i_end   = merge_path_search(A_dist, a_len, B_dist, b_len, end);
    int j_end   = end - i_end;

    int i = i_begin;
    int j = j_begin;
    int out = begin;

    // Serial merge for this lane's segment
    while (out < end) {
        // Pick the smaller current element, breaking ties in favor of A to keep stability
        float a_val = (i < a_len) ? A_dist[i] : CUDART_INF_F;
        float b_val = (j < b_len) ? B_dist[j] : CUDART_INF_F;

        bool take_a = (a_val <= b_val);
        if (take_a) {
            C_dist[out] = a_val;
            C_idx[out]  = A_idx[i];
            ++i;
        } else {
            C_dist[out] = b_val;
            C_idx[out]  = B_idx[j];
            ++j;
        }
        ++out;
    }
}

// Flush-and-merge: For a single warp, sort its buffer (ascending) and merge with current top-k.
// The result is stored back into top-k (sorted ascending). Buffer count is reset to zero.
__device__ __forceinline__ void warp_flush_merge(
    float *topk_dist, int *topk_idx,
    float *tmp_dist,  int *tmp_idx,
    float *buf_dist,  int *buf_idx,
    int k, int *buf_count_ptr, unsigned mask) {

    int lane = threadIdx.x & (WARP_SIZE - 1);
    // Load buffer count (uniform across warp)
    int buf_n = __shfl_sync(mask, (lane == 0 ? *buf_count_ptr : 0), 0);
    if (buf_n <= 0) {
        return;
    }

    // Sort the buffer using warp-synchronous bitonic sort (pad to next power-of-two with +INF)
    int n_pad = next_pow2(buf_n);
    for (int i = lane; i < n_pad; i += WARP_SIZE) {
        if (i >= buf_n) {
            buf_dist[i] = CUDART_INF_F;
            buf_idx[i]  = -1;
        }
    }
    __syncwarp(mask);
    warp_bitonic_sort_pairs(buf_dist, buf_idx, buf_n, n_pad, mask);
    __syncwarp(mask);

    // Merge the current top-k (size k, sorted ascending, potentially with +INF tail) with buffer (size buf_n, sorted)
    // Keep only the smallest k results.
    int out_len = k;     // We always produce exactly k outputs.
    int a_len   = k;
    int b_len   = buf_n;

    warp_merge_path_pairs(topk_dist, topk_idx, a_len,
                          buf_dist,  buf_idx,  b_len,
                          tmp_dist,  tmp_idx,  out_len, mask);
    __syncwarp(mask);

    // Copy merged result back to top-k arrays
    for (int i = lane; i < k; i += WARP_SIZE) {
        topk_dist[i] = tmp_dist[i];
        topk_idx[i]  = tmp_idx[i];
    }
    __syncwarp(mask);

    // Reset buffer count to zero
    if (lane == 0) {
        *buf_count_ptr = 0;
    }
    __syncwarp(mask);
}

// Main kernel
__global__ void knn_kernel_2d(
    const float2 * __restrict__ query, int query_count,
    const float2 * __restrict__ data,  int data_count,
    std::pair<int, float> * __restrict__ result,
    int k) {

    // Shared memory layout:
    // [ tile (TILE_POINTS float2) ]
    // [ For each warp in block:
    //     topk_dist[k], tmp_dist[k], buf_dist[WARP_BUFFER_CAP],
    //     topk_idx[k],  tmp_idx[k],  buf_idx[WARP_BUFFER_CAP],
    //     buf_count (int)
    // ]
    extern __shared__ unsigned char smem[];
    float2 *tile = reinterpret_cast<float2*>(smem);

    // Compute per-warp base pointers in shared memory
    // Compute sizes (in elements and bytes)
    const int per_warp_floats = 2 * k + WARP_BUFFER_CAP; // topk_dist[k] + tmp_dist[k] + buf_dist[cap]
    const int per_warp_ints   = 2 * k + WARP_BUFFER_CAP; // topk_idx[k]  + tmp_idx[k]  + buf_idx[cap]
    const size_t per_warp_bytes =
        per_warp_floats * sizeof(float) +
        per_warp_ints   * sizeof(int) +
        sizeof(int); // buf_count

    unsigned char *ptr = reinterpret_cast<unsigned char*>(tile + TILE_POINTS);

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane          = threadIdx.x & (WARP_SIZE - 1);
    int global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (global_warp >= query_count) return;

    unsigned warp_mask = __activemask();

    // Compute per-warp base pointer region
    unsigned char *warp_base = ptr + warp_in_block * per_warp_bytes;

    // Dist arrays
    float *topk_dist = reinterpret_cast<float*>(warp_base);
    float *tmp_dist  = topk_dist + k;
    float *buf_dist  = tmp_dist  + k;

    // Index arrays (after float arrays)
    int   *topk_idx  = reinterpret_cast<int*>(buf_dist + WARP_BUFFER_CAP);
    int   *tmp_idx   = topk_idx + k;
    int   *buf_idx   = tmp_idx  + k;

    // Buffer count pointer (after int arrays)
    int   *buf_count_ptr = reinterpret_cast<int*>(buf_idx + WARP_BUFFER_CAP);

    // Initialize top-k with +INF and idx = -1; initialize buffer count to 0
    for (int i = lane; i < k; i += WARP_SIZE) {
        topk_dist[i] = CUDART_INF_F;
        topk_idx[i]  = -1;
    }
    if (lane == 0) {
        *buf_count_ptr = 0;
    }
    __syncwarp(warp_mask);

    // Load query point for this warp's query into registers and broadcast
    float2 q;
    if (lane == 0) {
        q = query[global_warp];
    }
    q.x = __shfl_sync(warp_mask, q.x, 0);
    q.y = __shfl_sync(warp_mask, q.y, 0);

    // Current threshold tau = worst (largest) distance in top-k (last element since top-k is sorted ascending).
    float tau = CUDART_INF_F;

    // Process data in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        int tile_size = min(TILE_POINTS, data_count - tile_start);

        // Load tile into shared memory cooperatively by the whole block
        for (int t = threadIdx.x; t < tile_size; t += blockDim.x) {
            tile[t] = data[tile_start + t];
        }
        __syncthreads(); // synchronize entire block before using 'tile'

        // Process tile in "rounds": each lane computes distances for ROUND_BATCH points per round
        for (int base = 0; base < tile_size; base += WARP_SIZE * ROUND_BATCH) {
            // Lane-local candidate buffers for this round (in registers)
            float cand_d[ROUND_BATCH];
            int   cand_i[ROUND_BATCH];
            int   acc = 0;

            // Compute up to ROUND_BATCH candidates per lane
            #pragma unroll
            for (int m = 0; m < ROUND_BATCH; ++m) {
                int rel = base + lane * ROUND_BATCH + m;
                if (rel < tile_size) {
                    float2 p = tile[rel];
                    float d2 = dist2(q, p);
                    // Global index of this data point
                    int gidx = tile_start + rel;
                    // Check against current threshold
                    if (d2 < tau) {
                        cand_d[acc] = d2;
                        cand_i[acc] = gidx;
                        ++acc;
                    }
                }
            }

            // Compute warp-wide prefix and total of accepted candidates
            int total = 0;
            int excl  = warp_excl_prefix_sum(acc, total, warp_mask);

            // If buffer would overflow, flush first (uniform decision by lane 0)
            int need_flush = 0;
            if (lane == 0) {
                int cur = *buf_count_ptr;
                if (cur + total > WARP_BUFFER_CAP) {
                    need_flush = 1;
                }
            }
            need_flush = __shfl_sync(warp_mask, need_flush, 0);
            if (need_flush) {
                // Flush and merge buffer into top-k, update tau
                warp_flush_merge(topk_dist, topk_idx, tmp_dist, tmp_idx, buf_dist, buf_idx, k, buf_count_ptr, warp_mask);
                __syncwarp(warp_mask);
                // Update tau from last element of top-k (sorted ascending)
                tau = topk_dist[k - 1];
            }

            // Append this round's accepted candidates into the warp buffer contiguously
            int base_off = 0;
            if (lane == 0) {
                base_off = *buf_count_ptr;
            }
            base_off = __shfl_sync(warp_mask, base_off, 0);

            // Each lane writes its accepted candidates at positions [base_off + excl ... base_off + excl + acc)
            for (int j = 0; j < acc; ++j) {
                int pos = base_off + excl + j;
                // pos is guaranteed < WARP_BUFFER_CAP due to prior flush if needed
                buf_dist[pos] = cand_d[j];
                buf_idx[pos]  = cand_i[j];
            }
            __syncwarp(warp_mask);

            // Lane 0 updates buffer count
            if (lane == 0) {
                *buf_count_ptr = base_off + total;
            }
            __syncwarp(warp_mask);
        }

        __syncthreads(); // ensure tile is no longer used before it may be overwritten
    }

    // Final flush for remaining candidates in the buffer
    int rem = __shfl_sync(warp_mask, (lane == 0 ? *buf_count_ptr : 0), 0);
    if (rem > 0) {
        warp_flush_merge(topk_dist, topk_idx, tmp_dist, tmp_idx, buf_dist, buf_idx, k, buf_count_ptr, warp_mask);
        __syncwarp(warp_mask);
    }

    // Write out results for this query: top-k are sorted ascending
    const int out_base = global_warp * k;
    for (int j = lane; j < k; j += WARP_SIZE) {
        // result[out_base + j] = { index, distance }
        std::pair<int, float> out_pair;
        out_pair.first  = topk_idx[j];
        out_pair.second = topk_dist[j];
        result[out_base + j] = out_pair;
    }
}

// Host interface
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k) {
    // Choose launch configuration
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE, 1, 1);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    // Compute dynamic shared memory size
    const size_t tile_bytes = static_cast<size_t>(TILE_POINTS) * sizeof(float2);
    const size_t per_warp_floats = static_cast<size_t>(2 * k + WARP_BUFFER_CAP); // float arrays
    const size_t per_warp_ints   = static_cast<size_t>(2 * k + WARP_BUFFER_CAP); // int arrays
    const size_t per_warp_bytes  = per_warp_floats * sizeof(float) +
                                   per_warp_ints   * sizeof(int) +
                                   sizeof(int); // buffer count
    const size_t smem_bytes = tile_bytes + static_cast<size_t>(WARPS_PER_BLOCK) * per_warp_bytes;

    // Set kernel attribute to allow large dynamic shared memory if needed (CC >= 7.0)
    cudaFuncSetAttribute(knn_kernel_2d,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));

    // Launch kernel
    knn_kernel_2d<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k);
}