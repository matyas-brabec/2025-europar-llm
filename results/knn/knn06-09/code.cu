#include <cuda_runtime.h>
#include <cuda.h>
#include <utility>

// This implementation computes k-NN for 2D points using a single warp (32 threads) per query.
// It tiles the data points into shared memory, computes squared L2 distances, and maintains
// per-warp shared-memory buffers:
//   - An intermediate result of the current top-k neighbors (indices and distances), kept sorted.
//   - A candidate buffer (indices and distances) with capacity k, and a shared counter updated via atomicAdd.
// Whenever the candidate buffer becomes full, the warp sorts the candidates and merges them with
// the intermediate result (taking the k smallest). After processing all tiles, any remaining
// candidates are merged once more. The final top-k pairs are written to the output.
//
// Design notes:
// - Each warp handles exactly one query.
// - The candidate buffer and the intermediate top-k arrays are private to each warp and reside in shared memory.
// - A per-warp shared integer tracks the current number of candidates in the buffer; atomicAdd is used
//   to allocate slots in the buffer for new candidates.
// - When the buffer is full, all 32 threads in the warp cooperatively sort and merge.
// - Distances are squared Euclidean distances; no sqrt is performed.
// - k is a power of two between 32 and 1024 (inclusive).
// - No additional device memory is allocated beyond dynamic shared memory.
//
// The host function run_knn sets up the kernel launch, including choosing the number of warps per block,
// computing the dynamic shared memory size, opting into large shared memory if supported, and choosing
// a suitable tile size.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Define a plain-old-data pair type that matches std::pair<int, float> layout for device writes.
struct PairIF {
    int   first;
    float second;
};

// Round up 'x' to the nearest multiple of 'align' (align must be power-of-two).
static inline size_t round_up(size_t x, size_t align) {
    return (x + (align - 1)) & ~(align - 1);
}

// Return the next power-of-two >= x. Assumes x > 0 and x <= 1024 in this context.
__device__ __forceinline__ int next_pow2(int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

// In-warp bitonic sort on shared arrays (dist[], idx[]) of given length 'len' (power-of-two).
// The arrays are accessed cooperatively by the warp's 32 threads. Sorting is ascending by distance.
// Each pair (i, j=i^stride) is processed exactly once by the condition (j > i) to avoid races.
__device__ __forceinline__ void warp_bitonic_sort(float* dist, int* idx, int len, unsigned mask) {
    for (int size = 2; size <= len; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x % WARP_SIZE; i < len; i += WARP_SIZE) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = ((i & size) == 0);
                    float di = dist[i];
                    float dj = dist[j];
                    int   ii = idx[i];
                    int   ij = idx[j];
                    bool do_swap = (up && di > dj) || (!up && di < dj);
                    if (do_swap) {
                        dist[i] = dj;
                        dist[j] = di;
                        idx[i]  = ij;
                        idx[j]  = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// Merge-path search to find the partition index 'a_idx' for a given diagonal 'diag' in the merge of A and B.
// A and B are sorted ascending arrays of lengths a_len and b_len, respectively.
// Returns a_idx such that a_idx + b_idx = diag with the standard merge invariants.
__device__ __forceinline__ int merge_path_search(const float* A, int a_len, const float* B, int b_len, int diag) {
    int a_min = max(0, diag - b_len);
    int a_max = min(diag, a_len);
    while (a_min < a_max) {
        int a_idx = (a_min + a_max) >> 1;
        int b_idx = diag - a_idx;
        float A_val = (a_idx < a_len) ? A[a_idx] : CUDART_INF_F;
        float B_prev = (b_idx > 0) ? B[b_idx - 1] : -CUDART_INF_F;
        if (A_val < B_prev) {
            a_min = a_idx + 1;
        } else {
            a_max = a_idx;
        }
    }
    return a_min;
}

// Merge the first 'k_out' smallest elements from two sorted arrays:
//   A (a_len), B (b_len) into out (out_dist/out_idx).
// Work is partitioned across the warp using merge-path. The arrays must be ascending.
// This function does not assume a_len == b_len; it supports merging any lengths.
__device__ __forceinline__ void warp_merge_k_smallest(
    const float* A_dist, const int* A_idx, int a_len,
    const float* B_dist, const int* B_idx, int b_len,
    float* out_dist, int* out_idx, int k_out, unsigned mask)
{
    // Partition output range [0, k_out) into per-lane segments.
    int lane = threadIdx.x % WARP_SIZE;
    int seg  = (k_out + WARP_SIZE - 1) / WARP_SIZE; // ceil
    int start = lane * seg;
    if (start >= k_out) {
        __syncwarp(mask);
        return;
    }
    int end = min(start + seg, k_out);

    // Find starting positions in A and B for 'start' and 'end'
    int a_start = merge_path_search(A_dist, a_len, B_dist, b_len, start);
    int b_start = start - a_start;

    int a_end = merge_path_search(A_dist, a_len, B_dist, b_len, end);
    int b_end = end - a_end;

    int ai = a_start;
    int bi = b_start;
    for (int out = start; out < end; ++out) {
        bool takeA = (bi >= b_end) || (ai < a_end && A_dist[ai] <= B_dist[bi]);
        if (takeA) {
            out_dist[out] = A_dist[ai];
            out_idx[out]  = A_idx[ai];
            ++ai;
        } else {
            out_dist[out] = B_dist[bi];
            out_idx[out]  = B_idx[bi];
            ++bi;
        }
    }
    __syncwarp(mask);
}

// Flush and merge the candidate buffer with the intermediate top-k for a single warp.
// - top_dist/top_idx: the warp's current top-k (sorted ascending).
// - cand_dist/cand_idx: the warp's candidate buffer (first n_cand entries are valid).
// - n_cand: number of valid candidates (0 < n_cand <= k).
// Returns the updated max_distance (k-th smallest), broadcast to all lanes.
__device__ __forceinline__ float warp_flush_and_merge(
    float* top_dist, int* top_idx, int k,
    float* cand_dist, int* cand_idx, int n_cand, unsigned mask)
{
    if (n_cand <= 0) {
        float md = 0.0f;
        if ((threadIdx.x % WARP_SIZE) == 0) md = top_dist[k - 1];
        md = __shfl_sync(mask, md, 0);
        return md;
    }

    // Pad candidates to next power of two with +INF so bitonic sort can be applied.
    int len_pow2 = next_pow2(n_cand);
    for (int i = threadIdx.x % WARP_SIZE; i < (len_pow2 - n_cand); i += WARP_SIZE) {
        int pos = n_cand + i;
        if (pos < len_pow2) {
            cand_dist[pos] = CUDART_INF_F;
            cand_idx[pos]  = -1;
        }
    }
    __syncwarp(mask);

    // Sort the first len_pow2 entries (real + padded) ascending by distance.
    warp_bitonic_sort(cand_dist, cand_idx, len_pow2, mask);

    // Merge top-k (A) with n_cand (B) to keep k smallest; write into cand_* as temporary output.
    warp_merge_k_smallest(top_dist, top_idx, k,
                          cand_dist, cand_idx, n_cand,
                          cand_dist, cand_idx, k, mask);

    // Copy merged result back into top_*.
    for (int i = threadIdx.x % WARP_SIZE; i < k; i += WARP_SIZE) {
        top_dist[i] = cand_dist[i];
        top_idx[i]  = cand_idx[i];
    }
    __syncwarp(mask);

    // Update and broadcast new max_distance (k-th smallest).
    float new_max = 0.0f;
    if ((threadIdx.x % WARP_SIZE) == 0) new_max = top_dist[k - 1];
    new_max = __shfl_sync(mask, new_max, 0);
    return new_max;
}

// Kernel templated on the number of warps per block.
template<int WARPS_PER_BLOCK>
__global__ void knn_kernel_2d(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    PairIF* __restrict__ result,
    int k, int tile_elems)
{
    // Declare dynamic shared memory and compute layout:
    // [0 .. tile_elems*sizeof(float2))                     -> data tile
    // [aligned .. aligned + WARPS_PER_BLOCK*sizeof(int))   -> candidate counters (per warp)
    // [.. + WPB*k*sizeof(float)]                           -> top_dist for all warps
    // [.. + WPB*k*sizeof(int)]                             -> top_idx for all warps
    // [.. + WPB*k*sizeof(float)]                           -> cand_dist for all warps
    // [.. + WPB*k*sizeof(int)]                             -> cand_idx for all warps
    extern __shared__ unsigned char smem[];
    unsigned char* base = smem;

    float2* tile = reinterpret_cast<float2*>(base);
    size_t off = tile_elems * sizeof(float2);
    off = round_up(off, 16);

    int* cand_counts = reinterpret_cast<int*>(base + off);
    off += WARPS_PER_BLOCK * sizeof(int);
    off = round_up(off, 16);

    float* top_dist_all = reinterpret_cast<float*>(base + off);
    off += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(float);

    int* top_idx_all = reinterpret_cast<int*>(base + off);
    off += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(int);

    float* cand_dist_all = reinterpret_cast<float*>(base + off);
    off += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(float);

    int* cand_idx_all = reinterpret_cast<int*>(base + off);
    // off += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(int); // not needed further

    // Warp info
    int lane        = threadIdx.x & (WARP_SIZE - 1);
    int warp_in_blk = threadIdx.x >> 5;  // threadIdx.x / WARP_SIZE
    int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;

    bool warp_active = (global_warp < query_count);
    unsigned full_mask = 0xFFFFFFFFu;

    // Pointers to this warp's private shared-memory regions.
    float* top_dist = top_dist_all + size_t(warp_in_blk) * size_t(k);
    int*   top_idx  = top_idx_all  + size_t(warp_in_blk) * size_t(k);
    float* cand_dist= cand_dist_all+ size_t(warp_in_blk) * size_t(k);
    int*   cand_idx = cand_idx_all + size_t(warp_in_blk) * size_t(k);
    int*   cand_cnt_ptr = cand_counts + warp_in_blk;

    // Initialize per-warp state.
    if (lane == 0) {
        *cand_cnt_ptr = 0;
    }
    for (int i = lane; i < k; i += WARP_SIZE) {
        top_dist[i] = CUDART_INF_F;
        top_idx[i]  = -1;
    }
    __syncwarp(full_mask);

    // Local max_distance (k-th smallest) starts as +Inf. Updated after each merge.
    float max_d = CUDART_INF_F;

    // Main loop: process data in tiles.
    for (int tile_base = 0; tile_base < data_count; tile_base += tile_elems) {
        int tile_size = data_count - tile_base;
        if (tile_size > tile_elems) tile_size = tile_elems;

        // Load this tile into shared memory cooperatively by the whole block.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile[i] = data[tile_base + i];
        }
        __syncthreads();

        // Broadcast this warp's query point.
        float2 q = make_float2(0.f, 0.f);
        if (warp_active) {
            if (lane == 0) q = query[global_warp];
            q.x = __shfl_sync(full_mask, q.x, 0);
            q.y = __shfl_sync(full_mask, q.y, 0);
        }

        // Each warp processes the tile.
        if (warp_active) {
            for (int t = lane; t < tile_size; t += WARP_SIZE) {
                float2 p = tile[t];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                float d2 = dx * dx + dy * dy; // squared Euclidean distance

                if (d2 < max_d) {
                    bool pending = true;
                    int  gidx    = tile_base + t;

                    // Attempt to push this candidate; if buffer fills, flush; if overflow, flush and retry.
                    while (pending) {
                        int old = atomicAdd(cand_cnt_ptr, 1);
                        bool accept   = (old < k);
                        bool overflow = (old >= k);

                        if (accept) {
                            cand_dist[old] = d2;
                            cand_idx[old]  = gidx;
                        }

                        unsigned over_mask = __ballot_sync(full_mask, pending && overflow);
                        if (over_mask) {
                            // Overshoot: some lanes overflowed; bring count back to k and flush.
                            int overshoot = __popc(over_mask);
                            int leader    = __ffs(over_mask) - 1;
                            if (lane == leader) {
                                atomicSub(cand_cnt_ptr, overshoot);
                            }
                            __syncwarp(full_mask);

                            // Flush buffer (exactly k candidates present after subtraction).
                            float new_max = warp_flush_and_merge(top_dist, top_idx, k,
                                                                 cand_dist, cand_idx, k, full_mask);
                            max_d = new_max;

                            if (lane == 0) *cand_cnt_ptr = 0;
                            __syncwarp(full_mask);

                            // Re-check with tightened max_d whether this candidate should still be inserted.
                            pending = (d2 < max_d);
                            continue;
                        }

                        // Check if we just filled the last slot exactly (no overflow).
                        unsigned fill_mask = __ballot_sync(full_mask, pending && accept && (old == (k - 1)));
                        if (fill_mask) {
                            __syncwarp(full_mask);

                            // Flush buffer (k candidates) and reset.
                            float new_max = warp_flush_and_merge(top_dist, top_idx, k,
                                                                 cand_dist, cand_idx, k, full_mask);
                            max_d = new_max;

                            if (lane == 0) *cand_cnt_ptr = 0;
                            __syncwarp(full_mask);

                            // This candidate already accepted; nothing to retry.
                            pending = false;
                            continue;
                        }

                        // Accepted and buffer not full: done.
                        if (accept) {
                            pending = false;
                        } else {
                            // Should not reach here; overflow handled above.
                            pending = false;
                        }
                    } // while pending
                } // if (d2 < max_d)
            } // for (t)
        } // if warp_active

        __syncthreads(); // Ensure all warps done with this tile before reloading shared memory.
    } // for tiles

    // Final flush of remaining candidates (if any).
    if (warp_active) {
        int cc = 0;
        if (lane == 0) cc = *cand_cnt_ptr;
        cc = __shfl_sync(full_mask, cc, 0);

        if (cc > 0) {
            float new_max = warp_flush_and_merge(top_dist, top_idx, k,
                                                 cand_dist, cand_idx, cc, full_mask);
            (void)new_max; // max_d updated if needed; not used further.
            if (lane == 0) *cand_cnt_ptr = 0;
            __syncwarp(full_mask);
        }

        // Write final top-k for this query.
        for (int i = lane; i < k; i += WARP_SIZE) {
            size_t out_pos = size_t(global_warp) * size_t(k) + size_t(i);
            result[out_pos].first  = top_idx[i];
            result[out_pos].second = top_dist[i];
        }
    }
}

// Host-side launcher. Chooses kernel configuration and dynamic shared memory size, and launches the kernel.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Choose warps per block based on k to keep shared memory usage within limits.
    // For larger k, use fewer warps per block to accommodate per-warp buffers.
    int warps_per_block = (k >= 512) ? 4 : 8;
    int threads_per_block = warps_per_block * WARP_SIZE;

    // Determine the maximum shared memory per block (opt-in if available).
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int smem_optin = 0;
    cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    int smem_limit = (smem_optin > 0) ? smem_optin : prop.sharedMemPerBlock;

    // Shared memory usage per block:
    // - Candidate counters: warps_per_block * sizeof(int)
    // - Per-warp arrays: 2 * k * (sizeof(float) + sizeof(int)) bytes per warp = 16 * k bytes per warp
    size_t per_warp_bytes = size_t(16) * size_t(k);
    size_t counters_bytes = size_t(warps_per_block) * sizeof(int);

    // Decide tile size (number of float2 points per tile) to fit into shared memory.
    // Total dynamic shared memory = tile_bytes + counters + (per_warp_bytes * warps_per_block).
    size_t fixed_bytes = round_up(counters_bytes, 16) + per_warp_bytes * size_t(warps_per_block);
    // Leave a small safety margin.
    size_t safety_margin = 0;
    if (smem_limit > 0) {
        safety_margin = 0; // We rely on exact computation with alignment.
    }

    size_t max_tile_bytes = (smem_limit > fixed_bytes + safety_margin)
                          ? (smem_limit - fixed_bytes - safety_margin)
                          : size_t(0);
    // Ensure tile has at least WARP_SIZE elements; otherwise, reduce warps per block.
    int tile_elems = 0;
    if (max_tile_bytes >= sizeof(float2)) {
        tile_elems = int(max_tile_bytes / sizeof(float2));
    }
    if (tile_elems < WARP_SIZE) {
        // Fallback: reduce warps_per_block to 4 if possible to get more shared memory headroom.
        if (warps_per_block == 8) {
            warps_per_block = 4;
            threads_per_block = warps_per_block * WARP_SIZE;
            per_warp_bytes = size_t(16) * size_t(k);
            counters_bytes = size_t(warps_per_block) * sizeof(int);
            fixed_bytes = round_up(counters_bytes, 16) + per_warp_bytes * size_t(warps_per_block);
            max_tile_bytes = (smem_limit > fixed_bytes + safety_margin)
                           ? (smem_limit - fixed_bytes - safety_margin)
                           : size_t(0);
            tile_elems = (max_tile_bytes >= sizeof(float2)) ? int(max_tile_bytes / sizeof(float2)) : 0;
        }
        if (tile_elems < WARP_SIZE) {
            // As a last resort, set tile_elems to WARP_SIZE (will fit on A100/H100 with opt-in).
            tile_elems = WARP_SIZE;
        }
    }

    // Use as large a tile as possible but not larger than data_count to avoid wasting load iterations.
    if (tile_elems > data_count) tile_elems = data_count;
    if (tile_elems < 1) tile_elems = 1;

    // Compute dynamic shared memory size (must match kernel's layout).
    size_t tile_bytes = size_t(tile_elems) * sizeof(float2);
    size_t dyn_smem_bytes = round_up(tile_bytes, 16)
                          + round_up(counters_bytes, 16)
                          + size_t(warps_per_block) * size_t(k) * sizeof(float)  // top_dist
                          + size_t(warps_per_block) * size_t(k) * sizeof(int)    // top_idx
                          + size_t(warps_per_block) * size_t(k) * sizeof(float)  // cand_dist
                          + size_t(warps_per_block) * size_t(k) * sizeof(int);   // cand_idx

    // Opt-in to large shared memory if supported.
    if (smem_optin > 0) {
        // We need to set the attribute per kernel instantiation.
        if (warps_per_block == 8) {
            cudaFuncSetAttribute(knn_kernel_2d<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes);
        } else {
            cudaFuncSetAttribute(knn_kernel_2d<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem_bytes);
        }
    }

    // Grid configuration: one warp per query.
    int total_warps = (query_count + 0); // equals query_count
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    PairIF* out_pairs = reinterpret_cast<PairIF*>(const_cast<std::pair<int, float>*>(result));

    if (warps_per_block == 8) {
        knn_kernel_2d<8><<<blocks, threads_per_block, dyn_smem_bytes>>>(
            query, query_count, data, data_count, out_pairs, k, tile_elems);
    } else {
        knn_kernel_2d<4><<<blocks, threads_per_block, dyn_smem_bytes>>>(
            query, query_count, data, data_count, out_pairs, k, tile_elems);
    }
    // Optional: synchronize or check errors if desired by the caller.
}