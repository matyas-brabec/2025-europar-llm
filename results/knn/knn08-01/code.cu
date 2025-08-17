#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one warp (32 threads) to process one query point.
// Each warp maintains its intermediate k-NN result in registers, distributed
// so that each thread stores k/32 consecutive elements. Candidates are buffered
// in shared memory per warp and merged into the intermediate result using the
// bitonic sort-based procedure described in the prompt.
//
// The data points are processed in batches (tiles) loaded into shared memory
// by the entire block. Each warp independently computes distances from the
// cached data points to its own query point, filters them by the current
// max_distance (distance of the k-th neighbor), and uses warp ballot to compact
// and insert qualifying candidates into its per-warp candidate buffer. When the
// buffer fills (or after the last batch), the warp merges the buffer with its
// intermediate result using the specified bitonic sort and merge operations.
//
// The code assumes:
// - k is a power of two, 32 <= k <= 1024.
// - data_count >= k.
// - query_count and data_count are sufficiently large to benefit from GPU parallelism.
// - Input arrays are device-allocated via cudaMalloc.
// - No additional device memory is allocated; only shared memory is used.
//
// Hyper-parameters chosen:
// - WARPS_PER_BLOCK = 8 (i.e., 256 threads/block), balancing shared memory usage and occupancy.
// - TILE_SIZE = 4096 points per batch cached in shared memory.
//   With k=1024, this uses per-block shared memory approximately:
//     8 warps * (2 buffers * k * (int+float)) + TILE_SIZE * sizeof(float2)
//     = 8 * (2 * 1024 * 8) + 4096 * 8 = 131072 + 32768 = 163840 bytes,
//   which fits in A100/H100 (after enabling large dynamic shared memory).

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 800
#endif

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 4096
#endif

// Ensure we are on 32-thread warps.
static_assert(WARPS_PER_BLOCK * 32 <= 1024, "Too many threads per block");
static_assert((TILE_SIZE & (TILE_SIZE - 1)) == 0 || TILE_SIZE > 0, "TILE_SIZE must be positive");

#define FULL_MASK 0xffffffffu

// Fast INFINITY constant without extra includes.
__device__ __forceinline__ float float_inf() { return __int_as_float(0x7f800000); }

// Warp-wide bitonic sort for k elements distributed across the warp.
// Each thread holds 'seg' consecutive elements in registers: dist[0..seg-1], idx[0..seg-1].
// The global index of the m-th element in a thread with lane 'lane' is i = lane*seg + m.
//
// This function implements the standard bitonic sort network (ascending order), using:
// - Intra-thread register swaps when the XOR-stride J is less than 'seg'.
// - Cross-thread swaps via warp shuffles when J >= 'seg'.
//   In the cross-thread case, each thread updates its own element as either min or max
//   depending on the sort direction for that position.
//
// Parameters:
// - dist: per-thread distances (register array, size up to 32).
// - idx: per-thread indices (register array, size up to 32).
// - seg: number of elements per thread (k / 32).
// - k: total number of elements across the warp (power of two).
__device__ __forceinline__ void warp_bitonic_sort_registers(float dist[32], int idx[32], int seg, int k) {
    const unsigned mask = FULL_MASK;
    int lane = threadIdx.x & 31;

    // Outer loop: size of subsequences (Ksize) increases as powers of two.
    for (int Ksize = 2; Ksize <= k; Ksize <<= 1) {
        // Inner loop: swap distance J decreases by halves.
        for (int J = Ksize >> 1; J > 0; J >>= 1) {
            if (J >= seg) {
                // Cross-thread compare-exchange with partner lane = lane ^ (J / seg)
                int delta = J / seg;
                // For each local element index, exchange with partner's same local index
                #pragma unroll
                for (int m = 0; m < 32; ++m) {
                    if (m >= seg) break;
                    int gi = lane * seg + m;
                    // Determine sorting direction (ascending if (gi & Ksize) == 0)
                    bool up = ((gi & Ksize) == 0);

                    float a = dist[m];
                    int ai = idx[m];
                    float b = __shfl_xor_sync(mask, a, delta);
                    int bi = __shfl_xor_sync(mask, ai, delta);

                    // Select min for up, max for down
                    if (up) {
                        if (b < a) { dist[m] = b; idx[m] = bi; }
                    } else {
                        if (b > a) { dist[m] = b; idx[m] = bi; }
                    }
                }
            } else {
                // Intra-thread compare-exchange within register array.
                int j_intra = J;
                // Process each pair once: only when partner_m > m.
                #pragma unroll
                for (int m = 0; m < 32; ++m) {
                    if (m >= seg) break;
                    int p = m ^ j_intra;
                    if (p > m && p < seg) {
                        int gi = lane * seg + m;
                        bool up = ((gi & Ksize) == 0);

                        float a = dist[m], b = dist[p];
                        int ai = idx[m], bi = idx[p];
                        // Swap if (a > b) == up
                        bool do_swap = ((a > b) == up);
                        if (do_swap) {
                            dist[m] = b; dist[p] = a;
                            idx[m] = bi; idx[p] = ai;
                        }
                    }
                }
            }
            // No need for __syncwarp() here; shuffles are warp-synchronous by design,
            // and intra-thread operations affect only local registers.
        }
    }
}

// Merge the per-warp candidate buffer in shared memory with the intermediate
// sorted result stored in registers, following the specified algorithm:
//
// 0. Invariant: register-held result is sorted ascending.
// 1. Swap contents so that the candidate buffer is moved into registers:
//    - Copy registers -> shared result buffer.
//    - Copy shared candidate buffer -> registers (after padding to k if needed).
// 2. Sort the register-held buffer ascending via warp_bitonic_sort_registers.
// 3. Merge: for each position i, set regs[i] = min( regs[i], shared_res[k-1-i] ).
//    This yields a bitonic sequence of length k in registers.
// 4. Sort the registers again ascending to restore the invariant.
// 5. Reset the candidate count to zero.
// 6. Return the updated max_distance (= distance of the k-th neighbor).
//
// Parameters:
// - warp_local_id: warp index within the block [0..WARPS_PER_BLOCK-1].
// - lane: lane index within the warp [0..31].
// - k: number of neighbors.
// - seg: elements per thread = k / 32.
// - smem_cand_idx, smem_cand_dist: per-warp candidate buffer arrays of size WARPS_PER_BLOCK * k.
// - smem_res_idx, smem_res_dist: per-warp scratch arrays of size WARPS_PER_BLOCK * k.
// - smem_cand_count: per-warp candidate counts.
// - regs_dist, regs_idx: per-thread register arrays (size seg).
__device__ __forceinline__ float warp_flush_and_merge(
    int warp_local_id, int lane, int k, int seg,
    int* __restrict__ smem_cand_idx,
    float* __restrict__ smem_cand_dist,
    int* __restrict__ smem_res_idx,
    float* __restrict__ smem_res_dist,
    int* __restrict__ smem_cand_count,
    float regs_dist[32],
    int regs_idx[32]
) {
    const unsigned mask = FULL_MASK;
    const float INF = float_inf();
    int warp_base = warp_local_id * k;

    // Read current candidate count and broadcast
    int count = 0;
    if (lane == 0) count = smem_cand_count[warp_local_id];
    count = __shfl_sync(mask, count, 0);

    // Pad candidate buffer with INF to reach exactly k elements
    if (count < k) {
        int pad = k - count;
        for (int t = lane; t < pad; t += 32) {
            int pos = warp_base + count + t;
            smem_cand_idx[pos] = -1;
            smem_cand_dist[pos] = INF;
        }
    }
    __syncwarp();

    // 1. Swap contents: move register-held result into shared result buffer,
    //    and move candidate buffer into registers.
    #pragma unroll
    for (int m = 0; m < 32; ++m) {
        if (m >= seg) break;
        int gi = lane * seg + m;
        // Store current intermediate result to shared 'res'
        smem_res_idx[warp_base + gi] = regs_idx[m];
        smem_res_dist[warp_base + gi] = regs_dist[m];
    }
    __syncwarp();
    #pragma unroll
    for (int m = 0; m < 32; ++m) {
        if (m >= seg) break;
        int gi = lane * seg + m;
        // Load candidate buffer to registers
        regs_idx[m] = smem_cand_idx[warp_base + gi];
        regs_dist[m] = smem_cand_dist[warp_base + gi];
    }
    __syncwarp();

    // 2. Sort the candidate buffer now in registers
    warp_bitonic_sort_registers(regs_dist, regs_idx, seg, k);
    __syncwarp();

    // 3. Merge: regs[i] = min( regs[i], res[k-1-i] ) to form bitonic sequence
    #pragma unroll
    for (int m = 0; m < 32; ++m) {
        if (m >= seg) break;
        int gi = lane * seg + m;
        int jr = k - 1 - gi; // reverse index
        float bdist = smem_res_dist[warp_base + jr];
        int bidx = smem_res_idx[warp_base + jr];
        if (bdist < regs_dist[m]) {
            regs_dist[m] = bdist;
            regs_idx[m] = bidx;
        }
    }
    __syncwarp();

    // 4. Sort the merged bitonic sequence ascending
    warp_bitonic_sort_registers(regs_dist, regs_idx, seg, k);
    __syncwarp();

    // 5. Reset candidate count
    if (lane == 0) smem_cand_count[warp_local_id] = 0;

    // 6. Update max_distance (k-th neighbor = last element)
    float kth = 0.0f;
    if (lane == 31) kth = regs_dist[seg - 1];
    kth = __shfl_sync(mask, kth, 31);
    return kth;
}

__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data, int data_count,
    std::pair<int, float>* __restrict__ result,
    int k
) {
    // Dynamic shared memory layout:
    // [0]   float2 tile[TILE_SIZE]
    // [1]   int    cand_idx[WARPS_PER_BLOCK * k]
    // [2]   float  cand_dist[WARPS_PER_BLOCK * k]
    // [3]   int    res_idx[WARPS_PER_BLOCK * k]
    // [4]   float  res_dist[WARPS_PER_BLOCK * k]
    // [5]   int    cand_count[WARPS_PER_BLOCK]
    extern __shared__ unsigned char smem_raw[];
    unsigned char* ptr = smem_raw;

    // Align pointer to 8 bytes
    auto align_up = [](uintptr_t p, size_t a) { return (p + (a - 1)) & ~(a - 1); };
    ptr = reinterpret_cast<unsigned char*>(align_up(reinterpret_cast<uintptr_t>(ptr), 8));

    float2* tile = reinterpret_cast<float2*>(ptr);
    ptr += sizeof(float2) * TILE_SIZE;

    ptr = reinterpret_cast<unsigned char*>(align_up(reinterpret_cast<uintptr_t>(ptr), 8));
    int* smem_cand_idx = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * WARPS_PER_BLOCK * k;

    float* smem_cand_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * WARPS_PER_BLOCK * k;

    int* smem_res_idx = reinterpret_cast<int*>(ptr);
    ptr += sizeof(int) * WARPS_PER_BLOCK * k;

    float* smem_res_dist = reinterpret_cast<float*>(ptr);
    ptr += sizeof(float) * WARPS_PER_BLOCK * k;

    ptr = reinterpret_cast<unsigned char*>(align_up(reinterpret_cast<uintptr_t>(ptr), alignof(int)));
    int* smem_cand_count = reinterpret_cast<int*>(ptr);
    // (Optionally) ptr += sizeof(int) * WARPS_PER_BLOCK; // not needed further

    // Warp identification
    const int lane = threadIdx.x & 31;
    const int warp_local_id = threadIdx.x >> 5;                 // warp index within the block
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global_id = blockIdx.x * warps_per_block + warp_local_id;

    const bool active = (warp_global_id < query_count);

    // Elements per thread
    const int seg = k >> 5; // k / 32
    const float INF = float_inf();

    // Initialize per-warp candidate count to 0
    if (lane == 0) smem_cand_count[warp_local_id] = 0;

    // Load query point for this warp and broadcast to lanes
    float qx = 0.0f, qy = 0.0f;
    if (active && lane == 0) {
        float2 q = query[warp_global_id];
        qx = q.x; qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Initialize intermediate result in registers: all INF distances, idx = -1.
    float regs_dist[32];
    int regs_idx[32];
    #pragma unroll
    for (int m = 0; m < 32; ++m) {
        if (m >= seg) break;
        regs_dist[m] = INF;
        regs_idx[m] = -1;
    }

    // Initialize max_distance to INF (k-th neighbor distance)
    float max_distance = INF;

    // Process data in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_n = data_count - tile_start;
        if (tile_n > TILE_SIZE) tile_n = TILE_SIZE;

        // Load tile into shared memory by the whole block
        for (int i = threadIdx.x; i < tile_n; i += blockDim.x) {
            tile[i] = data[tile_start + i];
        }
        __syncthreads();

        if (active) {
            // Process this tile in groups of 32 points
            for (int base = 0; base < tile_n; base += 32) {
                int vi = base + lane;
                // Compute squared Euclidean distance for this lane's point
                float dist = INF;
                int gidx = -1;
                if (vi < tile_n) {
                    float2 p = tile[vi];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    // FMA for better perf/precision: dx*dx + dy*dy
                    dist = __fmaf_rn(dx, dx, dy * dy);
                    gidx = tile_start + vi; // global data index
                }

                // Filter by current max_distance
                bool is_candidate = (vi < tile_n) && (dist < max_distance);

                // Warp ballot to find how many candidates in this group
                unsigned mask = __ballot_sync(FULL_MASK, is_candidate);
                int nnz = __popc(mask);

                if (nnz > 0) {
                    // Reserve space; if overflow, flush and try again.
                    while (true) {
                        int old = 0;
                        if (lane == 0) old = smem_cand_count[warp_local_id];
                        old = __shfl_sync(FULL_MASK, old, 0);

                        if (old + nnz <= k) {
                            int base_off = 0;
                            if (lane == 0) {
                                base_off = old;
                                smem_cand_count[warp_local_id] = old + nnz;
                            }
                            base_off = __shfl_sync(FULL_MASK, base_off, 0);

                            // Per-lane offset within the nnz group
                            int prefix = __popc(mask & ((1u << lane) - 1));
                            if (is_candidate) {
                                int pos = warp_local_id * k + base_off + prefix;
                                smem_cand_idx[pos] = gidx;
                                smem_cand_dist[pos] = dist;
                            }
                            break; // stored successfully
                        } else {
                            // Flush buffer and merge with intermediate result
                            max_distance = warp_flush_and_merge(
                                warp_local_id, lane, k, seg,
                                smem_cand_idx, smem_cand_dist,
                                smem_res_idx, smem_res_dist,
                                smem_cand_count,
                                regs_dist, regs_idx
                            );
                            // Continue to reserve space after flush
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // After last tile, flush remaining candidates (if any)
    if (active) {
        int rem = 0;
        if (lane == 0) rem = smem_cand_count[warp_local_id];
        rem = __shfl_sync(FULL_MASK, rem, 0);
        if (rem > 0) {
            max_distance = warp_flush_and_merge(
                warp_local_id, lane, k, seg,
                smem_cand_idx, smem_cand_dist,
                smem_res_idx, smem_res_dist,
                smem_cand_count,
                regs_dist, regs_idx
            );
        }

        // Write out the final sorted k-NN for this query
        int out_base = warp_global_id * k;
        #pragma unroll
        for (int m = 0; m < 32; ++m) {
            if (m >= seg) break;
            int gi = (threadIdx.x & 31) * seg + m;
            // Store as std::pair<int,float>
            result[out_base + gi].first = regs_idx[m];
            result[out_base + gi].second = regs_dist[m];
        }
    }
}

// Host wrapper: configures and launches the kernel.
// Chooses WARPS_PER_BLOCK = 8 and TILE_SIZE = 4096 as described.
// Sets the kernel's dynamic shared memory attribute to the device's maximum
// opt-in shared memory to allow large shared memory usage (A100/H100).
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * 32;
    const int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Compute dynamic shared memory size needed per block
    size_t smem_bytes = 0;
    smem_bytes += sizeof(float2) * TILE_SIZE;                           // tile
    smem_bytes += sizeof(int)   * warps_per_block * k;                  // cand_idx
    smem_bytes += sizeof(float) * warps_per_block * k;                  // cand_dist
    smem_bytes += sizeof(int)   * warps_per_block * k;                  // res_idx
    smem_bytes += sizeof(float) * warps_per_block * k;                  // res_dist
    smem_bytes += sizeof(int)   * warps_per_block;                      // cand_count

    // Enable large dynamic shared memory if needed
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int maxOptin = prop.sharedMemPerBlockOptin; // e.g., 163840 (A100), 227328 (H100)
    if ((int)smem_bytes > prop.sharedMemPerBlock) {
        int attr = (smem_bytes <= (size_t)maxOptin) ? (int)smem_bytes : maxOptin;
        cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, attr);
    }

    // Launch kernel
    knn_kernel<<<blocks, threads_per_block, smem_bytes>>>(query, query_count, data, data_count, result, k);
}