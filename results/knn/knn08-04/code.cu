#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one warp (32 threads) to each query point.
// Each warp maintains an intermediate result (k nearest neighbors) distributed across
// the registers of its threads. Each thread stores k/32 consecutive elements.
// A per-warp candidate buffer of size k is kept in shared memory. Data points are
// processed in batches loaded into shared memory by the entire thread block.
// Candidates with distance < max_distance (k-th neighbor) are compacted into the buffer
// using warp ballot-based packing. When the buffer is full, it is merged into the
// intermediate result using a bitonic sort-based pipeline as described in the prompt.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef MAX_K
#define MAX_K 1024
#endif

#ifndef MAX_SEG
#define MAX_SEG (MAX_K / WARP_SIZE) // Maximum per-thread segment length
#endif

#ifndef TILE_POINTS
#define TILE_POINTS 4096 // Number of data points cached per block per batch (float2 -> 8 bytes -> 32KB)
#endif

// Utility: minimum of two (distance, index) pairs with distance as primary key.
// In case of tie, choose arbitrary (here: the first argument).
__device__ __forceinline__ void pair_min(float& da, int& ia, const float db, const int ib) {
    if (db < da) {
        da = db;
        ia = ib;
    }
}

// Warp-wide bitonic sort of a distributed array of length K = S * 32, where each thread
// holds S consecutive elements in vals[0..S-1] and idxs[0..S-1]. The sort is ascending.
// Cross-thread exchanges use warp shuffles; intra-thread exchanges swap in registers.
//
// The implementation follows the serial bitonic sort structure, splitting each compare-exchange
// step into either cross-thread (stride >= S) or intra-thread (stride < S) operations. For the
// cross-thread case we use the standard "keep min/max" formulation which is safe to apply
// symmetrically on both sides without an explicit "if (l > i)" guard. For the intra-thread
// case we explicitly perform each swap once per pair (partner > self).
__device__ __forceinline__ void warp_bitonic_sort_distributed(float vals[MAX_SEG], int idxs[MAX_SEG], const int S, const int K) {
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // size spans the subsequence size in the bitonic network (2, 4, 8, ..., K)
    for (unsigned size = 2; size <= (unsigned)K; size <<= 1) {
        // stride halves each iteration: size/2, size/4, ..., 1
        for (unsigned stride = size >> 1; stride > 0; stride >>= 1) {

            if (stride >= (unsigned)S) {
                // Cross-thread exchanges at fixed local index s
                const unsigned partner = stride / (unsigned)S; // XOR distance in lanes

                #pragma unroll
                for (int s = 0; s < MAX_SEG; ++s) {
                    if (s >= S) break;

                    const int i = lane * S + s;
                    const bool up = ((i & size) == 0);

                    float self_v = vals[s];
                    int   self_i = idxs[s];

                    float other_v = __shfl_xor_sync(FULL_MASK, self_v, partner);
                    int   other_i = __shfl_xor_sync(FULL_MASK, self_i, partner);

                    const bool take_self = up ? (self_v <= other_v) : (self_v >= other_v);

                    vals[s] = take_self ? self_v : other_v;
                    idxs[s] = take_self ? self_i : other_i;
                }
            } else {
                // Intra-thread exchanges: partner index differs within the local segment
                const int j = (int)stride;

                #pragma unroll
                for (int s = 0; s < MAX_SEG; ++s) {
                    if (s >= S) break;

                    const int partner = s ^ j;
                    if (partner > s) {
                        const int i = (threadIdx.x & (WARP_SIZE - 1)) * S + s;
                        const bool up = ((i & size) == 0);

                        float a = vals[s];  int ia = idxs[s];
                        float b = vals[partner]; int ib = idxs[partner];

                        const bool swap_needed = up ? (a > b) : (a < b);
                        if (swap_needed) {
                            vals[s]       = b;  idxs[s]       = ib;
                            vals[partner] = a;  idxs[partner] = ia;
                        }
                    }
                }
            }
        }
    }
}

// Merge the full candidate buffer (length K in shared memory) with the current intermediate
// result in registers (also length K), using the pipeline:
// 1) Swap content so that the buffer is in registers (load buffer into regs, store regs into buffer).
// 2) Sort the registers ascending via warp_bitonic_sort_distributed.
// 3) Merge registers and buffer into registers by taking per-position minima between
//    reg[i] and buffer[K - 1 - i]; the result is bitonic and contains the best K elements.
// 4) Sort the registers ascending again via warp_bitonic_sort_distributed.
__device__ __forceinline__ void merge_full_buffer_with_intermediate(
    float r_vals[MAX_SEG], int r_idxs[MAX_SEG],
    float* __restrict__ buf_vals, int* __restrict__ buf_idxs,
    const int S, const int K)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Step 1: swap content between registers (intermediate result) and shared memory (buffer)
    #pragma unroll
    for (int s = 0; s < MAX_SEG; ++s) {
        if (s >= S) break;
        const int pos = lane * S + s;
        const float tmp_v = buf_vals[pos];
        const int   tmp_i = buf_idxs[pos];

        buf_vals[pos] = r_vals[s];
        buf_idxs[pos] = r_idxs[s];

        r_vals[s] = tmp_v;
        r_idxs[s] = tmp_i;
    }
    __syncwarp();

    // Step 2: sort the buffer (now in registers) ascending
    warp_bitonic_sort_distributed(r_vals, r_idxs, S, K);
    __syncwarp();

    // Step 3: per-position minima between register value at i and shared buffer at (K - 1 - i)
    #pragma unroll
    for (int s = 0; s < MAX_SEG; ++s) {
        if (s >= S) break;
        const int i = lane * S + s;
        const int j = K - 1 - i;

        const float ov = buf_vals[j];
        const int   oi = buf_idxs[j];

        // r_vals[s] = min(r_vals[s], ov) with index paired
        if (ov < r_vals[s]) {
            r_vals[s] = ov;
            r_idxs[s] = oi;
        }
    }
    __syncwarp();

    // Step 4: sort the merged bitonic sequence ascending
    warp_bitonic_sort_distributed(r_vals, r_idxs, S, K);
    __syncwarp();
}

// CUDA kernel: one warp per query
__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    std::pair<int, float>* __restrict__ result,
    int K)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const bool warp_active = (warp_global < query_count);

    // Dynamic shared memory layout:
    // [ tile (TILE_POINTS * sizeof(float2)) ]
    // [ cand_counts (warps_per_block * sizeof(int)) ]
    // [ cand_dists (warps_per_block * K * sizeof(float)) ]
    // [ cand_idxs  (warps_per_block * K * sizeof(int)) ]
    extern __shared__ unsigned char smem[];
    unsigned char* ptr = smem;

    float2* tile = reinterpret_cast<float2*>(ptr);
    ptr += TILE_POINTS * sizeof(float2);

    int* cand_counts = reinterpret_cast<int*>(ptr);
    ptr += warps_per_block * sizeof(int);

    float* cand_dists_base = reinterpret_cast<float*>(ptr);
    ptr += (size_t)warps_per_block * K * sizeof(float);

    int* cand_idxs_base = reinterpret_cast<int*>(ptr);
    // ptr += (size_t)warps_per_block * K * sizeof(int); // not needed further

    // Pointers for this warp's candidate buffer (shared memory)
    float* warp_cand_dists = cand_dists_base + (size_t)warp_in_block * K;
    int*   warp_cand_idxs  = cand_idxs_base  + (size_t)warp_in_block * K;
    // Candidate count for this warp (shared memory)
    int& cand_count = cand_counts[warp_in_block];

    // S = K / 32 elements per thread; K is a power of two and 32 <= K <= 1024 => 1 <= S <= 32
    const int S = K >> 5;

    // Intermediate result in registers: sorted ascending at all times (invariant).
    float r_vals[MAX_SEG];
    int   r_idxs[MAX_SEG];

    // Initialize intermediate result to +inf (distances) and -1 (indices).
    #pragma unroll
    for (int s = 0; s < MAX_SEG; ++s) {
        if (s < S) {
            r_vals[s] = CUDART_INF_F;
            r_idxs[s] = -1;
        }
    }

    // Each warp loads its query point to registers.
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        const float2 q = query[warp_global];
        qx = q.x; qy = q.y;
    }

    // Initialize candidate count
    if (lane == 0) cand_count = 0;
    __syncthreads();

    // max_distance is the k-th (last) distance in the intermediate result (ascending).
    float max_distance = CUDART_INF_F;

    // Process data in batches cached into shared memory
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        const int tile_count = min(TILE_POINTS, data_count - base);

        // Load tile from global memory cooperatively
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            tile[t] = data[base + t];
        }
        __syncthreads();

        if (warp_active) {
            // Iterate over points in the tile, one per lane in a strided manner
            for (int t = lane; t < tile_count; t += WARP_SIZE) {
                const float2 p = tile[t];
                const float dx = p.x - qx;
                const float dy = p.y - qy;
                const float dist = dx * dx + dy * dy;
                const int   idx  = base + t;

                // Filter by current max_distance
                const bool is_candidate = (dist < max_distance);
                const unsigned FULL_MASK = 0xFFFFFFFFu;
                const unsigned mask = __ballot_sync(FULL_MASK, is_candidate);
                const int cnt = __popc(mask);

                if (cnt > 0) {
                    const int rank = __popc(mask & ((1u << lane) - 1));
                    int room = 0;

                    if (lane == 0) room = K - cand_count;
                    room = __shfl_sync(FULL_MASK, room, 0);

                    if (cnt <= room) {
                        int base_pos = 0;
                        if (lane == 0) { base_pos = cand_count; cand_count += cnt; }
                        base_pos = __shfl_sync(FULL_MASK, base_pos, 0);

                        if (is_candidate) {
                            const int pos = base_pos + rank;
                            warp_cand_dists[pos] = dist;
                            warp_cand_idxs[pos]  = idx;
                        }

                        // If we just filled the buffer exactly, merge it
                        if (cnt == room && room > 0) {
                            __syncwarp();
                            merge_full_buffer_with_intermediate(r_vals, r_idxs, warp_cand_dists, warp_cand_idxs, S, K);
                            if (lane == 0) cand_count = 0;
                            // Update max_distance from lane 31's last local element
                            const float kth = r_vals[S - 1];
                            max_distance = __shfl_sync(FULL_MASK, kth, WARP_SIZE - 1);
                        }
                    } else {
                        // cnt > room: fill remaining room (if any), then merge, then place leftovers
                        if (room > 0) {
                            if (is_candidate && rank < room) {
                                const int pos = (K - room) + rank; // cand_count + rank, since cand_count = K - room
                                warp_cand_dists[pos] = dist;
                                warp_cand_idxs[pos]  = idx;
                            }
                            __syncwarp();
                            merge_full_buffer_with_intermediate(r_vals, r_idxs, warp_cand_dists, warp_cand_idxs, S, K);
                            if (lane == 0) cand_count = 0;
                            const float kth = r_vals[S - 1];
                            max_distance = __shfl_sync(FULL_MASK, kth, WARP_SIZE - 1);
                        }

                        // Place leftover candidates into fresh buffer
                        const int leftover = cnt - room;
                        if (leftover > 0) {
                            if (is_candidate && rank >= room) {
                                const int new_rank = rank - room;
                                warp_cand_dists[new_rank] = dist;
                                warp_cand_idxs[new_rank]  = idx;
                            }
                            if (lane == 0) cand_count = leftover;
                        }
                    }
                }
            }
        }

        __syncthreads(); // Ensure tile is no longer used before loading the next one
    }

    // After all batches, flush remaining candidates (if any)
    if (warp_active) {
        const unsigned FULL_MASK = 0xFFFFFFFFu;
        int count_now = 0;
        if (lane == 0) count_now = cand_count;
        count_now = __shfl_sync(FULL_MASK, count_now, 0);

        if (count_now > 0) {
            // Pad the rest with +inf and invalid indices
            for (int pos = lane; pos < (K - count_now); pos += WARP_SIZE) {
                warp_cand_dists[count_now + pos] = CUDART_INF_F;
                warp_cand_idxs[count_now + pos]  = -1;
            }
            __syncwarp();

            // Merge the padded full buffer
            merge_full_buffer_with_intermediate(r_vals, r_idxs, warp_cand_dists, warp_cand_idxs, S, K);
            if (lane == 0) cand_count = 0;
            const float kth = r_vals[S - 1];
            max_distance = __shfl_sync(FULL_MASK, kth, WARP_SIZE - 1);
        }

        // Write final sorted k nearest neighbors to output
        #pragma unroll
        for (int s = 0; s < MAX_SEG; ++s) {
            if (s >= S) break;
            const int pos = lane * S + s;
            const int out_idx = warp_global * K + pos;
            result[out_idx].first  = r_idxs[s];
            result[out_idx].second = r_vals[s];
        }
    }
}

// Host entry point. Launches the kernel with 8 warps (256 threads) per block.
// Shared memory allocation:
//   TILE_POINTS * sizeof(float2)                         // tile cache
// + warps_per_block * sizeof(int)                        // candidate counts per warp
// + warps_per_block * k * (sizeof(float) + sizeof(int))  // per-warp candidate buffers
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;
    const int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    const size_t tile_bytes = (size_t)TILE_POINTS * sizeof(float2);
    const size_t counts_bytes = (size_t)warps_per_block * sizeof(int);
    const size_t cand_bytes = (size_t)warps_per_block * (size_t)k * (sizeof(float) + sizeof(int));
    const size_t shared_bytes = tile_bytes + counts_bytes + cand_bytes;

    // Increase dynamic shared memory limit if needed
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);

    knn_kernel<<<num_blocks, threads_per_block, shared_bytes>>>(query, query_count, data, data_count, result, k);
}