#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one warp (32 threads) per query. Each thread keeps k/32
// nearest neighbors in its registers as two arrays: distances (float) and indices (int).
// Data points are processed in batches cached into shared memory by the entire block.
// Each warp maintains a shared candidate buffer of size k (distances + indices) and a shared
// counter for the number of currently stored candidates. When the buffer is full (or after
// the last batch), the buffer is merged into the intermediate result in registers using
// a bitonic-sort based merge as described in the prompt.
//
// The implementation uses warp ballot to count candidates, warp shuffles to perform
// cross-lane compare-exchange steps of the bitonic sort, and shared memory for both
// the data tiles and per-warp candidate buffers.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable hyper-parameters
static constexpr int BLOCK_THREADS = 256;                       // 8 warps per block
static constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;
static constexpr int TILE_POINTS = 4096;                        // number of data points cached per batch (32KB for float2)
static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// Struct with the same memory layout as std::pair<int,float> for device writes.
// We will reinterpret_cast the result pointer to this type in the kernel.
struct PairIF { int first; float second; };

__device__ __forceinline__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
__device__ __forceinline__ int warp_id_in_block() { return threadIdx.x / WARP_SIZE; }

// Warp-wide bitonic sort of K = items_per_thread * WARP_SIZE elements distributed such that
// each lane holds items_per_thread consecutive elements in arrays dist[] and idx[].
// The result is sorted in ascending order of dist[]. For cross-lane exchanges, we use shuffles.
// For intra-lane exchanges, we directly swap register values.
// K must be a power of two and items_per_thread must be a power of two.
__device__ __forceinline__ void warp_bitonic_sort(float dist[], int idx[], int items_per_thread, int K)
{
    const int lane = lane_id();

    // Bitonic sort network: size = 2,4,8,...,K
    for (int size = 2; size <= K; size <<= 1)
    {
        // Stride = size/2, size/4, ..., 1
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            if (stride >= items_per_thread)
            {
                // Cross-lane compare-exchange. The partner lane differs by j_lane = stride / items_per_thread.
                const int j_lane = stride / items_per_thread;
                const int size_lane = size / items_per_thread; // >= 2 in this branch
                const bool up_block = ((lane & size_lane) == 0); // direction for this lane's block
                const bool is_lower = ((lane & j_lane) == 0);

                // For each register index, exchange with partner lane's same register index.
#pragma unroll
                for (int t = 0; t < 32; ++t)
                {
                    if (t >= items_per_thread) break;
                    float other_d = __shfl_xor_sync(FULL_MASK, dist[t], j_lane);
                    int   other_i = __shfl_xor_sync(FULL_MASK, idx[t], j_lane);

                    // Decide whether to take partner's value.
                    bool take_other;
                    if (is_lower) {
                        take_other = up_block ? (dist[t] > other_d) : (dist[t] < other_d);
                    } else {
                        take_other = up_block ? (dist[t] < other_d) : (dist[t] > other_d);
                    }
                    if (take_other) { dist[t] = other_d; idx[t] = other_i; }
                }
            }
            else
            {
                // Intra-lane compare-exchange among register entries.
#pragma unroll
                for (int t = 0; t < 32; ++t)
                {
                    if (t >= items_per_thread) break;
                    int partner = t ^ stride;
                    if ((t & stride) == 0) {
                        // Determine direction based on global index.
                        int global_i = lane * items_per_thread + t;
                        bool up = ((global_i & size) == 0);
                        float a = dist[t];  int ai = idx[t];
                        float b = dist[partner]; int bi = idx[partner];
                        bool swap = up ? (a > b) : (a < b);
                        if (swap) {
                            dist[t] = b; idx[t] = bi;
                            dist[partner] = a; idx[partner] = ai;
                        }
                    }
                }
            }
        }
    }
}

// Merge the warp's shared candidate buffer (size <= K) with the current intermediate result
// stored in registers (dist_reg/idx_reg). The buffer is first padded to size K with +inf,
// then swapped with the registers so that the buffer is in registers. The buffer is sorted,
// merged with the previous result using reversed pairing to create a bitonic sequence of size K,
// and finally sorted again to obtain the updated intermediate result.
//
// s_cand_dist, s_cand_idx: pointers to the warp's candidate arrays in shared memory
// s_cand_count: pointer to the warp's candidate count in shared memory
// dist_reg, idx_reg: per-thread arrays holding items_per_thread elements of the intermediate result
// items_per_thread, K: as above
// max_dist: reference to warp-local variable storing current k-th (largest) distance, updated here
__device__ __forceinline__
void warp_flush_and_merge(float* s_cand_dist, int* s_cand_idx, int* s_cand_count,
                          float dist_reg[], int idx_reg[], int items_per_thread, int K,
                          float& max_dist)
{
    const int lane = lane_id();
    int count = 0;
    if (lane == 0) count = *s_cand_count;
    count = __shfl_sync(FULL_MASK, count, 0);

    // Pad the remainder of the candidate buffer with +inf to complete size K.
    const float INF = CUDART_INF_F;
    for (int p = lane; p < K; p += WARP_SIZE) {
        if (p >= count) {
            s_cand_dist[p] = INF;
            s_cand_idx[p]  = -1;
        }
    }
    __syncwarp();

    // Step 1: Swap candidates into registers; previous result goes to shared memory.
#pragma unroll
    for (int t = 0; t < 32; ++t)
    {
        if (t >= items_per_thread) break;
        int pos = lane * items_per_thread + t;
        float tmp_d = dist_reg[t];
        int   tmp_i = idx_reg[t];
        float new_d = s_cand_dist[pos];
        int   new_i = s_cand_idx[pos];
        dist_reg[t] = new_d;  idx_reg[t] = new_i;
        s_cand_dist[pos] = tmp_d; s_cand_idx[pos] = tmp_i;
    }
    __syncwarp();

    // Step 2: Sort the buffer now in registers.
    warp_bitonic_sort(dist_reg, idx_reg, items_per_thread, K);
    __syncwarp();

    // Step 3: Merge with previous result (now in shared memory) using reversed pairing.
#pragma unroll
    for (int t = 0; t < 32; ++t)
    {
        if (t >= items_per_thread) break;
        int pos = lane * items_per_thread + t;
        int rev = K - 1 - pos; // reverse index for previous result
        float other_d = s_cand_dist[rev];
        int   other_i = s_cand_idx[rev];
        if (other_d < dist_reg[t]) {
            dist_reg[t] = other_d;
            idx_reg[t] = other_i;
        }
    }
    __syncwarp();

    // Step 4: Sort the merged bitonic sequence to get updated intermediate result.
    warp_bitonic_sort(dist_reg, idx_reg, items_per_thread, K);
    __syncwarp();

    // Update max_dist to the k-th (largest) element, which resides at global index K-1,
    // i.e., lane 31 and local index items_per_thread-1.
    float kth_dist = __shfl_sync(FULL_MASK, dist_reg[items_per_thread - 1], WARP_SIZE - 1);
    if (lane == 0) *s_cand_count = 0;
    max_dist = kth_dist;
    __syncwarp();
}

// Kernel: one warp processes one query.
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           PairIF* __restrict__ result,
                           int k)
{
    extern __shared__ unsigned char smem_raw[];
    // Layout shared memory:
    // [tile points][cand_dist all warps][cand_idx all warps][cand_count per warp]
    size_t offset = 0;

    float2* s_tile = reinterpret_cast<float2*>(smem_raw + offset);
    offset += size_t(TILE_POINTS) * sizeof(float2);

    // Align to 16 bytes
    offset = (offset + 15) & ~size_t(15);

    float* s_all_cand_dist = reinterpret_cast<float*>(smem_raw + offset);
    offset += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(float);

    offset = (offset + 15) & ~size_t(15);

    int* s_all_cand_idx = reinterpret_cast<int*>(smem_raw + offset);
    offset += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(int);

    offset = (offset + 15) & ~size_t(15);

    int* s_cand_count = reinterpret_cast<int*>(smem_raw + offset);
    offset += size_t(WARPS_PER_BLOCK) * sizeof(int);
    (void)offset; // suppress unused warning

    const int lane = lane_id();
    const int warp_in_block = warp_id_in_block();
    const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const bool warp_active = (global_warp < query_count);

    const int items_per_thread = k / WARP_SIZE; // k guaranteed multiple of 32

    // Pointers to this warp's candidate buffer in shared memory
    float* s_cand_dist = s_all_cand_dist + warp_in_block * k;
    int*   s_cand_idx  = s_all_cand_idx  + warp_in_block * k;

    // Initialize candidate count to 0 per warp
    if (lane == 0) s_cand_count[warp_in_block] = 0;
    __syncthreads();

    // Load the query point and broadcast to the warp
    float2 q = make_float2(0.0f, 0.0f);
    if (warp_active) {
        if (lane == 0) q = query[global_warp];
        q.x = __shfl_sync(FULL_MASK, q.x, 0);
        q.y = __shfl_sync(FULL_MASK, q.y, 0);
    }

    // Initialize intermediate result in registers: +inf distances and invalid indices.
    float best_dist[32];
    int   best_idx[32];
#pragma unroll
    for (int t = 0; t < 32; ++t) {
        if (t < items_per_thread) {
            best_dist[t] = CUDART_INF_F;
            best_idx[t]  = -1;
        }
    }

    // max_distance initialized to +inf; updated after each merge.
    float max_dist = CUDART_INF_F;

    // Process data in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS)
    {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_POINTS) tile_size = TILE_POINTS;

        // Cooperative load of tile into shared memory
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            s_tile[i] = data[tile_start + i];
        }
        __syncthreads();

        if (warp_active)
        {
            // Iterate over the tile elements; each lane processes a strided subset.
            for (int i = lane; i < tile_size; i += WARP_SIZE)
            {
                float2 p = s_tile[i];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                float d2 = dx * dx + dy * dy;
                int   idx = tile_start + i;

                // Filter by current max_dist
                bool is_cand = d2 < max_dist;
                unsigned mask = __ballot_sync(FULL_MASK, is_cand);
                int n = __popc(mask);
                if (n > 0)
                {
                    int old_count = 0, avail = 0, base_pos = 0;
                    if (lane == 0) {
                        old_count = s_cand_count[warp_in_block];
                        avail = k - old_count;
                        base_pos = old_count;
                        // Temporarily update count to avoid races within warp; final value corrected below if overflow.
                        s_cand_count[warp_in_block] = (n <= avail) ? (old_count + n) : k;
                    }
                    old_count = __shfl_sync(FULL_MASK, old_count, 0);
                    avail     = __shfl_sync(FULL_MASK, avail, 0);
                    base_pos  = __shfl_sync(FULL_MASK, base_pos, 0);

                    // Rank within the warp's current candidate set
                    unsigned lane_mask_lt = mask & ((1u << lane) - 1u);
                    int rank = __popc(lane_mask_lt);

                    // First portion fits before buffer is full
                    bool take_first = is_cand && (rank < avail);
                    if (take_first) {
                        int pos = base_pos + rank;
                        s_cand_dist[pos] = d2;
                        s_cand_idx[pos]  = idx;
                    }
                    __syncwarp();

                    // Handle overflow: flush buffer and attempt to add remaining candidates.
                    if (n > avail)
                    {
                        // Flush/merge to update best results and max_dist
                        warp_flush_and_merge(s_cand_dist, s_cand_idx, &s_cand_count[warp_in_block],
                                             best_dist, best_idx, items_per_thread, k, max_dist);

                        // Remaining candidates are those with rank >= avail. Re-filter with updated max_dist.
                        bool rem_cand = is_cand && (rank >= avail) && (d2 < max_dist);
                        unsigned mask2 = __ballot_sync(FULL_MASK, rem_cand);
                        int n2 = __popc(mask2);
                        if (n2 > 0) {
                            int base2 = 0;
                            if (lane == 0) {
                                base2 = s_cand_count[warp_in_block]; // should be 0 after flush
                                s_cand_count[warp_in_block] = base2 + n2;
                            }
                            base2 = __shfl_sync(FULL_MASK, base2, 0);
                            unsigned mask2_lt = mask2 & ((1u << lane) - 1u);
                            int rank2 = __popc(mask2_lt);
                            if (rem_cand) {
                                int pos2 = base2 + rank2;
                                s_cand_dist[pos2] = d2;
                                s_cand_idx[pos2]  = idx;
                            }
                        }
                    }
                }
            } // end per-tile per-warp loop
        }

        __syncthreads();
    } // end tiles loop

    // After the last tile, if there are still candidates in the buffer, flush them.
    if (warp_active) {
        int cnt = 0;
        if (lane == 0) cnt = s_cand_count[warp_in_block];
        cnt = __shfl_sync(FULL_MASK, cnt, 0);
        if (cnt > 0) {
            warp_flush_and_merge(s_cand_dist, s_cand_idx, &s_cand_count[warp_in_block],
                                 best_dist, best_idx, items_per_thread, k, max_dist);
        }

        // Write out the result in ascending order to global memory.
        // For query 'global_warp', its result segment starts at result[global_warp * k].
        PairIF* out = result + (size_t(global_warp) * size_t(k));
#pragma unroll
        for (int t = 0; t < 32; ++t) {
            if (t >= items_per_thread) break;
            int pos = lane * items_per_thread + t;
            out[pos].first  = best_idx[t];
            out[pos].second = best_dist[t];
        }
    }
}

// Host wrapper. Launches the kernel with appropriate configuration and shared memory.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Grid configuration: one warp per query.
    int warps_needed = query_count;
    int blocks = (warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks);
    dim3 block(BLOCK_THREADS);

    // Compute dynamic shared memory size:
    // tile + per-warp candidate buffers (float dist + int idx) + per-warp candidate counts
    size_t smem_size = 0;
    smem_size += size_t(TILE_POINTS) * sizeof(float2);
    // Align to 16 bytes
    smem_size = (smem_size + 15) & ~size_t(15);
    smem_size += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(float); // cand dist
    smem_size = (smem_size + 15) & ~size_t(15);
    smem_size += size_t(WARPS_PER_BLOCK) * size_t(k) * sizeof(int);   // cand idx
    smem_size = (smem_size + 15) & ~size_t(15);
    smem_size += size_t(WARPS_PER_BLOCK) * sizeof(int);               // cand count

    // Opt-in larger dynamic shared memory if necessary (A100/H100 support > 96KB).
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);

    // Launch kernel. Cast result pointer to device-friendly layout.
    PairIF* result_pairif = reinterpret_cast<PairIF*>(result);
    knn_kernel<<<grid, block, smem_size>>>(query, query_count, data, data_count, result_pairif, k);
    // Caller is assumed to perform cudaDeviceSynchronize or error checks if desired.
}