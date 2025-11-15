#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ----------------------------------------------------------------------------------
// k-NN implementation for 2D points (float2) using CUDA.
// - One warp (32 threads) processes one query point.
// - Each warp maintains its k nearest neighbors in registers.
// - Candidates are accumulated in a per-warp buffer in shared memory.
// - When the buffer fills (or at the end), the buffer is merged with the
//   intermediate result using a warp-wide bitonic sort + merge scheme.
// ----------------------------------------------------------------------------------

// Simple POD type matching the layout (int, float) for device-side access.
// We rely on the host providing a std::pair<int,float> array with identical layout.
struct PairIF {
    int   first;
    float second;
};

// Internal neighbor representation.
struct Neighbor {
    float dist;
    int   idx;
};

// Warp utilities.
__device__ __forceinline__ int lane_id()
{
    return threadIdx.x & 31;
}

__device__ __forceinline__ int warp_id_in_block()
{
    return threadIdx.x >> 5;
}

// Forward declarations of device helpers.
__device__ __forceinline__ void warp_bitonic_sort(
    float *dist, int *idx,
    int k, int k_per_thread,
    int lane, unsigned full_mask);

__device__ __forceinline__ void flush_candidate_buffer(
    Neighbor *warp_buf,
    int k, int k_per_thread,
    float *best_dist, int *best_idx,
    int &cand_count,
    float &max_dist,
    int lane, unsigned full_mask);

// ----------------------------------------------------------------------------------
// Warp-wide bitonic sort for k elements distributed over a warp.
// - k is a power of two, and k is a multiple of 32: k = 32 * k_per_thread.
// - Each thread holds k_per_thread elements in registers.
// - Global index mapping:
//   global_index = offset * 32 + lane (offset in [0, k_per_thread-1]).
// ----------------------------------------------------------------------------------
__device__ __forceinline__ void warp_bitonic_sort(
    float *dist, int *idx,
    int k, int k_per_thread,
    int lane, unsigned full_mask)
{
    const int WARP_SIZE = 32;

    // Outer loop: size of the subsequences being merged (Batcher's bitonic).
    for (int size = 2; size <= k; size <<= 1) {

        // Inner loop: distance between compared elements.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {

            if (stride >= WARP_SIZE) {
                // --------------------------------------------------------------
                // Comparisons within a thread (same lane, different offsets).
                // Each pair (offset, partner_offset) is processed only once.
                // --------------------------------------------------------------
                int stride_off = stride >> 5;  // stride / 32

                for (int o = 0; o < k_per_thread; ++o) {
                    int partner_o = o ^ stride_off;
                    // Process each pair only once (partner_o > o).
                    if (partner_o > o && partner_o < k_per_thread) {

                        int gi = (o << 5) | lane;  // global index of element at offset o
                        bool up = ((gi & size) == 0);

                        float v_i = dist[o];
                        float v_p = dist[partner_o];
                        int   i_i = idx[o];
                        int   i_p = idx[partner_o];

                        bool cond = (v_i > v_p);
                        // If up == true: keep smaller at gi, larger at partner.
                        // If up == false: keep larger at gi, smaller at partner.
                        if ((cond && up) || (!cond && !up)) {
                            float tmpd = v_i; v_i = v_p; v_p = tmpd;
                            int   tmpi = i_i; i_i = i_p; i_p = tmpi;
                        }

                        dist[o]          = v_i;
                        idx[o]           = i_i;
                        dist[partner_o]  = v_p;
                        idx[partner_o]   = i_p;
                    }
                }
            } else {
                // --------------------------------------------------------------
                // Comparisons across threads (different lanes, same offset).
                // Implemented with warp shuffles; each comparator is realized
                // symmetrically at both ends.
                // --------------------------------------------------------------
                for (int o = 0; o < k_per_thread; ++o) {
                    int gi = (o << 5) | lane;  // global index of this element

                    float self_dist = dist[o];
                    int   self_idx  = idx[o];

                    float other_dist = __shfl_xor_sync(full_mask, self_dist, stride);
                    int   other_idx  = __shfl_xor_sync(full_mask, self_idx,  stride);

                    bool up = ((gi & size) == 0);

                    // If up == true: keep min; if up == false: keep max.
                    bool swap = ((self_dist > other_dist) == up);
                    if (swap) {
                        self_dist = other_dist;
                        self_idx  = other_idx;
                    }

                    dist[o] = self_dist;
                    idx[o]  = self_idx;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------------
// Merge the per-warp candidate buffer (in shared memory) with the intermediate
// k-NN result (in registers).
//
// - best_dist / best_idx : current top-k in ascending order (registers).
// - warp_buf             : shared memory buffer [0..k-1] for candidate points.
// - cand_count           : number of valid candidates stored in warp_buf.
// - max_dist             : distance of the current k-th neighbor.
// ----------------------------------------------------------------------------------
__device__ __forceinline__ void flush_candidate_buffer(
    Neighbor *warp_buf,
    int k, int k_per_thread,
    float *best_dist, int *best_idx,
    int &cand_count,
    float &max_dist,
    int lane, unsigned full_mask)
{
    if (cand_count == 0) {
        return;
    }

    const int WARP_SIZE = 32;

    // If the candidate buffer is not completely filled, pad the remainder
    // with max_dist so that we can treat it like a full buffer.
    if (cand_count < k) {
        for (int i = cand_count + lane; i < k; i += WARP_SIZE) {
            warp_buf[i].dist = max_dist;
            warp_buf[i].idx  = -1;
        }
        __syncwarp(full_mask);
    }

    // -------------------------------------------------------------------------
    // Step 1: Swap the contents of the buffer (shared) and the intermediate
    // result (registers) so that the buffer becomes resident in registers.
    // After this step:
    // - warp_buf[...]     contains the old intermediate result.
    // - best_dist/best_idx contain the previous candidate buffer contents.
    // -------------------------------------------------------------------------
    for (int o = 0; o < k_per_thread; ++o) {
        int gi = (o << 5) | lane;  // global index for this register element

        // Old intermediate result (from registers).
        Neighbor tmp;
        tmp.dist = best_dist[o];
        tmp.idx  = best_idx[o];

        // Current candidate from shared buffer.
        Neighbor buf_val = warp_buf[gi];

        // Swap: write old intermediate result to shared memory.
        warp_buf[gi] = tmp;

        // Store candidates into registers.
        best_dist[o] = buf_val.dist;
        best_idx[o]  = buf_val.idx;
    }
    __syncwarp(full_mask);

    // -------------------------------------------------------------------------
    // Step 2: Sort the buffer (now in registers) in ascending order using
    // warp-wide bitonic sort.
    // -------------------------------------------------------------------------
    warp_bitonic_sort(best_dist, best_idx, k, k_per_thread, lane, full_mask);

    // -------------------------------------------------------------------------
    // Step 3: Merge the sorted buffer (registers) and intermediate result
    // (shared) into a bitonic sequence:
    //
    // For each global index i:
    //   merged[i] = min( buffer[i], intermediate[k - 1 - i] )
    //
    // In terms of our data layout:
    // - buffer[i] is in registers (best_dist/best_idx).
    // - intermediate[k - 1 - i] is in warp_buf[ mirrored_index ].
    // -------------------------------------------------------------------------
    for (int o = 0; o < k_per_thread; ++o) {
        int gi      = (o << 5) | lane;   // index in buffer (registers)
        int gi_rev  = k - 1 - gi;        // mirrored index in shared

        Neighbor r = warp_buf[gi_rev];
        float rd   = r.dist;
        int   ri   = r.idx;

        float bd   = best_dist[o];
        int   bi   = best_idx[o];

        // Keep the closer of the two.
        if (rd < bd) {
            best_dist[o] = rd;
            best_idx[o]  = ri;
        }
    }
    __syncwarp(full_mask);

    // -------------------------------------------------------------------------
    // Step 4: Sort the merged result (still in registers) in ascending order.
    // This yields the updated intermediate result.
    // -------------------------------------------------------------------------
    warp_bitonic_sort(best_dist, best_idx, k, k_per_thread, lane, full_mask);

    // Update max_dist to the distance of the k-th nearest neighbor (last element).
    int last_index  = k - 1;
    int last_offset = last_index >> 5;   // which offset within a thread
    int last_lane   = last_index & 31;   // which lane owns it

    float last_val = best_dist[last_offset];
    max_dist = __shfl_sync(full_mask, last_val, last_lane);

    // Candidate buffer is now conceptually empty.
    cand_count = 0;
}

// ----------------------------------------------------------------------------------
// CUDA kernel: each warp processes one query point.
// - query_count   : number of query points.
// - data_count    : number of data points.
// - query         : [query_count] array of float2.
// - data          : [data_count] array of float2.
// - result        : [query_count * k] array of PairIF (index, distance).
// - k             : number of nearest neighbors (power of two, 32 <= k <= 1024).
// ----------------------------------------------------------------------------------
template<int WARPS_PER_BLOCK, int TILE_SIZE>
__global__ void knn_kernel(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    PairIF * __restrict__ result,
    int k)
{
    const int WARP_SIZE  = 32;
    const int lane       = lane_id();
    const int warp_in_blk= warp_id_in_block();
    const int warp_global= blockIdx.x * WARPS_PER_BLOCK + warp_in_blk;
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Per-thread local storage for top-k neighbors.
    // k is in [32, 1024], so k_per_thread in [1, 32].
    const int k_per_thread = k / WARP_SIZE;

    // Shared memory layout:
    // [0 .. TILE_SIZE-1] : float2 data tile
    // [TILE_SIZE .. ]    : per-warp candidate buffers (Neighbor[WARPS_PER_BLOCK][k])
    extern __shared__ unsigned char smem[];
    float2 *s_data = reinterpret_cast<float2*>(smem);
    Neighbor *s_candidates_base = reinterpret_cast<Neighbor*>(s_data + TILE_SIZE);

    // Pointer to this warp's candidate buffer in shared memory.
    Neighbor *warp_buf = s_candidates_base + warp_in_blk * k;

    // Determine whether this warp corresponds to a valid query.
    bool active = (warp_global < query_count);

    // Load query point once per warp and broadcast.
    float2 q;
    if (active) {
        if (lane == 0) {
            q = query[warp_global];
        }
        q.x = __shfl_sync(FULL_MASK, q.x, 0);
        q.y = __shfl_sync(FULL_MASK, q.y, 0);
    }

    // Initialize intermediate result in registers: distances set to +inf, indices to -1.
    float best_dist[32];
    int   best_idx[32];
    for (int o = 0; o < k_per_thread; ++o) {
        best_dist[o] = FLT_MAX;
        best_idx[o]  = -1;
    }

    // Candidate buffer state (replicated across threads in the warp).
    int   cand_count = 0;
    float max_dist   = FLT_MAX;  // distance of current k-th neighbor

    // Process the data points in tiles loaded into shared memory.
    for (int base = 0; base < data_count; base += TILE_SIZE) {
        int tile_count = data_count - base;
        if (tile_count > TILE_SIZE) tile_count = TILE_SIZE;

        // Load tile into shared memory by the entire block.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_data[i] = data[base + i];
        }
        __syncthreads();

        // Each active warp processes the tile for its query.
        if (active) {
            for (int t = 0; t < tile_count; t += WARP_SIZE) {
                int idx_in_tile = t + lane;
                bool valid = (idx_in_tile < tile_count);

                float dist  = FLT_MAX;
                int   d_idx = -1;

                if (valid) {
                    float2 p = s_data[idx_in_tile];
                    float dx = p.x - q.x;
                    float dy = p.y - q.y;
                    dist  = dx * dx + dy * dy;     // squared Euclidean distance
                    d_idx = base + idx_in_tile;    // global data index
                }

                // Filter by max_dist and add to candidate buffer using warp ballot.
                bool is_candidate = valid && (dist < max_dist);

                unsigned mask      = __ballot_sync(FULL_MASK, is_candidate);
                int      n_in_warp = __popc(mask);

                if (n_in_warp > 0) {
                    // If adding these candidates would overflow the buffer, flush first.
                    if (cand_count + n_in_warp > k) {
                        flush_candidate_buffer(
                            warp_buf, k, k_per_thread,
                            best_dist, best_idx,
                            cand_count, max_dist,
                            lane, FULL_MASK);

                        // max_dist has been updated; recompute candidacy.
                        is_candidate = valid && (dist < max_dist);
                        mask         = __ballot_sync(FULL_MASK, is_candidate);
                        n_in_warp    = __popc(mask);

                        if (n_in_warp == 0) {
                            continue;
                        }
                    }

                    // Compute position in the buffer for this thread's candidate.
                    int prefix = __popc(mask & ((1u << lane) - 1));

                    if (is_candidate) {
                        int pos = cand_count + prefix;
                        warp_buf[pos].dist = dist;
                        warp_buf[pos].idx  = d_idx;
                    }

                    // Update candidate count (warp-uniform).
                    cand_count += n_in_warp;
                }
            }
        }

        __syncthreads();
    }

    // After all tiles are processed, flush any remaining candidates.
    if (active && cand_count > 0) {
        flush_candidate_buffer(
            warp_buf, k, k_per_thread,
            best_dist, best_idx,
            cand_count, max_dist,
            lane, FULL_MASK);
    }

    // Write final top-k results to global memory.
    if (active) {
        int base_out = warp_global * k;

        // Global index mapping: g = offset * 32 + lane.
        for (int o = 0; o < k_per_thread; ++o) {
            int g = (o << 5) | lane;  // position in [0, k-1]
            PairIF out;
            out.first  = best_idx[o];
            out.second = best_dist[o];
            result[base_out + g] = out;
        }
    }
}

// ----------------------------------------------------------------------------------
// Host entry point: launches the CUDA kernel.
// Assumptions (per problem statement):
// - query, data, result are device pointers allocated with cudaMalloc.
// - data_count >= k.
// - k is a power of two, 32 <= k <= 1024.
// ----------------------------------------------------------------------------------
void run_knn(
    const float2 *query, int query_count,
    const float2 *data,  int data_count,
    std::pair<int, float> *result,
    int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Configuration parameters:
    // - WARPS_PER_BLOCK: number of queries processed per block.
    // - TILE_SIZE: number of data points cached per block in shared memory.
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
    constexpr int TILE_SIZE = 1024;

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Shared memory size:
    //   TILE_SIZE * sizeof(float2) for the data tile
    // + WARPS_PER_BLOCK * k * sizeof(Neighbor) for per-warp candidate buffers.
    size_t shared_bytes =
        TILE_SIZE * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(Neighbor);

    PairIF *raw_result = reinterpret_cast<PairIF*>(result);

    knn_kernel<WARPS_PER_BLOCK, TILE_SIZE>
        <<<grid_dim, block_dim, shared_bytes>>>(
            query, query_count,
            data,  data_count,
            raw_result, k);

    // For simplicity, synchronize here. In a larger application, the caller
    // might prefer to manage stream synchronization externally.
    cudaDeviceSynchronize();
}