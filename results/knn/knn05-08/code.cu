#include <cuda_runtime.h>
#include <utility>

// This implementation assigns one warp (32 threads) per query.
// It processes the data points in tiles loaded into shared memory,
// maintains an intermediate top-k result per query in registers (distributed across warp lanes),
// buffers up to k candidate neighbors per warp in shared memory, and whenever the
// buffer is full (or at the end of processing), it sorts the buffer and merges it
// with the intermediate result using a parallel merge-path algorithm.
// The output contains indices of the nearest data points and their distances (squared L2).

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Choose a conservative configuration that fits A100's 164 KB shared memory limit for all k in [32, 1024].
// - Warps per block: 4 (128 threads per block)
// - Per-warp shared memory: 3 * k pairs (dist+idx) = 24*k bytes
// - Tile size (data points per block): 6144 float2 = 48 KB
// Total: 4*24k + 48KB <= 164KB for k <= 1024
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int TILE_POINTS = 6144; // 48 KB of float2

struct PairIF { int first; float second; };

__device__ __forceinline__ unsigned lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}
__device__ __forceinline__ unsigned warp_id_in_block() {
    return threadIdx.x >> 5;
}
__device__ __forceinline__ unsigned lane_mask_lt() {
    // Mask with bits set for lanes less than current lane
    return (1u << lane_id()) - 1u;
}

__device__ __forceinline__ float2 ld_float2(const float2* ptr) {
    // Use regular global load; modern GPUs will cache automatically.
    return *ptr;
}

__device__ __forceinline__ void warp_sync() {
#if __CUDA_ARCH__ >= 700
    __syncwarp();
#else
    // For older arch, no-op; but our target hardware is H100/A100 (sm80+)
#endif
}

// Bitonic sort for pairs (dist, idx) stored in shared memory.
// N must be a power of two. Sorting order: ascending by distance.
__device__ void bitonic_sort_pairs_shared(float* dist, int* idx, int N) {
    const unsigned full_mask = 0xFFFFFFFFu;
    unsigned lane = lane_id();
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = lane; i < N; i += WARP_SIZE) {
                int j = i ^ stride;
                if (j > i) {
                    bool ascending = ((i & size) == 0);
                    float di = dist[i];
                    float dj = dist[j];
                    int ii = idx[i];
                    int ij = idx[j];
                    bool cond = ascending ? (di > dj) : (di < dj);
                    if (cond) {
                        dist[i] = dj; dist[j] = di;
                        idx[i]  = ij; idx[j] = ii;
                    }
                }
            }
            warp_sync();
        }
    }
    (void)full_mask;
}

// Merge-path partition search: find a where
// A[a-1] <= B[b] and B[b-1] <= A[a], with a + b = diag.
// Arrays are sorted ascending. Uses sentinels -INF/+INF for out-of-bounds.
__device__ int merge_path_search(const float* A, int nA, const float* B, int nB, int diag) {
    int low = max(0, diag - nB);
    int high = min(diag, nA);
    while (low <= high) {
        int a = (low + high) >> 1;
        int b = diag - a;
        float A_left = (a > 0)    ? A[a - 1] : -CUDART_INF_F;
        float A_right = (a < nA)  ? A[a]     : CUDART_INF_F;
        float B_left = (b > 0)    ? B[b - 1] : -CUDART_INF_F;
        float B_right = (b < nB)  ? B[b]     : CUDART_INF_F;
        if (A_left <= B_right && B_left <= A_right) {
            return a;
        } else if (A_left > B_right) {
            high = a - 1;
        } else {
            low = a + 1;
        }
    }
    // Fallback; should not reach here for valid inputs
    return min(max(0, low), nA);
}

// Merge top-k of two sorted arrays A (nA=k) and B (nB<=k) into output (k elements).
// A_in_dist/idx: input A (size k)
// B_dist/idx:    input B (size k, with +INF padding in tail if nB < k)
// out_dist/idx:  output buffer (size k)
// Uses one warp cooperatively; each lane merges CHUNK = k / 32 outputs.
__device__ void merge_topk_warp(const float* A_in_dist, const int* A_in_idx,
                                const float* B_dist, const int* B_idx,
                                float* out_dist, int* out_idx,
                                int k) {
    const int lane = lane_id();
    const int CHUNK = k / WARP_SIZE;
    const int out_begin = lane * CHUNK;
    const int out_end   = out_begin + CHUNK;

    // Compute partitions for this lane
    int a_begin = merge_path_search(A_in_dist, k, B_dist, k, out_begin);
    int b_begin = out_begin - a_begin;
    int a_end   = merge_path_search(A_in_dist, k, B_dist, k, out_end);
    int b_end   = out_end - a_end;

    int i = a_begin;
    int j = b_begin;
    for (int out_pos = out_begin; out_pos < out_end; ++out_pos) {
        // Choose the smaller current element
        float a_val = (i < a_end) ? A_in_dist[i] : CUDART_INF_F;
        float b_val = (j < b_end) ? B_dist[j]    : CUDART_INF_F;
        if (a_val <= b_val) {
            out_dist[out_pos] = a_val;
            out_idx [out_pos] = A_in_idx[i];
            ++i;
        } else {
            out_dist[out_pos] = b_val;
            out_idx [out_pos] = B_idx[j];
            ++j;
        }
    }
    warp_sync();
}

// Kernel: one warp per query.
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           PairIF* __restrict__ result,
                           int k) {
    extern __shared__ unsigned char shared_bytes[];
    // Shared memory layout:
    // [0 .. TILE_POINTS-1] float2 tile_data
    // Then per-warp segments, each consisting of:
    //   cand_dist[k], cand_idx[k],
    //   A_in_dist[k], A_in_idx[k],
    //   out_dist[k],  out_idx[k]
    // Then cand_count[WARPS_PER_BLOCK]
    float2* tile_points = reinterpret_cast<float2*>(shared_bytes);
    size_t tile_bytes = size_t(TILE_POINTS) * sizeof(float2);
    unsigned char* p = shared_bytes + tile_bytes;

    size_t per_warp_bytes = size_t(k) * (sizeof(float) + sizeof(int)) * 3; // 3 arrays of pairs
    unsigned wid = warp_id_in_block();
    unsigned lane = lane_id();
    unsigned full_mask = 0xFFFFFFFFu;

    unsigned char* warp_base = p + wid * per_warp_bytes;

    float* cand_dist = reinterpret_cast<float*>(warp_base);
    int*   cand_idx  = reinterpret_cast<int*>(warp_base + size_t(k) * sizeof(float));

    float* A_in_dist = reinterpret_cast<float*>(warp_base + size_t(k) * (sizeof(float) + sizeof(int)));
    int*   A_in_idx  = reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(A_in_dist) + size_t(k) * sizeof(float));

    float* out_dist  = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(A_in_idx) + size_t(k) * sizeof(int));
    int*   out_idx   = reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(out_dist) + size_t(k) * sizeof(float));

    // cand_count array placed after all per-warp buffers
    int* cand_count_base = reinterpret_cast<int*>(p + WARPS_PER_BLOCK * per_warp_bytes);
    int& cand_count = cand_count_base[wid];

    // Assign one warp per query
    int query_index = int(blockIdx.x) * WARPS_PER_BLOCK + int(wid);
    bool warp_active = (query_index < query_count);

    // Initialize warp-private intermediate result in registers (distributed across lanes)
    const int CHUNK = k / WARP_SIZE;
    float res_dist[32]; // maximum CHUNK is 32; we will use only first CHUNK
    int   res_idx[32];
    for (int i = 0; i < CHUNK; ++i) {
        res_dist[i] = CUDART_INF_F;
        res_idx[i]  = -1;
    }
    float kth_threshold = CUDART_INF_F;

    // Initialize candidate buffer in shared memory for this warp
    if (lane == 0) cand_count = 0;
    warp_sync();
    for (int i = lane; i < k; i += WARP_SIZE) {
        cand_dist[i] = CUDART_INF_F;
        cand_idx [i] = -1;
    }
    warp_sync();

    // Load query point for this warp
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        float2 q = query[query_index];
        qx = q.x; qy = q.y;
    }
    // Broadcast qx, qy within the warp to ensure they are defined in all lanes
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);

    // Helper lambdas capture by reference 'cand_count', 'cand_dist', 'cand_idx', 'A_in_*', 'out_*', 'res_*', 'k', 'kth_threshold'
    auto sort_candidates_and_merge = [&]() {
        // Sort candidate buffer (size k with +INF padding) in shared memory
        bitonic_sort_pairs_shared(cand_dist, cand_idx, k);

        // Copy current intermediate result from registers to shared memory A_in_*
        // Global order corresponds to concatenation of lanes in lane order.
        int base = int(lane) * CHUNK;
        for (int i = 0; i < CHUNK; ++i) {
            A_in_dist[base + i] = res_dist[i];
            A_in_idx [base + i] = res_idx[i];
        }
        warp_sync();

        // Merge top-k of A_in_* and cand_* into out_*
        merge_topk_warp(A_in_dist, A_in_idx, cand_dist, cand_idx, out_dist, out_idx, k);

        // Copy merged result back into registers
        for (int i = 0; i < CHUNK; ++i) {
            res_dist[i] = out_dist[base + i];
            res_idx [i] = out_idx [base + i];
        }
        warp_sync();

        // Update kth_threshold (k-th nearest distance), which is the last element of the global list
        float tail = res_dist[CHUNK - 1];
        kth_threshold = __shfl_sync(full_mask, tail, WARP_SIZE - 1);

        // Reset candidate buffer
        if (lane == 0) cand_count = 0;
        warp_sync();
        for (int i = lane; i < k; i += WARP_SIZE) {
            cand_dist[i] = CUDART_INF_F;
            cand_idx [i] = -1;
        }
        warp_sync();
    };

    auto try_append_candidate = [&](float d, int idx) {
        // Try to append this candidate, merging as needed; re-evaluate against threshold after each merge.
        while (true) {
            // If not better than current k-th, skip
            if (!(d < kth_threshold)) break;

            unsigned mask = __ballot_sync(full_mask, d < kth_threshold);
            int num = __popc(mask);
            if (num == 0) break; // nobody has a valid candidate now

            // Reserve space or merge
            int start = 0;
            if (lane == 0) start = cand_count;
            start = __shfl_sync(full_mask, start, 0);
            int space = k - start;
            if (space >= num) {
                // Append all valid candidates
                int rank = __popc(mask & lane_mask_lt());
                int pos = start + rank;
                if (d < kth_threshold) {
                    cand_dist[pos] = d;
                    cand_idx [pos] = idx;
                }
                if (lane == 0) cand_count = start + num;
                warp_sync();
                break;
            } else {
                // Not enough space: merge current buffer with intermediate result
                sort_candidates_and_merge();
                // After merge, kth_threshold is updated; loop to re-evaluate acceptance
            }
        }
    };

    // Iterate over data in tiles loaded into shared memory by entire block
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        int tile_count = min(TILE_POINTS, data_count - base);

        // Load tile
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            tile_points[t] = ld_float2(data + base + t);
        }
        __syncthreads();

        // Each active warp processes the tile
        if (warp_active) {
            for (int j = lane; j < tile_count; j += WARP_SIZE) {
                float2 p = tile_points[j];
                float dx = qx - p.x;
                float dy = qy - p.y;
                float d = fmaf(dx, dx, dy * dy); // squared L2
                int   idx = base + j;

                // Try to append this candidate into the warp's buffer
                try_append_candidate(d, idx);
            }
        }

        __syncthreads(); // ensure all warps finished using tile before it's overwritten
    }

    // After last tile, flush remaining candidates
    if (warp_active) {
        int count = 0;
        if (lane == 0) count = cand_count;
        count = __shfl_sync(full_mask, count, 0);
        if (count > 0) {
            sort_candidates_and_merge();
        }

        // Write out final top-k for this query in row-major order
        int base_out = query_index * k;
        int global_base = int(lane) * CHUNK;
        for (int i = 0; i < CHUNK; ++i) {
            int out_pos = base_out + global_base + i;
            result[out_pos].first  = res_idx[i];
            result[out_pos].second = res_dist[i];
        }
    }
}

// Host function: launch the kernel with appropriate configuration and shared memory size.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Dynamic shared memory size:
    // Tile: TILE_POINTS * sizeof(float2)
    // Per-warp: 3 * k * (sizeof(float) + sizeof(int))
    // Counts: WARPS_PER_BLOCK * sizeof(int)
    size_t tile_bytes = size_t(TILE_POINTS) * sizeof(float2);
    size_t per_warp_bytes = size_t(k) * (sizeof(float) + sizeof(int)) * 3;
    size_t smem_bytes = tile_bytes + WARPS_PER_BLOCK * per_warp_bytes + WARPS_PER_BLOCK * sizeof(int);

    // Opt-in to large dynamic shared memory if necessary
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes));

    PairIF* out = reinterpret_cast<PairIF*>(result);
    knn_kernel<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, out, k);
}