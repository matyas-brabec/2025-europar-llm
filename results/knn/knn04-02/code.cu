#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// Optimized KNN for 2D points using one warp per query, shared-memory tiling for data points,
// batched candidate insertion, and warp-parallel merge via rank-based scatter.

// Notes:
// - k is assumed to be a power of two in [32, 1024].
// - Each warp (32 threads) handles one query.
// - The block loads a tile of data points to shared memory; each warp scans the tile against its query.
// - Per-warp intermediate top-k (indices + distances) is stored in shared memory.
// - Candidate updates are batched: each lane buffers a few better-than-worst candidates,
//   then the warp sorts and merges them with the current top-k using a parallel rank-scatter.
// - Uses no additional device allocations.

namespace knn_detail {

// Simple POD replacement with the same layout as std::pair<int, float>.
struct PairIF { int first; float second; };

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;             // 4 warps (128 threads) per block balances shared memory and occupancy.
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int TILE_POINTS = 2048;              // Number of data points per shared-memory tile (8 bytes per point -> 16KB).
constexpr int MAX_CAND_PER_LANE = 8;           // Buffered candidates per lane before a merge.
constexpr int MAX_CAND_TOTAL = WARP_SIZE * MAX_CAND_PER_LANE; // Max candidates per warp per batch (<= 256).

// Warp utilities
__device__ __forceinline__ unsigned full_mask() { return 0xffffffffu; }

__device__ __forceinline__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }

__device__ __forceinline__ int warp_id_in_block() { return threadIdx.x >> 5; }

// Warp-exclusive prefix sum of val (32 threads assumed active in warp)
__device__ __forceinline__ int warp_exclusive_prefix_sum(int val, unsigned mask = 0xffffffffu) {
    int lane = lane_id();
    int res = val;
    #pragma unroll
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
        int n = __shfl_up_sync(mask, res, d);
        if (lane >= d) res += n;
    }
    return res - val; // exclusive
}

// Warp-reduction sum (32 threads)
__device__ __forceinline__ int warp_reduce_sum(int val, unsigned mask = 0xffffffffu) {
    #pragma unroll
    for (int d = WARP_SIZE >> 1; d > 0; d >>= 1) {
        val += __shfl_down_sync(mask, val, d);
    }
    return val;
}

// Count of elements < x in sorted ascending array arr[0..n-1]
__device__ __forceinline__ int count_less(const float* arr, int n, float x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        float v = arr[mid];
        if (v < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Count of elements <= x in sorted ascending array arr[0..n-1]
__device__ __forceinline__ int count_less_equal(const float* arr, int n, float x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        float v = arr[mid];
        if (v <= x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Compute next power of two >= n, for n > 0 and n <= 1024 (small bounds here)
__device__ __forceinline__ int next_pow2(int n) {
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4;
    n |= n >> 8; n |= n >> 16;
    return n + 1;
}

// Bitonic sort (ascending) on shared-memory arrays (dists, idxs) of length n,
// using warp-synchronous parallelism. Elements beyond n up to next power-of-two
// are assumed initialized to +inf and dummy index.
__device__ __forceinline__ void bitonic_sort_shared(float* dists, int* idxs, int n) {
    const unsigned mask = full_mask();
    const int L = next_pow2(n);
    // Sort network over length L
    for (int k = 2; k <= L; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each lane processes multiple indices strided by warp
            for (int i = lane_id(); i < L; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    float ai = dists[i];
                    float aj = dists[ixj];
                    int ii = idxs[i];
                    int ij = idxs[ixj];
                    // Compare in ascending order when 'up' is true, descending otherwise
                    bool swap = (ai > aj) ^ (!up);
                    if (swap) {
                        dists[i] = aj; dists[ixj] = ai;
                        idxs[i] = ij; idxs[ixj] = ii;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

template<int K>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           PairIF* __restrict__ result) {
    // Shared memory layout:
    // - sTile: cached data points for current tile
    // - Per-warp arrays for top-K, candidate buffer, and merge buffer
    __shared__ float2 sTile[TILE_POINTS];

    __shared__ float sTopDist[WARPS_PER_BLOCK * K];
    __shared__ int   sTopIdx[WARPS_PER_BLOCK * K];

    __shared__ float sCandDist[WARPS_PER_BLOCK * MAX_CAND_TOTAL];
    __shared__ int   sCandIdx[WARPS_PER_BLOCK * MAX_CAND_TOTAL];

    __shared__ float sMergeDist[WARPS_PER_BLOCK * (K + MAX_CAND_TOTAL)];
    __shared__ int   sMergeIdx[WARPS_PER_BLOCK * (K + MAX_CAND_TOTAL)];

    const int lane = lane_id();
    const int warpInBlock = warp_id_in_block();
    const int warpGlobal = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;
    const bool warpActive = (warpGlobal < query_count);
    const unsigned wmask = full_mask();

    // Pointers to this warp's sections in shared memory
    float* topDist = sTopDist + warpInBlock * K;
    int*   topIdx  = sTopIdx  + warpInBlock * K;

    float* candDist = sCandDist + warpInBlock * MAX_CAND_TOTAL;
    int*   candIdx  = sCandIdx  + warpInBlock * MAX_CAND_TOTAL;

    float* mergeDist = sMergeDist + warpInBlock * (K + MAX_CAND_TOTAL);
    int*   mergeIdx  = sMergeIdx  + warpInBlock * (K + MAX_CAND_TOTAL);

    float2 q = make_float2(0.f, 0.f);
    if (warpActive) {
        // Load this warp's query point to registers
        if (lane == 0) {
            q = query[warpGlobal];
        }
        // Broadcast within warp
        q.x = __shfl_sync(wmask, q.x, 0);
        q.y = __shfl_sync(wmask, q.y, 0);

        // Initialize top-K arrays to +inf and -1
        for (int i = lane; i < K; i += WARP_SIZE) {
            topDist[i] = CUDART_INF_F;
            topIdx[i]  = -1;
        }
    }

    __syncthreads(); // Ensure shared memory initialized before use in tiles

    // For each tile of data points
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        const int tileCount = min(TILE_POINTS, data_count - base);

        // Load tile into shared memory cooperatively by all threads in block
        for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
            sTile[i] = data[base + i];
        }
        __syncthreads();

        if (warpActive) {
            // Local buffered candidates per lane
            float bufDist[MAX_CAND_PER_LANE];
            int   bufIdx[MAX_CAND_PER_LANE];
            int bufCount = 0;

            // Current worst distance in top-K (ascending order -> worst at index K-1)
            float worst = topDist[K - 1];

            // Process tile: each lane handles points strided by warp size
            for (int i = lane; i < tileCount; i += WARP_SIZE) {
                float2 p = sTile[i];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                float dist = fmaf(dx, dx, dy * dy);
                if (dist < worst) {
                    // Buffer this candidate
                    bufDist[bufCount] = dist;
                    bufIdx[bufCount] = base + i;
                    bufCount++;

                    // Flush buffered candidates if full
                    if (bufCount == MAX_CAND_PER_LANE) {
                        // Sort the local buffer (small n) ascending by distance (insertion sort)
                        for (int a = 1; a < bufCount; ++a) {
                            float vd = bufDist[a];
                            int   vi = bufIdx[a];
                            int b = a - 1;
                            while (b >= 0 && bufDist[b] > vd) {
                                bufDist[b + 1] = bufDist[b];
                                bufIdx[b + 1] = bufIdx[b];
                                --b;
                            }
                            bufDist[b + 1] = vd;
                            bufIdx[b + 1] = vi;
                        }

                        // Warp-wide gather of candidates into shared memory
                        int myCount = bufCount;
                        int myOffset = warp_exclusive_prefix_sum(myCount, wmask);
                        int total = __shfl_sync(wmask, myOffset + myCount, WARP_SIZE - 1);

                        // Write my buffer into warp's candidate array
                        for (int j = 0; j < myCount; ++j) {
                            candDist[myOffset + j] = bufDist[j];
                            candIdx[myOffset + j]  = bufIdx[j];
                        }
                        // Pad remaining up to next_pow2(total) with +inf
                        int L = next_pow2(max(1, total));
                        for (int j = myOffset + myCount; j < L; j += WARP_SIZE) {
                            int idx = j - myOffset - myCount + lane; // Distribute padding across lanes
                            int pos = myOffset + myCount + idx;
                            if (pos < L) {
                                candDist[pos] = CUDART_INF_F;
                                candIdx[pos]  = -1;
                            }
                        }
                        __syncwarp(wmask);

                        if (total > 0) {
                            // Bitonic sort of candidates (ascending)
                            bitonic_sort_shared(candDist, candIdx, total);

                            // Parallel rank-scatter merge of top-K (A) and candidates (C)
                            // A: topDist/topIdx of size K, ascending
                            // C: candDist/candIdx of size 'total', ascending
                            // Compute ranks and scatter into mergeDist/mergeIdx of size K + total
                            // Ranks for A elements: i + count_less(C, A[i])
                            for (int iA = lane; iA < K; iA += WARP_SIZE) {
                                float v = topDist[iA];
                                int r = iA + count_less(candDist, total, v);
                                mergeDist[r] = v;
                                mergeIdx[r]  = topIdx[iA];
                            }
                            // Ranks for C elements: j + count_less_equal(A, C[j])
                            for (int j = lane; j < total; j += WARP_SIZE) {
                                float v = candDist[j];
                                int r = j + count_less_equal(topDist, K, v);
                                mergeDist[r] = v;
                                mergeIdx[r]  = candIdx[j];
                            }
                            __syncwarp(wmask);

                            // Write back first K elements into top-K arrays (ascending)
                            for (int iA = lane; iA < K; iA += WARP_SIZE) {
                                topDist[iA] = mergeDist[iA];
                                topIdx[iA]  = mergeIdx[iA];
                            }
                            __syncwarp(wmask);

                            // Refresh worst
                            worst = topDist[K - 1];
                        }

                        // Reset buffer
                        bufCount = 0;
                    }
                }
            }

            // Flush remaining buffered candidates (same as above)
            if (bufCount > 0) {
                for (int a = 1; a < bufCount; ++a) {
                    float vd = bufDist[a];
                    int   vi = bufIdx[a];
                    int b = a - 1;
                    while (b >= 0 && bufDist[b] > vd) {
                        bufDist[b + 1] = bufDist[b];
                        bufIdx[b + 1] = bufIdx[b];
                        --b;
                    }
                    bufDist[b + 1] = vd;
                    bufIdx[b + 1] = vi;
                }

                int myCount = bufCount;
                int myOffset = warp_exclusive_prefix_sum(myCount, wmask);
                int total = __shfl_sync(wmask, myOffset + myCount, WARP_SIZE - 1);

                for (int j = 0; j < myCount; ++j) {
                    candDist[myOffset + j] = bufDist[j];
                    candIdx[myOffset + j]  = bufIdx[j];
                }
                int L = next_pow2(max(1, total));
                for (int j = myOffset + myCount; j < L; j += WARP_SIZE) {
                    int idx = j - myOffset - myCount + lane;
                    int pos = myOffset + myCount + idx;
                    if (pos < L) {
                        candDist[pos] = CUDART_INF_F;
                        candIdx[pos]  = -1;
                    }
                }
                __syncwarp(wmask);

                if (total > 0) {
                    bitonic_sort_shared(candDist, candIdx, total);

                    for (int iA = lane; iA < K; iA += WARP_SIZE) {
                        float v = topDist[iA];
                        int r = iA + count_less(candDist, total, v);
                        mergeDist[r] = v;
                        mergeIdx[r]  = topIdx[iA];
                    }
                    for (int j = lane; j < total; j += WARP_SIZE) {
                        float v = candDist[j];
                        int r = j + count_less_equal(topDist, K, v);
                        mergeDist[r] = v;
                        mergeIdx[r]  = candIdx[j];
                    }
                    __syncwarp(wmask);

                    for (int iA = lane; iA < K; iA += WARP_SIZE) {
                        topDist[iA] = mergeDist[iA];
                        topIdx[iA]  = mergeIdx[iA];
                    }
                    __syncwarp(wmask);
                }
            }
        }

        __syncthreads(); // Done using current tile; proceed to load next tile
    }

    if (warpActive) {
        // Write final top-K results for this query
        PairIF* out = result + warpGlobal * K;
        for (int i = lane; i < K; i += WARP_SIZE) {
            PairIF p;
            p.first = topIdx[i];
            p.second = topDist[i];
            out[i] = p;
        }
    }
}

} // namespace knn_detail

// Host callable entry point
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    using namespace knn_detail;

    if (query_count <= 0 || data_count <= 0) return;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Reinterpret result pointer to POD pair with identical layout
    PairIF* d_out = reinterpret_cast<PairIF*>(const_cast<std::pair<int,float>*>(result));

    // Dispatch based on k (power of two between 32 and 1024)
    switch (k) {
        case 32:
            knn_kernel<32><<<grid, block>>>(query, query_count, data, data_count, d_out);
            break;
        case 64:
            knn_kernel<64><<<grid, block>>>(query, query_count, data, data_count, d_out);
            break;
        case 128:
            knn_kernel<128><<<grid, block>>>(query, query_count, data, data_count, d_out);
            break;
        case 256:
            knn_kernel<256><<<grid, block>>>(query, query_count, data, data_count, d_out);
            break;
        case 512:
            knn_kernel<512><<<grid, block>>>(query, query_count, data, data_count, d_out);
            break;
        case 1024:
            knn_kernel<1024><<<grid, block>>>(query, query_count, data, data_count, d_out);
            break;
        default:
            // Unsupported k; do nothing. Preconditions guarantee k is valid.
            break;
    }
}