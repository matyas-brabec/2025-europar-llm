#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// This implementation assigns one CUDA warp (32 threads) per query point.
// The data points are processed in tiles that are cooperatively loaded into shared memory by the entire block.
// Each warp maintains a private intermediate top-k structure split evenly across its 32 lanes.
// Specifically, each lane keeps the best L = k/32 candidates it has seen (sorted ascending by distance),
// and at the end, the warp performs a multiway merge (via warp-level shuffles) to produce the final k neighbors.
//
// Key properties:
// - k is a power of two in [32, 1024], so L = k / 32 is an integer in [1, 32].
// - Distances are squared Euclidean (no sqrt), computed with an FMA for precision and speed.
// - No additional device memory is allocated (uses only shared memory and registers).
// - Threads in a warp communicate via __shfl_sync. Block-wide __syncthreads() is used around shared-memory tiles.
// - The per-lane top-L is kept sorted ascending to make insertion O(L) and the final merge simple.
// - Result is written as contiguous std::pair<int, float> per query, ordered by increasing distance.

struct PairIF {
    int   first;
    float second;
};

static inline __device__ float sqdist2D(const float2 a, const float qx, const float qy) {
    float dx = a.x - qx;
    float dy = a.y - qy;
    // dx*dx + dy*dy using one FMA
    return fmaf(dy, dy, dx * dx);
}

static inline __device__ unsigned full_warp_mask() {
#if __CUDACC_VER_MAJOR__ >= 9
    return 0xFFFFFFFFu;
#else
    return 0xFFFFFFFFu;
#endif
}

// Warp-level argmin on a float value; returns pair {min_value, min_lane}.
struct MinValLane {
    float val;
    int   lane;
};

static inline __device__ MinValLane warp_argmin(float val) {
    unsigned mask = full_warp_mask();
    int lane = threadIdx.x & 31;
    float v = val;
    int   l = lane;

    // Tree reduction to find min value and corresponding lane (tie-break by smaller lane id)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(mask, v, offset);
        int   ol = __shfl_down_sync(mask, l, offset);
        // Choose the smaller value; in case of tie, choose the smaller lane id
        if ((ov < v) || ((ov == v) && (ol < l))) {
            v = ov;
            l = ol;
        }
    }
    // Broadcast the winner to all lanes
    MinValLane res;
    res.val  = __shfl_sync(mask, v, 0);
    res.lane = __shfl_sync(mask, l, 0);
    return res;
}

__global__ void knn2d_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    PairIF* __restrict__ result,
    int k,
    int tile_points // number of data points per shared-memory tile
) {
    extern __shared__ float2 smem[]; // shared-memory cache for data tiles

    const int warpSize_ = 32;
    const int lane      = threadIdx.x & (warpSize_ - 1);
    const int warp      = threadIdx.x >> 5; // warp index within the block
    const int warpsPerBlock = blockDim.x >> 5;

    const int qid = blockIdx.x * warpsPerBlock + warp; // global query index served by this warp
    const bool active = (qid < query_count);

    // Load and broadcast the query point to the warp's lanes
    float qx = 0.0f, qy = 0.0f;
    if (active) {
        float2 qv;
        if (lane == 0) qv = query[qid];
        unsigned mask = full_warp_mask();
        qx = __shfl_sync(mask, qv.x, 0);
        qy = __shfl_sync(mask, qv.y, 0);
    }
    __syncwarp(); // Ensure all lanes have qx, qy (mainly for readability; warps execute in lock-step)

    // Per-lane top-L container (ascending order: best (smallest) at [0], worst at [L-1])
    const int L = k >> 5; // k / 32, guaranteed >= 1 by problem constraints
    float bestDists[32];
    int   bestIdx[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (i < L) {
            bestDists[i] = CUDART_INF_F;
            bestIdx[i]   = -1;
        }
    }

    // Process the data in tiles loaded into shared memory by the entire block
    for (int base = 0; base < data_count; base += tile_points) {
        int tileCount = tile_points;
        if (base + tileCount > data_count) tileCount = data_count - base;

        // Cooperative load: each thread in the block loads some float2s
        for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
            smem[i] = data[base + i];
        }
        __syncthreads();

        // Each warp computes distances from its query to all points in the tile
        if (active) {
            for (int i = lane; i < tileCount; i += warpSize_) {
                float d = sqdist2D(smem[i], qx, qy);

                // Fast reject if not better than current worst
                if (d < bestDists[L - 1]) {
                    // Insert d into ascending-sorted list bestDists[0..L-1], shifting larger elements right.
                    int gidx = base + i;
                    int j = L - 1;
                    // Move larger elements one position to the right
                    while (j > 0 && d < bestDists[j - 1]) {
                        bestDists[j] = bestDists[j - 1];
                        bestIdx[j]   = bestIdx[j - 1];
                        --j;
                    }
                    bestDists[j] = d;
                    bestIdx[j]   = gidx;
                }
            }
        }
        __syncthreads();
    }

    // Final multiway merge across lanes of the warp:
    // Each lane provides its next smallest candidate from its local sorted list.
    // We repeat k times to select the global top-k in ascending order.
    if (active) {
        int outBase = qid * k;
        int ptr = 0; // index into this lane's bestDists/bestIdx

        unsigned mask = full_warp_mask();
        for (int out = 0; out < k; ++out) {
            float myCand = (ptr < L) ? bestDists[ptr] : CUDART_INF_F;
            // Find the winning lane with the smallest candidate
            MinValLane m = warp_argmin(myCand);
            int winLane = m.lane;
            float winVal = m.val;

            // Broadcast the corresponding index from the winning lane
            int myCandIdx = (ptr < L) ? bestIdx[ptr] : -1;
            int winIdx = __shfl_sync(mask, myCandIdx, winLane);

            if (lane == 0) {
                result[outBase + out].first  = winIdx;
                result[outBase + out].second = winVal;
            }
            // The lane that contributed the winner advances its pointer
            if (lane == winLane) {
                ++ptr;
            }
        }
    }
}

void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Tunable parameters: threads per block and shared-memory tile size (in points).
    //  - Use 256 threads per block (8 warps / block) to balance occupancy and shared memory usage.
    //  - Use a tile of 4096 points (32 KB at 8 bytes per float2) to keep SMEM within typical limits.
    const int threads_per_block = 256;
    const int warps_per_block   = threads_per_block / 32;
    const int tile_points       = 4096;

    // Grid size: one warp per query
    const int total_warps = (query_count + warps_per_block - 1) / warps_per_block;
    dim3 grid(total_warps, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    // Dynamic shared memory size: tile_points float2s
    size_t shmem_bytes = static_cast<size_t>(tile_points) * sizeof(float2);

    // Launch kernel. Reinterpret the result pointer to a simple POD with identical layout.
    knn2d_kernel<<<grid, block, shmem_bytes>>>(
        query, query_count,
        data, data_count,
        reinterpret_cast<PairIF*>(result),
        k,
        tile_points
    );

    // Note: Error checking is intentionally omitted; callers may insert cudaGetLastError()/cudaDeviceSynchronize() as needed.
}