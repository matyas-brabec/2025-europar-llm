#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// This implementation assigns exactly one warp (32 threads) to each query point.
// It processes the dataset in tiles cached in shared memory. Each warp maintains
// a private k-NN result set distributed across its 32 lanes in registers: each lane
// holds k/32 entries. Distances for 32 candidates are computed in parallel,
// then the warp collaboratively inserts the best candidates into the current k-NN
// set by repeatedly identifying the smallest new candidate and replacing the
// current global worst (largest) entry in the k-NN set. At the end, each warp
// writes its k results to shared memory and performs a warp-scope bitonic sort
// to produce results sorted by ascending distance, then writes to global memory.

struct PairIF {
    int   first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "PairIF must match std::pair<int,float> layout");

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Utilities for warp-level operations.
static __forceinline__ __device__ unsigned warp_full_mask() {
    return 0xFFFFFFFFu;
}

static __forceinline__ __device__ int lane_id() {
#if __CUDACC_VER_MAJOR__ >= 9
    return threadIdx.x & (WARP_SIZE - 1);
#else
    return threadIdx.x % WARP_SIZE;
#endif
}

struct ArgPairF {
    float val;
    int   idx;
};

// Warp-wide argmax reduction: returns (max value, lane index containing it) broadcast to all lanes.
static __forceinline__ __device__ ArgPairF warp_argmax(float v) {
    unsigned mask = warp_full_mask();
    int l = lane_id();
    float maxv = v;
    int   maxi = l;
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(mask, maxv, offset);
        int   oi = __shfl_down_sync(mask, maxi, offset);
        if (ov > maxv) {
            maxv = ov;
            maxi = oi;
        }
    }
    ArgPairF r;
    r.val = __shfl_sync(mask, maxv, 0);
    r.idx = __shfl_sync(mask, maxi, 0);
    return r;
}

// Warp-wide argmin reduction: returns (min value, lane index containing it) broadcast to all lanes.
static __forceinline__ __device__ ArgPairF warp_argmin(float v) {
    unsigned mask = warp_full_mask();
    int l = lane_id();
    float minv = v;
    int   mini = l;
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(mask, minv, offset);
        int   oi = __shfl_down_sync(mask, mini, offset);
        if (ov < minv) {
            minv = ov;
            mini = oi;
        }
    }
    ArgPairF r;
    r.val = __shfl_sync(mask, minv, 0);
    r.idx = __shfl_sync(mask, mini, 0);
    return r;
}

// Find local argmax across S entries in register arrays.
template <int MAX_S>
static __forceinline__ __device__
void local_argmax(const float (&dist)[MAX_S], int S, float &maxv, int &maxp) {
    maxv = -CUDART_INF_F;
    maxp = 0;
#pragma unroll
    for (int i = 0; i < MAX_S; ++i) {
        if (i < S) {
            float v = dist[i];
            if (v > maxv) {
                maxv = v;
                maxp = i;
            }
        }
    }
}

// Copy per-lane stripe of k results to warp scratch shared memory.
// Each lane L writes S entries at positions [L*S .. L*S+S-1].
template <int MAX_S>
static __forceinline__ __device__
void write_warp_scratch(PairIF *warpScratch, int lane, const int (&idx)[MAX_S], const float (&dist)[MAX_S], int S) {
#pragma unroll
    for (int i = 0; i < MAX_S; ++i) {
        if (i < S) {
            int pos = lane * S + i;
            warpScratch[pos].first  = idx[i];
            warpScratch[pos].second = dist[i];
        }
    }
}

// Warp-scope bitonic sort for PairIF array in shared memory.
// - count must be a power of two (here, k).
// - Sorting by ascending .second (distance).
static __forceinline__ __device__
void warp_bitonic_sort_pairs(PairIF *arr, int count) {
    unsigned mask = warp_full_mask();
    int lane = lane_id();
    // Standard in-place bitonic sort; threads in a warp cooperatively process all indices.
    for (int k = 2; k <= count; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each thread handles multiple indices in strides of WARP_SIZE.
            for (int i = lane; i < count; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    PairIF ai = arr[i];
                    PairIF aj = arr[ixj];
                    bool swap = up ? (ai.second > aj.second) : (ai.second < aj.second);
                    if (swap) {
                        arr[i]   = aj;
                        arr[ixj] = ai;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

/// @FIXED
/// extern "C" __global__
__global__
void knn_kernel_2d(const float2 * __restrict__ query,
                   int query_count,
                   const float2 * __restrict__ data,
                   int data_count,
                   PairIF * __restrict__ result,
                   int k,
                   int tile_points,
                   int warps_per_block)
{
    // Dynamic shared memory layout:
    // [0 .. tile_points-1] float2 tile of data points
    // [tile_points .. tile_points + warps_per_block*k - 1] PairIF per-warp scratch
    extern __shared__ unsigned char smem_uchar[];
    float2 *s_points = reinterpret_cast<float2*>(smem_uchar);
    PairIF *s_pairs  = reinterpret_cast<PairIF*>(s_points + tile_points);

    int tid  = threadIdx.x;
    int lane = lane_id();
    int warp = tid >> 5;

    // Query index handled by this warp.
    int qidx = blockIdx.x * warps_per_block + warp;
    bool has_query = (qidx < query_count);

    // Number of per-lane slots in k-NN set.
    // k is guaranteed to be a power of two between 32 and 1024 inclusive.
    const int MAX_S = 32;
    int S = k >> 5; // k / 32
    // Intermediate k-NN kept in registers: lane-striped layout, S entries per lane.
    float best_dist[MAX_S];
    int   best_idx [MAX_S];
#pragma unroll
    for (int i = 0; i < MAX_S; ++i) {
        if (i < S) {
            best_dist[i] = CUDART_INF_F;
            best_idx[i]  = -1;
        }
    }

    // Load query point into registers and broadcast to all lanes in this warp.
    float qx = 0.0f, qy = 0.0f;
    if (has_query && lane == 0) {
        float2 q = query[qidx];
        qx = q.x; qy = q.y;
    }
    unsigned full = warp_full_mask();
    qx = __shfl_sync(full, qx, 0);
    qy = __shfl_sync(full, qy, 0);

    // Process data in tiles.
    for (int base = 0; base < data_count; base += tile_points) {
        int tile_count = data_count - base;
        if (tile_count > tile_points) tile_count = tile_points;

        // Load tile into shared memory cooperatively across the whole block.
        for (int i = tid; i < tile_count; i += blockDim.x) {
            s_points[i] = data[base + i];
        }
        __syncthreads();

        // Each warp processes this tile against its query point.
        if (has_query) {
            // Process the tile in groups of 32 candidates so each lane contributes one candidate.
            for (int offset = 0; offset < tile_count; offset += WARP_SIZE) {
                int local_idx = offset + lane;

                float cand_dist = CUDART_INF_F;
                int   cand_idx  = -1;

                if (local_idx < tile_count) {
                    float2 p = s_points[local_idx];
                    float dx = qx - p.x;
                    float dy = qy - p.y;
                    cand_dist = dx * dx + dy * dy;
                    cand_idx  = base + local_idx;
                }

                // Compute current global threshold T = maximum distance among the current k-NN set.
                float lmaxv;
                int   lmaxp;
                local_argmax<MAX_S>(best_dist, S, lmaxv, lmaxp);
                ArgPairF gmax = warp_argmax(lmaxv);
                float T = gmax.val;

                // Determine which candidates beat the current threshold.
                bool pass = (cand_dist < T);
                unsigned passmask = __ballot_sync(full, pass);

                // Insert all passing candidates one-by-one in order of increasing distance.
                while (passmask) {
                    float v = pass ? cand_dist : CUDART_INF_F;
                    ArgPairF mn = warp_argmin(v);
                    int bestLane = mn.idx;
                    float bestVal = mn.val;
                    int bestId = __shfl_sync(full, cand_idx, bestLane);

                    // Find current global worst location (lane and position).
                    local_argmax<MAX_S>(best_dist, S, lmaxv, lmaxp);
                    gmax = warp_argmax(lmaxv);
                    int worstLane = gmax.idx;
                    int worstPos  = __shfl_sync(full, lmaxp, worstLane);

                    // Replace worst with the best candidate.
                    if (lane == worstLane) {
                        best_dist[worstPos] = bestVal;
                        best_idx [worstPos] = bestId;
                    }

                    // Remove this candidate from further consideration in this group.
                    if (lane == bestLane) {
                        pass = false;
                        cand_dist = CUDART_INF_F;
                    }
                    passmask = __ballot_sync(full, pass);
                }
            }
        }

        __syncthreads(); // Synchronize before next tile load.
    }

    // After processing all tiles, each warp has its k results distributed across lanes.
    // Move to shared memory scratch private to the warp and sort ascending by distance.
    if (has_query) {
        PairIF *warpScratch = s_pairs + warp * k;

        // Write in lane-striped order: positions lane*S + i.
        write_warp_scratch<MAX_S>(warpScratch, lane, best_idx, best_dist, S);
        __syncwarp(full);

        // Warp-scope bitonic sort on k items in shared memory.
        warp_bitonic_sort_pairs(warpScratch, k);
        __syncwarp(full);

        // Write sorted results to global memory at positions result[qidx * k + j].
        PairIF *dst = result + qidx * k;
        for (int i = lane; i < k; i += WARP_SIZE) {
            dst[i] = warpScratch[i];
        }
    }
}

// Host-side launcher that configures tile size and block dimensions based on available shared memory.
// - Uses dynamic shared memory for both the data tile and per-warp sorting scratch.
// - Attempts to opt-in to maximum dynamic shared memory per block for data center GPUs (A100/H100).
/// @FIXED
/// extern "C"

void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Determine device properties and available dynamic shared memory.
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxOptin = 0;
    /// @FIXED
    /// cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxDynamicSharedMemoryPerBlockOptin, device);
    cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    size_t maxDynSmem = (maxOptin > 0) ? (size_t)maxOptin : (size_t)prop.sharedMemPerBlock;

    // Choose number of warps per block to balance occupancy and shared memory usage.
    // Try from 8 warps down to 1, ensuring we can allocate at least 32 points per tile.
    int chosenWarpsPerBlock = 8;
    int threadsPerBlock = 0;
    int tile_points = 0;

    // We need shared memory for:
    // - tile_points * sizeof(float2)
    // - warps_per_block * k * sizeof(PairIF)  (per-warp sorting scratch)
    for (int wpb = 8; wpb >= 1; wpb >>= 1) {
        int tpb = wpb * WARP_SIZE;
        if (tpb > prop.maxThreadsPerBlock) continue;

        size_t scratchBytes = (size_t)wpb * (size_t)k * sizeof(PairIF);
        if (scratchBytes >= maxDynSmem) continue; // no room for tile

        int tpoints = (int)((maxDynSmem - scratchBytes) / sizeof(float2));
        if (tpoints >= WARP_SIZE) {
            chosenWarpsPerBlock = wpb;
            threadsPerBlock = tpb;
            tile_points = tpoints;
            break;
        }
    }

    if (threadsPerBlock == 0) {
        // Fallback: minimal configuration with 1 warp; tile points at least 32 or as many as fit.
        chosenWarpsPerBlock = 1;
        threadsPerBlock = WARP_SIZE;
        size_t scratchBytes = (size_t)chosenWarpsPerBlock * (size_t)k * sizeof(PairIF);
        size_t remaining = (maxDynSmem > scratchBytes) ? (maxDynSmem - scratchBytes) : 0;
        tile_points = (int)(remaining / sizeof(float2));
        if (tile_points < WARP_SIZE) tile_points = WARP_SIZE; // ensure progress
    }

    // Total dynamic shared memory required for kernel launch.
    size_t smemBytes = (size_t)tile_points * sizeof(float2)
                     + (size_t)chosenWarpsPerBlock * (size_t)k * sizeof(PairIF);

    // Opt-in to large dynamic shared memory if supported.
    cudaFuncSetAttribute(knn_kernel_2d, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
    cudaFuncSetCacheConfig(knn_kernel_2d, cudaFuncCachePreferShared);

    int blocks = (query_count + chosenWarpsPerBlock - 1) / chosenWarpsPerBlock;

    // Launch kernel. The result pointer is reinterpreted as PairIF to match device layout.
    knn_kernel_2d<<<blocks, threadsPerBlock, smemBytes>>>(
        query, query_count,
        data, data_count,
        reinterpret_cast<PairIF*>(result),
        k,
        tile_points,
        chosenWarpsPerBlock
    );
}
