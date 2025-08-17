#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdint.h>

// This implementation computes k-NN for 2D points where each query is handled by a single warp (32 threads).
// It follows the requirements:
// - Batches (tiles) of data points are cooperatively loaded into shared memory by the full block.
// - Each warp processes one query and caches candidate neighbors in a per-warp shared memory buffer.
// - Candidates closer than the current max_distance (distance of the k-th best) are added using atomicAdd
//   (warp-aggregated) to a per-warp candidate buffer. Whenever the buffer is full, it is merged with the
//   intermediate result using warp-cooperative routines (reductions via warp shuffles).
// - After all batches are processed, if the candidate buffer is not empty, a final merge is performed.
// - The intermediate result (top-k) is kept unsorted during processing. At the end, results are written
//   to global memory in ascending distance order using repeated warp-cooperative arg-min selection.
//
// Performance notes:
// - Per-warp intermediate result and candidate buffer are placed in shared memory to avoid global memory round-trips.
// - Memory footprint per warp is ~16*k bytes (best + candidates), with k in [32, 1024]. With 8 warps per block and
//   tile size 4096 points, the total dynamic shared memory used is below the 164KB per-block limit on A100, and also
//   well within H100 limits.
// - Warp-cooperative reductions are used to find argmax/argmin efficiently.
// - Warp-aggregated atomicAdd is used to reduce contention when appending candidates.

// Simple POD equivalent of std::pair<int,float> for device-side writes.
// The host run_knn casts the std::pair<int,float>* to this type. Layout is identical on typical ABIs.
struct PairIF {
    int   first;
    float second;
};

// Warp size is assumed 32 on NVIDIA GPUs.
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Utility: get lane id in a warp
static __device__ __forceinline__ int lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}

// Utility: get warp id within the block
static __device__ __forceinline__ int warp_id_in_block() {
    return threadIdx.x >> 5;
}

// Utility: full warp mask
static __device__ __forceinline__ unsigned full_mask() {
    return 0xFFFFFFFFu;
}

// Warp-cooperative argmax over an array "arr" of length k stored in shared memory.
// Each lane scans a strided subset and keeps its local max, then we reduce across the warp.
// Returns (maxVal, maxIdx) via references. Only lane 0 holds meaningful outputs;
// other lanes receive broadcasted values at the end for convenience.
static __device__ __forceinline__ void warp_argmax_shared(const float* arr, int k, float& outVal, int& outIdx) {
    const int lane = lane_id();
    float localMax = -CUDART_INF_F;
    int localIdx = -1;
    for (int i = lane; i < k; i += WARP_SIZE) {
        float v = arr[i];
        if (v > localMax) {
            localMax = v;
            localIdx = i;
        }
    }
    // Warp reduction: argmax
    unsigned mask = full_mask();
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float othV = __shfl_down_sync(mask, localMax, offset);
        int   othI = __shfl_down_sync(mask, localIdx, offset);
        if (othV > localMax) {
            localMax = othV;
            localIdx = othI;
        }
    }
    // Broadcast lane 0 result to all lanes so every lane can read the current max if needed.
    outVal = __shfl_sync(mask, localMax, 0);
    outIdx = __shfl_sync(mask, localIdx, 0);
}

// Warp-cooperative argmin over an array "arr" of length k stored in shared memory.
// Each lane scans a strided subset and keeps its local min, then reduce across the warp.
static __device__ __forceinline__ void warp_argmin_shared(const float* arr, int k, float& outVal, int& outIdx) {
    const int lane = lane_id();
    float localMin = CUDART_INF_F;
    int localIdx = -1;
    for (int i = lane; i < k; i += WARP_SIZE) {
        float v = arr[i];
        if (v < localMin) {
            localMin = v;
            localIdx = i;
        }
    }
    unsigned mask = full_mask();
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float othV = __shfl_down_sync(mask, localMin, offset);
        int   othI = __shfl_down_sync(mask, localIdx, offset);
        if (othV < localMin) {
            localMin = othV;
            localIdx = othI;
        }
    }
    outVal = __shfl_sync(mask, localMin, 0);
    outIdx = __shfl_sync(mask, localIdx, 0);
}

// Merge per-warp candidate buffer into per-warp intermediate top-k result.
// - bestIdx, bestDist: arrays of length k (shared), storing current top-k (unsorted).
// - candIdx, candDist: arrays of length >= candCount (shared), storing candidates.
// - candCount: number of candidates in the buffer (shared).
// - maxDistRef: in/out, distance of current k-th nearest (the maximum in bestDist).
// Implementation details:
// - We maintain bestDist/Idx as an unsorted set of k best. We repeatedly select candidates one-by-one,
//   and if a candidate distance is less than current maxDist, we replace the worst element.
// - The "worst" (max) is found via warp-cooperative argmax reduction.
// - Complexity per merge is O(k * candCount / WARP_SIZE) comparisons, acceptable for k<=1024.
static __device__ __forceinline__ void merge_candidates_warp(
    int* bestIdx, float* bestDist, int k,
    const int* candIdx, const float* candDist, int& candCount,
    float& maxDistRef)
{
    // Compute current worst (max) among bestDist
    float curMax;
    int curMaxIdx;
    warp_argmax_shared(bestDist, k, curMax, curMaxIdx);

    // Process all candidates sequentially; lane 0 performs stores, warp helps to find max when replaced.
    // We read cand arrays directly; they belong exclusively to this warp's region in shared memory.
    const int lane = lane_id();
    for (int t = 0; t < candCount; ++t) {
        // Only lane 0 performs the comparison and potential replacement using the current curMax.
        float cd;
        int ci;
        if (lane == 0) {
            cd = candDist[t];
            ci = candIdx[t];
            // Insert if closer than current k-th nearest
            if (cd < curMax) {
                bestDist[curMaxIdx] = cd;
                bestIdx[curMaxIdx]  = ci;
            }
        }
        // Broadcast lane0's decision to all lanes: we need to know if replacement happened to refresh curMax.
        unsigned mask = full_mask();
        float cd_b = __shfl_sync(mask, cd, 0);
        int   ci_b = __shfl_sync(mask, ci, 0);
        bool replaced = __shfl_sync(mask, (int)(cd_b < curMax), 0) != 0;

        // If replaced, recompute the current worst among bestDist; else keep previous curMax.
        if (replaced) {
            // Ensure the replacement is visible to all lanes before reading bestDist
            __syncwarp();
            warp_argmax_shared(bestDist, k, curMax, curMaxIdx);
        }
        // Otherwise, no change to curMax/curMaxIdx
    }

    // Update shared max distance reference and reset candidate count.
    if (lane == 0) {
        maxDistRef = curMax;
        candCount = 0;
    }
    __syncwarp();
}

// KNN kernel: each warp processes one query point.
// Shared memory layout (dynamic):
// [0]                          : float2 tilePoints[TILE_POINTS]
// [tilePoints * sizeof(float2)]: per-warp arrays:
//   bestDist  [warpsPerBlock * k] (float)
//   bestIdx   [warpsPerBlock * k] (int)
//   candDist  [warpsPerBlock * k] (float)
//   candIdx   [warpsPerBlock * k] (int)
//   candCount [warpsPerBlock]     (int)
//   maxDist   [warpsPerBlock]     (float)
__global__ void knn_kernel_2d(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    PairIF* __restrict__ result, int k,
    int tile_points) // number of points per tile loaded into shared memory
{
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_ptr = smem_raw;

    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    const int warpInBlock   = warp_id_in_block();
    const int lane          = lane_id();

    // Per-block shared tiles for data points
    float2* tile = reinterpret_cast<float2*>(smem_ptr);
    size_t tile_bytes = (size_t)tile_points * sizeof(float2);
    smem_ptr += tile_bytes;

    // Helper lambda to align pointer to alignment bytes
    auto align_ptr = [&smem_ptr](size_t alignment) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(smem_ptr);
        size_t mis = addr % alignment;
        if (mis != 0) smem_ptr += (alignment - mis);
    };

    // Align to 16 bytes for safety
    align_ptr(16);

    // Per-warp arrays in shared memory
    float* bestDist_all = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += (size_t)warpsPerBlock * k * sizeof(float);
    int*   bestIdx_all  = reinterpret_cast<int*>(smem_ptr);
    smem_ptr += (size_t)warpsPerBlock * k * sizeof(int);
    float* candDist_all = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += (size_t)warpsPerBlock * k * sizeof(float);
    int*   candIdx_all  = reinterpret_cast<int*>(smem_ptr);
    smem_ptr += (size_t)warpsPerBlock * k * sizeof(int);

    // Per-warp counters and max distance
    int*   candCount_all = reinterpret_cast<int*>(smem_ptr);
    smem_ptr += (size_t)warpsPerBlock * sizeof(int);
    float* maxDist_all   = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += (size_t)warpsPerBlock * sizeof(float);

    // Compute global warp id and query index handled by this warp
    int warpGlobal = (blockIdx.x * warpsPerBlock) + warpInBlock;
    if (warpGlobal >= query_count) return; // nothing to do

    // Pointers to this warp's slices
    float* bestDist = bestDist_all + (size_t)warpInBlock * k;
    int*   bestIdx  = bestIdx_all  + (size_t)warpInBlock * k;
    float* candDist = candDist_all + (size_t)warpInBlock * k;
    int*   candIdx  = candIdx_all  + (size_t)warpInBlock * k;
    int&   candCountRef = candCount_all[warpInBlock];
    float& maxDistRef   = maxDist_all[warpInBlock];

    // Initialize per-warp intermediate result to +inf distances and -1 indices
    for (int i = lane; i < k; i += WARP_SIZE) {
        bestDist[i] = CUDART_INF_F;
        bestIdx[i]  = -1;
    }
    if (lane == 0) {
        candCountRef = 0;
        maxDistRef = CUDART_INF_F; // current "k-th nearest" distance
    }
    __syncwarp();

    // Load this warp's query point
    float2 q = query[warpGlobal];
    float qx = q.x, qy = q.y;

    // Process data points in tiles
    for (int tileBase = 0; tileBase < data_count; tileBase += tile_points) {
        int count = data_count - tileBase;
        if (count > tile_points) count = tile_points;

        // Block-cooperative load of the tile into shared memory
        for (int idx = threadIdx.x; idx < count; idx += blockDim.x) {
            tile[idx] = data[tileBase + idx];
        }
        __syncthreads(); // ensure the tile is fully loaded

        // Each warp processes all points in the tile
        for (int i = lane; i < count; i += WARP_SIZE) {
            float2 p = tile[i];
            float dx = p.x - qx;
            float dy = p.y - qy;
            float dist = fmaf(dx, dx, dy * dy); // squared L2 distance
            int   gidx = tileBase + i;

            // Determine if this distance is a candidate (strictly closer than current maxDist)
            float curMax = maxDistRef; // read current shared maxDist for this warp
            int want = (dist < curMax) ? 1 : 0;

            // Build a mask of lanes that want to insert for this "round"
            unsigned mask = __ballot_sync(full_mask(), want != 0);
            int nToInsert = __popc(mask);

            if (nToInsert > 0) {
                // Ensure there is enough space in the candidate buffer. If not, merge first.
                if (lane == 0) {
                    int cc = candCountRef;
                    if (cc + nToInsert > k) {
                        // Merge before we insert these candidates to free up space and update maxDist
                        // Note: This merge uses only warp-local structures; no need for block sync.
                        merge_candidates_warp(bestIdx, bestDist, k, candIdx, candDist, candCountRef, maxDistRef);
                    }
                }
                __syncwarp();
                // After merge, maxDist may have changed; recompute predicate and mask
                curMax = maxDistRef;
                want = (dist < curMax) ? 1 : 0;
                mask = __ballot_sync(full_mask(), want != 0);
                nToInsert = __popc(mask);

                if (nToInsert > 0) {
                    // Warp-aggregated atomicAdd to reserve a contiguous slice in candidate buffer
                    int base = 0;
                    if (lane == 0) {
                        base = atomicAdd(&candCountRef, nToInsert);
                    }
                    base = __shfl_sync(full_mask(), base, 0);

                    // Compute each lane's offset among the inserting lanes
                    int rank = __popc(mask & ((1u << lane) - 1));
                    if (want) {
                        int pos = base + rank;
                        if (pos < k) {
                            candDist[pos] = dist;
                            candIdx[pos]  = gidx;
                        }
                    }
                    __syncwarp();

                    // If we just filled the buffer exactly to capacity, trigger a merge now.
                    if (lane == 0) {
                        if (candCountRef >= k) {
                            merge_candidates_warp(bestIdx, bestDist, k, candIdx, candDist, candCountRef, maxDistRef);
                        }
                    }
                    __syncwarp();
                }
            }
        } // end tile inner loop

        __syncthreads(); // ensure all warps are done reading the tile before loading the next
    } // end tiles loop

    // Final merge if any candidates remain
    if (lane == 0) {
        if (candCountRef > 0) {
            merge_candidates_warp(bestIdx, bestDist, k, candIdx, candDist, candCountRef, maxDistRef);
        }
    }
    __syncwarp();

    // At this point, bestDist/bestIdx hold the k nearest neighbors (unsorted).
    // We must output them in ascending distance order. We perform k repeated warp-cooperative argmin selections.
    // After selecting a min at position selIdx, we mark bestDist[selIdx] = +inf to exclude it from further selections.
    // Lane 0 writes the outputs to global memory.
    int outBase = warpGlobal * k;
    for (int j = 0; j < k; ++j) {
        float mnVal;
        int mnIdx;
        warp_argmin_shared(bestDist, k, mnVal, mnIdx);
        // Write result j (j-th nearest neighbor)
        if (lane == 0) {
            PairIF out;
            out.first  = bestIdx[mnIdx];
            out.second = mnVal;
            result[outBase + j] = out;
            // Mark as used
            bestDist[mnIdx] = CUDART_INF_F;
        }
        __syncwarp();
    }
}

// Host interface
extern "C" void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Choose block and grid configuration.
    // We use 256 threads per block (8 warps). Each warp processes one query.
    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock / WARP_SIZE;

    // Compute number of blocks to cover all queries (one warp per query).
    int totalWarps = query_count;
    int blocks = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Prefer tile size of 4096 points for good bandwidth, adjust down if shared memory is tight.
    int tilePoints = 4096;

    // Compute dynamic shared memory requirement:
    // tilePoints * sizeof(float2) + warpsPerBlock * (2*k*(sizeof(float)+sizeof(int)) + sizeof(int) + sizeof(float))
    size_t perWarpMem = (size_t)2 * k * (sizeof(float) + sizeof(int)) + sizeof(int) + sizeof(float);
    size_t smemBytes = (size_t)tilePoints * sizeof(float2) + (size_t)warpsPerBlock * perWarpMem;

    // Query device shared memory limits to ensure our configuration fits.
    int device = 0;
    cudaGetDevice(&device);
    int maxOptin = 0;
    cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (maxOptin <= 0) {
        // Fallback to legacy limit if opt-in attribute is not exposed
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        maxOptin = (int)prop.sharedMemPerBlockOptin;
        if (maxOptin <= 0) maxOptin = (int)prop.sharedMemPerBlock;
    }

    // Adjust tilePoints down if needed to fit within the maximum dynamic shared memory
    // Reserve a small alignment cushion (e.g., 256 bytes).
    const size_t cushion = 256;
    if (smemBytes + cushion > (size_t)maxOptin) {
        size_t maxForTile = (size_t)maxOptin - (size_t)warpsPerBlock * perWarpMem;
        if ((long long)maxForTile < 0) maxForTile = 0;
        tilePoints = (int)(maxForTile / sizeof(float2));
        // Make tilePoints a positive multiple of threadsPerBlock for balanced loads
        if (tilePoints < threadsPerBlock) tilePoints = threadsPerBlock;
        tilePoints = (tilePoints / threadsPerBlock) * threadsPerBlock;
        if (tilePoints <= 0) tilePoints = threadsPerBlock; // ensure positivity
        smemBytes = (size_t)tilePoints * sizeof(float2) + (size_t)warpsPerBlock * perWarpMem;
    }

    // Opt-in to the required dynamic shared memory size.
    cudaFuncSetAttribute(knn_kernel_2d, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
    // Prefer shared memory carveout (100% if possible) to maximize shared memory availability.
    cudaFuncSetAttribute(knn_kernel_2d, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    // Launch the kernel
    PairIF* result_alias = reinterpret_cast<PairIF*>(result);
    knn_kernel_2d<<<blocks, threadsPerBlock, smemBytes>>>(query, query_count, data, data_count, result_alias, k, tilePoints);
}