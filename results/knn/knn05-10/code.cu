#include <cuda_runtime.h>
#include <utility>

// This implementation computes k-nearest neighbors (k-NN) for 2D points using a warp-per-query design.
// - One warp (32 threads) processes one query.
// - All data points are processed in batches (tiles) loaded into shared memory by the whole block.
// - Each warp maintains:
//    * An intermediate top-k result (indices + distances) as a "private" copy distributed across its 32 lanes in registers.
//      Each lane stores k/32 entries (with k in [32, 1024], power-of-two, so k/32 <= 32).
//    * A per-warp candidate buffer of size k (indices + distances) in shared memory.
// - For each data batch:
//    * The block loads a tile of points to shared memory.
//    * Each warp computes distances from its query to all points in the tile and selectively appends better candidates
//      (distances smaller than the current k-th best) to its buffer using warp-aggregated push.
//    * When a buffer is full, the warp merges its candidates with the intermediate top-k.
// - After all batches, if a buffer is non-empty, it merges remaining candidates.
// - Finally, each warp sorts its intermediate top-k in ascending distance (selection via warp reductions) and writes the
//   results as std::pair<int,float> to the output.
//
// Design details and optimizations:
// - Distances are squared L2. Single-precision floats with FMA used where possible (fmaf).
// - Intermediate top-k is stored entirely in per-lane registers (arrays of length up to 32).
// - The k-th best distance ("worst" within the current top-k) is tracked warp-wide and used as threshold.
// - Merging: candidates are processed sequentially per warp; replacement targets are found by:
//     (a) each lane keeps (and recomputes when changed) the max of its local segment,
//     (b) warp-wide argmax reduction finds the global worst to replace.
//   This yields small per-candidate overhead with k/32 <= 32.
// - Candidate buffer append uses warp-aggregated compaction with ballot/popc/rank and safe handling when the buffer
//   overflows; flushing triggers a merge followed by re-evaluation of remaining candidates against the new threshold.
// - Shared memory usage per block = warpsPerBlock * k * sizeof(Candidate) + tilePoints * sizeof(float2).
//   Host code queries device limits and chooses warpsPerBlock and tilePoints that fit. It also opts-in to high dynamic
//   shared memory when supported (A100/H100).
// - Threads per block = warpsPerBlock * 32. A good default is 8 warps per block (256 threads), with fallbacks to 4/2/1
//   if shared memory is tight for large k or constrained devices.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

struct Candidate {
    float dist;
    int   idx;
};

// Utility: per-warp lane id and warp id within a block
static __device__ __forceinline__ int lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}
static __device__ __forceinline__ int warp_id_in_block() {
    return threadIdx.x >> 5;
}

// Utility: return a mask of bits less than lane id (for prefix rank)
static __device__ __forceinline__ unsigned lane_mask_lt() {
    return (1u << lane_id()) - 1u;
}

// Warp-wide argmax reduction on values, returning both maximum value and owning lane.
// Ties are broken in favor of higher lane indices to keep deterministic behavior.
struct ArgMaxPair {
    float val;
    int lane;
};
static __device__ __forceinline__ ArgMaxPair warp_argmax(float v, unsigned mask) {
    ArgMaxPair a{v, lane_id()};
    // Tree reduction using shfl_down. Only uses active lanes specified by 'mask'.
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float v_other = __shfl_down_sync(mask, a.val, offset);
        int   l_other = __shfl_down_sync(mask, a.lane, offset);
        if (v_other > a.val || (v_other == a.val && l_other > a.lane)) {
            a.val  = v_other;
            a.lane = l_other;
        }
    }
    // Broadcast the final result to all lanes (from lane 0 after reduction)
    a.val  = __shfl_sync(mask, a.val, 0);
    a.lane = __shfl_sync(mask, a.lane, 0);
    return a;
}

// Warp-wide max reduction (returns maximum value)
static __device__ __forceinline__ float warp_max(float v, unsigned mask) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float v_other = __shfl_down_sync(mask, v, offset);
        v = v > v_other ? v : v_other;
    }
    return __shfl_sync(mask, v, 0);
}

// Merge a warp's candidate buffer (size 'bufCount') into the warp's private top-k.
// - resDist/resIdx are register arrays local to each lane with length 'itemsPerLane' (k/32).
// - The function updates warpWorstDist to the new k-th best distance after merging.
// - bufCount is set to 0 at the end.
template <int WARPS_PER_BLOCK>
static __device__ __forceinline__
void merge_buffer_with_topk(Candidate* cand_base, int k, int& bufCount,
                            float* resDist, int* resIdx, int itemsPerLane,
                            float& warpWorstDist) {
    const unsigned full_mask = 0xFFFFFFFFu;
    // Compute each lane's local maximum (worst in its segment).
    float localWorst = resDist[0];
    int localWorstPos = 0;
    #pragma unroll
    for (int i = 1; i < 32; ++i) {
        if (i >= itemsPerLane) break;
        float v = resDist[i];
        if (v > localWorst) { localWorst = v; localWorstPos = i; }
    }
    // Initialize warp-wide worst
    ArgMaxPair am = warp_argmax(localWorst, full_mask);
    warpWorstDist = am.val;

    // Sequentially process candidates in the buffer. Each iteration reads one candidate and attempts replacement.
    for (int i = 0; i < bufCount; ++i) {
        int reader_lane = i & (WARP_SIZE - 1);
        float candDist = __shfl_sync(full_mask, cand_base[i].dist, reader_lane);
        int   candIdx  = __shfl_sync(full_mask, cand_base[i].idx,  reader_lane);

        // Skip if not better than current k-th best
        if (!(candDist < warpWorstDist)) continue;

        // Find which lane owns the global worst entry
        ArgMaxPair gw = warp_argmax(localWorst, full_mask);
        int ownerLane = gw.lane;
        int ownerPos  = __shfl_sync(full_mask, localWorstPos, ownerLane);

        // Owner lane replaces its worst entry with the candidate and recomputes its local worst
        if (lane_id() == ownerLane) {
            resDist[ownerPos] = candDist;
            resIdx[ownerPos]  = candIdx;
            // Recompute local worst after update
            float lw = resDist[0];
            int lp = 0;
            #pragma unroll
            for (int t = 1; t < 32; ++t) {
                if (t >= itemsPerLane) break;
                float val = resDist[t];
                if (val > lw) { lw = val; lp = t; }
            }
            localWorst = lw;
            localWorstPos = lp;
        }
        // Update global worst
        warpWorstDist = warp_max(localWorst, full_mask);
    }
    // Clear the buffer
    if (lane_id() == 0) bufCount = 0;
    __syncwarp();
}

// The main kernel template. WARPS_PER_BLOCK must be 1, 2, 4, or 8 (256 threads = 8 warps is a good default).
template <int WARPS_PER_BLOCK>
__global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                           const float2* __restrict__ data,  int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k, int tilePoints) {
    static_assert(WARPS_PER_BLOCK >= 1 && WARPS_PER_BLOCK <= 8, "WARPS_PER_BLOCK in [1,8]");
    static_assert(WARP_SIZE == 32, "This kernel assumes warp size is 32");

    const int lane = lane_id();
    const int warpInBlock = warp_id_in_block();
    const int warpGlobal = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;

    const unsigned full_mask = 0xFFFFFFFFu;

    // Shared memory layout:
    // [ per-warp candidate buffers (WARPS_PER_BLOCK * k entries of Candidate) ] [ data tile (tilePoints float2) ]
    extern __shared__ unsigned char smem_raw[];
    Candidate* smemCandidates = reinterpret_cast<Candidate*>(smem_raw);
    Candidate* warpCandBase   = smemCandidates + warpInBlock * k;
    float2*    sData          = reinterpret_cast<float2*>(smemCandidates + WARPS_PER_BLOCK * k);

    const bool activeWarp = (warpGlobal < query_count);

    // Load the query point and broadcast within the warp.
    float qx = 0.0f, qy = 0.0f;
    if (activeWarp) {
        if (lane == 0) {
            float2 q = query[warpGlobal];
            qx = q.x; qy = q.y;
        }
        qx = __shfl_sync(full_mask, qx, 0);
        qy = __shfl_sync(full_mask, qy, 0);
    }

    // Items per lane in the intermediate top-k
    const int itemsPerLane = k >> 5; // since k is a power-of-two >= 32, divisible by 32
    // Private top-k per warp, distributed across lanes, held in registers.
    // Each lane stores itemsPerLane entries.
    float resDist[32];
    int   resIdx[32];

    // Initialize intermediate results to +inf dist and invalid index.
    const float INF = CUDART_INF_F;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        resDist[i] = INF;
        resIdx[i]  = -1;
    }

    // Candidate buffer count (per warp), maintained by lane 0 and broadcast as needed
    int bufCount = 0;

    // Current k-th best distance in the intermediate result (initially INF)
    float warpWorstDist = INF;

    // Process data in tiles loaded into shared memory by the whole block
    for (int tileStart = 0; tileStart < data_count; tileStart += tilePoints) {
        int count = data_count - tileStart;
        if (count > tilePoints) count = tilePoints;

        // Cooperative load of the data tile
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            sData[i] = data[tileStart + i];
        }
        __syncthreads();

        // Each active warp processes the tile: compute distances and push candidates
        if (activeWarp) {
            for (int pos = lane; pos < count; pos += WARP_SIZE) {
                float2 p = sData[pos];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = fmaf(dx, dx, dy * dy);
                int   idx  = tileStart + pos;

                // Only consider if strictly better than current k-th best
                bool pending = (dist < warpWorstDist);

                // Warp-aggregated append with buffer-full handling and re-evaluation after merges
                while (true) {
                    unsigned mask = __ballot_sync(full_mask, pending);
                    int nActive = __popc(mask);
                    if (nActive == 0) break;

                    int base = 0;
                    int bufCountWarp = __shfl_sync(full_mask, bufCount, 0);
                    int slotsLeft = k - bufCountWarp;

                    if (slotsLeft == 0) {
                        // Merge buffer into top-k to free space and tighten the threshold
                        merge_buffer_with_topk<WARPS_PER_BLOCK>(warpCandBase, k, bufCount, resDist, resIdx, itemsPerLane, warpWorstDist);
                        // Re-evaluate with the new threshold
                        pending = (dist < warpWorstDist);
                        continue;
                    }

                    int emitCount = (nActive < slotsLeft) ? nActive : slotsLeft;
                    int rank = __popc(mask & lane_mask_lt());
                    bool willWrite = pending && (rank < emitCount);

                    if (lane == 0) base = bufCount;
                    base = __shfl_sync(full_mask, base, 0);

                    if (willWrite) {
                        warpCandBase[base + rank].dist = dist;
                        warpCandBase[base + rank].idx  = idx;
                    }

                    if (lane == 0) bufCount = base + emitCount;
                    __syncwarp();

                    // If all active lanes wrote, we're done with this candidate.
                    if (nActive <= slotsLeft) {
                        pending = false;
                        break;
                    }

                    // Otherwise, update 'pending' to keep only those that didn't fit; the buffer is now full -> next loop merges.
                    pending = pending && (rank >= emitCount);
                }
            }
        }

        __syncthreads();
    }

    // After the final tile, merge remaining buffered candidates
    if (activeWarp && bufCount > 0) {
        merge_buffer_with_topk<WARPS_PER_BLOCK>(warpCandBase, k, bufCount, resDist, resIdx, itemsPerLane, warpWorstDist);
    }

    // Final step: output the top-k results sorted by ascending distance.
    // We perform a selection sort using warp-wide reductions:
    //   - each lane scans its local segment to find the current minimum,
    //   - a warp reduction finds the global minimum and its owner,
    //   - the owner marks it as used by setting its distance to INF,
    //   - lane 0 writes the selected pair to global memory.
    if (activeWarp) {
        // Local arrays are already in resDist/resIdx.
        auto* out = result + warpGlobal * k;

        for (int sel = 0; sel < k; ++sel) {
            // Local min and its position
            float localMin = resDist[0];
            int localPos = 0;
            #pragma unroll
            for (int i = 1; i < 32; ++i) {
                if (i >= itemsPerLane) break;
                float v = resDist[i];
                if (v < localMin) { localMin = v; localPos = i; }
            }

            // Warp-wide argmin via argmax on negative values to avoid another helper
            // or use max with inverted sense: We simulate argmin by flipping sign and argmax
            float negLocalMin = -localMin;
            ArgMaxPair gm = warp_argmax(negLocalMin, full_mask);
            int ownerLane = gm.lane;
            int ownerPos  = __shfl_sync(full_mask, localPos, ownerLane);
            float bestDist = -gm.val;
            int bestIdx    = __shfl_sync(full_mask, resIdx[ownerPos], ownerLane);

            if (lane == 0) {
                out[sel].first  = bestIdx;
                out[sel].second = bestDist;
            }

            // Mark selected item as used by setting distance to INF
            if (lane == ownerLane) {
                resDist[ownerPos] = INF;
                // resIdx[ownerPos] may remain as-is; distance INF prevents reselection.
            }
            __syncwarp();
        }
    }
}

// Helper to choose a launch configuration and dynamic shared memory size that fits, then launch the kernel.
static inline void launch_knn_kernel(const float2* query, int query_count,
                                     const float2* data,  int data_count,
                                     std::pair<int, float>* result, int k) {
    // Candidate buffers require warpsPerBlock * k candidates, each 8 bytes.
    // Data tile requires tilePoints * 8 bytes.
    // We try WARPS_PER_BLOCK in {8,4,2,1} to fit into device limits.
    int device = 0;
    cudaGetDevice(&device);

    int maxDynSmemOptin = 0;
    int maxDynSmemDefault = 0;
    cudaDeviceGetAttribute(&maxDynSmemOptin,   cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaDeviceGetAttribute(&maxDynSmemDefault, cudaDevAttrMaxSharedMemoryPerBlock,      device);

    int maxDynSmem = maxDynSmemOptin > 0 ? maxDynSmemOptin : maxDynSmemDefault;
    if (maxDynSmem <= 0) {
        // Fallback to a conservative number if querying fails (rare)
        maxDynSmem = 96 * 1024;
    }

    // Try different WARPS_PER_BLOCK to fit shared memory constraints.
    int chosenWarps = 0;
    int blockThreads = 0;
    int tilePoints   = 0;
    size_t dynSmem   = 0;

    auto try_config = [&](int warpsPerBlock) {
        size_t candBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) * sizeof(Candidate);
        if (candBytes + sizeof(float2) > static_cast<size_t>(maxDynSmem)) return false;
        int maxTilePts = static_cast<int>((maxDynSmem - candBytes) / sizeof(float2));
        if (maxTilePts <= 0) return false;
        // Round tile points down to multiple of blockDim.x for coalesced loads
        int threads = warpsPerBlock * WARP_SIZE;
        int rounded = (maxTilePts / threads) * threads;
        if (rounded <= 0) return false;

        chosenWarps = warpsPerBlock;
        blockThreads = threads;
        tilePoints = rounded;
        dynSmem = candBytes + static_cast<size_t>(tilePoints) * sizeof(float2);
        return true;
    };

    if (!try_config(8)) {
        if (!try_config(4)) {
            if (!try_config(2)) {
                if (!try_config(1)) {
                    // As a last resort, fallback to minimal viable tilePoints with 1 warp/block
                    chosenWarps = 1;
                    blockThreads = WARP_SIZE;
                    size_t candBytes = static_cast<size_t>(chosenWarps) * static_cast<size_t>(k) * sizeof(Candidate);
                    int minTilePts = blockThreads; // at least one per thread
                    size_t needed = candBytes + static_cast<size_t>(minTilePts) * sizeof(float2);
                    // If still not fitting, clamp to device default max
                    if (needed > static_cast<size_t>(maxDynSmem)) {
                        // Reduce tile points to fit
                        int maxTilePts = static_cast<int>((maxDynSmem - candBytes) / sizeof(float2));
                        if (maxTilePts < blockThreads) maxTilePts = blockThreads; // even if exceeding, will set attr
                        tilePoints = maxTilePts / blockThreads * blockThreads;
                        dynSmem = candBytes + static_cast<size_t>(tilePoints) * sizeof(float2);
                    } else {
                        tilePoints = minTilePts;
                        dynSmem = needed;
                    }
                }
            }
        }
    }

    // Grid size: one warp per query
    int gridBlocks = (query_count + chosenWarps - 1) / chosenWarps;

    // Opt-in to large dynamic shared memory if supported
    // Set preferred shared memory carveout to maximize shared memory
    // These calls are best-effort; ignore errors if not supported.
    if (maxDynSmemOptin > 0) {
        cudaFuncSetAttribute(knn_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxDynSmem);
        cudaFuncSetAttribute(knn_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxDynSmem);
        cudaFuncSetAttribute(knn_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxDynSmem);
        cudaFuncSetAttribute(knn_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxDynSmem);
#if defined(cudaFuncAttributePreferredSharedMemoryCarveout)
        cudaFuncSetAttribute(knn_kernel<8>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(knn_kernel<4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(knn_kernel<2>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaFuncSetAttribute(knn_kernel<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
#endif
    }

    // Launch with the chosen configuration
    switch (chosenWarps) {
        case 8:
            knn_kernel<8><<<gridBlocks, 8 * WARP_SIZE, dynSmem>>>(query, query_count, data, data_count, result, k, tilePoints);
            break;
        case 4:
            knn_kernel<4><<<gridBlocks, 4 * WARP_SIZE, dynSmem>>>(query, query_count, data, data_count, result, k, tilePoints);
            break;
        case 2:
            knn_kernel<2><<<gridBlocks, 2 * WARP_SIZE, dynSmem>>>(query, query_count, data, data_count, result, k, tilePoints);
            break;
        default:
            knn_kernel<1><<<gridBlocks, 1 * WARP_SIZE, dynSmem>>>(query, query_count, data, data_count, result, k, tilePoints);
            break;
    }
}

// Public API entry point
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Assumptions: pointers are device memory; k is power-of-two in [32, 1024]; data_count >= k.
    // The kernel is designed to handle large arrays efficiently with tiling and warp-per-query processing.
    launch_knn_kernel(query, query_count, data, data_count, result, k);
}