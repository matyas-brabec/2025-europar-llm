#include <cuda_runtime.h>
#include <cuda.h>
#include <utility>

// This implementation assigns one warp (32 threads) per query point.
// Each warp processes the full dataset in tiles cached in shared memory.
// For each tile, every lane maintains a small per-thread top-L (L = k/32) list
// of its best candidates in shared memory (descending order: worst at front).
// Then, the warp performs a multiway merge (via warp shuffles) of the 32 per-lane
// lists to form the tile's k best candidates and merges them with the current
// per-warp global top-k (also kept in shared memory).
// The per-warp global result is maintained in descending order (worst to best).
// After all tiles are processed, the final per-query k-NN indices/distances are
// written out in ascending order (best to worst).
//
// Notes:
// - Distances are squared L2 (no sqrt), as requested.
// - k is a power of two in [32, 1024], so L = k/32 is an integer in [1, 32].
// - No additional device memory is allocated; only dynamic shared memory is used.
// - Uses warp shuffles for fast intra-warp communication and __syncthreads for tile barriers.
// - Tunables: WARPS_PER_BLOCK and TILE_SIZE control occupancy and shared memory usage.
//   With WARPS_PER_BLOCK=4 and TILE_SIZE=2048, the maximum shared memory per block
//   is ~114 KB for k=1024, which fits on A100/H100 with opt-in dynamic shared memory.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable parameters: chosen to balance shared memory usage and occupancy on A100/H100.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 2048
#endif

static __device__ __forceinline__ float sq_l2_dist_2d(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // FMA for better throughput/precision
    return fmaf(dx, dx, dy * dy);
}

// Warp-level argmax over 32 lanes.
// Inputs:
//   myVal    - candidate value in each lane
// Outputs (same in all lanes after return):
//   maxVal   - maximum value across the warp
//   maxLane  - lane index (0..31) that holds the maximum (lowest lane on tie)
static __device__ __forceinline__ void warp_argmax(float myVal, float &maxVal, int &maxLane) {
    const unsigned mask = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    float bestVal = myVal;
    int bestLane = lane;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float othVal = __shfl_down_sync(mask, bestVal, offset);
        int   othLan = __shfl_down_sync(mask, bestLane, offset);
        if (othVal > bestVal || (othVal == bestVal && othLan < bestLane)) {
            bestVal = othVal;
            bestLane = othLan;
        }
    }
    // Broadcast reduction result from lane 0 to all
    maxVal  = __shfl_sync(mask, bestVal, 0);
    maxLane = __shfl_sync(mask, bestLane, 0);
}

// CUDA kernel: one warp per query.
__global__ void knn2d_kernel(const float2 *__restrict__ query,
                             int query_count,
                             const float2 *__restrict__ data,
                             int data_count,
                             std::pair<int, float> *__restrict__ result,
                             int k) {
    extern __shared__ unsigned char smem_raw[];
    // Shared tile of data points (float2)
    float2 *sData = reinterpret_cast<float2*>(smem_raw);

    const int tid   = threadIdx.x;
    const int lane  = tid & (WARP_SIZE - 1);
    const int warp  = tid >> 5; // warp id within block [0..WARPS_PER_BLOCK-1]
    const int warps_per_block = blockDim.x / WARP_SIZE;

    // Per-warp shared memory layout after the tile:
    // [globalDist k floats][globalIdx k ints][localDist k floats][localIdx k ints][outDist k floats][outIdx k ints]
    size_t tile_bytes = sizeof(float2) * TILE_SIZE;
    unsigned char *warp_base = smem_raw + tile_bytes + static_cast<size_t>(warp) * (static_cast<size_t>(k) * (sizeof(float) + sizeof(int)) * 3u);
    float *gDist = reinterpret_cast<float*>(warp_base);
    int   *gIdx  = reinterpret_cast<int*>(gDist + k);
    float *lDist = reinterpret_cast<float*>(gIdx  + k);
    int   *lIdx  = reinterpret_cast<int*>(lDist + k);
    float *oDist = reinterpret_cast<float*>(lIdx  + k);
    int   *oIdx  = reinterpret_cast<int*>(oDist + k);

    // Grid-stride over queries, with one warp per query.
    int warp_global = blockIdx.x * warps_per_block + warp;
    int warp_stride = gridDim.x * warps_per_block;

    for (int q = warp_global; q < query_count; q += warp_stride) {
        // Load query point and broadcast within warp
        float2 qpt;
        if (lane == 0) qpt = query[q];
        qpt.x = __shfl_sync(0xFFFFFFFFu, qpt.x, 0);
        qpt.y = __shfl_sync(0xFFFFFFFFu, qpt.y, 0);

        // Initialize per-warp global top-k (descending; worst at front)
        for (int j = lane; j < k; j += WARP_SIZE) {
            gDist[j] = CUDART_INF_F;
            gIdx[j]  = -1;
        }
        __syncwarp();

        // Iterate over data in tiles cached in shared memory
        for (int t0 = 0; t0 < data_count; t0 += TILE_SIZE) {
            int nThisTile = min(TILE_SIZE, data_count - t0);

            // Cooperative load of the tile
            for (int i = tid; i < nThisTile; i += blockDim.x) {
                sData[i] = data[t0 + i];
            }
            __syncthreads(); // ensure tile is loaded

            // Per-thread local top-L in shared memory (descending; worst at front)
            const int L = k / WARP_SIZE; // k is guaranteed to be multiple of 32
            int base = lane * L;
#pragma unroll
            for (int s = 0; s < L; ++s) {
                lDist[base + s] = CUDART_INF_F;
                lIdx [base + s] = -1;
            }

            // Each lane processes its strided subset of the tile
            for (int i = lane; i < nThisTile; i += WARP_SIZE) {
                float2 p = sData[i];
                float d = sq_l2_dist_2d(p, qpt);
                int   idx = t0 + i;

                // Bubble-insert into descending local list
                float candD = d;
                int   candI = idx;
#pragma unroll
                for (int s = 0; s < L; ++s) {
                    float prevD = lDist[base + s];
                    int   prevI = lIdx [base + s];
                    // If candidate is better (smaller), push it down the list
                    if (candD < prevD) {
                        lDist[base + s] = candD;
                        lIdx [base + s] = candI;
                        candD = prevD;
                        candI = prevI;
                    }
                }
            }

            // Multiway merge of 32 local lists (descending) with the existing global top-k (descending).
            // We stream the tile's k candidates via repeated warp-argmax over the 32 per-lane lists,
            // including +INF placeholders for exhausted positions to keep list length = L per lane.
            // Then merge the streamed tile sequence with the global sequence, keeping the k smallest.
            // We implement this by:
            // 1) Skipping the largest k elements from the union of both sequences.
            // 2) Emitting the next k elements (descending) into oDist/oIdx.
            // Finally, copy oDist/oIdx back to gDist/gIdx.

            // Per-lane pointers into local lists (start at first valid element).
            // Count valid (finite) entries in each local list; INF means unused.
            int valid = 0;
#pragma unroll
            for (int s = 0; s < L; ++s) {
                valid += (lDist[base + s] < CUDART_INF_F) ? 1 : 0;
            }
            int posLocal = L - valid; // first valid entry (descending list), may be == L if valid == 0

            float curD = (posLocal < L) ? lDist[base + posLocal] : CUDART_INF_F;
            int   curI = (posLocal < L) ? lIdx [base + posLocal] : -1;

            // Skip the largest k elements from the union of A (gDist) and streamed S (curD across lanes).
            int aPos = 0; // position in gDist (descending)
            for (int skip = 0; skip < k; ++skip) {
                // Compute max over the 32 lanes' current candidates (curD)
                float sMax; int sLane;
                warp_argmax(curD, sMax, sLane);

                // Head of A (global) sequence
                float aHead = (aPos < k) ? gDist[aPos] : -CUDART_INF_F; // if exhausted, treat as -inf so S wins

                // Prefer A on ties to keep deterministic behavior
                bool takeA = (aHead >= sMax);
                // Lane 0 decides; broadcast decision and sLane to all lanes
                unsigned mask = 0xFFFFFFFFu;
                int takeA_int = __shfl_sync(mask, (int)takeA, 0);

                if (takeA_int) {
                    // Consume from A
                    if (lane == 0) {
                        ++aPos;
                    }
                } else {
                    // Consume from S: the winning lane advances its pointer
                    if (lane == sLane) {
                        ++posLocal;
                        if (posLocal < L) {
                            curD = lDist[base + posLocal];
                            curI = lIdx [base + posLocal];
                        } else {
                            curD = CUDART_INF_F; // exhausted: act as +inf (worst) placeholder
                            curI = -1;
                        }
                    }
                }
            }

            // Now emit the next k elements (descending) into output buffer
            for (int out = 0; out < k; ++out) {
                float sMax; int sLane;
                warp_argmax(curD, sMax, sLane);

                float aHead = (aPos < k) ? gDist[aPos] : -CUDART_INF_F;
                int   aHeadI = (aPos < k) ? gIdx[aPos] : -1;

                bool takeA = (aHead >= sMax);
                unsigned mask = 0xFFFFFFFFu;
                int takeA_int = __shfl_sync(mask, (int)takeA, 0);

                if (lane == 0) {
                    if (takeA_int) {
                        oDist[out] = aHead;
                        oIdx [out] = aHeadI;
                    } else {
                        // get winning lane's curI via shuffle
                        int winIdx = __shfl_sync(mask, curI, sLane);
                        oDist[out] = sMax;
                        oIdx [out] = winIdx;
                    }
                }

                if (takeA_int) {
                    if (lane == 0) ++aPos;
                } else {
                    if (lane == sLane) {
                        ++posLocal;
                        if (posLocal < L) {
                            curD = lDist[base + posLocal];
                            curI = lIdx [base + posLocal];
                        } else {
                            curD = CUDART_INF_F;
                            curI = -1;
                        }
                    }
                }
            }

            // Copy output back to global per-warp buffers
            for (int j = lane; j < k; j += WARP_SIZE) {
                gDist[j] = oDist[j];
                gIdx [j] = oIdx [j];
            }
            __syncwarp();

            __syncthreads(); // before loading next tile
        } // end tile loop

        // Write final results in ascending order (best to worst)
        // gDist/gIdx are descending; output should be ascending -> reverse order
        std::pair<int, float> *out = result + static_cast<size_t>(q) * k;
        for (int j = lane; j < k; j += WARP_SIZE) {
            int src = k - 1 - j;
            // Reverse with per-lane stride: compute actual source index for this lane's j
            // j varies as lane, lane+32, lane+64, ...
            // For correctness, compute real position as:
            int jj = j;
            int src_pos = k - 1 - jj;
            out[jj].first  = gIdx[src_pos];
            out[jj].second = gDist[src_pos];
        }
        __syncwarp();
    }
}

// Host-side launcher.
// query, data, result are device pointers (cudaMalloc'ed).
// query_count: number of queries, data_count: number of data points, k: power of two in [32, 1024].
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    const int warps_per_block = WARPS_PER_BLOCK;
    const int block_threads = warps_per_block * WARP_SIZE;

    // Grid: one warp per query (grid-stride inside kernel handles leftovers)
    int grid_blocks = (query_count + warps_per_block - 1) / warps_per_block;
    if (grid_blocks <= 0) grid_blocks = 1;

    // Dynamic shared memory size:
    // - TILE_SIZE float2's for the data tile
    // - For each warp:
    //     6 arrays of length k: gDist, gIdx, lDist, lIdx, oDist, oIdx
    //     (float/int are 4 bytes each) => 6 * k * 4 bytes = 24*k bytes per warp
    size_t tile_bytes = sizeof(float2) * TILE_SIZE;
    size_t per_warp_bytes = static_cast<size_t>(k) * (sizeof(float) + sizeof(int)) * 3u; // 3 pairs of (float/int)
    size_t shared_bytes = tile_bytes + static_cast<size_t>(warps_per_block) * per_warp_bytes;

    // Opt-in to use large dynamic shared memory if needed
    cudaFuncSetAttribute(knn2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
#if defined(cudaFuncAttributePreferredSharedMemoryCarveout)
    cudaFuncSetAttribute(knn2d_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100); // Prefer shared mem
#endif

    dim3 block(block_threads);
    dim3 grid(grid_blocks);
    knn2d_kernel<<<grid, block, shared_bytes>>>(query, query_count, data, data_count, result, k);
}