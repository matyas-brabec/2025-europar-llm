#include <cuda_runtime.h>
#include <utility>
#include <cmath>

// This implementation computes k-NN for 2D points (float2) using one warp per query.
// It processes the database in shared-memory tiles, keeps an intermediate top-k
// per warp in shared memory, and accumulates candidates into a per-warp shared
// buffer. When the buffer fills (or at the end), it is sorted and merged with the
// intermediate result using a warp-parallel merge-path algorithm. The algorithm
// uses only dynamic shared memory and no additional device allocations.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Small helper for integer division with rounding up
static inline int div_up(int x, int y) { return (x + y - 1) / y; }

// Device-side utility: next power of two >= x (for x >= 1)
__device__ __forceinline__ int next_pow2_ceil(int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// Warp-scope bitonic sort of pairs (dist, idx) stored in shared memory.
// - dist, idx: base pointers to the array for this warp
// - n: number of valid elements to sort (must be a power of two)
// The routine sorts ascending by distance. It assumes all threads in the warp
// participate and uses __syncwarp at each stage.
__device__ __forceinline__ void warp_bitonic_sort_pairs(float* dist, int* idx, int n) {
    const unsigned full_mask = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Bitonic sort network using XOR indexing; n must be a power of two.
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each lane processes multiple elements in stride of warp size.
            for (int i = lane; i < n; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i && ixj < n) {
                    bool up = ((i & k) == 0);
                    float di = dist[i];
                    float dj = dist[ixj];
                    int ii = idx[i];
                    int ij = idx[ixj];
                    bool swap = (di > dj) == up;
                    if (swap) {
                        dist[i] = dj; idx[i] = ij;
                        dist[ixj] = di; idx[ixj] = ii;
                    }
                }
            }
            __syncwarp(full_mask);
        }
    }
}

// Merge-path diagonal search for merging two sorted arrays A (size nA) and B (size nB).
// Finds the partition i along diagonal 'diag' such that i in [max(0, diag - nB), min(diag, nA)]
// and B[j-1] <= A[i] with j = diag - i. Returns i; j can be derived as diag - i.
// This variant uses sentinels -INF/INF via explicit conditionals.
__device__ __forceinline__ int merge_path_search(const float* A, int nA, const float* B, int nB, int diag) {
    int i_min = max(0, diag - nB);
    int i_max = min(diag, nA);
    while (i_min < i_max) {
        int i = (i_min + i_max) >> 1;
        int j = diag - i;

        float a_i      = (i < nA) ? A[i]     : INFINITY;
        float a_i_minus= (i > 0)  ? A[i - 1] : -INFINITY;
        float b_j      = (j < nB) ? B[j]     : INFINITY;
        float b_j_minus= (j > 0)  ? B[j - 1] : -INFINITY;

        // We are looking for the smallest i such that b_{j-1} <= a_i
        if (b_j_minus > a_i) {
            i_min = i + 1;
        } else {
            i_max = i;
        }
    }
    return i_min;
}

// Warp-parallel merge of two sorted lists (ascending by distance) producing the first 'k' elements.
// The output is written to outDist/outIdx. Work is partitioned evenly across the warp using merge-path.
// nA can be k (current intermediate result), nB is the number of candidates (<= k).
__device__ __forceinline__ void warp_merge_sorted_topk(
    const float* A, const int* Aidx, int nA,
    const float* B, const int* Bidx, int nB,
    float* outDist, int* outIdx, int k)
{
    const unsigned full_mask = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    // k is a power of two and >= 32, so r >= 1
    int r = k / WARP_SIZE;
    int out_begin = lane * r;
    int out_end   = out_begin + r; // exclusive

    int i0 = merge_path_search(A, nA, B, nB, out_begin);
    int j0 = out_begin - i0;
    int i1 = merge_path_search(A, nA, B, nB, out_end);
    int j1 = out_end - i1;

    int ia = i0;
    int ib = j0;

    // Merge the assigned segment
    for (int outPos = out_begin; outPos < out_end; ++outPos) {
        bool takeA = (ia < i1) && ( (ib >= j1) || (A[ia] <= B[ib]) );
        if (takeA) {
            outDist[outPos] = A[ia];
            outIdx[outPos]  = Aidx[ia];
            ++ia;
        } else {
            outDist[outPos] = B[ib];
            outIdx[outPos]  = Bidx[ib];
            ++ib;
        }
    }

    __syncwarp(full_mask);
}

// Compute squared Euclidean distance between two float2 points.
__device__ __forceinline__ float sq_l2(const float2& a, const float2& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // Use FMA for better throughput
    return fmaf(dx, dx, dy * dy);
}

// The main kernel: one warp processes one query.
__global__ void knn_warp_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    std::pair<int, float>* __restrict__ result,
    int k, int tile_points)
{
    extern __shared__ unsigned char smem_raw[];

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x >> 5; // / 32
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    if (warp_global >= query_count) return;
    const unsigned full_mask = 0xFFFFFFFFu;

    // Shared memory layout
    // [0] Shared tile of data points: float2 tile[tile_points]
    // [1] Per-warp arrays: resDist[W*k], resIdx[W*k], bufDist[W*k], bufIdx[W*k], bufCount[W]
    // The per-warp slices are contiguous segments of size k.
    unsigned char* p = smem_raw;

    // Align float2 tile base
    float2* tile = reinterpret_cast<float2*>(p);
    p += sizeof(float2) * tile_points;

    // Align to 4 bytes for subsequent arrays
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    addr = (addr + 3) & ~uintptr_t(3);
    p = reinterpret_cast<unsigned char*>(addr);

    float* resDistAll = reinterpret_cast<float*>(p);
    p += sizeof(float) * (warps_per_block * k);

    int* resIdxAll = reinterpret_cast<int*>(p);
    p += sizeof(int) * (warps_per_block * k);

    float* bufDistAll = reinterpret_cast<float*>(p);
    p += sizeof(float) * (warps_per_block * k);

    int* bufIdxAll = reinterpret_cast<int*>(p);
    p += sizeof(int) * (warps_per_block * k);

    int* bufCountAll = reinterpret_cast<int*>(p);
    // p += sizeof(int) * warps_per_block; // Not needed further

    // Per-warp slices
    float* resDist = resDistAll + warp_in_block * k;
    int*   resIdx  = resIdxAll  + warp_in_block * k;
    float* bufDist = bufDistAll + warp_in_block * k;
    int*   bufIdx  = bufIdxAll  + warp_in_block * k;
    int&   bufCountRef = bufCountAll[warp_in_block];

    // Load the query point into registers and broadcast to the warp
    float qx = 0.f, qy = 0.f;
    if (lane == 0) {
        float2 q = query[warp_global];
        qx = q.x; qy = q.y;
    }
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);
    const float2 qpt = make_float2(qx, qy);

    // Initialize intermediate result with +INF distances and invalid indices.
    const float INF = INFINITY;
    for (int i = lane; i < k; i += WARP_SIZE) {
        resDist[i] = INF;
        resIdx[i]  = -1;
    }
    if (lane == 0) bufCountRef = 0;
    __syncwarp(full_mask);

    // Helper lambda to flush candidate buffer: sort candidates and merge into res.
    auto flush_merge = [&](int candCount) {
        // candCount can be 0..k
        // If zero, nothing to do.
        if (candCount <= 0) return;

        // Prepare candidate buffer: pad to next power-of-two length with INF
        int sortLen = next_pow2_ceil(candCount);
        // Fill padding with INF to not affect sorting
        for (int i = lane + candCount; i < sortLen; i += WARP_SIZE) {
            bufDist[i] = INF;
            bufIdx[i]  = -1;
        }
        __syncwarp(full_mask);

        // Sort the candidate buffer (ascending)
        warp_bitonic_sort_pairs(bufDist, bufIdx, sortLen);

        // Merge top-k from res (size k) and buf (size candCount) into buf as output
        warp_merge_sorted_topk(resDist, resIdx, k, bufDist, bufIdx, candCount, bufDist, bufIdx, k);

        // Swap: buf now holds the new intermediate result; reuse old res as the next buffer
        __syncwarp(full_mask);
        float* tmpDist = resDist; resDist = bufDist; bufDist = tmpDist;
        int*   tmpIdx  = resIdx;  resIdx  = bufIdx;  bufIdx  = tmpIdx;

        // Reset candidate count
        if (lane == 0) bufCountRef = 0;
        __syncwarp(full_mask);
    };

    // Process the data in tiles loaded by the entire block
    for (int base = 0; base < data_count; base += tile_points) {
        int tileCount = min(tile_points, data_count - base);

        // Cooperative load of the tile into shared memory
        for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
            tile[i] = data[base + i];
        }
        __syncthreads(); // Ensure the tile is fully loaded before any warp uses it

        // Each warp processes its assigned query against the shared tile
        for (int j = lane; j < tileCount; j += WARP_SIZE) {
            float2 pnt = tile[j];
            float d = sq_l2(qpt, pnt);
            int   idx = base + j;

            // Check against current threshold (k-th distance in intermediate result)
            // We always compare to the current intermediate result; candidates do not affect
            // the threshold until merged.
            float threshold = resDist[k - 1];
            bool accept = (d < threshold);

            // Warp-scope ballot of accepted candidates
            unsigned mask = __ballot_sync(full_mask, accept);
            int accepted = __popc(mask);

            if (accepted > 0) {
                // If adding all accepted would overflow the buffer, flush first (if buffer not empty).
                int need_merge = 0;
                if (lane == 0) {
                    int count = bufCountRef;
                    need_merge = (count > 0) && (count + accepted > k);
                }
                need_merge = __shfl_sync(full_mask, need_merge, 0);
                if (need_merge) {
                    // Flush existing buffer
                    int prevCount = 0;
                    if (lane == 0) prevCount = bufCountRef;
                    prevCount = __shfl_sync(full_mask, prevCount, 0);
                    flush_merge(prevCount);
                    // Update threshold and recompute acceptance after merge
                    threshold = resDist[k - 1];
                    accept = (d < threshold);
                    mask = __ballot_sync(full_mask, accept);
                    accepted = __popc(mask);
                    if (accepted == 0) {
                        // No longer a candidate after improved threshold; continue
                        continue;
                    }
                }

                // Reserve positions in candidate buffer
                int basePos = 0;
                if (lane == 0) {
                    basePos = bufCountRef;
                    bufCountRef = basePos + accepted;
                }
                basePos = __shfl_sync(full_mask, basePos, 0);

                // Compute each accepting lane's offset using prefix popcount
                int rank = __popc(mask & ((1u << lane) - 1u));
                if (accept) {
                    int pos = basePos + rank;
                    bufDist[pos] = d;
                    bufIdx[pos]  = idx;
                }
                __syncwarp(full_mask);

                // If buffer is exactly full now, flush it
                int do_flush = 0;
                if (lane == 0) do_flush = (bufCountRef == k);
                do_flush = __shfl_sync(full_mask, do_flush, 0);
                if (do_flush) {
                    flush_merge(k);
                }
            }
        }

        __syncthreads(); // Ensure all warps are done with the tile before loading the next one
    }

    // After all tiles, flush any remaining candidates
    int remaining = 0;
    if (lane == 0) remaining = bufCountRef;
    remaining = __shfl_sync(full_mask, remaining, 0);
    if (remaining > 0) {
        flush_merge(remaining);
    }

    // Write out the final top-k for this query
    size_t outBase = static_cast<size_t>(warp_global) * static_cast<size_t>(k);
    for (int i = lane; i < k; i += WARP_SIZE) {
        result[outBase + i].first  = resIdx[i];
        result[outBase + i].second = resDist[i];
    }
}

// Host function to select launch parameters and run the kernel
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Query device shared memory capability to choose warps per block and tile size.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    // Maximum dynamic shared memory per block (opt-in) if supported; fallback to default.
    int maxOptinSmem = 0;
    cudaDeviceGetAttribute(&maxOptinSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    size_t maxSmem = maxOptinSmem > 0 ? static_cast<size_t>(maxOptinSmem) : static_cast<size_t>(prop.sharedMemPerBlock);

    // Choose warps per block: try a few options, prefer more warps if shared memory allows.
    int chosenWarps = 4; // start optimistic
    for (; chosenWarps >= 1; --chosenWarps) {
        // Per-block shared memory footprint for per-warp buffers: 16 bytes per entry, 2*k entries per warp = 16*k bytes per warp
        size_t perWarpSmem = static_cast<size_t>(16 * k);
        // Reserve a minimum tile of at least 2048 points (16 KB) if possible; adjust later precisely
        size_t overhead = perWarpSmem * chosenWarps + sizeof(int) * chosenWarps;
        if (overhead + sizeof(float2) * 1024 <= maxSmem) break;
    }
    if (chosenWarps < 1) chosenWarps = 1;

    // Compute the maximum tile size that fits in shared memory with the chosen number of warps
    size_t perWarpSmem = static_cast<size_t>(16 * k);
    size_t baseOverhead = perWarpSmem * chosenWarps + sizeof(int) * chosenWarps;

    // Leave a small safety margin
    const size_t safety = 0;

    size_t maxTileBytes = 0;
    if (maxSmem > baseOverhead + safety) {
        maxTileBytes = maxSmem - baseOverhead - safety;
    } else {
        maxTileBytes = sizeof(float2) * 256; // minimal fallback
    }

    int tile_points = static_cast<int>(maxTileBytes / sizeof(float2));
    // Cap tile_points to a reasonable size and at least 1024 for efficiency if possible
    tile_points = max(256, min(tile_points, 16384));
    // Ensure tile_points is not zero
    if (tile_points <= 0) tile_points = 256;

    // Recompute actual shared memory size required
    size_t smemBytes = sizeof(float2) * static_cast<size_t>(tile_points)
                     + perWarpSmem * chosenWarps // resDist+resIdx+bufDist+bufIdx (16*k bytes per warp)
                     + sizeof(int) * chosenWarps; // bufCount

    // Request opt-in larger shared memory if needed/available
    cudaFuncSetAttribute(knn_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(min(smemBytes, maxSmem)));

    dim3 block(chosenWarps * WARP_SIZE, 1, 1);
    dim3 grid(div_up(query_count, chosenWarps), 1, 1);

    knn_warp_kernel<<<grid, block, smemBytes>>>(query, query_count, data, data_count, result, k, tile_points);
}