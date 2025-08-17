#include <cuda_runtime.h>
#include <utility>

// Optimized CUDA kernel for k-NN (2D points, squared Euclidean distance) using one warp per query.
// Requirements implemented:
// - k is power of two in [32, 1024], so per-thread register storage m = k/32 in [1, 32].
// - Each warp maintains an intermediate result (top-k) distributed across thread registers: each thread has m consecutive elements.
// - Each warp has a candidate buffer of size k in shared memory (indices and distances) with a shared counter.
// - Data points are processed in batches; batches are loaded into shared memory by the entire thread block.
// - Warp ballot is used to select candidates whose distance < max_distance and to compute write positions in the buffer.
// - When the buffer is full, a merge is performed as specified using bitonic sort and warp shuffles.
// - After the final batch, any remaining candidates are merged.
// - Implementation uses warp shuffles and shared memory barriers appropriately.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunables: 8 warps per block = 256 threads. TILE_POINTS controls shared-memory tile size for cached data points.
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)
#endif
#ifndef TILE_POINTS
#define TILE_POINTS 4096   // 4096*8B = 32KB per block for data tiles
#endif

// Maximum per-thread register storage when k=1024 => m = 1024/32 = 32
#define MAX_M 32

// Utility: squared Euclidean distance for float2
__device__ __forceinline__ float squared_l2(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // Use FMA for slightly better precision/performance
    return fmaf(dy, dy, dx * dx);
}

// Bitonic sort across K elements distributed across a warp, each thread holds m consecutive elements.
// - dist[idx] and id[idx] arrays are of length m in registers (per thread).
// - Global index i of element r in lane 'lane' is i = lane*m + r.
// - We implement the serial bitonic algorithm in parallel across the warp:
//   For j >= m, compare/exchange across threads using __shfl_xor_sync with delta = j/m.
//   For j < m, compare/exchange within a thread between registers at indices r and r^j (only once per pair).
__device__ __forceinline__ void warp_bitonic_sort(float dist[MAX_M], int id[MAX_M], int m, int K, unsigned lane) {
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Outer loop over subsequence sizes
    for (int kstage = 2; kstage <= K; kstage <<= 1) {
        // Inner loop over stride
        for (int j = kstage >> 1; j > 0; j >>= 1) {

            if (j >= m) {
                // Cross-lane compare-exchange
                unsigned delta = (unsigned)(j / m); // because j and m are powers of two, j % m == 0 when j >= m
                for (int r = 0; r < m; ++r) {
                    float myVal = dist[r];
                    int   myIdx = id[r];

                    float othVal = __shfl_xor_sync(FULL_MASK, myVal, delta, WARP_SIZE);
                    int   othIdx = __shfl_xor_sync(FULL_MASK, myIdx, delta, WARP_SIZE);

                    int i_global = (int)(lane * (unsigned)m + (unsigned)r);
                    bool up = ((i_global & kstage) == 0);

                    // Compare and select for this element side
                    bool takeOther = up ? (myVal > othVal) : (myVal < othVal);
                    if (takeOther) {
                        dist[r] = othVal;
                        id[r]   = othIdx;
                    }
                }
                __syncwarp(FULL_MASK);
            } else {
                // Intra-lane compare-exchange between registers
                for (int r = 0; r < m; ++r) {
                    int ro = r ^ j;
                    if (ro > r) {
                        float a = dist[r], b = dist[ro];
                        int ia = id[r],   ib = id[ro];

                        int i_global = (int)(lane * (unsigned)m + (unsigned)r);
                        bool up = ((i_global & kstage) == 0);

                        bool swap = up ? (a > b) : (a < b);
                        if (swap) {
                            dist[r]  = b;  id[r]  = ib;
                            dist[ro] = a;  id[ro] = ia;
                        }
                    }
                }
                __syncwarp(FULL_MASK);
            }
        }
    }
}

// Merge the warp's candidate buffer (in shared memory) with the warp's intermediate result (in registers).
// Steps implemented:
// 0. Intermediate result (in registers) is assumed sorted ascending.
// 1. Swap the content of the buffer (shared) and the intermediate result (registers) so that the buffer is in registers.
//    We do a true swap per element between registers and shared memory.
// 2. Sort the buffer now held in registers using bitonic sort (ascending).
// 3. Merge: for each position i in [0,k), compute j = k-1-i, read shared element at j,
//    and set registers[i] = min(registers[i], shared[j]). This forms a bitonic sequence.
// 4. Sort the merged sequence in registers using bitonic sort to obtain updated intermediate result.
// Finally, update maxDistance to the last element (k-th neighbor).
__device__ __forceinline__ void merge_buffer_with_result(
    float dist[MAX_M], int id[MAX_M],           // registers (intermediate result in/out)
    float* sBufDistWarp, int* sBufIdxWarp,      // shared memory buffer for this warp (size k)
    int k, int m, unsigned lane,                // sizes and lane id
    float& maxDistance,                         // in/out: updated after merge
    int bufCount                                // current number of valid candidates in shared buffer
) {
    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // If buffer not full, pad remaining slots with +inf and invalid index.
    if (bufCount < k) {
        for (int r = 0; r < m; ++r) {
            int i = (int)(lane * (unsigned)m + (unsigned)r);
            if (i >= bufCount && i < k) {
                sBufDistWarp[i] = CUDART_INF_F;
                sBufIdxWarp[i]  = -1;
            }
        }
        __syncwarp(FULL_MASK);
    }

    // Step 1: Swap between registers (result) and shared (buffer).
    for (int r = 0; r < m; ++r) {
        int i = (int)(lane * (unsigned)m + (unsigned)r);
        float tmpDist = dist[r];
        int   tmpIdx  = id[r];
        float bufDist = sBufDistWarp[i];
        int   bufIdx  = sBufIdxWarp[i];
        // Swap: registers receive buffer's content; shared receives previous registers' content.
        dist[r]         = bufDist;
        id[r]           = bufIdx;
        sBufDistWarp[i] = tmpDist;
        sBufIdxWarp[i]  = tmpIdx;
    }
    __syncwarp(FULL_MASK);

    // Step 2: Sort the buffer now in registers.
    warp_bitonic_sort(dist, id, m, k, lane);

    // Step 3: Merge by taking element-wise min of (reg[i], shared[k-1-i]).
    for (int r = 0; r < m; ++r) {
        int i = (int)(lane * (unsigned)m + (unsigned)r);
        int j = k - 1 - i;

        float sDist = sBufDistWarp[j];
        int   sIdx  = sBufIdxWarp[j];

        if (sDist < dist[r]) {
            dist[r] = sDist;
            id[r]   = sIdx;
        }
    }
    __syncwarp(FULL_MASK);

    // Step 4: Sort merged sequence to restore ascending order.
    warp_bitonic_sort(dist, id, m, k, lane);

    // Update maxDistance = distance of the k-th neighbor (last element).
    float kth = dist[m - 1];
    // Broadcast from lane 31
    maxDistance = __shfl_sync(FULL_MASK, kth, WARP_SIZE - 1, WARP_SIZE);
}

// Kernel implementing k-NN for 2D points using one warp per query.
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result,
    int k,
    int tile_points  // set to TILE_POINTS by host
) {
    // Sanity: warp size assumed 32
    static_assert(WARP_SIZE == 32, "Warp size must be 32");

    // Compute lane and warp identifiers
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & (WARP_SIZE - 1);
    const unsigned warp_in_block = tid >> 5; // warp id within block
    const unsigned warps_per_block = blockDim.x >> 5;
    const unsigned warp_global = blockIdx.x * warps_per_block + warp_in_block;

    if ((int)warp_global >= query_count) return;

    // Each warp processes one query
    const float2 q = query[warp_global];

    // Register storage for intermediate result: top-k distances and indices.
    // Each thread stores m = k/32 consecutive elements in ascending order.
    const int m = k / WARP_SIZE;
    float regDist[MAX_M];
    int   regIdx[MAX_M];

    // Initialize registers to +inf and invalid indices.
    for (int r = 0; r < m; ++r) {
        regDist[r] = CUDART_INF_F;
        regIdx[r]  = -1;
    }

    // Shared memory layout:
    // [0]   float sBufDist[warps_per_block * k]
    // [..]  int   sBufIdx [warps_per_block * k]
    // [..]  int   sBufCount[warps_per_block]
    // [..]  float2 sTile[tile_points]
    extern __shared__ unsigned char smem[];
    size_t offset = 0;

    auto align_up = [](size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); };

    // Distances buffer
    offset = align_up(offset, alignof(float));
    float* sBufDist = reinterpret_cast<float*>(smem + offset);
    offset += (size_t)warps_per_block * (size_t)k * sizeof(float);

    // Indices buffer
    offset = align_up(offset, alignof(int));
    int* sBufIdx = reinterpret_cast<int*>(smem + offset);
    offset += (size_t)warps_per_block * (size_t)k * sizeof(int);

    // Buffer counts
    offset = align_up(offset, alignof(int));
    int* sBufCount = reinterpret_cast<int*>(smem + offset);
    offset += (size_t)warps_per_block * sizeof(int);

    // Data tile cache
    offset = align_up(offset, alignof(float2));
    float2* sTile = reinterpret_cast<float2*>(smem + offset);
    // The shared memory allocation from host ensures enough space for tile_points float2s.

    // Per-warp views into shared memory
    float* sBufDistWarp = sBufDist + warp_in_block * k;
    int*   sBufIdxWarp  = sBufIdx  + warp_in_block * k;

    // Initialize candidate buffer count for this warp
    if (lane == 0) sBufCount[warp_in_block] = 0;
    __syncwarp();

    // Initial max distance is +infinity
    float maxDistance = CUDART_INF_F;

    // Process data in tiles cached into shared memory by the whole block
    for (int tileStart = blockIdx.y * tile_points; tileStart < data_count; tileStart += gridDim.y * tile_points) {
        int remaining = data_count - tileStart;
        int tileCount = remaining > tile_points ? tile_points : remaining;

        // Cooperative load of the tile
        for (int t = threadIdx.x; t < tileCount; t += blockDim.x) {
            sTile[t] = data[tileStart + t];
        }
        __syncthreads();

        // Each warp processes the cached tile
        for (int base = 0; base < tileCount; base += WARP_SIZE) {
            int idxInTile = base + lane;
            float2 p;
            float d = CUDART_INF_F;
            int globalIdx = -1;

            if (idxInTile < tileCount) {
                p = sTile[idxInTile];
                d = squared_l2(q, p);
                globalIdx = tileStart + idxInTile;
            }

            // Filter by maxDistance
            unsigned FULL_MASK = 0xFFFFFFFFu;
            unsigned candMask = __ballot_sync(FULL_MASK, d < maxDistance);
            // Process candidates possibly in multiple rounds if buffer capacity is limited
            while (candMask) {
                // Determine available slots in the buffer
                int bufCount = sBufCount[warp_in_block];
                int available = k - bufCount;
                if (available == 0) {
                    // Merge buffer with current result
                    merge_buffer_with_result(regDist, regIdx, sBufDistWarp, sBufIdxWarp, k, m, lane, maxDistance, bufCount);
                    if (lane == 0) sBufCount[warp_in_block] = 0;
                    __syncwarp(FULL_MASK);
                    bufCount = 0;
                    available = k;
                }

                if (available > WARP_SIZE) available = WARP_SIZE;

                // Take the first 'available' set bits from candMask
                unsigned before = candMask & ((1u << lane) - 1u);
                int rankInCand = __popc(before);
                int isSet = ((candMask >> lane) & 1u);
                int takeThis = isSet && (rankInCand < available);
                unsigned takeMask = __ballot_sync(FULL_MASK, takeThis);
                int takeCount = __popc(takeMask);

                // Reserve positions in buffer and broadcast base offset
                int basePos = 0;
                if (lane == 0) {
                    basePos = sBufCount[warp_in_block];
                    sBufCount[warp_in_block] = basePos + takeCount;
                }
                basePos = __shfl_sync(FULL_MASK, basePos, 0, WARP_SIZE);

                if (takeThis) {
                    int rankTake = __popc(takeMask & ((1u << lane) - 1u));
                    int writePos = basePos + rankTake;
                    // Write candidate to shared buffer
                    sBufDistWarp[writePos] = d;
                    sBufIdxWarp[writePos]  = globalIdx;
                }

                // Remove the taken subset; loop continues if there are more candidates
                candMask &= ~takeMask;
                __syncwarp(FULL_MASK);
            }
        }
        __syncthreads();
    }

    // After the last tiles, merge remaining candidates if any
    int finalCount = sBufCount[warp_in_block];
    if (finalCount > 0) {
        merge_buffer_with_result(regDist, regIdx, sBufDistWarp, sBufIdxWarp, k, m, lane, maxDistance, finalCount);
        if (lane == 0) sBufCount[warp_in_block] = 0;
        __syncwarp();
    }

    // Write final sorted top-k results to global memory
    // Output layout: result[query * k + j] = pair(index, distance), j in [0,k)
    std::pair<int, float>* out = result + ((size_t)warp_global * (size_t)k);
    for (int r = 0; r < m; ++r) {
        int pos = (int)(lane * (unsigned)m + (unsigned)r);
        // regDist/regIdx are sorted ascending
        out[pos].first  = regIdx[r];
        out[pos].second = regDist[r];
    }
}

// Host API wrapper
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Choose grid/block configuration.
    // - One warp per query: with WARPS_PER_BLOCK warps per block, we need ceil(query_count / WARPS_PER_BLOCK) blocks along x.
    // - gridDim.y is used to pipeline tiles of data across blocks in y-dimension; choose a small factor (e.g., 1 or 2).
    //   We'll use 1 by default to maximize per-block shared memory for tiles.
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    // Dynamic shared memory size:
    // Per block:
    // - Candidate buffers: WARPS_PER_BLOCK * (k*(sizeof(float) + sizeof(int)) + sizeof(int))
    // - Data tile: TILE_POINTS * sizeof(float2)
    size_t smem_bytes = 0;
    smem_bytes += (size_t)WARPS_PER_BLOCK * (size_t)k * sizeof(float); // distances
    smem_bytes += (size_t)WARPS_PER_BLOCK * (size_t)k * sizeof(int);   // indices
    smem_bytes += (size_t)WARPS_PER_BLOCK * sizeof(int);               // counts
    smem_bytes += (size_t)TILE_POINTS * sizeof(float2);                // data tile

    // Launch kernel
    knn_kernel<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k, TILE_POINTS);

    // No device sync or error checks here; assume caller handles them if needed.
}