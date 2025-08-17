#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

// This implementation computes k-NN for 2D points using one warp (32 threads) per query.
// The algorithm processes the data points in shared-memory tiles. For each query:
// - An intermediate result of k nearest neighbors is kept sorted in ascending order across warp registers,
//   distributed so that each thread keeps k/32 consecutive elements (chunk = k/32).
// - A per-warp candidate buffer in shared memory accumulates up to k candidates at a time.
// - Distances below the current max_distance (k-th best so far) are appended to the candidate buffer using
//   warp ballot and lane-local prefix to place them contiguously.
// - When the candidate buffer becomes full (or at the end), it is merged with the intermediate result by:
//     1) Sorting the candidate buffer in ascending order using a warp-wide Bitonic sort.
//     2) Producing the minima between cand[i] and reversed intermediate[k-1-i] into registers, forming a bitonic sequence of k elements.
//     3) Sorting this merged bitonic sequence with a warp-wide Bitonic sort to obtain the updated intermediate result.
// - The Bitonic sort is implemented with warp shuffles for inter-thread compare-exchange when the partner is in another lane,
//   and with in-register swaps when the partner is within the same thread (j < chunk). For cross-lane exchanges, the compared
//   elements always share the same index within each thread's registers because of the block distribution (consecutive elements per thread).

// NOTE ABOUT std::pair IN DEVICE MEMORY:
// The device kernel writes results via a POD struct with identical layout to std::pair<int,float>.
struct PairIF {
    int   first;
    float second;
};

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Number of data points to cache per block in shared memory per tile.
// 2048 points * 8 bytes/point = 16KB shared for the tile.
// Candidate buffer per warp at worst k=1024 -> 8KB. With 8 warps/block = 64KB.
// Total ~80KB (plus minor overhead), safely under 96KB common per-block SMEM without special attributes.
#ifndef TILE_POINTS
#define TILE_POINTS 2048
#endif

// Compute squared Euclidean distance between two float2 points.
__device__ __forceinline__ float sq_l2(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // Use FMA for precision and throughput: dx*dx + dy*dy
    return fmaf(dx, dx, dy * dy);
}

// Warp-wide Bitonic sort for a distributed array of length K = k, where each lane owns 'chunk = k/32' consecutive elements.
// - dist[0..chunk-1], idx[0..chunk-1] are the registers for the local chunk in each thread/lane.
// - The logical global position of local element r in lane L is g = L*chunk + r.
// - For j >= chunk, partner resides in another lane but at the same local offset: lane' = L ^ (j/chunk), offset r.
// - For j < chunk, partner resides in the same lane at offset r^j.
// This function sorts the k elements across the warp in ascending order by distance.
__device__ __forceinline__ void warp_bitonic_sort_blocked(int k, int chunk, float dist[], int idx[]) {
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int base = lane * chunk;

    // Outer Bitonic build/merge stages
    for (int Kstage = 2; Kstage <= k; Kstage <<= 1) {
        // Inter-lane compare-exchange stages (partners in different lanes, same local offset)
        for (int j = Kstage >> 1; j >= chunk; j >>= 1) {
            const int partner_lane = lane ^ (j / chunk);
            // For each local register
            for (int r = 0; r < chunk; ++r) {
                float a = dist[r];
                int   ia = idx[r];
                // Exchange with partner lane, same register index r
                float b = __shfl_sync(FULL_MASK, a, partner_lane);
                int   ib = __shfl_sync(FULL_MASK, ia, partner_lane);
                // Direction of comparison for this position
                const bool up = (((base + r) & Kstage) == 0);
                // For 'up', keep min at this position; for 'down', keep max at this position.
                const bool take_partner = ((a > b) == up);
                if (take_partner) { a = b; ia = ib; }
                dist[r] = a; idx[r] = ia;
            }
            __syncwarp();
        }
        // Intra-lane compare-exchange stages (partners in same lane, offsets differ by j)
        for (int j = (chunk > (Kstage >> 1) ? (Kstage >> 1) : (chunk - 1)); j >= 1; j >>= 1) {
            // Process each pair exactly once: if r2 > r perform the swap for (r, r2)
            for (int r = 0; r < chunk; ++r) {
                int r2 = r ^ j;
                if (r2 > r) {
                    float a = dist[r];   int ia = idx[r];
                    float b = dist[r2];  int ib = idx[r2];
                    // For intra-lane pairs, the 'i' used for direction is the smaller index => base + r (since r2 > r)
                    const bool up = (((base + r) & Kstage) == 0);
                    const bool do_swap = ((a > b) == up);
                    if (do_swap) {
                        float tmp = a; a = b; b = tmp;
                        int   ti  = ia; ia = ib; ib = ti;
                    }
                    dist[r]  = a;  idx[r]  = ia;
                    dist[r2] = b;  idx[r2] = ib;
                }
            }
            __syncwarp();
        }
    }
}

// Merge the per-warp shared-memory candidate buffer with the current intermediate result in registers.
// After completion:
// - best_dist/idx hold the updated intermediate result (sorted ascending).
// - maxDist is updated to the k-th (largest) element.
// The shared candidate count is reset to 0.
__device__ __forceinline__ void warp_flush_merge(
    int k, int chunk,
    float best_dist[], int best_idx[],
    float buf_dist[], int buf_idx[],
    volatile int* cand_count_ptr, float* cand_dist_s, int* cand_idx_s,
    float &maxDist)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int base = lane * chunk;

    // Read candidate count (warp-private) and broadcast
    int count = 0;
    if (lane == 0) count = *cand_count_ptr;
    count = __shfl_sync(FULL_MASK, count, 0);

    // Load candidates into registers; pad with +INF where no candidate exists
    for (int r = 0; r < chunk; ++r) {
        const int g = base + r;
        if (g < count) {
            buf_dist[r] = cand_dist_s[g];
            buf_idx[r]  = cand_idx_s[g];
        } else {
            buf_dist[r] = FLT_MAX;
            buf_idx[r]  = -1;
        }
    }
    __syncwarp();

    // Step 2: Sort the buffer in ascending order
    warp_bitonic_sort_blocked(k, chunk, buf_dist, buf_idx);

    // Step 3: Merge minima with reversed intermediate result in registers.
    // For global logical position p = base + r, the reversed partner index is q = k-1-p.
    // Its lane is lane' = 31 - lane and offset r' = chunk - 1 - r.
    for (int r = 0; r < chunk; ++r) {
        const int partner_lane = (WARP_SIZE - 1) - lane; // 31 - lane
        // Provide per-lane variable as the source for shfl; all lanes use the same r, so offsets match.
        float best_rev_val = __shfl_sync(FULL_MASK, best_dist[chunk - 1 - r], partner_lane);
        int   best_rev_idx = __shfl_sync(FULL_MASK, best_idx[chunk - 1 - r], partner_lane);
        float a = buf_dist[r];
        int   ia = buf_idx[r];
        if (best_rev_val < a) { a = best_rev_val; ia = best_rev_idx; }
        buf_dist[r] = a;
        buf_idx[r]  = ia;
    }
    __syncwarp();

    // Step 4: Sort the merged result (bitonic sequence) in ascending order
    warp_bitonic_sort_blocked(k, chunk, buf_dist, buf_idx);

    // Copy merged sorted result back into intermediate registers
    for (int r = 0; r < chunk; ++r) {
        best_dist[r] = buf_dist[r];
        best_idx[r]  = buf_idx[r];
    }
    __syncwarp();

    // Update maxDist to the k-th (largest) element
    float kth_local = (lane == (WARP_SIZE - 1)) ? best_dist[chunk - 1] : 0.0f;
    float kth = __shfl_sync(FULL_MASK, kth_local, (WARP_SIZE - 1));
    maxDist = kth;

    // Reset candidate buffer count
    if (lane == 0) *cand_count_ptr = 0;
    __syncwarp();
}

// CUDA kernel: One warp per query
__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    PairIF* __restrict__ result, int k)
{
    // Thread identification
    const int lane             = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block    = threadIdx.x >> 5;                 // threadIdx.x / 32
    const int warps_per_block  = blockDim.x >> 5;
    const int warp_global      = blockIdx.x * warps_per_block + warp_in_block;
    const bool warpActive      = (warp_global < query_count);
    const unsigned FULL_MASK   = 0xFFFFFFFFu;

    // Each thread maintains 'chunk = k/32' consecutive elements in registers
    const int chunk = k >> 5; // k/32; k is power of two between 32 and 1024

    // Shared memory organization:
    // [ float2 tile[TILE_POINTS] ][ int cand_count[warps_per_block] ]
    // [ float cand_dist[warps_per_block * k] ][ int cand_idx[warps_per_block * k] ]
    extern __shared__ unsigned char smem_raw[];
    float2* smem_points = reinterpret_cast<float2*>(smem_raw);
    size_t  smem_off    = TILE_POINTS * sizeof(float2);

    int*   cand_count_s = reinterpret_cast<int*>(smem_raw + smem_off);
    smem_off += warps_per_block * sizeof(int);

    float* cand_dist_s  = reinterpret_cast<float*>(smem_raw + smem_off);
    smem_off += static_cast<size_t>(warps_per_block) * k * sizeof(float);

    int*   cand_idx_s   = reinterpret_cast<int*>(smem_raw + smem_off);
    // smem_off ends here

    // Pointers to this warp's candidate buffer in shared memory
    volatile int* my_cand_count = cand_count_s + warp_in_block;
    float* my_cand_dist = cand_dist_s + warp_in_block * k;
    int*   my_cand_idx  = cand_idx_s  + warp_in_block * k;

    // Initialize candidate buffer count to zero
    if (lane == 0) *my_cand_count = 0;

    // Intermediate result stored in registers: sorted ascending
    float best_dist[WARP_SIZE]; // allocate max 32; only [0..chunk-1] are used
    int   best_idx [WARP_SIZE];
    #pragma unroll
    for (int r = 0; r < WARP_SIZE; ++r) {
        if (r < chunk) {
            best_dist[r] = FLT_MAX;
            best_idx[r]  = -1;
        }
    }

    // Max distance (current k-th neighbor distance), replicated in all lanes
    float maxDist = FLT_MAX;

    // Load query point, broadcast across the warp
    float qx = 0.0f, qy = 0.0f;
    if (warpActive && lane == 0) {
        float2 q = query[warp_global];
        qx = q.x; qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);
    const float2 qpt = make_float2(qx, qy);

    // Iterate over data tiles
    for (int tileBase = 0; tileBase < data_count; tileBase += TILE_POINTS) {
        const int tileCount = min(TILE_POINTS, data_count - tileBase);

        // Load tile to shared memory cooperatively by the whole block
        for (int t = threadIdx.x; t < tileCount; t += blockDim.x) {
            smem_points[t] = data[tileBase + t];
        }
        __syncthreads();

        // Process this tile: each warp computes distances for its query
        if (warpActive) {
            // Iterate over points in the tile; one per lane each iteration
            for (int t = lane; t < tileCount; t += WARP_SIZE) {
                float2 dp = smem_points[t];
                float d = sq_l2(qpt, dp);

                // Filter by current maxDist and ballot across the warp
                const bool isCand = (d < maxDist);
                const unsigned m = __ballot_sync(FULL_MASK, isCand);
                const int n = __popc(m);

                if (n > 0) {
                    // Read current count (lane 0) and broadcast
                    int curCount = 0;
                    if (lane == 0) curCount = *my_cand_count;
                    curCount = __shfl_sync(FULL_MASK, curCount, 0);

                    // If adding n candidates would overflow the buffer, flush/merge first
                    int needFlush = (curCount + n > k);
                    needFlush = __shfl_sync(FULL_MASK, needFlush, 0);
                    if (needFlush) {
                        warp_flush_merge(k, chunk,
                                         best_dist, best_idx,
                                         // use these regs as temporary 'buf'
                                         best_dist /*unused*/, best_idx /*unused*/,
                                         my_cand_count, my_cand_dist, my_cand_idx,
                                         maxDist);
                        // After flush, cand_count == 0
                        curCount = 0;
                    }

                    // Reserve positions [curCount, curCount + n) for these candidates
                    if (lane == 0) *my_cand_count = curCount + n;
                    const int base = __shfl_sync(FULL_MASK, curCount, 0);

                    // Lane-local offset using prefix of ballot mask
                    const int pos = __popc(m & ((1u << lane) - 1));
                    if (isCand) {
                        const int writeIdx = base + pos;
                        my_cand_dist[writeIdx] = d;
                        my_cand_idx [writeIdx] = tileBase + t;
                    }
                    __syncwarp();

                    // If buffer just became full, flush immediately
                    int newCount = 0;
                    if (lane == 0) newCount = *my_cand_count;
                    newCount = __shfl_sync(FULL_MASK, newCount, 0);
                    int doFlush = (newCount == k);
                    doFlush = __shfl_sync(FULL_MASK, doFlush, 0);
                    if (doFlush) {
                        warp_flush_merge(k, chunk,
                                         best_dist, best_idx,
                                         // reuse scratch regs
                                         best_dist /*unused*/, best_idx /*unused*/,
                                         my_cand_count, my_cand_dist, my_cand_idx,
                                         maxDist);
                    }
                }

                __syncwarp(); // Warp-level barrier for safety between iterations
            }
        }

        __syncthreads(); // Ensure tile is no longer in use before loading the next one
    }

    // After processing all tiles, flush any remaining candidates
    if (warpActive) {
        int remaining = 0;
        if (lane == 0) remaining = *my_cand_count;
        remaining = __shfl_sync(FULL_MASK, remaining, 0);
        if (remaining > 0) {
            warp_flush_merge(k, chunk,
                             best_dist, best_idx,
                             // reuse scratch regs
                             best_dist /*unused*/, best_idx /*unused*/,
                             my_cand_count, my_cand_dist, my_cand_idx,
                             maxDist);
        }
    }

    // Write back results: for query 'warp_global', sorted nearest neighbors in ascending order.
    if (warpActive) {
        PairIF* out = result + static_cast<size_t>(warp_global) * k;
        const int base = lane * chunk;
        for (int r = 0; r < chunk; ++r) {
            const int g = base + r;
            out[g].first  = best_idx[r];
            out[g].second = best_dist[r];
        }
    }
}

// Host API entry point.
// query: device pointer to float2 queries of size query_count
// data:  device pointer to float2 data points of size data_count
// result: device pointer to std::pair<int,float> array of size query_count * k (contiguous)
// k: power of two in [32, 1024], data_count >= k
/// @FIXED
/// extern "C"

void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Select threads/block and warps/block. Use 8 warps per block (256 threads).
    const int warpsPerBlock = 8;
    const int threadsPerBlock = warpsPerBlock * WARP_SIZE;
    const int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Dynamic shared memory size:
    // - Tile: TILE_POINTS * sizeof(float2)
    // - Candidate counts: warpsPerBlock * sizeof(int)
    // - Candidate buffers (dist + idx): warpsPerBlock * k * (sizeof(float) + sizeof(int))
    size_t smem_size = static_cast<size_t>(TILE_POINTS) * sizeof(float2)
                     + static_cast<size_t>(warpsPerBlock) * sizeof(int)
                     + static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    // Launch kernel. We reinterpret the result pointer to a POD struct with identical layout.
    knn_kernel<<<blocks, threadsPerBlock, smem_size>>>(
        query, query_count,
        data,  data_count,
        reinterpret_cast<PairIF*>(result),
        k);

    // Synchronize to ensure completion before returning.
    cudaDeviceSynchronize();
}
