#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <limits>

// This implementation assigns one full warp (32 threads) to process a single query point.
// Each warp maintains a distributed top-k structure across its 32 lanes.
// The dataset is processed in shared-memory cached tiles for better memory bandwidth utilization.
// Within each tile, we further process data in sub-chunks to limit per-thread register pressure.
// For each sub-chunk, each lane computes distances for a small number (<= 16) of points, sorts them locally,
// and then the warp performs a multiway merge by repeatedly inserting the current best candidate into the
// global top-k (distributed across lanes), replacing the current worst element. Warp-wide argmin/argmax
// reductions via shuffles are used to coordinate insertions. The intermediate top-k per query is stored
// in shared memory, so no additional global memory allocations are required.
// After processing all tiles, the warp performs a distributed bitonic sort across its k results to produce
// the final ascending order required by the API, and then writes the (index, distance) pairs to the result.

namespace {

// Hardware-tuned constants. Adjust with care; they balance occupancy, shared memory use, and arithmetic.
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;       // 8 warps -> 256 threads per block
constexpr int TILE_SIZE = 4096;          // Number of data points per shared-memory tile (float2 => 32KB)
constexpr int MAX_CHUNK = 512;           // Sub-chunk size per warp per merge step (must be multiple of 32)
// MAX_CHUNK/32 = 16 local candidates per lane; keeps register usage modest.

// Warp utilities
__device__ __forceinline__ int lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}

__device__ __forceinline__ int warp_id_in_block() {
    return threadIdx.x >> 5;
}

__device__ __forceinline__ int warps_per_block() {
    return blockDim.x >> 5;
}

// Compute squared Euclidean distance between a query point q and data point p.
__device__ __forceinline__ float l2_sq(const float2& q, const float2& p) {
    float dx = q.x - p.x;
    float dy = q.y - p.y;
    // FMA for precision/perf: dx*dx + dy*dy
    return fmaf(dx, dx, dy * dy);
}

// Warp-wide argmax reduction: returns (maxVal, maxLane) broadcast to all lanes in the warp.
__device__ __forceinline__ void warp_argmax(float v, int lane, float& outVal, int& outLane, unsigned mask = 0xffffffffu) {
    float bestVal = v;
    int bestLane = lane;
    // Tree reduction using shuffles
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float othVal = __shfl_down_sync(mask, bestVal, offset);
        int othLane = __shfl_down_sync(mask, bestLane, offset);
        if (othVal > bestVal) {
            bestVal = othVal;
            bestLane = othLane;
        }
    }
    // Broadcast winner to all lanes
    outVal = __shfl_sync(mask, bestVal, 0);
    outLane = __shfl_sync(mask, bestLane, 0);
}

// Warp-wide argmin reduction: returns (minVal, minLane) broadcast to all lanes in the warp.
__device__ __forceinline__ void warp_argmin(float v, int lane, float& outVal, int& outLane, unsigned mask = 0xffffffffu) {
    float bestVal = v;
    int bestLane = lane;
    // Tree reduction using shuffles
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float othVal = __shfl_down_sync(mask, bestVal, offset);
        int othLane = __shfl_down_sync(mask, bestLane, offset);
        if (othVal < bestVal) {
            bestVal = othVal;
            bestLane = othLane;
        }
    }
    // Broadcast winner to all lanes
    outVal = __shfl_sync(mask, bestVal, 0);
    outLane = __shfl_sync(mask, bestLane, 0);
}

// Simple insertion sort for very small arrays in registers (ascending by value).
template <int N>
__device__ __forceinline__ void insertion_sort_asc(float (&vals)[N], int (&idxs)[N], int count) {
    #pragma unroll
    for (int i = 1; i < N; ++i) {
        if (i >= count) break;
        float keyV = vals[i];
        int keyI = idxs[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > keyV) {
            vals[j + 1] = vals[j];
            idxs[j + 1] = idxs[j];
            --j;
        }
        vals[j + 1] = keyV;
        idxs[j + 1] = keyI;
    }
}

// Swap utility
__device__ __forceinline__ void swap_float(float& a, float& b) { float t = a; a = b; b = t; }
__device__ __forceinline__ void swap_int(int& a, int& b) { int t = a; a = b; b = t; }

} // namespace

// Kernel implementing warp-per-query KNN with shared-memory tiling and warp-cooperative top-k maintenance.
__global__ void knn_warp_kernel(const float2* __restrict__ query,
                                int query_count,
                                const float2* __restrict__ data,
                                int data_count,
                                std::pair<int, float>* __restrict__ result,
                                int k) {
    // Map threads to warps and queries
    const int lane = lane_id();
    const int wid = warp_id_in_block();
    const int wpb = warps_per_block();
    const int warp_global = blockIdx.x * wpb + wid;
    if (warp_global >= query_count) return;

    // Shared memory layout:
    // [0, TILE_SIZE) float2 sData
    // then W warps' A-dist arrays (float): size W*k
    // then W warps' A-idx arrays (int): size W*k
    extern __shared__ unsigned char smem_raw[];
    float2* sData = reinterpret_cast<float2*>(smem_raw);
    size_t offset = static_cast<size_t>(TILE_SIZE) * sizeof(float2);

    float* smem_A_dist_all = reinterpret_cast<float*>(smem_raw + offset);
    offset += static_cast<size_t>(wpb) * static_cast<size_t>(k) * sizeof(float);

    int* smem_A_idx_all = reinterpret_cast<int*>(smem_raw + offset);
    // No further shared memory allocations.

    // Per-warp A arrays in shared memory
    float* A_dist = smem_A_dist_all + static_cast<size_t>(wid) * static_cast<size_t>(k);
    int*   A_idx  = smem_A_idx_all  + static_cast<size_t>(wid) * static_cast<size_t>(k);

    // Query point for this warp
    const float2 q = query[warp_global];

    // Per-thread chunk size parameters
    const int m = k / WARP_SIZE;                    // elements per lane for A
    const int chunk = (k < MAX_CHUNK ? k : MAX_CHUNK); // choose sub-chunk size
    const int bmax = MAX_CHUNK / WARP_SIZE;         // max local candidates per lane (compile-time 16)
    int bslots = chunk / WARP_SIZE;                 // runtime slots per lane for sub-chunk

    // Initialize A with +inf distances and invalid indices (-1), keep local descending order
    // We place the local worst at position 0 for each lane's block of size m.
    for (int i = 0; i < m; ++i) {
        const int g = lane * m + i; // global index into A arrays for this lane
        A_dist[g] = CUDART_INF_F;   // Initially worst at head (position 0)
        A_idx[g] = -1;
    }
    __syncwarp();

    // Process data in shared-memory tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        const int tile_count = min(TILE_SIZE, data_count - tile_start);

        // Load tile into shared memory cooperatively
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            sData[t] = data[tile_start + t];
        }
        __syncthreads();

        // Within this tile, process in sub-chunks sized 'chunk'
        for (int sub = 0; sub < tile_count; sub += chunk) {
            const int sub_count = min(chunk, tile_count - sub);
            // Number of local candidates per lane for this sub-chunk
            bslots = (sub_count + WARP_SIZE - 1) / WARP_SIZE; // ceil
            if (bslots > bmax) bslots = bmax;

            // Local candidate buffers in registers (ascending after sort)
            float bvals[bmax];
            int   bidxs[bmax];

            // Fill local candidates with distances and global indices
            #pragma unroll
            for (int i = 0; i < bmax; ++i) {
                int idx_in_sub = i * WARP_SIZE + lane; // sub-chunk-local index processed by this lane
                int sidx = sub + idx_in_sub;
                if (i < bslots && sidx < tile_count) {
                    // Valid candidate
                    float2 p = sData[sidx];
                    float dist = l2_sq(q, p);
                    bvals[i] = dist;
                    bidxs[i] = tile_start + sidx;
                } else {
                    // Pad with +inf (ignored)
                    bvals[i] = CUDART_INF_F;
                    bidxs[i] = -1;
                }
            }

            // Sort local candidates ascending so bvals[head] is current best within lane
            insertion_sort_asc<bmax>(bvals, bidxs, bslots);

            // Initialize per-lane pointer into B list
            int phead = 0;

            // Prepare initial worst (tau) across A: read this lane's local worst at position 0
            float laneWorst = A_dist[lane * m + 0];
            float tau; int worstLane;
            warp_argmax(laneWorst, lane, tau, worstLane);

            // Warp-cooperative multiway merge: repeatedly insert best among lanes if it improves tau
            while (true) {
                // Each lane contributes its current best candidate (or +inf if exhausted)
                float candVal = (phead < bslots ? bvals[phead] : CUDART_INF_F);
                int   candIdx = (phead < bslots ? bidxs[phead] : -1);

                float bestVal; int bestLane;
                warp_argmin(candVal, lane, bestVal, bestLane);
                // Broadcast best candidate's index from bestLane
                int bestIdx = __shfl_sync(0xffffffffu, candIdx, bestLane);

                // If the best available candidate does not improve over current worst, stop
                if (!(bestVal < tau)) break;

                // Replace current worst in A with best candidate and restore local descending order
                if (lane == worstLane) {
                    // Replace head (position 0) and bubble down to maintain descending order
                    int base = lane * m;
                    A_dist[base + 0] = bestVal;
                    A_idx [base + 0] = bestIdx;
                    // Bubble down until order A[0] >= A[1] >= ... holds for this lane's segment
                    for (int pos = 0; pos + 1 < m; ++pos) {
                        // If next is larger than current, swap
                        if (A_dist[base + pos] < A_dist[base + pos + 1]) {
                            swap_float(A_dist[base + pos], A_dist[base + pos + 1]);
                            swap_int  (A_idx [base + pos], A_idx [base + pos + 1]);
                        } else {
                            break;
                        }
                    }
                }
                // Advance head in best lane's local candidate list
                if (lane == bestLane) {
                    ++phead;
                }

                // Ensure A updates are visible before next tau computation
                __syncwarp();

                // Recompute tau = global worst across A for next iteration
                laneWorst = A_dist[lane * m + 0];
                warp_argmax(laneWorst, lane, tau, worstLane);
            }

            // End sub-chunk loop for this tile segment
            __syncwarp();
        }

        __syncthreads(); // Ensure all warps done reading this tile before loading the next
    }

    // Final sorting of A across the warp: produce ascending order required by API
    // We perform an in-place bitonic sort over k elements stored in shared memory at A_dist/A_idx.
    // Global index mapping for element g in [0, k): belongs to lane = g / m, slot = g % m.
    // Each thread iterates over its local m elements and participates in pairwise compare-exchange using shared memory.
    // After each compare-exchange stage, we synchronize the warp to ensure memory coherence.

    // Bitonic sort network (ascending)
    for (int size = 2; size <= k; size <<= 1) {
        // Stride starts at size/2 and halves down to 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each lane handles its m elements
            for (int s = 0; s < m; ++s) {
                int g = lane * m + s;           // global index in [0, k)
                int partner = g ^ stride;       // partner index for compare-exchange
                if (partner > g && partner < k) {
                    // Determine comparison direction (up = ascending; down = descending) for this pair
                    bool up = ((g & size) == 0);
                    float v1 = A_dist[g];
                    float v2 = A_dist[partner];
                    int i1 = A_idx[g];
                    int i2 = A_idx[partner];
                    // For ascending sort, swap if v1 > v2 (and vice versa for descending)
                    bool swapNeeded = up ? (v1 > v2) : (v1 < v2);
                    if (swapNeeded) {
                        A_dist[g] = v2; A_idx[g] = i2;
                        A_dist[partner] = v1; A_idx[partner] = i1;
                    }
                }
            }
            __syncwarp();
        }
    }

    // Write results to global memory: ascending by distance, result[q*k + j] = (index, distance)
    const int outBase = warp_global * k;
    for (int s = lane; s < k; s += WARP_SIZE) {
        result[outBase + s].first  = A_idx[s];
        result[outBase + s].second = A_dist[s];
    }
}

// Host interface
void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    // Basic validation (assumed valid per problem statement); k is power of two between 32 and 1024 inclusive
    const int wpb = WARPS_PER_BLOCK;
    const int threads_per_block = wpb * WARP_SIZE;
    const int blocks = (query_count + wpb - 1) / wpb;

    // Dynamic shared memory size per block:
    // TILE_SIZE * sizeof(float2) for data tile +
    // WARPS_PER_BLOCK * k * (sizeof(float) + sizeof(int)) for per-warp A arrays
    size_t smem_bytes =
        static_cast<size_t>(TILE_SIZE) * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    knn_warp_kernel<<<blocks, threads_per_block, smem_bytes>>>(query, query_count, data, data_count, result, k);
    // The caller is responsible for CUDA error checking and stream synchronization if desired.
}