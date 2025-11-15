#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// ================================================================
// Configuration constants
// ================================================================

static constexpr int WARP_SIZE          = 32;
static constexpr int MAX_K              = 1024;                // Maximum allowed k
static constexpr int MAX_K_PER_THREAD   = MAX_K / WARP_SIZE;   // 1024 / 32 = 32
static constexpr int WARPS_PER_BLOCK    = 4;                   // 4 warps per block
static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int BATCH_SIZE         = 1536;                // Must fit shared memory

// ================================================================
// Warp-level utilities
// ================================================================

/**
 * Warp-wide bitonic sort (ascending) over 32 lanes.
 * Each lane provides one key (distance) and associated value (index).
 * After the call, keys are sorted ascending across lanes 0..31.
 *
 * Implemented using __shfl_xor_sync with standard bitonic network.
 */
template <typename T, typename U>
__device__ __forceinline__ void warp_bitonic_sort_asc(T &key, U &val) {
    const unsigned mask = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    #pragma unroll
    for (int k = 2; k <= WARP_SIZE; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            T  other_key = __shfl_xor_sync(mask, key, j);
            U  other_val = __shfl_xor_sync(mask, val, j);
            bool up = ((lane & k) == 0);
            if ((key > other_key) == up) {
                key = other_key;
                val = other_val;
            }
        }
    }
}

/**
 * Warp-wide argmax reduction.
 * All lanes provide a value; returns the maximum value and its lane index
 * in maxVal and maxLane (broadcast to all lanes).
 */
__device__ __forceinline__
void warp_argmax(float val, float &maxVal, int &maxLane) {
    const unsigned mask = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    float bestVal = val;
    int   bestLane = lane;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float otherVal  = __shfl_down_sync(mask, bestVal,  offset);
        int   otherLane = __shfl_down_sync(mask, bestLane, offset);
        if (otherVal > bestVal) {
            bestVal  = otherVal;
            bestLane = otherLane;
        }
    }

    bestVal  = __shfl_sync(mask, bestVal,  0);
    bestLane = __shfl_sync(mask, bestLane, 0);

    maxVal  = bestVal;
    maxLane = bestLane;
}

/**
 * Bitonic sort on an array in shared memory using a single warp.
 * - dist/idx: base pointers to the warp's region in shared memory.
 * - n:        number of elements to sort (power of two, <= MAX_K).
 * - lane:     lane id in the warp (0..31).
 *
 * Uses standard block-level bitonic sort algorithm, but only 32 threads
 * cooperate to sort up to 1024 elements living in shared memory.
 */
__device__ __forceinline__
void warp_bitonic_sort_shared(float *dist, int *idx, int n, int lane) {
    // Bitonic sort for ascending order
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread processes multiple indices spaced by WARP_SIZE
            for (int i = lane; i < n; i += WARP_SIZE) {
                int partner = i ^ stride;
                if (partner > i && partner < n) {
                    bool up = ((i & size) == 0); // ascending for lower half
                    float a = dist[i];
                    float b = dist[partner];
                    if ((a > b) == up) {
                        // swap distances
                        dist[i]       = b;
                        dist[partner] = a;
                        // swap indices
                        int ia   = idx[i];
                        idx[i]       = idx[partner];
                        idx[partner] = ia;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// ================================================================
// k-NN kernel
// ================================================================

/**
 * Each warp processes a single query point.
 * - For each query, we maintain a distributed top-k structure:
 *   each lane holds k/32 candidates (distance/index pairs) sorted ascending.
 * - The block iterates over data in batches cached in shared memory.
 * - For each batch, all threads cooperatively load points into shared memory.
 * - Each warp then processes the batch: every iteration, each lane computes
 *   the distance of one candidate; the 32 candidates are warp-sorted and
 *   cooperatively inserted into the warp's top-k list.
 * - At the end, each warp gathers its top-k into shared memory, performs
 *   a warp-level bitonic sort over k elements, and writes the sorted k-NN
 *   to the output array.
 */
/// @FIXED
/// extern "C"
__global__ void knn_kernel_2d(
    const float2 * __restrict__ query,
    int query_count,
    const float2 * __restrict__ data,
    int data_count,
    std::pair<int, float> * __restrict__ result,
    int k
) {
    // Dynamic shared memory layout:
    // [ float2 sh_data[BATCH_SIZE] |
    //   float  sh_topk_dist[WARPS_PER_BLOCK * MAX_K] |
    //   int    sh_topk_idx [WARPS_PER_BLOCK * MAX_K] ]
    extern __shared__ __align__(16) unsigned char smem[];
    float2 *sh_data = reinterpret_cast<float2*>(smem);

    float *sh_topk_dist = reinterpret_cast<float*>(
        sh_data + BATCH_SIZE
    );
    int *sh_topk_idx = reinterpret_cast<int*>(
        sh_topk_dist + WARPS_PER_BLOCK * MAX_K
    );

    const unsigned FULL_MASK = 0xFFFFFFFFu;

    int tid           = threadIdx.x;
    int warpInBlock   = tid / WARP_SIZE;
    int lane          = tid & (WARP_SIZE - 1);
    int warpGlobal    = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;
    bool warpActive   = (warpGlobal < query_count);

    // Each warp is assigned exactly one query index
    int q_idx = warpGlobal;

    // Load query point for active warp, broadcast to all lanes
    float qx = 0.0f;
    float qy = 0.0f;
    if (warpActive && lane == 0) {
        float2 q = query[q_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Per-lane portion of top-k list
    int k_per_thread = k / WARP_SIZE;  // k is power of two and >= 32
    float best_dist[MAX_K_PER_THREAD];
    int   best_idx [MAX_K_PER_THREAD];

    // Initialize local top-k to +inf / -1
    #pragma unroll
    for (int i = 0; i < MAX_K_PER_THREAD; ++i) {
        best_dist[i] = FLT_MAX;
        best_idx[i]  = -1;
    }

    // Main loop: iterate over data in batches cached in shared memory
    for (int base = 0; base < data_count; base += BATCH_SIZE) {
        int batch_size = data_count - base;
        if (batch_size > BATCH_SIZE) batch_size = BATCH_SIZE;

        // Block-wide cooperative load into shared memory
        for (int i = tid; i < batch_size; i += blockDim.x) {
            sh_data[i] = data[base + i];
        }
        __syncthreads();

        // Each warp processes the cached batch
        // We traverse the batch in groups of 32 candidates so that each
        // lane produces one candidate per iteration.
        for (int jBase = 0; jBase < batch_size; jBase += WARP_SIZE) {
            int j = jBase + lane;

            float cand_dist = FLT_MAX;
            int   cand_idx  = -1;

            if (warpActive && j < batch_size) {
                float2 p = sh_data[j];
                float dx = p.x - qx;
                float dy = p.y - qy;
                cand_dist = dx * dx + dy * dy;
                cand_idx  = base + j;
            }

            // Sort the 32 candidates in the warp by distance (ascending)
            warp_bitonic_sort_asc(cand_dist, cand_idx);

            // Cooperatively insert the sorted candidates into the
            // distributed top-k structure.
            for (int r = 0; r < WARP_SIZE; ++r) {
                // Broadcast r-th smallest candidate in this group to all lanes
                float d   = __shfl_sync(FULL_MASK, cand_dist, r);
                int   idx = __shfl_sync(FULL_MASK, cand_idx,  r);

                // If candidate index is invalid, no more valid candidates
                if (idx < 0) break;

                if (!warpActive) continue;

                // Compute global worst (largest distance) among current top-k
                float localWorst = best_dist[k_per_thread - 1];
                float maxVal;
                int   maxLane;
                warp_argmax(localWorst, maxVal, maxLane);

                // If the candidate is not better than the worst, and since
                // candidates are processed in ascending order within this
                // group, no subsequent candidate in this group can improve.
                if (d >= maxVal) break;

                // The lane owning the global worst element inserts the new
                // candidate into its local sorted list (ascending order).
                if (lane == maxLane) {
                    int pos = k_per_thread - 1;
                    // Insertion into local list with removal of current worst
                    while (pos > 0 && d < best_dist[pos - 1]) {
                        best_dist[pos] = best_dist[pos - 1];
                        best_idx[pos]  = best_idx[pos - 1];
                        --pos;
                    }
                    best_dist[pos] = d;
                    best_idx[pos]  = idx;
                }
            } // candidates in this group
        } // groups of WARP_SIZE within batch

        __syncthreads();
    } // batches

    // Gather top-k from per-lane arrays into shared memory and
    // perform a warp-level bitonic sort over the k elements.
    int warpBase = warpInBlock * MAX_K;
    float *warp_dist = sh_topk_dist + warpBase;
    int   *warp_idx  = sh_topk_idx  + warpBase;

    if (warpActive) {
        // Copy each lane's local k_per_thread elements into warp-shared arrays
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_THREAD; ++i) {
            if (i < k_per_thread) {
                int pos = lane * k_per_thread + i; // 0..k-1
                warp_dist[pos] = best_dist[i];
                warp_idx [pos] = best_idx[i];
            }
        }

        __syncwarp();

        // Sort first k elements in shared memory using the warp
        warp_bitonic_sort_shared(warp_dist, warp_idx, k, lane);

        __syncwarp();

        // Write sorted k-NN results to global memory
        int out_base = q_idx * k;
        for (int pos = lane; pos < k; pos += WARP_SIZE) {
            int out_idx = out_base + pos;
            result[out_idx].first  = warp_idx[pos];
            result[out_idx].second = warp_dist[pos];
        }
    }
}

// ================================================================
// Host wrapper
// ================================================================

/**
 * Host interface:
 *   query       - device pointer to query points (size query_count)
 *   query_count - number of queries
 *   data        - device pointer to data points (size data_count)
 *   data_count  - number of data points
 *   result      - device pointer to output pairs (size query_count * k)
 *   k           - number of neighbors (power of two, 32..1024)
 */
void run_knn(
    const float2 *query,
    int query_count,
    const float2 *data,
    int data_count,
    std::pair<int, float> *result,
    int k
) {
    // Compute grid configuration: one warp per query
    int warps_needed = query_count;
    int blocks = (warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Shared memory size per block
    size_t shmem_size =
        sizeof(float2) * BATCH_SIZE +
        WARPS_PER_BLOCK * MAX_K * (sizeof(float) + sizeof(int));

    dim3 grid(blocks);
    dim3 block(THREADS_PER_BLOCK);

    knn_kernel_2d<<<grid, block, shmem_size>>>(
        query,
        query_count,
        data,
        data_count,
        result,
        k
    );

    // Optional error checking/synchronization can be added here if desired
}
