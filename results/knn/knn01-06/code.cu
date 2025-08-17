#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <cfloat>

// Optimized brute-force k-NN (squared L2) for 2D points.
// Strategy:
// - Assign one warp per query.
// - The block cooperatively tiles the data points into shared memory to maximize reuse across warps.
// - Within a warp, each lane scans a strided subset of the tile and maintains its top-L candidates,
//   where L = k / 32 (k is guaranteed to be a power of two between 32 and 1024, so L in [1, 32]).
//   For each lane, we store L best candidates in small unsorted buffers and track the current threshold
//   (the maximum among them); when a better candidate is found, we replace the current max and recompute
//   the new max by scanning L elements. Since replacements are relatively infrequent after warmup,
//   this is efficient and avoids heavy per-candidate O(log L) heap costs.
// - After scanning all data, each lane sorts its L candidates ascending (insertion sort; L <= 32).
// - The 32 sorted lists (one per lane) are stored in shared memory per warp and then merged by lane 0
//   using a small min-heap of size 32 to produce the final k sorted results, which are written to 'result'.
//
// Shared memory layout per block (dynamic):
//   [tile of float2 points][warp_idx_buf (W*k ints)][warp_dist_buf (W*k floats)]
//   W = WARPS_PER_BLOCK
//
// The shared memory size (worst case k=1024) is chosen to stay within 48KB default per-block limit
// without requiring opt-in attributes: WARPS_PER_BLOCK=4, TILE_SIZE=2048 => 16KB (tile) + 32KB (merge buffers) = 48KB.
//
// Thread-block configuration: 4 warps (128 threads). Grid size covers all queries.
// Modern GPUs (A100/H100) handle many such blocks concurrently; the dominant cost is memory bandwidth.

// Tunable kernel parameters (chosen to fit 48KB shared mem limit at k=1024):
#ifndef KNN_WARPS_PER_BLOCK
#define KNN_WARPS_PER_BLOCK 4
#endif

#ifndef KNN_TILE_SIZE
#define KNN_TILE_SIZE 2048
#endif

static constexpr int WARP_SIZE = 32;

// Small per-lane top-L container with threshold ("tau") tracking.
// - Stores up to MaxL = 32 best candidates (distance, index).
// - Insertion: if not full, append and update tau; else if new dist < tau, replace tau slot and recompute tau in O(L).
// - After scanning, supports in-place insertion sort to ascending order.
struct LaneTopL {
    float dist[WARP_SIZE]; // capacity up to 32
    int   idx[WARP_SIZE];
    int   count;
    int   capacity;
    float tau;     // current maximum distance among stored
    int   tau_pos; // index of the current maximum

    __device__ inline void init(int cap) {
        capacity = cap;
        count = 0;
        tau = -FLT_MAX;
        tau_pos = -1;
        // No need to initialize buffers here.
    }

    __device__ inline void consider(float d, int id) {
        if (count < capacity) {
            dist[count] = d;
            idx[count]  = id;
            if (d > tau) {
                tau = d;
                tau_pos = count;
            }
            ++count;
        } else {
            if (d < tau) {
                // Replace current maximum and recompute tau.
                dist[tau_pos] = d;
                idx[tau_pos]  = id;
                // Recompute max across the L elements.
                int n = capacity;
                float maxd = dist[0];
                int maxp = 0;
                #pragma unroll
                for (int i = 1; i < WARP_SIZE; ++i) {
                    if (i >= n) break;
                    float v = dist[i];
                    if (v > maxd) { maxd = v; maxp = i; }
                }
                tau = maxd;
                tau_pos = maxp;
            }
        }
    }

    __device__ inline void sort_ascending() {
        // Simple insertion sort; capacity <= 32.
        int n = count;
        for (int i = 1; i < n; ++i) {
            float keyd = dist[i];
            int   keyi = idx[i];
            int j = i - 1;
            while (j >= 0 && dist[j] > keyd) {
                dist[j + 1] = dist[j];
                idx[j + 1]  = idx[j];
                --j;
            }
            dist[j + 1] = keyd;
            idx[j + 1]  = keyi;
        }
    }
};

// Kernel to compute k-NN for 2D points (squared Euclidean distances).
template <int WARPS_PER_BLOCK, int TILE_SIZE>
__global__ void knn2d_kernel(const float2* __restrict__ query,
                             int query_count,
                             const float2* __restrict__ data,
                             int data_count,
                             std::pair<int, float>* __restrict__ result,
                             int k)
{
    // Shared memory:
    extern __shared__ unsigned char smem[];
    float2* s_points = reinterpret_cast<float2*>(smem);
    const size_t s_points_bytes = static_cast<size_t>(TILE_SIZE) * sizeof(float2);

    int* s_idx = reinterpret_cast<int*>(smem + s_points_bytes);
    float* s_dist = reinterpret_cast<float*>(s_idx + WARPS_PER_BLOCK * k);

    // Thread and warp identifiers
    const int tid = threadIdx.x;
    const int warp_local = tid / WARP_SIZE;  // warp index within block [0..WARPS_PER_BLOCK-1]
    const int lane = tid % WARP_SIZE;

    const int warp_global_q = blockIdx.x * WARPS_PER_BLOCK + warp_local; // query index processed by this warp
    const bool active_warp = (warp_global_q < query_count);

    // Per-warp shared memory buffers for merging:
    int*   warp_idx_buf  = s_idx  + warp_local * k;
    float* warp_dist_buf = s_dist + warp_local * k;

    // Load query point once per warp and broadcast to lanes.
    float qx = 0.0f, qy = 0.0f;
    if (active_warp) {
        if (lane == 0) {
            float2 q = query[warp_global_q];
            qx = q.x;
            qy = q.y;
        }
        unsigned mask = 0xFFFFFFFFu;
        qx = __shfl_sync(mask, qx, 0);
        qy = __shfl_sync(mask, qy, 0);
    }

    // Each lane maintains top-L candidates.
    const int L = k >> 5; // k / 32, guaranteed integer (k power-of-two >= 32)
    LaneTopL top;
    if (active_warp) {
        top.init(L);
    }

    // Loop over data in tiles cooperatively loaded by the block.
    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_SIZE) {
        int tile_elems = data_count - tile_base;
        if (tile_elems > TILE_SIZE) tile_elems = TILE_SIZE;

        // Cooperative load of the tile into shared memory.
        for (int i = tid; i < tile_elems; i += blockDim.x) {
            s_points[i] = data[tile_base + i];
        }
        __syncthreads();

        // Each active warp processes the tile: strided access per lane.
        if (active_warp) {
            for (int i = lane; i < tile_elems; i += WARP_SIZE) {
                float2 p = s_points[i];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float d = dx * dx + dy * dy; // squared L2
                int idx = tile_base + i;
                top.consider(d, idx);
            }
        }

        __syncthreads(); // ensure tile not used before next load
    }

    // After processing all data: sort each lane's top-L ascending and write to per-warp shared memory.
    if (active_warp) {
        top.sort_ascending();
        // Store lane's sorted list at [lane * L, lane * L + L)
        int base = lane * L;
        #pragma unroll
        for (int j = 0; j < WARP_SIZE; ++j) {
            if (j >= L) break;
            warp_dist_buf[base + j] = top.dist[j];
            warp_idx_buf [base + j] = top.idx[j];
        }
    }

    // Synchronize lanes within the warp before merging.
    __syncwarp();

    // Lane 0 performs a 32-way merge using a small min-heap and writes directly to the result array.
    if (active_warp && lane == 0) {
        // Min-heap over the current head of each lane's list.
        // Heap arrays: distances, indices, and source list IDs.
        float hDist[WARP_SIZE];
        int   hIdx [WARP_SIZE];
        int   hList[WARP_SIZE];
        int   hSize = WARP_SIZE;

        // Current positions within each lane's list.
        int pos[WARP_SIZE];
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) pos[i] = 0;

        // Initialize heap with the first element from each of the 32 lists.
        // All lists have length L >= 1 because k >= 32 (guaranteed).
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            int base = i * L;
            hDist[i] = warp_dist_buf[base + 0];
            hIdx[i]  = warp_idx_buf [base + 0];
            hList[i] = i;
            pos[i]   = 1;
        }

        // Helper: heapify-down for min-heap at position 0.
        auto heapify_down = [&](int start, int size) {
            int i = start;
            while (true) {
                int l = 2 * i + 1;
                int r = l + 1;
                int smallest = i;
                if (l < size && hDist[l] < hDist[smallest]) smallest = l;
                if (r < size && hDist[r] < hDist[smallest]) smallest = r;
                if (smallest == i) break;
                // swap i and smallest
                float td = hDist[i]; int ti = hIdx[i]; int tl = hList[i];
                hDist[i] = hDist[smallest]; hIdx[i] = hIdx[smallest]; hList[i] = hList[smallest];
                hDist[smallest] = td; hIdx[smallest] = ti; hList[smallest] = tl;
                i = smallest;
            }
        };

        // Build initial heap (bottom-up).
        for (int i = (hSize >> 1) - 1; i >= 0; --i) {
            heapify_down(i, hSize);
        }

        // Merge and write results.
        const int out_base = warp_global_q * k;
        for (int out = 0; out < k; ++out) {
            // Root of heap is the smallest current head.
            float bestD = hDist[0];
            int   bestI = hIdx[0];
            int   list  = hList[0];

            // Write to result (ascending order).
            result[out_base + out].first  = bestI;
            result[out_base + out].second = bestD;

            // Advance in the source list; if exhausted, remove from heap.
            if (pos[list] < L) {
                int base = list * L;
                hDist[0] = warp_dist_buf[base + pos[list]];
                hIdx[0]  = warp_idx_buf [base + pos[list]];
                // hList[0] remains 'list'
                pos[list]++;
                heapify_down(0, hSize);
            } else {
                // Replace root with last element and shrink heap.
                --hSize;
                if (hSize > 0) {
                    hDist[0] = hDist[hSize];
                    hIdx[0]  = hIdx[hSize];
                    hList[0] = hList[hSize];
                    heapify_down(0, hSize);
                }
            }
        }
    }
}

// Host API: launches the CUDA kernel.
// query: pointer to query_count float2 points.
// data: pointer to data_count float2 points.
// result: pointer to query_count * k std::pair<int, float> output (indices and squared distances).
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Kernel configuration
    constexpr int WARPS_PER_BLOCK = KNN_WARPS_PER_BLOCK;  // 4 warps
    constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE; // 128 threads
    constexpr int TILE_SIZE = KNN_TILE_SIZE; // 2048 points per tile

    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    // Dynamic shared memory size: tile + per-warp merge buffers (indices + distances).
    size_t smem_bytes = static_cast<size_t>(TILE_SIZE) * sizeof(float2)
                      + static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(int)
                      + static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(k) * sizeof(float);

    // Launch kernel
    knn2d_kernel<WARPS_PER_BLOCK, TILE_SIZE><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k);

    // The function assumes that the caller handles synchronization and error checking if needed.
}