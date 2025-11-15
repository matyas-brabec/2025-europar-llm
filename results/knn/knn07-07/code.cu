// CUDA k-NN (k nearest neighbors) for 2D points.
// Each query point is processed by a single warp (32 threads).
// The kernel uses shared-memory tiling for the data points and per-warp
// shared-memory buffers for candidate neighbors and the intermediate result.

#include <cuda_runtime.h>
#include <utility>

// Internal POD equivalent of std::pair<int,float> for device code.
// Layout is compatible with std::pair<int,float> for common ABIs.
struct PairIntFloat {
    int   first;
    float second;
};

// Tunable parameters.
static constexpr int WARP_SIZE         = 32;
static constexpr int WARPS_PER_BLOCK   = 4;    // 4 warps -> 128 threads per block
static constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
static constexpr int MAX_K             = 1024; // maximum supported k (power of two)
static constexpr int TILE_SIZE         = 3072; // number of data points cached per tile

// -----------------------------------------------------------------------------
// Serial Bitonic Sort on (distance, index) pairs.
// This is executed by a single thread (lane 0 of the warp), operating on
// shared-memory arrays of length n (n is a power of two, 32 <= n <= MAX_K).
// -----------------------------------------------------------------------------
__device__ __forceinline__
void bitonic_sort_serial(float *dist, int *idx, int n)
{
    // Bitonic sort network as in the provided pseudocode.
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; ++i) {
                int l = i ^ j;
                if (l > i) {
                    bool ascending = ((i & k) == 0);
                    float di = dist[i];
                    float dl = dist[l];
                    int   ii = idx[i];
                    int   il = idx[l];
                    bool swap =
                        ( ascending && di > dl) ||
                        (!ascending && di < dl);
                    if (swap) {
                        dist[i] = dl;
                        dist[l] = di;
                        idx[i]  = il;
                        idx[l]  = ii;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Flush and merge the per-warp candidate buffer into the intermediate result.
// This function is called only by lane 0 of the corresponding warp.
// The intermediate result is always kept sorted in ascending order.
// Steps:
//   1. If the buffer is not full, pad it up to k with large distances.
//   2. Sort the buffer (k elements) using bitonic sort (ascending).
//   3. Merge buffer and intermediate result into a bitonic sequence:
//         merged[i] = min( intermediate[i], buffer[k - 1 - i] ),
//      and store it back into intermediate[].
//   4. Sort the merged sequence (k elements) using bitonic sort.
//   5. Update max_distance (distance of the k-th nearest neighbor).
//   6. Reset candidate_count to zero.
// -----------------------------------------------------------------------------
__device__ __forceinline__
void flush_buffer_for_warp(
    int warpInBlock,
    int k,
    float intermediate_dist[][MAX_K],
    int   intermediate_idx[][MAX_K],
    float candidate_dist[][MAX_K],
    int   candidate_idx[][MAX_K],
    int   candidate_count[],
    float warp_max_dist[]
)
{
    int count = candidate_count[warpInBlock];
    if (count <= 0)
        return;

    // Pad buffer with large distances if it is not full.
    if (count < k) {
        float padVal = warp_max_dist[warpInBlock];
        // padVal is initially +inf; after the first merge it is the k-th distance.
        for (int i = count; i < k; ++i) {
            candidate_dist[warpInBlock][i] = padVal;
            candidate_idx[warpInBlock][i]  = -1;
        }
    }

    // Step 1: Sort the candidate buffer (ascending).
    bitonic_sort_serial(candidate_dist[warpInBlock], candidate_idx[warpInBlock], k);

    // Step 2: Merge buffer and intermediate result into a bitonic sequence.
    for (int i = 0; i < k; ++i) {
        int j = k - 1 - i;
        float distA = intermediate_dist[warpInBlock][i];
        float distB = candidate_dist[warpInBlock][j];
        int   idxA  = intermediate_idx[warpInBlock][i];
        int   idxB  = candidate_idx[warpInBlock][j];

        if (distB < distA) {
            intermediate_dist[warpInBlock][i] = distB;
            intermediate_idx[warpInBlock][i]  = idxB;
        }
        // else keep intermediate[i] as is
    }

    // Step 3: Sort the merged sequence (still stored in intermediate_*) (ascending).
    bitonic_sort_serial(intermediate_dist[warpInBlock], intermediate_idx[warpInBlock], k);

    // Step 4: Update max_distance: distance of the k-th nearest neighbor.
    warp_max_dist[warpInBlock] = intermediate_dist[warpInBlock][k - 1];

    // Step 5: Reset candidate buffer count.
    candidate_count[warpInBlock] = 0;
}

// -----------------------------------------------------------------------------
// Kernel: one warp (32 threads) processes one query point.
// All warps in a block cooperatively cache data points into shared memory.
// Each warp maintains:
//   - intermediate_dist / intermediate_idx: its current top-k neighbors (sorted).
//   - candidate_dist / candidate_idx: a candidate buffer of size k.
//   - candidate_count: count of elements in the candidate buffer.
//   - warp_max_dist: distance of the k-th nearest neighbor.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(
    const float2 * __restrict__ query,
    int            query_count,
    const float2 * __restrict__ data,
    int            data_count,
    PairIntFloat * __restrict__ result,
    int            k
)
{
    // Shared memory: cached data tile, per-warp candidate and intermediate arrays.
    __shared__ float2 shared_data[TILE_SIZE];

    __shared__ float intermediate_dist[WARPS_PER_BLOCK][MAX_K];
    __shared__ int   intermediate_idx [WARPS_PER_BLOCK][MAX_K];

    __shared__ float candidate_dist   [WARPS_PER_BLOCK][MAX_K];
    __shared__ int   candidate_idx    [WARPS_PER_BLOCK][MAX_K];
    __shared__ int   candidate_count  [WARPS_PER_BLOCK];

    __shared__ float warp_max_dist    [WARPS_PER_BLOCK];

    const int lane        = threadIdx.x & (WARP_SIZE - 1);
    const int warpInBlock = threadIdx.x / WARP_SIZE;
    const int globalWarp  = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;
    const int queryIdx    = globalWarp; // one warp per query

    const bool validWarp  = (queryIdx < query_count);

    // Load query coordinates into registers for the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (validWarp && lane == 0) {
        float2 q = query[queryIdx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(0xFFFFFFFFu, qx, 0);
    qy = __shfl_sync(0xFFFFFFFFu, qy, 0);

    // Initialize per-warp intermediate result and control variables.
    if (validWarp) {
        // Initialize intermediate result to "infinite" distances, sorted.
        for (int i = lane; i < k; i += WARP_SIZE) {
            intermediate_dist[warpInBlock][i] = CUDART_INF_F;
            intermediate_idx [warpInBlock][i] = -1;
        }
        if (lane == 0) {
            candidate_count[warpInBlock] = 0;
            warp_max_dist  [warpInBlock] = CUDART_INF_F; // initial max_distance
        }
    }

    // Process data in tiles cached into shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {
        int tileSize = data_count - tileStart;
        if (tileSize > TILE_SIZE) tileSize = TILE_SIZE;

        // Load the current tile of data points into shared memory.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            shared_data[i] = data[tileStart + i];
        }
        __syncthreads();

        if (validWarp) {
            // Each warp iterates over the tile, processing WARP_SIZE points at a time.
            for (int tBase = 0; tBase < tileSize; tBase += WARP_SIZE) {
                int t = tBase + lane;

                float dist      = CUDART_INF_F;
                int   dataIndex = -1;

                if (t < tileSize) {
                    float2 p = shared_data[t];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    dist      = dx * dx + dy * dy;     // squared Euclidean distance
                    dataIndex = tileStart + t;
                }

                // Sequentially present each lane's candidate to lane 0 for insertion.
                // This uses warp shuffles for intra-warp communication.
                #pragma unroll
                for (int src = 0; src < WARP_SIZE; ++src) {
                    float cd = __shfl_sync(0xFFFFFFFFu, dist,      src);
                    int   ci = __shfl_sync(0xFFFFFFFFu, dataIndex, src);

                    if (lane == 0 && ci >= 0) {
                        float maxDist = warp_max_dist[warpInBlock];

                        if (cd < maxDist) {
                            // Reserve a slot in the candidate buffer using atomicAdd,
                            // as requested in the specification.
                            int pos = atomicAdd(&candidate_count[warpInBlock], 1);

                            // Store candidate (distance, index).
                            candidate_dist[warpInBlock][pos] = cd;
                            candidate_idx [warpInBlock][pos] = ci;

                            // If the buffer has just become full, flush it.
                            if (pos + 1 == k) {
                                flush_buffer_for_warp(
                                    warpInBlock, k,
                                    intermediate_dist, intermediate_idx,
                                    candidate_dist, candidate_idx,
                                    candidate_count, warp_max_dist
                                );
                            }
                        }
                    }
                }
            }
        }

        // Ensure all warps are done using shared_data before loading the next tile.
        __syncthreads();
    }

    // After processing all tiles, flush any remaining candidates.
    if (validWarp && lane == 0) {
        if (candidate_count[warpInBlock] > 0) {
            flush_buffer_for_warp(
                warpInBlock, k,
                intermediate_dist, intermediate_idx,
                candidate_dist, candidate_idx,
                candidate_count, warp_max_dist
            );
        }

        // Write the final sorted k nearest neighbors for this query to global memory.
        PairIntFloat *out = result + queryIdx * k;
        for (int i = 0; i < k; ++i) {
            out[i].first  = intermediate_idx [warpInBlock][i];
            out[i].second = intermediate_dist[warpInBlock][i];
        }
    }
}

// -----------------------------------------------------------------------------
// Public C++ interface.
// query      : pointer to query_count 2D points (float2) in device memory.
// data       : pointer to data_count  2D points (float2) in device memory.
// result     : pointer to query_count * k std::pair<int,float> in device memory.
// k          : number of nearest neighbors (power of two, 32 <= k <= 1024).
// -----------------------------------------------------------------------------
void run_knn(
    const float2 *query,
    int           query_count,
    const float2 *data,
    int           data_count,
    std::pair<int, float> *result,
    int           k
)
{
    // Assume that the inputs satisfy: data_count >= k and 32 <= k <= MAX_K,
    // and that k is a power of two (as specified).

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Cast std::pair<int,float>* to internal POD type for the kernel.
    PairIntFloat *deviceResult = reinterpret_cast<PairIntFloat*>(result);

    knn_kernel<<<gridDim, blockDim>>>(
        query,
        query_count,
        data,
        data_count,
        deviceResult,
        k
    );

    // Optionally synchronize here; omitted for performance in typical pipelines.
    // cudaDeviceSynchronize();
}