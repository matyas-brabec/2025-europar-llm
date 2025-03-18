#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

//------------------------------------------------------------------------------
// This kernel implements k-nearest neighbors (k-NN) for 2D points.
// Each warp (32 threads) processes one query point.
// Each thread in the warp maintains a private candidate list (of size L = k/32)
// in registers sorted in ascending order (lowest/smallest distance first).
// Data points (from the "data" array) are processed in tiles loaded into shared
// memory. Each warp’s lanes process a disjoint subset of the tile (using lane id
// strides) and update their local candidate lists.  After all tiles are processed,
// the 32 candidate lists (one per warp lane) are merged (via warp shuffles) by
// lane 0 to form the final k candidates in sorted order.  The resulting neighbor
// indices and squared distances are written to the output array "result".
//------------------------------------------------------------------------------

#define TILE_SIZE 1024  // Number of data points to load per tile; tuned for shared memory.

__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Each warp (32 threads) processes one query.
    // Identify warp and lane within warp.
    int warpId_inBlock = threadIdx.x / 32;   // warp index within block
    int laneId = threadIdx.x % 32;             // lane (thread) index within warp
    int warpsPerBlock = blockDim.x / 32;
    // Global warp index = blockIdx.x * warpsPerBlock + warpId_inBlock.
    int globalWarpId = blockIdx.x * warpsPerBlock + warpId_inBlock;
    if (globalWarpId >= query_count) return;

    // Load the query point for this warp.
    float2 q = query[globalWarpId];

    // Each warp must ultimately produce k neighbors.
    // We partition the k candidates among the 32 lanes: each lane gets L = k/32 candidates.
    // k is assumed to be a power of 2 between 32 and 1024, so division is exact.
    const int L = k / 32;

    // Each thread allocates private candidate arrays in registers.
    // They are maintained in ascending order (best candidate at index 0, worst at index L-1).
    float local_dist[32];  // maximum L is 1024/32 = 32.
    int local_idx[32];
#pragma unroll
    for (int i = 0; i < L; i++) {
        local_dist[i] = FLT_MAX;
        local_idx[i] = -1;
    }

    // Shared memory tile to cache a batch of data points.
    extern __shared__ float2 shared_data[];  // size = TILE_SIZE * sizeof(float2)

    // Process the data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        // Compute number of points in this tile.
        int tile_elems = ((data_count - tile_start) < TILE_SIZE) ? (data_count - tile_start) : TILE_SIZE;

        // All threads in the block cooperatively load the tile into shared memory.
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            shared_data[i] = data[tile_start + i];
        }
        __syncthreads();  // Ensure the tile is loaded

        // Each warp processes the tile.
        // Each lane in the warp processes a subset of points from the tile (stride = warpSize).
        for (int i = laneId; i < tile_elems; i += 32) {
            float2 p = shared_data[i];
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float dist = dx * dx + dy * dy;
            int d_idx = tile_start + i;
            // Update local candidate list if this candidate is better than the worst stored.
            // The worst candidate is at index L-1 (since the list is sorted ascending).
            if (dist < local_dist[L - 1]) {
                // Insertion: find the proper position to keep list sorted.
                int pos = L - 1;
                while (pos > 0 && dist < local_dist[pos - 1]) {
                    local_dist[pos] = local_dist[pos - 1];
                    local_idx[pos] = local_idx[pos - 1];
                    pos--;
                }
                local_dist[pos] = dist;
                local_idx[pos] = d_idx;
            }
        }
        __syncwarp();      // Synchronize threads within the warp.
        __syncthreads();   // Synchronize all threads in block before next tile load.
    }

    // At this point, each lane in the warp holds L candidates.
    // We now need to merge the 32 sorted lists (from each lane) into one sorted list of k candidates.
    // For simplicity, lane 0 of the warp gathers all candidates via warp shuffles,
    // sorts them with insertion sort, and writes the final sorted k results.
    if (laneId == 0) {
        int merged_idx[1024];    // k maximum is 1024.
        float merged_dist[1024];
        // Gather candidate lists from all lanes.
        for (int r = 0; r < 32; r++) {
            for (int j = 0; j < L; j++) {
                int cand = __shfl_sync(0xFFFFFFFF, local_idx[j], r);
                float dval = __shfl_sync(0xFFFFFFFF, local_dist[j], r);
                merged_idx[r * L + j] = cand;
                merged_dist[r * L + j] = dval;
            }
        }
        // Sort the merged list in ascending order using insertion sort.
        for (int i = 1; i < k; i++) {
            int key_idx = merged_idx[i];
            float key_dist = merged_dist[i];
            int j = i - 1;
            while (j >= 0 && merged_dist[j] > key_dist) {
                merged_idx[j + 1] = merged_idx[j];
                merged_dist[j + 1] = merged_dist[j];
                j--;
            }
            merged_idx[j + 1] = key_idx;
            merged_dist[j + 1] = key_dist;
        }
        // Write the sorted k candidates to global memory.
        for (int i = 0; i < k; i++) {
            result[globalWarpId * k + i] = std::pair<int, float>(merged_idx[i], merged_dist[i]);
        }
    }
    // End of kernel.
}

//------------------------------------------------------------------------------
// Host interface: run_knn
// This function launches the knn_kernel and assumes that the input arrays
// (query, data, and result) are allocated on the device (using cudaMalloc).
// "query" holds query_count float2 elements.
// "data" holds data_count float2 elements.
// "result" is a device array that will receive query_count * k std::pair<int,float>
// entries in row-major order (each query's k nearest neighbors consecutively).
// k is a power-of-two between 32 and 1024.
//------------------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k) {
    // Choose kernel parameters.
    // We use blockDim.x = 128 threads (i.e. 4 warps per block).
    int blockSize = 128;
    int warpsPerBlock = blockSize / 32;  // 4 warps per block.
    // Each warp processes one query, so grid dimension counts warps.
    int gridSize = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Shared memory: allocate TILE_SIZE float2 elements.
    size_t sharedMemSize = TILE_SIZE * sizeof(float2);

    // Launch kernel.
    knn_kernel<<<gridSize, blockSize, sharedMemSize>>>(query, query_count,
                                                       data, data_count,
                                                       result, k);
    // Optionally check for errors:
    // cudaError_t err = cudaGetLastError();
    // if(err != cudaSuccess) { printf("Kernel launch error: %s\n", cudaGetErrorString(err)); }
}
