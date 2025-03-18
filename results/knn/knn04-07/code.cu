// This complete CUDA C++ source implements an optimized k–nearest neighbors (k‐NN)
// search on 2D points using a warp–synchronous algorithm. For each query point,
// one warp (32 threads) computes its k nearest data points (using squared Euclidean distance)
// by iterating over the data points in batches (“tiles”). Each thread in the warp
// processes a subset of points in the tile, locally collects promising candidates
// (i.e. points with distance less than the current global threshold), and then
// merges its local candidates with its private candidate list. The candidate list
// for the warp is distributed among the 32 threads (each thread holds k/32 entries)
// and is maintained sorted in increasing order by distance. After processing all
// tiles, the warp performs a final merge of the per‐thread candidate lists (using
// shared memory and a single–thread sort within the warp) and writes the final result
// to global memory.
//
// The code assumes k is a power‐of–two between 32 and 1024, inclusive. The kernel
// uses a tile size of 256 data points and a block size of 128 threads (4 warps per block).
//
// To build, compile with the latest CUDA toolkit and an up–to–date C++ compiler.
//
// Note: The code directly uses std::pair for output results.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>   // for FLT_MAX
#include <utility>   // for std::pair

// -----------------------------------------------------------------------------
// Device helper: warp–synchronous reduction for maximum (over float values).
// This function uses __shfl_down_sync to compute the maximum value among all threads in a warp.
__inline__ __device__ float warpReduceMax(float val) {
    // full mask for 32 threads
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// -----------------------------------------------------------------------------
// Host function run_knn: launches the CUDA kernel.
// Inputs:
//    query        - pointer to query points (float2 array)
//    query_count  - number of query points
//    data         - pointer to data points (float2 array)
//    data_count   - number of data points
//    result       - pointer to result array, where for each query the k nearest
//                   neighbors are stored as std::pair<int, float> (index, distance)
//    k            - number of nearest neighbors to find (power-of–two between 32 and 1024)
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k);
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k);

// -----------------------------------------------------------------------------
// The CUDA kernel: each warp processes one query point.
/// @FIXED
/// extern "C" __global__ void knn_kernel(const float2 *query, int query_count,
__global__ void knn_kernel(const float2 *query, int query_count,
                                      const float2 *data, int data_count,
                                      std::pair<int, float> *result, int k)
{
    // -------------------------------------------------------------------------
    // Algorithm hyper-parameters:
    const int TILE_SIZE = 256;   // Number of data points loaded per tile into shared memory.
    const int MAX_LOCAL = 8;     // Per-thread local buffer capacity (for candidates from a tile).
    // -------------------------------------------------------------------------
    // Identify the warp (of 32 threads) and lane within the warp.
    int lane   = threadIdx.x & 31;           // lane ID in [0,31]
    int warpId = threadIdx.x >> 5;           // warp ID within the block
    int warpsPerBlock = blockDim.x >> 5;       // number of warps per block
    // Each warp processes one query; compute the global query index.
    int queryIndex = blockIdx.x * warpsPerBlock + warpId;
    if (queryIndex >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[queryIndex];

    // Compute localK = k/32. Each warp holds k candidates, distributed evenly among 32 threads.
    int localK = k >> 5; // since k is power-of-two and divisible by 32.

    // Each thread maintains its candidate list in registers.
    // Allocate arrays of maximum size 32 (k max=1024 => localK max = 32).
    float cand[32];
    int   candIdx[32];
    // Initialize candidate list with FLT_MAX distances and invalid (-1) indices.
    for (int i = 0; i < localK; i++) {
        cand[i] = FLT_MAX;
        candIdx[i] = -1;
    }
    // The candidate list is maintained in increasing order (best = smallest distances).
    // Initially, all entries are FLT_MAX so it is trivially sorted.

    // Each thread also maintains a small local buffer (in registers) to accumulate
    // candidate distances from the current tile.
    float localBuf[MAX_LOCAL];
    int   localBufIdx[MAX_LOCAL];
    int localBufCount = 0;

    // -------------------------------------------------------------------------
    // Shared memory:
    // The first part of shared memory holds a tile of data points.
    extern __shared__ char sharedMem[];
    float2 *s_data = (float2*) sharedMem;  // Space for TILE_SIZE float2 elements.
    // The next part of shared memory is reserved for final warp candidate merge.
    // Each warp gets a contiguous region to hold k float values and k integer indices.
    // Layout:
    //   s_finalD: float array, size = (number_of_warps_per_block * k)
    //   s_finalIdx: int array, size = (number_of_warps_per_block * k)
    char *ptr = sharedMem + TILE_SIZE * sizeof(float2);
    float *s_finalD = (float*) ptr;
    int   *s_finalIdx = (int*)(ptr + warpsPerBlock * k * sizeof(float));

    // -------------------------------------------------------------------------
    // Process the data points in batches (tiles).
    for (int tile = 0; tile < data_count; tile += TILE_SIZE) {
        int tileSize = (data_count - tile) < TILE_SIZE ? (data_count - tile) : TILE_SIZE;
        // Cooperative load: all threads in the block load portions of the tile.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            s_data[i] = data[tile + i];
        }
        __syncthreads(); // Ensure the tile is fully loaded.

        // Compute the current global threshold by reducing the worst (largest) candidate
        // in the per-thread candidate lists. Each thread's worst candidate is cand[localK-1].
        float myWorst = cand[localK - 1];
        float globalThreshold = warpReduceMax(myWorst); // maximum among all lanes.

        // Each warp processes the tile: each thread iterates over data points in s_data,
        // striding by the warp size (32).
        for (int i = lane; i < tileSize; i += 32) {
            float2 pt = s_data[i];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float dist = dx * dx + dy * dy;
            // Only consider data points that are closer than the current global threshold.
            if (dist < globalThreshold) {
                // Add candidate info to the local buffer.
                localBuf[localBufCount] = dist;
                localBufIdx[localBufCount] = tile + i;  // global data index.
                localBufCount++;
                // When the local buffer is full, merge its contents with the candidate list.
                if (localBufCount == MAX_LOCAL) {
                    // --- Insertion sort the small local buffer (length MAX_LOCAL or less) ---
                    for (int a = 1; a < localBufCount; a++) {
                        float key = localBuf[a];
                        int keyIdx = localBufIdx[a];
                        int b = a - 1;
                        while (b >= 0 && localBuf[b] > key) {
                            localBuf[b+1] = localBuf[b];
                            localBufIdx[b+1] = localBufIdx[b];
                            b--;
                        }
                        localBuf[b+1] = key;
                        localBufIdx[b+1] = keyIdx;
                    }
                    // --- Merge the sorted local buffer with the candidate list ---
                    // Both arrays are sorted in increasing order.
                    // We'll merge cand[0..localK-1] and localBuf[0..localBufCount-1]
                    // and retain only the smallest localK elements.
                    int i1 = 0, i2 = 0, out = 0;
                    float temp[40];      // localK + MAX_LOCAL <= 32 + 8 = 40.
                    int   tempIdx[40];
                    while (out < localK && (i1 < localK || i2 < localBufCount)) {
                        float val1 = (i1 < localK) ? cand[i1] : FLT_MAX;
                        float val2 = (i2 < localBufCount) ? localBuf[i2] : FLT_MAX;
                        if (val1 <= val2) {
                            temp[out] = val1;
                            tempIdx[out] = (i1 < localK) ? candIdx[i1] : -1;
                            i1++;
                        } else {
                            temp[out] = val2;
                            tempIdx[out] = (i2 < localBufCount) ? localBufIdx[i2] : -1;
                            i2++;
                        }
                        out++;
                    }
                    // Copy the merged best candidates back to the candidate list.
                    for (int j = 0; j < localK; j++) {
                        cand[j] = temp[j];
                        candIdx[j] = tempIdx[j];
                    }
                    // Reset the local buffer.
                    localBufCount = 0;
                    // (globalThreshold will be recomputed at the next tile iteration.)
                    myWorst = cand[localK - 1];
                    globalThreshold = warpReduceMax(myWorst);
                }
            }
        }
        __syncthreads(); // Ensure all warps finish processing this tile before reusing s_data.
    }
    // End of tile loop.

    // After all tiles have been processed, merge any remaining items in the local buffer.
    if (localBufCount > 0) {
        // Sort the remaining local buffer.
        for (int a = 1; a < localBufCount; a++) {
            float key = localBuf[a];
            int keyIdx = localBufIdx[a];
            int b = a - 1;
            while (b >= 0 && localBuf[b] > key) {
                localBuf[b+1] = localBuf[b];
                localBufIdx[b+1] = localBufIdx[b];
                b--;
            }
            localBuf[b+1] = key;
            localBufIdx[b+1] = keyIdx;
        }
        int i1 = 0, i2 = 0, out = 0;
        float temp[40];
        int   tempIdx[40];
        while (out < localK && (i1 < localK || i2 < localBufCount)) {
            float val1 = (i1 < localK) ? cand[i1] : FLT_MAX;
            float val2 = (i2 < localBufCount) ? localBuf[i2] : FLT_MAX;
            if (val1 <= val2) {
                temp[out] = val1;
                tempIdx[out] = (i1 < localK) ? candIdx[i1] : -1;
                i1++;
            } else {
                temp[out] = val2;
                tempIdx[out] = (i2 < localBufCount) ? localBufIdx[i2] : -1;
                i2++;
            }
            out++;
        }
        for (int j = 0; j < localK; j++){
            cand[j] = temp[j];
            candIdx[j] = tempIdx[j];
        }
    }

    // -------------------------------------------------------------------------
    // Final merge: Each warp now has 32 sorted candidate lists (each of length localK)
    // stored in registers. We combine them to obtain the final sorted list of k candidates.
    // We'll write each thread's candidate list into shared memory using column–major order.
    //
    // The global final candidate array for this warp (of length k) is stored in s_finalD and s_finalIdx.
    int warpOffset = warpId * k;  // Each warp uses k contiguous elements in s_finalD/s_finalIdx.
    for (int j = 0; j < localK; j++) {
        int pos = j * 32 + lane; // column–major: thread 'lane' writes candidate element j.
        s_finalD[warpOffset + pos] = cand[j];
        s_finalIdx[warpOffset + pos] = candIdx[j];
    }
    __syncwarp();  // Synchronize the warp before final merge.

    // Let lane 0 of the warp perform a final, full sort on the k candidates.
    if (lane == 0) {
        // Allocate local arrays in registers (k is at most 1024).
        float finalD[1024];
        int   finalIdx[1024];
        // Copy the k candidates from shared memory.
        for (int i = 0; i < k; i++) {
            finalD[i] = s_finalD[warpOffset + i];
            finalIdx[i] = s_finalIdx[warpOffset + i];
        }
        // Use a simple insertion sort to sort the k candidates in increasing order.
        for (int i = 1; i < k; i++) {
            float key = finalD[i];
            int keyIdx = finalIdx[i];
            int j = i - 1;
            while (j >= 0 && finalD[j] > key) {
                finalD[j+1] = finalD[j];
                finalIdx[j+1] = finalIdx[j];
                j--;
            }
            finalD[j+1] = key;
            finalIdx[j+1] = keyIdx;
        }
        // Write the final sorted k nearest neighbors for this query to global memory.
        for (int i = 0; i < k; i++) {
            result[queryIndex * k + i] = std::pair<int, float>(finalIdx[i], finalD[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// Host function implementation that launches the k-NN kernel.
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                          const float2 *data, int data_count,
                          std::pair<int, float> *result, int k)
{
    // Choose block configuration:
    // Each block uses 128 threads (4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // One warp processes one query, so number of queries per block = warpsPerBlock.
    int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute required shared memory:
    // Shared memory = tile buffer + final merge buffers.
    // Tile buffer: TILE_SIZE * sizeof(float2)
    // Final merge buffers: (warpsPerBlock * k * (sizeof(float) + sizeof(int)))
    const int TILE_SIZE = 256;
    size_t shmTile = TILE_SIZE * sizeof(float2);
    size_t shmMerge = warpsPerBlock * k * (sizeof(float) + sizeof(int));
    size_t sharedMemSize = shmTile + shmMerge;

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // Synchronize and check for errors.
    cudaDeviceSynchronize();
}
