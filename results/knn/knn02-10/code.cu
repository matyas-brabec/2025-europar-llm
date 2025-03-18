#include <cuda_runtime.h>
#include <float.h>

// Our “Pair” type matches std::pair<int,float> in layout.
struct Pair {
    int first;
    float second;
};

// TILE_SIZE is the number of data points loaded per shared‐memory batch.
// This value can be tuned based on the target GPU shared memory capacity.
#define TILE_SIZE 1024

// The knn_kernel computes the k-nearest neighbors for exactly one query per warp.
// Each warp (32 threads) cooperates to process all data points in batches,
// maintains a private candidate list – partitioned evenly among the 32 threads – and then
// merges the per-thread candidate lists into a sorted global candidate list.
// The final candidate list (of length k) for the query (sorted in ascending order of distance)
// is stored in the results array.
__global__ void knn_kernel(const float2 * __restrict__ query, int query_count,
                           const float2 * __restrict__ data, int data_count,
                           Pair * __restrict__ results, int k) {
    // Each warp is assigned one query.
    // Compute warp index within the block.
    int warpId_inBlock = threadIdx.x / 32;
    int lane = threadIdx.x & 31;  // thread's lane within its warp

    // Total number of warps per block.
    int warpsPerBlock = blockDim.x / 32;
    // Global warp id (each warp processes one query).
    int global_warp_id = blockIdx.x * warpsPerBlock + warpId_inBlock;
    if (global_warp_id >= query_count)
        return;

    // All lanes in the warp load the same query point.
    float2 q = query[global_warp_id];

    // Each warp will ultimately return k candidates.
    // We partition the k candidates evenly among 32 threads:
    // L_local = k/32 (since k is a power of two between 32 and 1024, this is an integer).
    int L_local = k / 32;
    // Each thread allocates its private candidate arrays in registers.
    // We use fixed-size arrays (max size 32 needed for k=1024).
    float localDist[32];
    int localIdx[32];
    // Initialize candidate list with worst possible distances.
    for (int i = 0; i < L_local; i++) {
        localDist[i] = FLT_MAX;
        localIdx[i] = -1;
    }

    // Partition the shared memory.
    // The first part is used as a tile to cache a batch of data points.
    // The remainder is used for each warp’s candidate merge buffer.
    extern __shared__ char smem[];
    float2 *sharedData = (float2*)smem;
    // Each warp’s candidate merge buffer occupies k elements.
    // Its starting address is shifted by the size of the data tile.
    Pair *warpCandBuffer = (Pair*)(smem + TILE_SIZE * sizeof(float2));
    warpCandBuffer += warpId_inBlock * k;

    // Process all data points in batches that are loaded into shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {
        // Determine how many data points in this batch.
        int tileSize = (data_count - tileStart < TILE_SIZE) ? (data_count - tileStart) : TILE_SIZE;
        // Cooperative loading: use all block threads for copying into shared memory.
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            sharedData[i] = data[tileStart + i];
        }
        __syncthreads();  // ensure the data tile is loaded

        // Each warp processes the current tile.
        // Distribute indices among the 32 lanes.
        for (int i = lane; i < tileSize; i += 32) {
            float2 pt = sharedData[i];
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            // Compute squared Euclidean distance.
            float dist = dx * dx + dy * dy;
            int index = tileStart + i;
            // Since we maintain each thread’s candidate list sorted in ascending order
            // (best candidate at index 0, worst at index L_local-1), we only update
            // if the new distance is smaller than our worst candidate so far.
            if (dist < localDist[L_local - 1]) {
                // Insertion into the sorted list.
                int pos = L_local - 1;
                while (pos > 0 && dist < localDist[pos - 1]) {
                    localDist[pos] = localDist[pos - 1];
                    localIdx[pos] = localIdx[pos - 1];
                    pos--;
                }
                localDist[pos] = dist;
                localIdx[pos] = index;
            }
        }
        __syncthreads();  // ensure all warps finish processing this tile
    } // end for each tile

    // Each lane writes its private candidate list into the warp-local candidate merge buffer.
    // Each lane writes its L_local elements to a contiguous segment.
    for (int i = 0; i < L_local; i++) {
        warpCandBuffer[lane * L_local + i].first = localIdx[i];
        warpCandBuffer[lane * L_local + i].second = localDist[i];
    }
    __syncwarp();  // ensure the 32 lanes have populated the buffer

    // Now merge the 32 sorted candidate lists (each of length L_local) into one sorted list of length k.
    // Let only lane 0 of the warp perform the merge.
    // We perform a simple k-way merge over 32 arrays (each sorted in ascending order).
    if (lane == 0) {
        // "ptr" tracks the current index in each lane’s candidate list.
        int ptr[32];
#pragma unroll
        for (int j = 0; j < 32; j++) {
            ptr[j] = 0;
        }
        // For each output candidate position r (0 <= r < k), choose the best candidate among those available.
        for (int r = 0; r < k; r++) {
            float bestVal = FLT_MAX;
            int bestLane = -1;
            // Find the candidate with the minimal distance among the heads of the 32 arrays.
            for (int j = 0; j < 32; j++) {
                if (ptr[j] < L_local) {
                    float candidateVal = warpCandBuffer[j * L_local + ptr[j]].second;
                    if (candidateVal < bestVal) {
                        bestVal = candidateVal;
                        bestLane = j;
                    }
                }
            }
            // Write the chosen candidate into the final results.
            results[global_warp_id * k + r].first = warpCandBuffer[bestLane * L_local + ptr[bestLane]].first;
            results[global_warp_id * k + r].second = warpCandBuffer[bestLane * L_local + ptr[bestLane]].second;
            ptr[bestLane]++;  // move forward in the candidate list from the selected lane
        }
    }
}

// Host interface: run_knn launches the kernel with appropriate parameters.
// It is assumed that query, data, and result have been allocated with cudaMalloc.
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k) {
    // Choose 256 threads per block (8 warps per block).
    int blockSize = 256;
    int warpsPerBlock = blockSize / 32;
    // Compute the number of blocks required so that each warp processes one query.
    int gridSize = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    // Compute the required shared memory per block:
    //   - Data tile: TILE_SIZE * sizeof(float2)
    //   - Candidate merge buffer: (warpsPerBlock * k * sizeof(Pair))
    size_t sharedMemSize = TILE_SIZE * sizeof(float2) + warpsPerBlock * k * sizeof(Pair);
    // Launch the kernel.
    knn_kernel<<<gridSize, blockSize, sharedMemSize>>>(query, query_count, data, data_count,
                                                       reinterpret_cast<Pair*>(result), k);
    // Optionally, one might synchronize or check errors here.
    // cudaDeviceSynchronize();
}
