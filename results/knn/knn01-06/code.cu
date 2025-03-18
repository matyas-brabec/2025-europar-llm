// This file implements a CUDA kernel for the k‐nearest neighbors (k-NN)
// search for 2D points in Euclidean space. For each query point, the kernel
// finds the k data points (from a very large dataset) having the smallest
// squared Euclidean distances. The results (index and distance pairs) for each
// query are written in row‐major order to the output array.
// 
// We use a two‐phase strategy within each kernel block (each block processes
// one query):
//   1. Each thread in the block scans a portion of the data (via tiling) and
//      maintains in its registers a local candidate list (of fixed size LOCAL_SIZE)
//      of its best (smallest‐distance) results. We use a simple “max‐threshold”
//      update to maintain the local candidate list in an unsorted state.
//   2. The per‐thread results (each of size LOCAL_SIZE) are written into shared
//      memory (forming an array of TOTAL_CAND = blockDim.x * LOCAL_SIZE elements).
//      Then, a bitonic sort (implemented in shared memory) is performed on this
//      candidate array to sort all candidates in ascending order of distance.
//      Finally, the first k elements (k-nearest neighbors) are written back to global memory.
//
// The kernel uses tiling of the data array to improve global memory access efficiency.
// The shared memory used is partitioned into two parts:
//   - A tile buffer for the data points (float2) for coalesced loading.
//   - A candidate buffer for the union of per-thread candidate arrays.
//
// We assume that k is a power-of-two between 32 and 1024 and that data_count >= k.
// The kernel launch configuration sets one block per query and uses 256 threads per block.
//

#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// Define a structure to hold a candidate neighbor (data point index and squared distance).
// This simple structure is used both in registers and in shared memory.
struct Neighbor {
    int idx;
    float dist;
};

// __device__ function to swap two Neighbor elements.
__device__ inline void swap_neighbor(Neighbor &a, Neighbor &b) {
    Neighbor tmp = a;
    a = b;
    b = tmp;
}

// The main kernel: each block processes one query point.
// Threads in the block cooperatively scan the entire data array (tiled) and
// maintain local candidate lists. Then, each block merges and sorts all candidates
// to extract the k-nearest neighbors.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           std::pair<int, float> *result,
                           int k) {
    // Each block processes one query.
    int qIdx = blockIdx.x;
    if(qIdx >= query_count) return;

    // Load the query point from global memory.
    float2 q = query[qIdx];

    // Parameters for per-thread candidate storage.
    // LOCAL_SIZE: number of candidates each thread keeps (fixed).
    // THREADS: number of threads in the block (our kernel launch uses 256 threads per block).
    const int LOCAL_SIZE = 32;
    const int THREADS = blockDim.x;
    const int TOTAL_CAND = THREADS * LOCAL_SIZE; // Total number of candidate elements in the block.

    // Each thread will maintain an array of LOCAL_SIZE candidates in registers.
    // Initialize the local candidate list to "empty" (distance = FLT_MAX).
    Neighbor localCand[LOCAL_SIZE];
    int localCount = 0;  // Number of valid candidates stored.
    // 'localMax' and 'localMaxIdx' track the worst (largest) distance in the local candidate list.
    float localMax = -1.0f; 
    int localMaxIdx = -1;
    for (int i = 0; i < LOCAL_SIZE; i++) {
        localCand[i].idx = -1;
        localCand[i].dist = FLT_MAX;
    }

    // ---------------------------------------------------------------------------
    // Phase 1: Each thread processes a subset of the data points (using tiling)
    // and updates its local candidate list.
    // ---------------------------------------------------------------------------
    // We use tiling to load blocks of data points into shared memory for coalesced access.
    // TILE_SIZE: number of data points loaded per tile.
    const int TILE_SIZE = 256;

    // Calculate the amount of shared memory (dynamically allocated) used by the kernel:
    // The first TILE_SIZE float2 elements are used for the tile buffer.
    // The remainder is used for the candidate array. (Partitioning is done in the host function.)
    extern __shared__ char sharedBuffer[];
    float2 *sTile = reinterpret_cast<float2*>(sharedBuffer);
    // sCand will point to the candidate buffer placed after the tile buffer.
    Neighbor *sCand = reinterpret_cast<Neighbor*>(sharedBuffer + TILE_SIZE * sizeof(float2));

    // Loop over the data points in tiles.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {
        // Each thread loads one data point from global memory into shared memory (if in range).
        int loadIndex = tileStart + threadIdx.x;
        if (threadIdx.x < TILE_SIZE && loadIndex < data_count) {
            sTile[threadIdx.x] = data[loadIndex];
        }
        // Ensure the tile is loaded.
        __syncthreads();

        // Determine number of data points in the current tile.
        int curTileSize = (tileStart + TILE_SIZE <= data_count) ? TILE_SIZE : (data_count - tileStart);

        // Each thread iterates over the current tile.
        for (int j = 0; j < curTileSize; j++) {
            float2 dpt = sTile[j];
            float dx = dpt.x - q.x;
            float dy = dpt.y - q.y;
            float dist = dx * dx + dy * dy;
            int dataIdx = tileStart + j;

            // Upsert the candidate into the local candidate list.
            if (localCount < LOCAL_SIZE) {
                // There's still room in the candidate list: simply append.
                localCand[localCount].idx = dataIdx;
                localCand[localCount].dist = dist;
                // Update the tracked maximum candidate if needed.
                if (localCount == 0 || dist > localMax) {
                    localMax = dist;
                    localMaxIdx = localCount;
                }
                localCount++;
            } else {
                // Candidate list is full.
                // Check if the current candidate is better than the worst in our list.
                if (dist < localMax) {
                    // Replace the worst candidate.
                    localCand[localMaxIdx].idx = dataIdx;
                    localCand[localMaxIdx].dist = dist;
                    // Recompute the worst candidate in the local list.
                    localMax = localCand[0].dist;
                    localMaxIdx = 0;
                    for (int r = 1; r < LOCAL_SIZE; r++) {
                        float candDist = localCand[r].dist;
                        if (candDist > localMax) {
                            localMax = candDist;
                            localMaxIdx = r;
                        }
                    }
                }
            }
        }
        // Ensure all threads finish processing this tile before loading the next.
        __syncthreads();
    }
    // End of Phase 1.

    // ---------------------------------------------------------------------------
    // Phase 2: Write local candidate lists into shared memory and merge them.
    // ---------------------------------------------------------------------------
    // Each thread writes its LOCAL_SIZE candidates (its entire local candidate list)
    // into the shared candidate array sCand. The layout is such that each thread's block
    // of LOCAL_SIZE candidates occupies a contiguous segment.
    int baseIdx = threadIdx.x * LOCAL_SIZE;
    for (int i = 0; i < LOCAL_SIZE; i++) {
        sCand[baseIdx + i] = localCand[i];
    }
    __syncthreads();

    // Now sCand contains TOTAL_CAND = blockDim.x * LOCAL_SIZE candidates.
    // We perform an in-place bitonic sort on this candidate array in shared memory.
    // For bitonic sort, we assume TOTAL_CAND is a power-of-two.
    const int N = TOTAL_CAND; // Total number of elements to sort.
    // The bitonic sort algorithm uses two nested loops:
    // "size" (subsequence length) doubles each iteration, and for each "size" we
    // iterate over "stride" sizes, comparing and swapping elements conditionally.
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            // Each thread processes several indices in the candidate array.
            for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
                int partner = idx ^ stride;
                if (partner > idx) {
                    // Determine sorting direction.
                    // If (idx & size) == 0, then the subsequence is to be sorted in ascending order.
                    bool ascending = ((idx & size) == 0);
                    // Compare and swap elements if they are not in the proper order.
                    if (ascending) {
                        if (sCand[idx].dist > sCand[partner].dist) {
                            swap_neighbor(sCand[idx], sCand[partner]);
                        }
                    } else {
                        if (sCand[idx].dist < sCand[partner].dist) {
                            swap_neighbor(sCand[idx], sCand[partner]);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    // At this point, sCand is sorted in ascending order by the distance field.
    // The first k elements in sCand are the k-nearest neighbors for this query.

    // ---------------------------------------------------------------------------
    // Phase 3: Write the top k sorted results to global memory.
    // Each query's results are stored in row-major order in the output array.
    // For query qIdx, the output segment is result[qIdx * k ... qIdx * k + (k-1)].
    // ---------------------------------------------------------------------------
    int outBase = qIdx * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        // Note: A candidate with idx == -1 indicates an unused slot.
        // However, since data_count >= k and the search is exhaustive,
        // the first k elements should be valid.
        result[outBase + i] = std::make_pair(sCand[i].idx, sCand[i].dist);
    }
}

// Host C++ interface: run_knn
// Launch configuration: one block per query and 256 threads per block.
// The dynamically allocated shared memory is sized to accommodate both the
// tile buffer (for data points) and the candidate buffer.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Use 256 threads per block.
    int threadsPerBlock = 256;
    int blocks = query_count;

    // Define parameters same as in the kernel.
    const int TILE_SIZE = 256;   // Number of data points per tile.
    const int LOCAL_SIZE = 32;   // Number of candidates per thread.
    // Total candidate array size per block = threadsPerBlock * LOCAL_SIZE.
    // Compute required shared memory:
    //   sTile: TILE_SIZE * sizeof(float2)
    //   sCand: (threadsPerBlock * LOCAL_SIZE) * sizeof(Neighbor)
    int sharedMemSize = TILE_SIZE * sizeof(float2) + (threadsPerBlock * LOCAL_SIZE) * sizeof(Neighbor);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    // (Optionally, cudaDeviceSynchronize() can be added here for error checking.)
}