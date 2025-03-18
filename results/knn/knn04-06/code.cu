#include <cuda_runtime.h>
#include <utility>
#include <cstdio>
#include <cfloat>

//-------------------------------------------------------------------------------
// This CUDA kernel implements k-nearest neighbors (k-NN) for 2D points.
// Each warp processes one query point. Within a warp the 32 threads work together:
// • Each thread maintains a private (register‐stored) sorted list of L = k/32 candidate
//   neighbor records (each record is an index and a squared distance), sorted in ascending order
//   (so the element at index L-1 is the worst (largest) distance from that thread).
// • The input "data" points are processed by loading them in batches (tiles) into shared memory
//   (by the whole block). Then each warp iterates over the tile in a round‐robin fashion (each thread
//   processes every 32nd element), computes squared Euclidean distances from its query, and updates
//   its local candidate list using insertion sort if the computed distance is lower than its current worst.
// • At the end of each tile, the 32 threads in the warp merge their candidate lists into one sorted list
//   of k candidates. This is done by writing the distributed candidate lists into a contiguous region of
//   shared memory (a warp–local candidate buffer), then having lane 0 perform a serial bitonic sort on the
//   k elements, and finally redistributing the sorted candidates back to per–thread registers.
// • After processing all data batches, the final sorted candidate list (of size k, in ascending order) is
//   written to global memory in row-major order.
//-------------------------------------------------------------------------------

#define TILE_SIZE 1024  // Number of data points per shared-memory batch.

// Structure to hold a candidate neighbor (data point index and squared distance)
struct Candidate {
    int idx;
    float dist;
};

//--------------------------------------------------------------------------
// The k-NN kernel: each warp handles one query point.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp processes one query.
    // Compute global warp id and lane id.
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = global_thread_id / 32; // global warp index
    int lane = threadIdx.x & 31;         // lane index (0..31)

    if (warpId >= query_count)
        return; // extra threads do nothing

    // Load the query point for this warp.
    float2 q = query[warpId];

    // Each warp will maintain k candidate records split across its 32 threads.
    // Let L = k/32 be the number of candidate records per thread.
    int L = k / 32;

    // Each thread holds its local candidate list in registers.
    Candidate localCand[32]; // Maximum L can be up to 32 when k==1024.
    // Initialize candidate records to "infinite" distance.
    for (int i = 0; i < L; i++) {
        localCand[i].dist = FLT_MAX;
        localCand[i].idx = -1;
    }

    // Shared memory layout:
    // - First part: tile of data points (TILE_SIZE float2's).
    // - Second part: Warp-local candidate buffer for merging.
    extern __shared__ unsigned char shared_mem[];
    // Pointer to shared tile for data points.
    float2 *smemTile = reinterpret_cast<float2*>(shared_mem);
    // After the tile data, the remaining shared memory is used for per-warp candidate buffers.
    // Compute offset in bytes for tile data.
    size_t tileBytes = TILE_SIZE * sizeof(float2);
    Candidate *warpCandBuffer = reinterpret_cast<Candidate*>(shared_mem + tileBytes);
    // For each block, the candidate buffer is allocated for each warp in the block.
    // Number of warps per block.
    int warpsPerBlock = blockDim.x / 32;
    // Each warp gets k Candidate elements.
    // The pointer for this warp's candidate buffer:
    Candidate *myWarpBuffer = warpCandBuffer + (threadIdx.x / 32) * k;

    // Process the dataset in batches (tiles) loaded into shared memory.
    for (int batch = 0; batch < data_count; batch += TILE_SIZE) {
        // Number of data points in this tile.
        int tile_count = (data_count - batch < TILE_SIZE) ? (data_count - batch) : TILE_SIZE;

        // Cooperative loading: each thread loads one or more data points from global memory into shared memory.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            smemTile[i] = data[batch + i];
        }
        __syncthreads(); // Ensure tile is loaded.

        // Within the warp, process data points in the shared tile.
        // Each thread processes elements stride 32.
        for (int i = lane; i < tile_count; i += 32) {
            float2 dpt = smemTile[i];
            float dx = dpt.x - q.x;
            float dy = dpt.y - q.y;
            float d = dx*dx + dy*dy;  // squared Euclidean distance

            // Update local candidate list if new candidate is better than the worst in the list.
            // localCand is maintained in sorted order (ascending order: best candidate at index 0,
            // worst candidate at index L-1).
            if (d < localCand[L-1].dist) {
                // Find proper insertion position via a backward scan.
                int pos = L - 1;
                // Shift candidates down until the right spot is found.
                while (pos > 0 && d < localCand[pos-1].dist) {
                    localCand[pos] = localCand[pos-1];
                    pos--;
                }
                localCand[pos].dist = d;
                localCand[pos].idx = batch + i; // global index of the data point.
            }
        }
        // End processing tile.

        // --- Warp–level candidate merge ---
        // The goal is to merge the 32 per–thread candidate lists (each of length L)
        // into one sorted list of k candidates. The sorted order is ascending by distance.
        // Step 1: Each thread writes its register candidate list to the corresponding positions
        // in the warp's shared candidate buffer. We use an interleaved layout: thread with lane 'r'
        // writes its i-th candidate to index (r + i*32) so that all k = L*32 candidates are collected.
        for (int i = 0; i < L; i++) {
            myWarpBuffer[lane + i * 32] = localCand[i];
        }
        // Use warp-level synchronization. (sync within warp is achieved using __syncwarp)
        __syncwarp();

        // Step 2: Let lane 0 in the warp perform a serial bitonic sort on the k candidates in myWarpBuffer.
        // Note: k is a power-of-two (32 <= k <= 1024).
        if (lane == 0) {
            // Serial bitonic sort on myWarpBuffer (stored in shared memory).
            // The algorithm sorts in ascending order.
            for (int size = 2; size <= k; size <<= 1) {
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                    // Process all indices [0, k)
                    for (int i = 0; i < k; i++) {
                        int j = i ^ stride;
                        if (j > i) {
                            // Determine the direction: if (i & size)==0, sort ascending.
                            bool ascending = ((i & size) == 0);
                            // Swap if out of order.
                            if ( (ascending && myWarpBuffer[i].dist > myWarpBuffer[j].dist) ||
                                 (!ascending && myWarpBuffer[i].dist < myWarpBuffer[j].dist) )
                            {
                                Candidate tmp = myWarpBuffer[i];
                                myWarpBuffer[i] = myWarpBuffer[j];
                                myWarpBuffer[j] = tmp;
                            }
                        }
                    }
                }
            }
        }
        __syncwarp();  // Ensure sorted candidate buffer is ready.

        // Step 3: Redistribute the sorted candidate list back to the per-thread registers.
        // Each thread reads L candidates from positions: lane + i*32.
        for (int i = 0; i < L; i++) {
            localCand[i] = myWarpBuffer[lane + i * 32];
        }
        __syncthreads(); // Ensure entire block is finished with the tile before loading next.
    } // end for (batch)

    // After processing all batches, we perform one final merge and write the final sorted candidate list to global memory.
    // Final merge: Write registers to warp buffer.
    for (int i = 0; i < L; i++) {
        myWarpBuffer[lane + i * 32] = localCand[i];
    }
    __syncwarp();
    if (lane == 0) {
        // Serial bitonic sort on myWarpBuffer for final candidate list.
        for (int size = 2; size <= k; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = 0; i < k; i++) {
                    int j = i ^ stride;
                    if (j > i) {
                        bool ascending = ((i & size) == 0);
                        if ((ascending && myWarpBuffer[i].dist > myWarpBuffer[j].dist) ||
                            (!ascending && myWarpBuffer[i].dist < myWarpBuffer[j].dist))
                        {
                            Candidate tmp = myWarpBuffer[i];
                            myWarpBuffer[i] = myWarpBuffer[j];
                            myWarpBuffer[j] = tmp;
                        }
                    }
                }
            }
        }
    }
    __syncwarp();
    // Write the final sorted k candidates (the nearest neighbors) to global result.
    // The results are stored in row-major order: for query i, the neighbors are stored at offsets i*k ... i*k+k-1.
    int query_offset = warpId * k;
    for (int i = lane; i < k; i += 32) {
        Candidate cand = myWarpBuffer[i];
        result[query_offset + i] = std::make_pair(cand.idx, cand.dist);
    }
}

//-------------------------------------------------------------------------------
// Host interface function to launch the k-NN kernel.
// The function assumes that query, data, and result arrays have been allocated on the device.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose a block configuration. We want each warp to process one query.
    // Let's choose 128 threads per block (i.e., 4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute the size of shared memory:
    // Shared memory holds:
    //  - Data point tile: TILE_SIZE * sizeof(float2)
    //  - Warp candidate buffers: (warpsPerBlock * k * sizeof(Candidate))
    size_t sharedMemBytes = TILE_SIZE * sizeof(float2) + warpsPerBlock * k * sizeof(Candidate);

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(query, query_count, data, data_count, result, k);

    // Optionally, one may synchronize and check for errors here.
    cudaDeviceSynchronize();
}