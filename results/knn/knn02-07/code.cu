#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <utility>  // For std::pair

// -----------------------------------------------------------------------------
// This kernel implements the k-nearest neighbors (k-NN) computation for 2D points.
// Each query point is processed by one warp (32 threads). For each query the warp
// iterates over the data points in batches (tiles) loaded into shared memory.
// The warp maintains a private candidate list (of k nearest neighbors) in a max-heap,
// stored entirely in registers by lane 0. Each warp‐iteration, all 32 threads compute
// a candidate distance from the current batch and lane 0 collects these via warp shuffle.
// If a candidate distance is less than the current worst (largest) distance in the heap,
// lane 0 inserts it into the max‐heap (using heapify-down) so that at the end, the
// candidate heap contains the k best (smallest) squared distances found.
// Finally, lane 0 performs an in‐place heap sort (which yields ascending order)
// and writes the result (pairs of (data index, distance)) to global memory.
// -----------------------------------------------------------------------------
//
// Hyper-parameters:
// - TILE_SIZE: number of data points loaded into shared memory in each batch.
//   A value of 1024 is used (this is a tunable parameter).
// - Threads per warp: 32 (by definition) and each query is handled by one warp.
// - Threads per block: chosen as a multiple of 32 (here 128 threads/block => 4 warps per block).
//
// Note: We assume 'k' is a power of two between 32 and 1024 inclusive. Maximum k is 1024.
//       The candidate heap is maintained only by lane 0 of each warp in registers using
//       fixed-size arrays of length 1024 (only the first 'k' entries are used).
//
// The squared Euclidean distance is used (i.e. (dx*dx + dy*dy)).
// -----------------------------------------------------------------------------

// Define TILE_SIZE as a compile-time hyper-parameter.
#define TILE_SIZE 1024

// Kernel to compute k-NN for 2D points.
// Each warp (32 threads) processes one query point.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp handles one query.
    const int warpSize = 32;
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdInBlock = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    // Compute global warp (query) id.
    int warpsPerBlock = blockDim.x / warpSize;
    int queryIdx = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (queryIdx >= query_count)
        return;

    // Load query point.
    float2 q = query[queryIdx];

    // Declare shared memory tile (used by all threads in the block).
    extern __shared__ float2 shared_tile[];

    // -------------------------------------------------------------------------
    // Candidate heap maintained by lane 0 of this warp.
    // We use a max-heap of size 'k' stored in registers.
    // The heap property: the root (index 0) holds the largest distance.
    // Initially, all heap entries are set to FLT_MAX and index -1.
    // Only lane 0 will update this heap.
    // Since maximum k is 1024, we allocate fixed-size arrays of length 1024.
    // -------------------------------------------------------------------------
    float heap_d[1024];    // Candidate distances (squared Euclidean)
    int   heap_idx[1024];  // Candidate data indices
    if (lane == 0)
    {
        // Initialize the heap with worst possible distances.
        for (int i = 0; i < k; i++)
        {
            heap_d[i] = FLT_MAX;
            heap_idx[i] = -1;
        }
    }

    // Loop over data points in batches (tiles).
    for (int tile_offset = 0; tile_offset < data_count; tile_offset += TILE_SIZE)
    {
        // Load a tile of data into shared memory.
        // Each thread in the block loads multiple elements if needed.
        int tileLimit = TILE_SIZE;
        if (tile_offset + TILE_SIZE > data_count)
            tileLimit = data_count - tile_offset;
        for (int i = threadIdx.x; i < tileLimit; i += blockDim.x)
        {
            shared_tile[i] = data[tile_offset + i];
        }
        __syncthreads();

        // Process the tile in groups of warpSize (each warp iteration processes 32 candidates).
        for (int j = 0; j < tileLimit; j += warpSize)
        {
            int idxInTile = j + lane; // Each lane processes one element in this group.
            float d = FLT_MAX;        // Default distance if index is out-of-bound.
            int   dataIndex = -1;
            if (idxInTile < tileLimit)
            {
                float2 pt = shared_tile[idxInTile];
                float dx = pt.x - q.x;
                float dy = pt.y - q.y;
                d = dx * dx + dy * dy;
                dataIndex = tile_offset + idxInTile;
            }
            // Now, gather candidate distances from all 32 warp lanes.
            // Only lane 0 of the warp will update the global candidate heap.
            if (lane == 0)
            {
                // Process each candidate value from the 32 lanes.
                for (int l = 0; l < warpSize; l++)
                {
                    // Use warp shuffle to get the candidate from lane 'l'.
                    float cand = __shfl_sync(0xFFFFFFFF, d, l);
                    int   candIndex = __shfl_sync(0xFFFFFFFF, dataIndex, l);
                    // If the candidate distance is less than the worst distance in the heap,
                    // then insert it into the max-heap.
                    if (cand < heap_d[0])
                    {
                        // Replace the root with the new candidate.
                        heap_d[0] = cand;
                        heap_idx[0] = candIndex;
                        // Heapify-down to restore the max-heap property.
                        int i = 0;
                        while (true)
                        {
                            int left = 2 * i + 1;
                            int right = 2 * i + 2;
                            int largest = i;
                            if (left < k && heap_d[left] > heap_d[largest])
                                largest = left;
                            if (right < k && heap_d[right] > heap_d[largest])
                                largest = right;
                            if (largest != i)
                            {
                                float tmp = heap_d[i];
                                heap_d[i] = heap_d[largest];
                                heap_d[largest] = tmp;
                                int tmpi = heap_idx[i];
                                heap_idx[i] = heap_idx[largest];
                                heap_idx[largest] = tmpi;
                                i = largest;
                            }
                            else
                                break;
                        }
                    }
                }
            }
            // End of warp group processing. (No explicit __syncwarp() is needed since shuffles are warp-synchronous.)
        }
        __syncthreads(); // Synchronize before loading next tile.
    }

    // After processing all data tiles, lane 0 has the candidate max-heap containing the k nearest neighbors.
    // Now, lane 0 performs a heap sort to produce a sorted (ascending) list of (index, distance) pairs.
    if (lane == 0)
    {
        // Perform in-place heap sort.
        // Standard heapsort: for i = k-1 downto 1, swap heap root and element i, then heapify the root.
        for (int i = k - 1; i > 0; i--)
        {
            // Swap root with element i.
            float tmp = heap_d[0];
            heap_d[0] = heap_d[i];
            heap_d[i] = tmp;
            int tmpi = heap_idx[0];
            heap_idx[0] = heap_idx[i];
            heap_idx[i] = tmpi;
            // Heapify the root over the reduced heap (size = i).
            int j = 0;
            while (true)
            {
                int left = 2 * j + 1;
                int right = 2 * j + 2;
                int largest = j;
                if (left < i && heap_d[left] > heap_d[largest])
                    largest = left;
                if (right < i && heap_d[right] > heap_d[largest])
                    largest = right;
                if (largest != j)
                {
                    float t = heap_d[j];
                    heap_d[j] = heap_d[largest];
                    heap_d[largest] = t;
                    int ti = heap_idx[j];
                    heap_idx[j] = heap_idx[largest];
                    heap_idx[largest] = ti;
                    j = largest;
                }
                else
                {
                    break;
                }
            }
        }
        // The heap_d array is now sorted in ascending order.
        // Write the results for this query to global memory.
        int offset = queryIdx * k;
        for (int i = 0; i < k; i++)
        {
            result[offset + i] = std::pair<int, float>(heap_idx[i], heap_d[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// Host-facing C++ interface function.
// This function launches the CUDA kernel with appropriate grid and block dimensions.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose threads per block.
    // Each query is processed by one warp (32 threads).
    // We choose 128 threads per block (4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Compute grid dimensions so that each warp processes one query.
    int totalWarps = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    int blocks = totalWarps;  // one warp per query results in blocks = totalWarps

    // Calculate shared memory size per block (for TILE_SIZE float2 elements).
    size_t sharedMemSize = TILE_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);

    // Optionally, one can add error checking here (omitted for brevity).
    cudaDeviceSynchronize();
}