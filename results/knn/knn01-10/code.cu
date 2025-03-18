#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>

// -----------------------------------------------------------------------------
// This implementation computes the k-nearest neighbors (k-NN) for 2D points
// using a brute-force approach with tiled data loads into shared memory.
// Each thread processes one query point and maintains its own candidate list
// (a max‐heap of size k) in local memory. When a new point is found that is
// closer than the current farthest candidate (heap root), the heap is updated.
// At the end, the candidate list is heap‐sorted into ascending order (closest first)
// and written to global memory as an array of std::pair<int, float> values.
// 
// Assumptions:
//  - k is a power of two between 32 and 1024 (inclusive).
//  - data_count >= k.
//  - Query points, data points, and results arrays have been allocated with cudaMalloc.
//  - This code is tuned for modern NVIDIA GPUs like the A100/H100 using the latest CUDA toolkit.
// 
// The kernel uses a tile size of 256 data points (threadsPerBlock) loaded into shared
// memory to amortize global memory latency. Each thread in a block processes one query.
// -----------------------------------------------------------------------------

#define TILE_SIZE 256  // Number of elements loaded per tile; must equal blockDim.x.

//------------------------------------------------------------------------------
// Device function: heapifyDown
// Restores the max-heap property in the candidate heap starting at index "root".
// The heap is maintained in the arrays "dists" (squared distances) and "idxs"
// (corresponding data point indices).
//------------------------------------------------------------------------------
__device__ inline void heapifyDown(int heapSize, float *dists, int *idxs, int root)
{
    int current = root;
    while (true)
    {
        int left = (current << 1) + 1;
        int right = left + 1;
        int largest = current;
        if (left < heapSize && dists[left] > dists[largest])
            largest = left;
        if (right < heapSize && dists[right] > dists[largest])
            largest = right;
        if (largest != current)
        {
            // Swap dists and idxs between current and largest.
            float tmp = dists[current];
            dists[current] = dists[largest];
            dists[largest] = tmp;
            int tmpIdx = idxs[current];
            idxs[current] = idxs[largest];
            idxs[largest] = tmpIdx;
            current = largest;
        }
        else
        {
            break;
        }
    }
}

//------------------------------------------------------------------------------
// Device function: buildHeap
// Builds a max-heap from the candidate arrays using a bottom-up approach.
//------------------------------------------------------------------------------
__device__ inline void buildHeap(int heapSize, float *dists, int *idxs)
{
    // Start from the last parent and call heapifyDown.
    for (int i = (heapSize / 2) - 1; i >= 0; i--)
    {
        heapifyDown(heapSize, dists, idxs, i);
    }
}

//------------------------------------------------------------------------------
// Device function: heapSort
// Sorts the candidate heap (of size k) into ascending order via in-place heapsort.
// Since the candidate list is maintained as a max-heap, this procedure produces
// an array where the smallest elements (closest points) come first.
//------------------------------------------------------------------------------
__device__ inline void heapSort(int heapSize, float *dists, int *idxs)
{
    // Standard heapsort algorithm operating in-place.
    for (int i = heapSize - 1; i > 0; i--)
    {
        // Swap the max element (root) with the last element of the heap.
        float tmp = dists[0];
        dists[0] = dists[i];
        dists[i] = tmp;
        int tmpIdx = idxs[0];
        idxs[0] = idxs[i];
        idxs[i] = tmpIdx;
        // Restore the max-heap property on the reduced heap.
        heapifyDown(i, dists, idxs, 0);
    }
}

//------------------------------------------------------------------------------
// Kernel: knn_kernel
// For each query point, iterates over all data points in tiles. Each tile is
// loaded into shared memory cooperatively by the block. Each thread computes
// squared Euclidean distances from its query to the points in the tile and
// updates its candidate list (max-heap) if a nearer point is found.
// At the end, results are heap-sorted and written in ascending order.
//
// Parameters:
//  - query: Array of query points (float2)
//  - query_count: Number of query points.
//  - data: Array of data points (float2)
//  - data_count: Number of data points.
//  - result: Output array (row-major) where each query's k nn are stored.
//  - k: Number of nearest neighbors to find (k is a power of two between 32 and 1024).
//------------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Allocate shared memory for one tile of data points.
    extern __shared__ float2 sdata[];

    // Each thread processes one query point.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= query_count)
        return;

    // Load the query point.
    float2 q = query[tid];

    // Each thread maintains its own candidate list (max-heap) of up to k nearest neighbors.
    // We allocate fixed-size arrays of maximum capacity 1024.
    float cand_dists[1024]; // Squared distances.
    int   cand_idxs[1024];  // Data point indices.
    int cand_count = 0;     // Number of candidates seen so far.
    bool heapBuilt = false; // Flag to indicate if the candidate list has been heapified.

    // Loop over the data points in tiles.
    int numTiles = (data_count + blockDim.x - 1) / blockDim.x;
    for (int t = 0; t < numTiles; t++)
    {
        // Each thread in the block loads one data point into shared memory, if available.
        int dataIdx = t * blockDim.x + threadIdx.x;
        if (dataIdx < data_count)
        {
            sdata[threadIdx.x] = data[dataIdx];
        }
        // Synchronize to ensure the tile is fully loaded.
        __syncthreads();

        // Compute the number of elements in this tile.
        int tileSize = ((t + 1) * blockDim.x <= data_count) ? blockDim.x : (data_count - t * blockDim.x);

        // Each thread processes all points in the current tile.
        for (int j = 0; j < tileSize; j++)
        {
            // Compute the global index for this data point.
            int global_idx = t * blockDim.x + j;
            float2 d = sdata[j];

            // Compute squared Euclidean distance.
            float dx = q.x - d.x;
            float dy = q.y - d.y;
            float dist = dx * dx + dy * dy;

            // Update candidate list.
            if (cand_count < k)
            {
                // Append candidate if we haven't reached k points yet.
                cand_dists[cand_count] = dist;
                cand_idxs[cand_count]  = global_idx;
                cand_count++;
                if (cand_count == k)
                {
                    // Once we have k candidates, build the initial max-heap.
                    buildHeap(k, cand_dists, cand_idxs);
                    heapBuilt = true;
                }
            }
            else
            {
                // Candidate list is full; compare against the farthest (heap root).
                if (dist < cand_dists[0])
                {
                    // Replace the root with the new candidate and restore heap order.
                    cand_dists[0] = dist;
                    cand_idxs[0] = global_idx;
                    heapifyDown(k, cand_dists, cand_idxs, 0);
                }
            }
        }
        // Synchronize before loading the next tile.
        __syncthreads();
    }

    // At this point, each thread's candidate list holds k nearest neighbors (in a max-heap).
    // Perform heapsort to order candidates in ascending order (nearest first).
    heapSort(k, cand_dists, cand_idxs);

    // Write the sorted candidates to global memory.
    int base = tid * k;
    for (int j = 0; j < k; j++)
    {
        // Each result is stored as a std::pair<int, float>
        result[base + j].first  = cand_idxs[j];
        result[base + j].second = cand_dists[j];
    }
}

//------------------------------------------------------------------------------
// Host function: run_knn
// Sets up the execution configuration and launches the kernel.
//------------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose 256 threads per block (equal to TILE_SIZE).
    const int threadsPerBlock = TILE_SIZE;
    int blocks = (query_count + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate shared memory for one tile of data points.
    size_t sharedMemSize = threadsPerBlock * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);

    // Optionally, check for errors and synchronize:
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { ... }
    // cudaDeviceSynchronize();
}