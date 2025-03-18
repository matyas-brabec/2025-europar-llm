#include <cuda_runtime.h>
#include <utility>
#include <cstdio>

// Define the tile size for shared memory tiling of the data array.
// We choose TILE_SIZE equal to the number of threads per block (256),
// so that each thread loads one element per tile.
#define TILE_SIZE 256

// The CUDA kernel implements a brute‐force k‑nearest neighbors search for 2D points.
// Each thread processes one query point and scans the entire data array in tiles.
// For each query, the thread maintains a candidate list of size k (k is a power of 2 between 32 and 1024)
// stored as a max‐heap. The candidate list is first filled with the first k data points
// (taken in order) and then turned into a max heap. For every subsequent data point, if its squared
// Euclidean distance (L2 norm squared) to the query is smaller than the current maximum candidate,
// the candidate at the root is replaced and the heap is adjusted via reheapify.
// After processing all data points, the candidate list (which is in max‐heap order)
// is heap‐sorted into ascending order (nearest first) and written to the output.
//
// This implementation uses shared memory to load tiles of the data points to improve
// memory bandwidth utilization on modern NVIDIA GPUs such as A100 or H100.
 
// Note: The candidate list is maintained in per-thread local memory (using a fixed array of size 1024)
// since k is known to be no larger than 1024.
 
// Define an internal candidate structure that holds an index and its squared distance.
struct Candidate {
    int idx;
    float dist;
};

// __global__ kernel function for k-NN search
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each thread processes one query.
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= query_count)
        return;
        
    // Load the query point.
    float2 q = query[qid];
    
    // Allocate a local candidate array. Maximum k is 1024.
    Candidate cand[1024];
    int count = 0;         // Number of candidates collected so far.
    bool heapInitialized = false;  // Flag indicates whether the max heap has been built.
    
    // Declare shared memory tile for data points.
    __shared__ float2 s_tile[TILE_SIZE];
    
    // Process the data array in tiles. Each tile loads up to TILE_SIZE data points.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE)
    {
        // Each thread loads one element into shared memory if it is within bounds.
        int globalIdx = tileStart + threadIdx.x;
        int tileCount = ((data_count - tileStart) < TILE_SIZE) ? (data_count - tileStart) : TILE_SIZE;
        if(threadIdx.x < tileCount)
        {
            s_tile[threadIdx.x] = data[globalIdx];
        }
        // Ensure the tile is loaded before processing.
        __syncthreads();
        
        // Each thread iterates over the elements in the current tile.
        for (int i = 0; i < tileCount; i++)
        {
            int j = tileStart + i;  // global index of the data point
            float2 pt = s_tile[i];
            // Compute squared Euclidean distance between the query and this data point.
            float dx = q.x - pt.x;
            float dy = q.y - pt.y;
            float d = dx * dx + dy * dy;
            
            // If we haven't yet collected k candidates, simply add this candidate.
            if (count < k)
            {
                cand[count].idx = j;
                cand[count].dist = d;
                count++;
                // Once we have k candidates, build a max-heap using bottom-up heap construction.
                if (count == k)
                {
                    for (int h = (k / 2) - 1; h >= 0; h--)
                    {
                        int parent = h;
                        while (true)
                        {
                            int left = 2 * parent + 1;
                            int right = left + 1;
                            int largest = parent;
                            if (left < k && cand[left].dist > cand[largest].dist)
                                largest = left;
                            if (right < k && cand[right].dist > cand[largest].dist)
                                largest = right;
                            if (largest == parent)
                                break;
                            // Swap parent and largest child.
                            Candidate temp = cand[parent];
                            cand[parent] = cand[largest];
                            cand[largest] = temp;
                            parent = largest;
                        }
                    }
                    heapInitialized = true;
                }
            }
            else
            {
                // Now the candidate list is full and represented as a max-heap.
                // If the new distance is smaller than the maximum (at root), update the heap.
                if (d < cand[0].dist)
                {
                    cand[0].dist = d;
                    cand[0].idx = j;
                    // Reheapify (trickle down) from the root.
                    int parent = 0;
                    while (true)
                    {
                        int left = 2 * parent + 1;
                        int right = left + 1;
                        int largest = parent;
                        if (left < k && cand[left].dist > cand[largest].dist)
                            largest = left;
                        if (right < k && cand[right].dist > cand[largest].dist)
                            largest = right;
                        if (largest == parent)
                            break;
                        Candidate temp = cand[parent];
                        cand[parent] = cand[largest];
                        cand[largest] = temp;
                        parent = largest;
                    }
                }
            }
        }
        // Synchronize to ensure all threads have finished using the shared tile before it is overwritten.
        __syncthreads();
    }
    
    // At this point, the candidate array 'cand' holds the k nearest neighbors (in a max-heap order).
    // We now sort the candidate list into ascending order (nearest first) via heap sort.
    for (int i = k - 1; i > 0; i--)
    {
        // Swap the maximum element at cand[0] with the last element of the heap.
        Candidate temp = cand[0];
        cand[0] = cand[i];
        cand[i] = temp;
        
        int parent = 0;
        int heapSize = i;
        while (true)
        {
            int left = 2 * parent + 1;
            int right = left + 1;
            int largest = parent;
            if (left < heapSize && cand[left].dist > cand[largest].dist)
                largest = left;
            if (right < heapSize && cand[right].dist > cand[largest].dist)
                largest = right;
            if (largest == parent)
                break;
            temp = cand[parent];
            cand[parent] = cand[largest];
            cand[largest] = temp;
            parent = largest;
        }
    }
    
    // Write the sorted candidate list to the output result array.
    // For query qid, the neighbors are stored consecutively starting at result[qid * k].
    int base = qid * k;
    for (int i = 0; i < k; i++)
    {
        result[base + i].first = cand[i].idx;
        result[base + i].second = cand[i].dist;
    }
}

// Host function that launches the k-NN kernel.
// This function uses 256 threads per block, which is a good choice for modern GPUs.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose 256 threads per block.
    int threadsPerBlock = 256;
    int blocksPerGrid = (query_count + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel.
    knn_kernel<<<blocksPerGrid, threadsPerBlock>>>(query, query_count, data, data_count, result, k);
    
    // Synchronize to ensure kernel completion.
    cudaDeviceSynchronize();
}