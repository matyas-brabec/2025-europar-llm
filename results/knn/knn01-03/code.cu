#include <cuda_runtime.h>
#include <utility>
#include <limits>

// Maximum allowed value of k (as per problem statement, k is between 32 and 1024)
#define MAX_K 1024

// -----------------------------------------------------------------------------
// This struct holds a candidate data point: its index and squared Euclidean distance.
// It serves as the element type of the max‐heap that maintains the k best neighbors.
struct Candidate {
    int index;
    float dist;
};

// -----------------------------------------------------------------------------
// Device inline swap function for Candidate structures.
__device__ __forceinline__ void swapCandidate(Candidate &a, Candidate &b) {
    Candidate tmp = a;
    a = b;
    b = tmp;
}

// -----------------------------------------------------------------------------
// Device inline function that "heapifies down" from index i in a max‐heap of size heapSize.
// In our max‐heap, the candidate with the largest (i.e. worst) distance is at the root.
// This routine restores the max‐heap property after modification.
__device__ __forceinline__ void heapify_down(Candidate *heap, int heapSize, int i) {
    // Loop until the heap property is satisfied.
    while (true) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < heapSize && heap[left].dist > heap[largest].dist)
            largest = left;
        if (right < heapSize && heap[right].dist > heap[largest].dist)
            largest = right;
        if (largest == i)
            break;
        swapCandidate(heap[i], heap[largest]);
        i = largest;
    }
}

// -----------------------------------------------------------------------------
// Device function to perform an in-place heapsort on an array of Candidate elements.
// Since the candidates are stored in a max‐heap, this sort produces an array in ascending
// order (i.e. nearest neighbor first).
__device__ void heap_sort(Candidate *heap, int heapSize) {
    // Iteratively remove the maximum element (root of the heap) and rebuild the heap.
    for (int i = heapSize - 1; i > 0; i--) {
        swapCandidate(heap[0], heap[i]);
        heapify_down(heap, i, 0);
    }
}

// -----------------------------------------------------------------------------
// CUDA kernel that performs a brute‐force k–nearest neighbors search for 2D points.
// Each thread processes one query point by iterating over all data points and maintaining
// a local max‐heap (of size k) that holds the k closest data points seen so far.
// The squared Euclidean distance is used as the metric.
// After scanning all the data, the thread sorts its local k best candidates in ascending order
// and writes them to the global result array.
__global__ void knn_kernel(const float2 * __restrict__ query, int queryCount,
                           const float2 * __restrict__ data, int dataCount,
                           std::pair<int, float> * __restrict__ result, int k) {
    // Each thread works on one query.
    int queryIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queryIdx >= queryCount)
        return;

    // Load the query point from global memory.
    float2 q = query[queryIdx];

    // Allocate a local array of Candidates (max-heap) to hold the best k points.
    // We allocate MAX_K elements; only the first k elements are used (k is guaranteed <= MAX_K).
    Candidate heap[MAX_K];

    // Loop over all data points.
    // For each data point, compute its squared Euclidean distance to the query.
    // For the first k points, simply fill the heap array.
    // When the heap is full, build a max-heap and then, for all subsequent points, if the
    // distance is smaller than the worst distance (located at heap[0]), then replace it and heapify.
    for (int i = 0; i < dataCount; i++) {
        // Use __ldg to load the data point via read-only cache.
        float2 d = __ldg(&data[i]);
        float dx = q.x - d.x;
        float dy = q.y - d.y;
        float dist = dx * dx + dy * dy;

        if (i < k) {
            // Fill the initial heap array.
            heap[i].index = i;
            heap[i].dist = dist;
            // Once the first k points have been processed, build the max-heap.
            if (i == k - 1) {
                int start = (k / 2) - 1;
                for (int j = start; j >= 0; j--) {
                    heapify_down(heap, k, j);
                }
            }
        } else {
            // If the current point is closer than the farthest in our heap...
            if (dist < heap[0].dist) {
                // Replace the worst candidate with the current point.
                heap[0].index = i;
                heap[0].dist = dist;
                // Restore the heap property.
                heapify_down(heap, k, 0);
            }
        }
    }

    // At this point, the heap contains the k nearest neighbors (in max-heap order).
    // We now perform an in-place heapsort to obtain them in ascending order (nearest first).
    heap_sort(heap, k);

    // Write out the results to the global result array.
    // For the query point at index queryIdx, the j-th nearest neighbor is stored at
    // result[queryIdx * k + j].
    int offset = queryIdx * k;
    for (int j = 0; j < k; j++) {
        result[offset + j] = std::pair<int, float>(heap[j].index, heap[j].dist);
    }
}

// -----------------------------------------------------------------------------
// Host function that launches the CUDA kernel for k–nearest neighbors search.
// Input pointers (query, data, result) are assumed to be allocated on the device (via cudaMalloc).
// The kernel uses one thread per query and processes data in a brute-force manner.
// We choose a block size of 256 threads per block, which is a typical value for modern GPUs.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Ensure that k is within our supported maximum.
    if (k > MAX_K)
        k = MAX_K;

    // Choose the number of threads per block.
    int blockSize = 256;
    // Compute the grid size based on the number of queries.
    int gridSize = (query_count + blockSize - 1) / blockSize;

    // Launch the kernel with the chosen grid and block dimensions.
    knn_kernel<<<gridSize, blockSize>>>(query, query_count, data, data_count, result, k);

    // Synchronize to ensure that the kernel has completed.
    cudaDeviceSynchronize();
}