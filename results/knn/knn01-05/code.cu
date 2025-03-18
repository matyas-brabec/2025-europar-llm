#include <cuda_runtime.h>
#include <utility>
#include <limits>
// @FIXED
#include <cstdio>
// @FIXED
#include <math_constants.h>

// -----------------------------------------------------------------------------
// This file implements a brute‐force k‐nearest neighbors (k‐NN) algorithm in 2D.
// For each query point, we compute the squared Euclidean distance to all data
// points and then select the k closest ones. Since data_count is very large
// (millions) and query_count is in the thousands, we assign one thread block
// per query. In order to reduce register/local‐memory usage while guaranteeing
// correctness in the worst‐case (k up to 1024), we use a two‐phase algorithm:
//  1) Each thread in the block processes a disjoint subset of the data and maintains
//     its own local candidate list (a max–heap) of size k. (When a thread’s region
//     is large, its local heap will hold the best k candidates from its region.)
//  2) The block then cooperatively merges the k–candidate lists from each thread
//     (after sorting each one in ascending order) to produce the final sorted list
//     (ascending order of squared distance) of k–nearest neighbors.
//
// Because k can be large (up to 1024) and to keep the merge cost modest we choose a
// relatively small block size. In this implementation we use 16 threads per block.
// (Note: if k is much smaller, one might choose a larger block size; however, for
//  the worst-case k values targeted here, 16 threads per block offers a good tradeoff.)
//
// The shared memory usage per block is (blockDim.x * k * sizeof(Candidate)) bytes.
// For blockDim.x == 16 and k up to 1024, this is 16*1024*8 = 131072 bytes (128KB),
// which is acceptable on modern GPUs such as the NVIDIA A100/H100 (with maximized
// shared memory configuration).
// -----------------------------------------------------------------------------

// Candidate structure holds an index and its squared distance.
struct Candidate {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// Device kernel implementing k-NN for one query per block.
// Each block processes one query point (query[blockIdx.x]) and produces k nearest
// data points (index and squared distance), stored sorted in increasing order.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each block processes one query.
    int qid = blockIdx.x;
    if(qid >= query_count)
        return;

    // Load query point from global memory.
    float2 qPt = query[qid];

    // For best performance over huge data sets, each thread in the block
    // handles a portion (stride = blockDim.x) of data points.
    int tid = threadIdx.x;

    //---------------------------------------------------------------------------
    // Phase 1: Each thread computes its own k–nearest candidates (using a max–heap).
    // We allocate a local array "localHeap" to store candidate pairs.
    // Note: We allocate up to k candidates per thread (k is always <= 1024).
    //---------------------------------------------------------------------------
    // Use a fixed-size array (in local memory) to hold candidate heap.
    Candidate localHeap[1024];

    // Initialize local heap entries to "infinite" distance and invalid index.
    for (int j = 0; j < k; j++) {
        localHeap[j].dist = CUDART_INF_F; // Use CUDA's built-in infinity.
        localHeap[j].idx  = -1;
    }
    int heapSize = 0;  // number of valid candidates in the heap.

    // Iterate over data points assigned to this thread.
    for (int i = tid; i < data_count; i += blockDim.x)
    {
        float2 dPt = data[i];
        float dx = dPt.x - qPt.x;
        float dy = dPt.y - qPt.y;
        float dist = dx * dx + dy * dy; // squared Euclidean distance.

        if (heapSize < k) {
            // Heap is not yet full. Insert candidate.
            localHeap[heapSize].dist = dist;
            localHeap[heapSize].idx = i;
            heapSize++;
            if (heapSize == k) {
                // Build max–heap over the k candidates.
                for (int j = (k >> 1) - 1; j >= 0; j--) {
                    int parent = j;
                    while (true) {
                        int left = 2 * parent + 1;
                        int right = left + 1;
                        int largest = parent;
                        if (left < k && localHeap[left].dist > localHeap[largest].dist)
                            largest = left;
                        if (right < k && localHeap[right].dist > localHeap[largest].dist)
                            largest = right;
                        if (largest == parent)
                            break;
                        // Swap parent with the larger child.
                        Candidate temp = localHeap[parent];
                        localHeap[parent] = localHeap[largest];
                        localHeap[largest] = temp;
                        parent = largest;
                    }
                }
            }
        } else {
            // Heap is full; check if the new candidate is better than the worst so far.
            if (dist < localHeap[0].dist) {
                // Replace the root (worst candidate) with the new candidate.
                localHeap[0].dist = dist;
                localHeap[0].idx = i;
                // Sift down to restore the max–heap property.
                int parent = 0;
                while (true) {
                    int left = 2 * parent + 1;
                    int right = left + 1;
                    int largest = parent;
                    if (left < k && localHeap[left].dist > localHeap[largest].dist)
                        largest = left;
                    if (right < k && localHeap[right].dist > localHeap[largest].dist)
                        largest = right;
                    if (largest == parent)
                        break;
                    Candidate temp = localHeap[parent];
                    localHeap[parent] = localHeap[largest];
                    localHeap[largest] = temp;
                    parent = largest;
                }
            }
        }
    } // end for each data point

    //---------------------------------------------------------------------------
    // Phase 2: Merge per-thread candidate heaps into a global candidate list.
    // (a) First, sort each thread’s candidate list stored in localHeap in ascending
    //     order; we use a simple insertion sort since k is bounded (and k is small
    //     compared to data_count).
    //---------------------------------------------------------------------------
    for (int i = 1; i < k; i++) {
        Candidate key = localHeap[i];
        int j = i - 1;
        while (j >= 0 && localHeap[j].dist > key.dist) {
            localHeap[j+1] = localHeap[j];
            j--;
        }
        localHeap[j+1] = key;
    }

    // Shared memory for merging candidate lists.
    // Each thread writes its k sorted candidates into shared memory.
    // Shared memory size per block (in bytes) must be: blockDim.x * k * sizeof(Candidate)
    extern __shared__ Candidate sharedCandidates[];

    // Each thread copies its sorted candidate list into its designated region.
    for (int j = 0; j < k; j++) {
        sharedCandidates[tid * k + j] = localHeap[j];
    }
    __syncthreads();

    //---------------------------------------------------------------------------
    // Next, perform a binary-tree reduction to merge the sorted candidate lists.
    // At each step, pairs of lists (each of length k) are merged into a sorted list
    // that retains only the k best (smallest-distance) elements.
    //---------------------------------------------------------------------------
    int active = blockDim.x; // number of candidate lists currently in shared memory.
    while (active > 1) {
        int half = active >> 1; // active/2
        if (tid < half) {
            // Pointers to the two sorted lists to be merged.
            Candidate* listA = &sharedCandidates[tid * k];
            Candidate* listB = &sharedCandidates[(tid + half) * k];
            // Temporary local buffer to store merged result.
            Candidate merged[1024];
            int i_ptr = 0, j_ptr = 0, m_ptr = 0;
            // Merge the two lists (each of length k) by taking the k smallest.
            while (m_ptr < k && (i_ptr < k || j_ptr < k)) {
                Candidate cand;
                if (i_ptr < k && j_ptr < k) {
                    if (listA[i_ptr].dist <= listB[j_ptr].dist) {
                        cand = listA[i_ptr++];
                    } else {
                        cand = listB[j_ptr++];
                    }
                } else if (i_ptr < k) {
                    cand = listA[i_ptr++];
                } else {
                    cand = listB[j_ptr++];
                }
                merged[m_ptr++] = cand;
            }
            // Write the merged list back to listA.
            for (int x = 0; x < k; x++) {
                listA[x] = merged[x];
            }
        }
        __syncthreads();
        active = half;
    }

    // After reduction, thread 0's candidate list in sharedCandidates[0...k-1]
    // is the sorted (ascending) list of the k nearest neighbors.
    if (tid == 0) {
        for (int j = 0; j < k; j++) {
            // Write results in row-major order: for query qid, the j-th neighbor.
            result[(qid * k) + j] = std::pair<int, float>(sharedCandidates[j].idx, sharedCandidates[j].dist);
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface: run_knn
//   query       : array of float2 query points (device pointer)
//   query_count : number of query points
//   data        : array of float2 data points (device pointer)
//   data_count  : number of data points (>= k)
//   result      : output array of std::pair<int,float> (device pointer)
//                 Organized in row-major order: for query i, result[i*k+j] is the
//                 j-th nearest neighbor (data index and squared distance).
//   k           : number of nearest neighbors to select (power of two between 32 and 1024)
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block size: here we use 16 threads per block.
    int blockSize = 16;
    int gridSize = query_count; // one block per query.

    // Calculate the amount of shared memory needed per block:
    // sharedMem = blockSize * k * sizeof(Candidate)
    size_t sharedMemSize = blockSize * static_cast<size_t>(k) * sizeof(Candidate);

    // Launch the kernel.
    knn_kernel<<<gridSize, blockSize, sharedMemSize>>>(query, query_count, data, data_count, result, k);

    // Check for launch errors (optional).
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // In production code you would handle the error appropriately.
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Wait for the kernel to finish.
    cudaDeviceSynchronize();
}
