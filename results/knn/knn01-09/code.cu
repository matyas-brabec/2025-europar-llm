#include <cuda_runtime.h>
#include <cfloat>     // for FLT_MAX
#include <utility>    // for std::pair

// -----------------------------------------------------------------------------
// In this implementation we compute the k-nearest neighbors (k-NN) for 2D points
// using a GPU kernel. Each query is handled by one CUDA block and the data
// points are processed in a strided loop by the threads in that block.
// Each thread collects a local top-k candidate list (using an online max‐heap)
// from its assigned subset of data points. Then, the sorted local candidate
// lists (of exactly k elements each; if a thread processed fewer than k points,
// its list is padded with dummy entries with distance FLT_MAX) are written to
// shared memory. Finally, thread 0 in the block sequentially merges the sorted
// lists from all threads (each list is sorted in ascending order by distance)
// to produce the final sorted list of k nearest neighbors for the query.
// 
// We choose the number of threads per block (T) at launch such that the total
// shared memory used for the reduction stage (T * k * sizeof(std::pair<int, float>))
// is modest. For our implementation we require T*k*8 <= 49152 bytes (48KB).
// Hence, we choose T <= 6144/k. (k is always a power of two between 32 and 1024.)
// 
// The squared Euclidean distance is computed using standard arithmetic.
// The kernel assumes that the arrays "query", "data", and "result" are allocated
// on the device (via cudaMalloc). No additional device memory is allocated.
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// Device inline function: pushHeap
// Implements an online max-heap insertion for an array of candidate distances.
// The heap stores the best (lowest) k distances seen so far in a max-heap so that
// the worst candidate is always at heap[0].
// - heap: array of candidate distances (size = k)
// - heapIdx: corresponding array of candidate indices
// - size: current number of elements stored in the heap (0 <= size <= k)
// - K: maximum number of candidates to store (k)
// - d: new candidate distance
// - idx: new candidate index
// -----------------------------------------------------------------------------
__device__ inline void pushHeap(const int K, float d, int idx, float* heap, int* heapIdx, int &size) {
    if (size < K) {
        // Insert new element at the end and bubble up.
        int pos = size;
        heap[pos] = d;
        heapIdx[pos] = idx;
        size++;
        // Bubble up to maintain max-heap property.
        while (pos > 0) {
            int parent = (pos - 1) >> 1;
            if (heap[parent] < heap[pos]) {
                // Swap parent and current
                float tmp = heap[parent];
                heap[parent] = heap[pos];
                heap[pos] = tmp;
                int tmpi = heapIdx[parent];
                heapIdx[parent] = heapIdx[pos];
                heapIdx[pos] = tmpi;
                pos = parent;
            } else {
                break;
            }
        }
    } else {
        // Heap is full: if new candidate is better (smaller distance)
        // than the worst candidate at the root, replace and heapify down.
        if (d < heap[0]) {
            heap[0] = d;
            heapIdx[0] = idx;
            // Heapify down.
            int pos = 0;
            while (true) {
                int left = (pos << 1) + 1;
                int right = left + 1;
                int largest = pos;
                if (left < K && heap[left] > heap[largest])
                    largest = left;
                if (right < K && heap[right] > heap[largest])
                    largest = right;
                if (largest != pos) {
                    float tmp = heap[pos];
                    heap[pos] = heap[largest];
                    heap[largest] = tmp;
                    int tmpi = heapIdx[pos];
                    heapIdx[pos] = heapIdx[largest];
                    heapIdx[largest] = tmpi;
                    pos = largest;
                } else {
                    break;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Device inline function: sortArray
// Simple insertion sort that sorts an array (and its associated index array)
// in ascending order based on the float values.
// Sorting k elements (k is relatively small: 32 to 1024).
// -----------------------------------------------------------------------------
__device__ inline void sortArray(const int n, float* arr, int* idxArr) {
    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int keyIdx = idxArr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            idxArr[j + 1] = idxArr[j];
            j--;
        }
        arr[j + 1] = key;
        idxArr[j + 1] = keyIdx;
    }
}

// -----------------------------------------------------------------------------
// Device inline function: mergeSortedLists
// Merges two sorted arrays (each of length k) into a sorted output array,
// keeping only the first k (smallest) elements.
// - A: first list of distances (sorted in ascending order)
// - A_idx: corresponding indices for first list
// - B: second list of distances (sorted in ascending order)
// - B_idx: corresponding indices for second list
// - out: output array of distances (length = k)
// - out_idx: output indices array (length = k)
// -----------------------------------------------------------------------------
__device__ inline void mergeSortedLists(const int k,
                                          const float* A, const int* A_idx,
                                          const float* B, const int* B_idx,
                                          float* out, int* out_idx) {
    int i = 0, j = 0;
    for (int r = 0; r < k; r++) {
        if (i < k && (j >= k || A[i] <= B[j])) {
            out[r] = A[i];
            out_idx[r] = A_idx[i];
            i++;
        } else {
            out[r] = B[j];
            out_idx[r] = B_idx[j];
            j++;
        }
    }
}

// -----------------------------------------------------------------------------
// Define a local struct for neighbors (k-NN candidate info).
// This struct has the same memory layout as std::pair<int,float>.
// -----------------------------------------------------------------------------
struct Neighbor {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// Device kernel: knn_kernel
// Each block processes one query from the "query" array. Threads in the block
// cooperatively compute the k-nearest neighbors by scanning through all data
// points in a strided loop. Each thread builds a local candidate list using
// an online max-heap, then sorts its candidate list in ascending order.
// The sorted lists from all threads are written to shared memory and finally
// merged by thread 0 to produce the final sorted list of k neighbors for
// the query.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k)
{
    // Each block handles one query.
    int q = blockIdx.x;
    if (q >= query_count)
        return;
    
    // Load query point into register.
    float2 q_point = query[q];
    
    // Use blockDim.x threads in this block. (Chosen at launch.)
    const int T = blockDim.x;
    const int tid = threadIdx.x;
    
    // Maximum allowed k value per problem specification.
    const int MAXK = 1024;  
    // Local arrays to implement an online max-heap.
    // Only first 'k' elements (k <= MAXK) are used.
    float localDists[MAXK];
    int   localIdxs[MAXK];
    int heapSize = 0; // current number of candidates
    
    // Loop over data points in a strided manner.
    // Each thread processes data indices: tid, tid+T, tid+2*T, ...
    for (int i = tid; i < data_count; i += T) {
        float2 dPoint = data[i];
        float dx = q_point.x - dPoint.x;
        float dy = q_point.y - dPoint.y;
        float dist = dx * dx + dy * dy;
        // Insert the candidate into the local heap.
        pushHeap(k, dist, i, localDists, localIdxs, heapSize);
    }
    
    // If this thread processed fewer than k candidates, pad remaining entries
    // with dummy values (FLT_MAX and index -1). This ensures that each thread's
    // candidate list has exactly k entries.
    for (int i = heapSize; i < k; i++) {
        localDists[i] = FLT_MAX;
        localIdxs[i] = -1;
    }
    
    // Sort the local candidate list in ascending order by distance.
    // (After this, localDists[0] is the smallest distance.)
    sortArray(k, localDists, localIdxs);
    
    // -------------------------------------------------------------------------
    // Write the sorted local candidate list to shared memory.
    // We use dynamic shared memory. Each thread writes its k candidates into
    // a contiguous region: sharedNeighbors[tid * k ... tid * k + k - 1].
    // -------------------------------------------------------------------------
    extern __shared__ char shared_mem[];
    Neighbor* sharedNeighbors = reinterpret_cast<Neighbor*>(shared_mem);
    for (int j = 0; j < k; j++) {
        sharedNeighbors[tid * k + j].idx  = localIdxs[j];
        sharedNeighbors[tid * k + j].dist = localDists[j];
    }
    __syncthreads();
    
    // -------------------------------------------------------------------------
    // Reduction phase: Thread 0 in the block merges the sorted candidate lists
    // from all T threads to form the final sorted list of k nearest neighbors.
    // The merge is done sequentially since T is small.
    // -------------------------------------------------------------------------
    if (tid == 0) {
        // Allocate local arrays for the global merge result.
        float globalDists[MAXK];
        int   globalIdxs[MAXK];
        // Copy thread 0's sorted list from shared memory into global arrays.
        for (int j = 0; j < k; j++) {
            globalDists[j] = sharedNeighbors[j].dist; // thread 0's list starts at offset 0
            globalIdxs[j] = sharedNeighbors[j].idx;
        }
        
        // Temporary arrays to hold merged result.
        float mergedDists[MAXK];
        int   mergedIdxs[MAXK];
        
        // Merge candidate list from threads 1 ... T-1.
        for (int t = 1; t < T; t++) {
            // Copy sorted candidate list from thread t from shared memory.
            float otherListDists[MAXK];
            int   otherListIdxs[MAXK];
            int base = t * k;
            for (int j = 0; j < k; j++) {
                otherListDists[j] = sharedNeighbors[base + j].dist;
                otherListIdxs[j] = sharedNeighbors[base + j].idx;
            }
            // Merge globalDists and otherListDists into merged arrays.
            mergeSortedLists(k, globalDists, globalIdxs,
                                otherListDists, otherListIdxs,
                                mergedDists, mergedIdxs);
            // Copy merged result back to global arrays.
            for (int j = 0; j < k; j++) {
                globalDists[j] = mergedDists[j];
                globalIdxs[j] = mergedIdxs[j];
            }
        }
        
        // Write the final sorted k neighbors for query q into the output array.
        int outBase = q * k;
        for (int j = 0; j < k; j++) {
            result[outBase + j].first  = globalIdxs[j];
            result[outBase + j].second = globalDists[j];
        }
    }
}

// -----------------------------------------------------------------------------
// Host interface: run_knn
// This function configures and launches the CUDA kernel for k-NN computation.
// It selects an appropriate number of threads per block based on k, aiming
// to keep the shared memory usage (T * k * 8 bytes) below about 48KB.
// k is guaranteed to be a power-of-two between 32 and 1024.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose the number of threads per block, T, based on k.
    // We require: T * k * sizeof(std::pair<int, float>) <= 49152 bytes (48KB).
    // Since sizeof(std::pair<int,float>) is 8 bytes, that means T <= 6144 / k.
    int maxThreads = 6144 / k;
    if (maxThreads < 1)
        maxThreads = 1;
    // Round maxThreads down to the nearest power of two.
    int blockSize = 1;
    while (blockSize * 2 <= maxThreads)
        blockSize *= 2;
    
    // Compute the total shared memory required for the reduction stage.
    size_t sharedMemSize = blockSize * k * sizeof(std::pair<int, float>);
    
    // Launch one block per query.
    dim3 grid(query_count, 1, 1);
    dim3 block(blockSize, 1, 1);
    
    knn_kernel<<<grid, block, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    
    // Synchronize and check for errors.
    cudaDeviceSynchronize();
}