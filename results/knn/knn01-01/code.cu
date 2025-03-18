// This file implements an optimized CUDA kernel for 2D k‐nearest neighbors (k-NN) search.
// Each query point is processed cooperatively by one block of threads. The block loads
// tiles of the global data into shared memory (to improve DRAM bandwidth utilization)
// and the threads in the block partition the work of scanning the data tile for distances.
// Each thread accumulates a small “local” candidate list (of size L = ceil(k/blockDim.x)) in
// registers, sorted in ascending order (lowest distance first). After processing all data,
// the candidate lists from all threads are gathered into shared memory and one thread merges
// them and sorts the final k candidates into ascending order (i.e. k-th nearest neighbor order).
//
// Assumptions:
//   - 'query', 'data', and 'result' arrays are allocated with cudaMalloc.
//   - data_count >= k, and k is a power-of-two between 32 and 1024 (inclusive).
//   - The target hardware is a modern NVIDIA GPU (H100/A100) with the latest CUDA toolkit.
//
// Hyper-parameters:
//   - Each query is processed by one block with THREADS_PER_BLOCK threads (set to 256).
//   - The data array is processed in tiles of TILE_SIZE (256) elements.
//   - Each thread's candidate list size is L = ceil(k / THREADS_PER_BLOCK). For k smaller than
//     THREADS_PER_BLOCK, L becomes 1.
//
// NOTE: This kernel does not allocate any additional device memory besides shared memory.
//
#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// Structure to hold candidate neighbor information.
struct Candidate {
    float dist; // Squared Euclidean distance
    int idx;    // Index of the data point
};

// The CUDA kernel for k-NN.
// Each block processes one query point.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each block is assigned one query:
    int qid = blockIdx.x;
    if (qid >= query_count) return;
    
    // Load the query point.
    float2 q = query[qid];
    
    // Determine the candidate list size per thread.
    // L = ceil(k / blockDim.x). This code assumes blockDim.x is chosen such that L <= 32.
    int L = (k + blockDim.x - 1) / blockDim.x;
    const int MAX_L = 32;  // Maximum allowed L (should cover worst-case with THREADS_PER_BLOCK=256 and k<=1024)
    if (L > MAX_L) L = MAX_L;
    
    // Each thread will maintain a local list of best candidates (sorted in ascending order).
    // cand[i] stores the distance and candIdx[i] stores the corresponding data index.
    float cand[MAX_L];
    int candIdx[MAX_L];
    // Initialize the candidate list with large values.
#pragma unroll
    for (int i = 0; i < L; i++) {
        cand[i] = FLT_MAX;
        candIdx[i] = -1;
    }
    
    // Shared memory layout:
    // [0, cand_region_size)    : Candidate merging array (one Candidate per thread per local candidate)
    // [cand_region_size,  ... )  : Data tile shared memory (array of float2)
    //
    // Compute the region size (in bytes) needed for the candidates.
    int cand_region_size = blockDim.x * L * sizeof(Candidate);
    // The tile region will be placed immediately after the candidate region.
    // Define TILE_SIZE for data tiling.
    const int TILE_SIZE = 256;
    
    // Obtain the pointer to the entire shared memory block passed from the host.
    extern __shared__ char smem[];
    // Candidate merge region:
    Candidate *sCandidates = reinterpret_cast<Candidate*>(smem);
    // Data tile region starts at offset cand_region_size.
    float2 *tile_points = reinterpret_cast<float2*>(smem + cand_region_size);
    
    // Loop over all data points in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        int tile_elems = TILE_SIZE;
        if (tile_start + TILE_SIZE > data_count)
            tile_elems = data_count - tile_start;
        
        // Cooperative load: threads in the block load the current tile from global memory
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            tile_points[i] = data[tile_start + i];
        }
        __syncthreads();  // Ensure the tile is fully loaded.
        
        // Each thread processes a subset of the tile.
        for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            float2 dpt = tile_points[i];
            // Compute squared Euclidean distance between query and data point.
            float dx = q.x - dpt.x;
            float dy = q.y - dpt.y;
            float dist = dx * dx + dy * dy;
            int d_index = tile_start + i;
            
            // If this candidate is better than the worst candidate in the local list,
            // insert it into the sorted candidate array.
            if (dist < cand[L - 1]) {
                // Insertion sort: since cand[0..L-1] is maintained in ascending order,
                // cand[L-1] holds the largest (worst) value.
                int pos = L - 1;
                // Shift larger values to the right.
                while (pos > 0 && dist < cand[pos - 1]) {
                    cand[pos] = cand[pos - 1];
                    candIdx[pos] = candIdx[pos - 1];
                    pos--;
                }
                cand[pos] = dist;
                candIdx[pos] = d_index;
            }
        }
        __syncthreads();  // Ensure all threads finished processing this tile.
    }
    
    // Each thread writes its local candidate list to the candidate merge region in shared memory.
    for (int i = 0; i < L; i++) {
        sCandidates[threadIdx.x * L + i].dist = cand[i];
        sCandidates[threadIdx.x * L + i].idx  = candIdx[i];
    }
    __syncthreads();
    
    // Now, one thread (thread 0) in the block merges the candidate lists from all threads.
    if (threadIdx.x == 0) {
        int totalCandidates = blockDim.x * L;
        // Temporary array to hold the merged candidates.
        // The maximum possible total candidates is k (or slightly more, but k<=1024 by assumption).
        Candidate tmp[1024];
        for (int i = 0; i < totalCandidates; i++) {
            tmp[i] = sCandidates[i];
        }
        // Perform a simple insertion sort on tmp to sort all candidate entries in ascending order by distance.
        for (int i = 1; i < totalCandidates; i++) {
            Candidate key = tmp[i];
            int j = i - 1;
            while (j >= 0 && tmp[j].dist > key.dist) {
                tmp[j + 1] = tmp[j];
                j--;
            }
            tmp[j + 1] = key;
        }
        // Write the best k candidates into the result array.
        int outBase = qid * k;
        for (int i = 0; i < k; i++) {
            result[outBase + i].first  = tmp[i].idx;
            result[outBase + i].second = tmp[i].dist;
        }
    }
}

// Host function interface for k-NN
// This function launches the k-NN kernel.
// It configures each block to process one query point.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose a fixed number of threads per block.
    const int THREADS_PER_BLOCK = 256;
    // One block per query.
    int numBlocks = query_count;
    
    // Compute candidate list size per thread: L = ceil(k / THREADS_PER_BLOCK).
    int L = (k + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // For safety, restrict L to maximum 32 (as assumed in the kernel).
    if (L > 32) L = 32;
    
    // Shared memory: we need space for:
    //   - Candidate merge region: THREADS_PER_BLOCK * L * sizeof(Candidate)
    //   - Data tile: TILE_SIZE * sizeof(float2), where TILE_SIZE is set to 256.
    const int TILE_SIZE = 256;
    size_t sharedMemSize = THREADS_PER_BLOCK * L * sizeof(Candidate) + TILE_SIZE * sizeof(float2);
    
    // Launch the kernel.
    knn_kernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(query, query_count,
                                                                  data, data_count,
                                                                  result, k);
    // It is assumed that error checking and synchronization is performed by the caller if necessary.
}