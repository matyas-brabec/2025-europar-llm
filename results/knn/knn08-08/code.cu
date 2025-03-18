#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <utility>

// -----------------------------------------------------------------------------
// This implementation computes the k-nearest neighbors (using squared Euclidean
// distance) for 2D points. Each query is processed by a single warp (32 threads)
// that cooperatively maintains an intermediate sorted list of k candidates (distributed
// among the warp registers) and a candidate buffer in shared memory. Data points are
// processed in batches that are first loaded into shared memory. When the candidate
// buffer becomes full (i.e. holds k candidates), it is merged with the intermediate
// result using a two‐phase Bitonic Sort merging procedure (as described in the design).
//
// Notes on data layout within each warp:
// - The k candidates are distributed over 32 threads. Each thread stores k/32
//   consecutive candidates (in registers). The global linear index in the candidate
//   list is: global_idx = (lane id)*localCount + localIndex (with localCount = k/32).
// - The intermediate candidate list is kept sorted in ascending order (by distance).
// - The candidate buffer for each warp is allocated from shared memory (each warp
//   uses its own contiguous region).
//
// The Bitonic Sort algorithm below is implemented as an approximate distributed
// version using warp shuffle instructions. When two candidates are in different lanes
// (always at the same register offset) the value is exchanged using __shfl_sync; when
// local (in the same thread) a simple swap is performed.
// -----------------------------------------------------------------------------

// Structure for candidate data (index and squared distance).
struct Candidate {
    int index;
    float dist;
};

// Swap two Candidate values.
__device__ inline void swapCandidate(Candidate &a, Candidate &b) {
    Candidate tmp = a;
    a = b;
    b = tmp;
}

// -----------------------------------------------------------------------------
// This device function implements a warp-level Bitonic Sort on a list of k 
// Candidate elements that is distributed among the 32 warp lanes.
// Each thread holds localCount = k/32 candidates in the array "localCandidates".
// Sorting is done in ascending order (by candidate.distance).
//
// The algorithm loops over all stages as in the serial pseudocode, and for each
// candidate (indexed by its “global index”) remote exchanges are done using __shfl_sync.
// Note: This implementation is simplified and intended for moderate values of k 
// (k between 32 and 1024).
// -----------------------------------------------------------------------------
__device__ void bitonic_sort_warp(Candidate localCandidates[], int k) {
    const int warpSize = 32;
    int localCount = k / warpSize; // Number of candidates per thread.
    unsigned mask = 0xffffffff;
    // Loop over bitonic sort stages.
    for (int size = 2; size <= k; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            // Each thread goes over its local candidates.
            for (int r = 0; r < localCount; r++) {
                // Compute global index of this candidate.
                int lane = threadIdx.x % warpSize;
                int globalIdx = lane * localCount + r;
                int partnerIdx = globalIdx ^ stride;
                if (partnerIdx > globalIdx) {
                    // The sort direction is "up" if globalIdx's bit corresponding to 'size'
                    // is not set.
                    bool ascending = ((globalIdx & size) == 0);
                    Candidate myVal = localCandidates[r];
                    Candidate partnerVal;
                    int partnerLane = partnerIdx / localCount;
                    int partnerLocalIdx = partnerIdx % localCount;
                    if (partnerLane == lane) {
                        // The partner candidate is stored in the same thread.
                        partnerVal = localCandidates[partnerLocalIdx];
                    } else {
                        // Exchange the candidate from the partner lane. The exchanged value is
                        // always at the same register offset (partnerLocalIdx) in that lane.
                        partnerVal.index = __shfl_sync(mask, localCandidates[partnerLocalIdx].index, partnerLane);
                        partnerVal.dist  = __shfl_sync(mask, localCandidates[partnerLocalIdx].dist, partnerLane);
                    }
                    bool needSwap = ascending ? (myVal.dist > partnerVal.dist) : (myVal.dist < partnerVal.dist);
                    if (needSwap) {
                        if (partnerLane == lane) {
                            swapCandidate(localCandidates[r], localCandidates[partnerLocalIdx]);
                        }
                        else {
                            // For remote exchange, we simply update our own register.
                            // (The partner lane will perform the complementary update.)
                            localCandidates[r] = partnerVal;
                        }
                    }
                }
            }
            __syncwarp();
        }
    }
}

// -----------------------------------------------------------------------------
// Simple insertion sort on a thread's local candidate array (of size "count").
// Used to sort the candidates stored in registers (localCount elements per thread).
// Since count is small (k/32 <= 32), insertion sort is efficient.
// -----------------------------------------------------------------------------
__device__ void local_sort(Candidate localCandidates[], int count) {
    for (int i = 1; i < count; i++) {
        Candidate key = localCandidates[i];
        int j = i - 1;
        while (j >= 0 && localCandidates[j].dist > key.dist) {
            localCandidates[j + 1] = localCandidates[j];
            j--;
        }
        localCandidates[j + 1] = key;
    }
}

// -----------------------------------------------------------------------------
// Merge the intermediate result (stored in warp registers) and the candidate 
// buffer (also loaded into registers) into a new sorted candidate list in the 
// registers. The merge is performed as follows:
//  1. For each global index i (distributed among warp lanes), compute:
//         merged[i] = min( candidateBuffer[i], intermediate[k - i - 1] )
//     (min is chosen by comparing the distance).
//  2. The resulting merged list is a bitonic sequence which is then sorted using
//     the Bitonic Sort algorithm. The final sorted merged result is written back 
//     into the registers given by "intermediate".
// -----------------------------------------------------------------------------
__device__ void merge_registers(Candidate intermediate[], Candidate buffer[], int k) {
    const int warpSize = 32;
    int localCount = k / warpSize;
    unsigned mask = 0xffffffff;
    // For each candidate in the register array (global index computed from lane and local index)
    for (int r = 0; r < localCount; r++) {
        int lane = threadIdx.x % warpSize;
        int globalIdx = lane * localCount + r;
        // Compute the partner index from the opposite end.
        int partnerGlobal = k - globalIdx - 1;
        int partnerLane = partnerGlobal / localCount;
        int partnerLocalIdx = partnerGlobal % localCount;
        Candidate interVal;
        if (partnerLane == lane) {
            interVal = intermediate[partnerLocalIdx];
        } else {
            interVal.index = __shfl_sync(mask, intermediate[partnerLocalIdx].index, partnerLane);
            interVal.dist  = __shfl_sync(mask, intermediate[partnerLocalIdx].dist, partnerLane);
        }
        Candidate myBuff = buffer[r];
        Candidate merged = (myBuff.dist <= interVal.dist) ? myBuff : interVal;
        buffer[r] = merged;
    }
    __syncwarp();
    // Sort the merged result stored in buffer.
    bitonic_sort_warp(buffer, k);
    __syncwarp();
    // Copy sorted results back into the intermediate registers.
    for (int r = 0; r < localCount; r++) {
        intermediate[r] = buffer[r];
    }
    __syncwarp();
}

// -----------------------------------------------------------------------------
// Kernel for k-NN computation.
// Each warp processes one query point. The full procedure is:
//  1. Initialize an intermediate candidate result (of size k, distributed among warp registers)
//     with all distances set to FLT_MAX.
//  2. Process the data points in batches which are first loaded into shared memory.
//  3. For each data point in the batch, compute the squared Euclidean distance to the query.
//     If the distance is lower than the current worst candidate (max_distance), add the candidate
//     to the candidate buffer (stored in shared memory) using atomic updates (per warp).
//  4. When the candidate buffer becomes full (i.e. contains >= k candidates), merge it with the
//     intermediate result using the Bitonic Sort merge procedure described above.
//  5. After all batches are processed, merge any remaining candidates from the candidate buffer.
//  6. Write the final sorted list of k nearest neighbors (each as a pair of index and distance)
//     to the output result array.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k, int batch_size) {
    // Define warp parameters.
    const int warpSize = 32;
    int lane = threadIdx.x % warpSize;
    int warpIdInBlock = threadIdx.x / warpSize;
    int globalWarpId = (blockIdx.x * (blockDim.x / warpSize)) + warpIdInBlock;
    if (globalWarpId >= query_count)
        return; // Only process queries that exist.
        
    // Shared memory layout:
    // 1) Batch data: array of "batch_size" float2 elements.
    // 2) Candidate buffers: one per warp; each buffer has k Candidate elements.
    // 3) Candidate counts: one int per warp.
    extern __shared__ char shared_mem[];
    float2* s_data = (float2*)shared_mem;  // Batch data buffer.
    // Candidate buffer starts after s_data.
    Candidate* s_candidateBuffer = (Candidate*)(s_data + batch_size);
    // Candidate count array starts after candidate buffers.
    int warpsPerBlock = blockDim.x / warpSize;
    int* s_candidateCount = (int*)(s_candidateBuffer + warpsPerBlock * k);
    
    // Each warp uses its own candidate buffer.
    Candidate* warpCandidateBuffer = s_candidateBuffer + warpIdInBlock * k;
    int* warpCandidateCount = s_candidateCount + warpIdInBlock;
    
    if (lane == 0)
        *warpCandidateCount = 0;
    __syncwarp();
    
    // Each thread prepares its local portion of the intermediate result.
    int localCount = k / warpSize;  // Number of candidates per thread.
    Candidate intermediate[32];  // Maximum localCount is at most 32 (for k up to 1024).
    for (int i = 0; i < localCount; i++) {
        intermediate[i].index = -1;
        intermediate[i].dist = FLT_MAX;
    }
    __syncwarp();
    
    // Load the query point and broadcast it within the warp.
    float2 q = query[globalWarpId];
    q.x = __shfl_sync(0xffffffff, q.x, 0);
    q.y = __shfl_sync(0xffffffff, q.y, 0);
    
    // Retrieve the current worst (maximum) distance from the intermediate result.
    float max_distance = __shfl_sync(0xffffffff, intermediate[localCount - 1].dist, warpSize - 1);
    
    // Process the data points in batches.
    for (int batchStart = 0; batchStart < data_count; batchStart += batch_size) {
        int batchLength = (data_count - batchStart < batch_size) ? (data_count - batchStart) : batch_size;
        // Load batch of data points into shared memory by all threads in the block.
        for (int i = threadIdx.x; i < batchLength; i += blockDim.x) {
            s_data[i] = data[batchStart + i];
        }
        __syncthreads();
        
        // Each warp processes the batch (each lane handles different indices).
        for (int i = lane; i < batchLength; i += warpSize) {
            float2 d = s_data[i];
            float dx = d.x - q.x;
            float dy = d.y - q.y;
            float dist = dx * dx + dy * dy;
            // If the candidate is promising, add it to the warp's candidate buffer.
            if (dist < max_distance) {
                // Atomically get an insertion position in the candidate buffer.
                int pos = atomicAdd(warpCandidateCount, 1);
                if (pos < k) {
                    warpCandidateBuffer[pos].index = batchStart + i;
                    warpCandidateBuffer[pos].dist = dist;
                }
            }
        }
        __syncwarp();
        
        // If the candidate buffer is full, merge it.
        if (*warpCandidateCount >= k) {
            // Load candidate buffer into registers.
            Candidate buffer[32];  // Each thread gets its portion.
            for (int i = 0; i < localCount; i++) {
                int globalIdx = (threadIdx.x % warpSize) * localCount + i;
                if (globalIdx < *warpCandidateCount)
                    buffer[i] = warpCandidateBuffer[globalIdx];
                else {
                    buffer[i].index = -1;
                    buffer[i].dist  = FLT_MAX;
                }
            }
            __syncwarp();
            // Swap intermediate result registers and buffer registers.
            for (int i = 0; i < localCount; i++) {
                Candidate temp = intermediate[i];
                intermediate[i] = buffer[i];
                buffer[i] = temp;
            }
            __syncwarp();
            // Sort the candidate buffer (now in registers) using Bitonic Sort.
            bitonic_sort_warp(buffer, k);
            __syncwarp();
            // Merge the sorted buffer with the intermediate result.
            merge_registers(intermediate, buffer, k);
            __syncwarp();
            // Update max_distance from the updated intermediate result.
            max_distance = __shfl_sync(0xffffffff, intermediate[localCount - 1].dist, warpSize - 1);
            // Reset candidate buffer.
            if (lane == 0)
                *warpCandidateCount = 0;
            __syncwarp();
        }
        __syncthreads();
    }
    
    // After the last batch, merge any remaining candidates from the candidate buffer.
    if (*warpCandidateCount > 0) {
        Candidate buffer[32];
        for (int i = 0; i < localCount; i++) {
            int globalIdx = (threadIdx.x % warpSize) * localCount + i;
            if (globalIdx < *warpCandidateCount)
                buffer[i] = warpCandidateBuffer[globalIdx];
            else {
                buffer[i].index = -1;
                buffer[i].dist  = FLT_MAX;
            }
        }
        __syncwarp();
        // Swap intermediate registers and buffer.
        for (int i = 0; i < localCount; i++) {
            Candidate temp = intermediate[i];
            intermediate[i] = buffer[i];
            buffer[i] = temp;
        }
        __syncwarp();
        // Sort and merge.
        bitonic_sort_warp(buffer, k);
        __syncwarp();
        merge_registers(intermediate, buffer, k);
        __syncwarp();
        max_distance = __shfl_sync(0xffffffff, intermediate[localCount - 1].dist, warpSize - 1);
    }
    
    // Write the final k nearest neighbors to global memory.
    // The final sorted list (of k elements) is distributed among warp registers.
    int queryIdx = globalWarpId;
    for (int i = 0; i < localCount; i++) {
        int globalIdx = queryIdx * k + ((threadIdx.x % warpSize) * localCount + i);
        result[globalIdx] = std::make_pair(intermediate[i].index, intermediate[i].dist);
    }
}

// -----------------------------------------------------------------------------
// Host interface function. All pointers (query, data, result) are assumed to be 
// allocated on the device (via cudaMalloc). 'data_count' is always >= k and k is a 
// power of two between 32 and 1024.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Use 128 threads per block.
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock / 32;
    // Each warp processes one query so we need (query_count) warps.
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;
    
    // Choose a batch size for loading data points into shared memory.
    // This value can be tuned; here we use 1024.
    int batch_size = 1024;
    
    // Compute shared memory size per block:
    // s_data: batch_size * sizeof(float2)
    size_t shmem_data = batch_size * sizeof(float2);
    // Candidate buffers: one per warp, each of size k * sizeof(Candidate)
    size_t shmem_candidates = warpsPerBlock * k * sizeof(Candidate);
    // Candidate count array: one int per warp.
    size_t shmem_counts = warpsPerBlock * sizeof(int);
    size_t shmem_total = shmem_data + shmem_candidates + shmem_counts;
    
    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, shmem_total>>>(query, query_count, data, data_count, result, k, batch_size);
    cudaDeviceSynchronize();
}