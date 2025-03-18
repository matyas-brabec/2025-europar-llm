#include <cuda_runtime.h>
#include <utility>
#include <float.h>

// -----------------------------------------------------------------------------
// This implementation of k-nearest neighbors (kNN) uses one warp (32 threads)
// per query. Each warp maintains its own sorted list (of length k) of candidate
// nearest neighbors in registers, distributed such that each thread keeps
// k/32 consecutive candidates. A candidate is represented as a pair consisting
// of the index of the data point and its squared Euclidean distance from the
// query point.
// 
// To accelerate distance computation, data points are processed in batches
// (tiles) that are loaded into shared memory. For each data point loaded,
// every warp computes the distance from its query point and, if the distance is
// lower than the current worst (largest) candidate distance (max_distance),
// adds it into a per-warp candidate buffer (in shared memory) using warp-ballot
// instructions. When the candidate buffer has accumulated at least k candidates,
// it is merged with the intermediate result stored in registers using the
// following steps:
//   1. The candidate buffer is swapped into registers.
//   2. The candidate registers are sorted in ascending order using a
//      distributed Bitonic Sort (using warp shuffles).
//   3. The sorted candidate buffer registers and the current intermediate
//      result registers (which are already sorted) are merged by taking, for
//      each global index i (0 <= i < k), the minimum of the intermediate result
//      candidate at position i and the candidate from the candidate buffer at
//      the mirrored position (k - i - 1). This forms a bitonic sequence.
//   4. The merged sequence is then sorted with another round of Bitonic Sort to
//      produce the updated intermediate result.
// After all data batches have been processed (and remaining candidates merged),
// the final sorted k-nearest neighbors for each query are written back to global
// memory in row-major order.
// 
// The kernel uses two shared memory regions per block:
//   1. A tile (batch) of data points (float2) of fixed size TILE_SIZE.
//   2. Per-warp candidate buffers of size k (each warp's candidate buffer is
//      contiguous).
// The block configuration is chosen so that one warp handles one query.
// -----------------------------------------------------------------------------

// Define the tile size for data batches.
#define TILE_SIZE 256

// Candidate structure: stores a data point index and its squared distance.
struct Candidate {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// Device helper function: Performs an in-warp distributed Bitonic Sort on a
// candidate array that is distributed among the 32 threads of a warp.
// total   : total number of elements, i.e. k.
// r       : number of candidates stored per thread (r = k/32).
// The candidate array "reg" is stored in registers, with each thread owning r
// consecutive entries; the global index of an element held by a thread is:
//         global_idx = (lane * r) + local_idx.
// The algorithm uses warp shuffle instructions (__shfl_sync) to exchange values
// with elements held in another thread's registers.
// -----------------------------------------------------------------------------
__device__ inline void bitonicSortWarpDistributed(Candidate reg[], int total, int r) {
    int lane = threadIdx.x & 31;  // lane id in the warp

    // Iterate over bitonic sort stages.
    for (int size = 2; size <= total; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread loops over its r elements.
            for (int i = 0; i < r; i++) {
                int global_idx = lane * r + i;
                int partner = global_idx ^ stride;
                if (partner > global_idx) {
                    // Determine sorting order: ascending if the bit 'size' in global_idx is 0.
                    bool ascending = ((global_idx & size) == 0);
                    int partner_lane = partner / r;
                    int partner_local = partner % r;
                    Candidate partnerVal;
                    // Fetch partner candidate using warp shuffle if in a different lane.
                    if (partner_lane == lane) {
                        partnerVal = reg[partner_local];
                    } else {
                        partnerVal.dist = __shfl_sync(0xffffffff, reg[partner_local].dist, partner_lane);
                        partnerVal.idx  = __shfl_sync(0xffffffff, reg[partner_local].idx, partner_lane);
                    }
                    Candidate myVal = reg[i];
                    // Compare and update if needed (only updating local register).
                    bool cmp = ascending ? (myVal.dist > partnerVal.dist) : (myVal.dist < partnerVal.dist);
                    if (cmp) {
                        reg[i] = partnerVal;
                    }
                }
            }
            __syncwarp(0xffffffff);
        }
    }
}

// -----------------------------------------------------------------------------
// Device helper function: Merges two sorted candidate arrays (each of length k,
// distributed among the 32 threads) into one sorted array stored in the register
// array "inter".
// "inter" holds the current intermediate result (sorted ascending) and "buf"
// holds the sorted candidate buffer that was just swapped into registers.
// The merge is done by pairing element i from inter and element (k - i - 1) from
// buf (accessed in reversed order), taking the minimum of the two at each pair,
// and then sorting the resulting distributed array with Bitonic Sort.
// -----------------------------------------------------------------------------
__device__ inline void mergeBufferAndResult(Candidate inter[], Candidate buf[], int total, int r) {
    int lane = threadIdx.x & 31;
    Candidate merged[32]; // Temporary storage per thread (max r = 32)

    // For each element owned by the current thread:
    for (int i = 0; i < r; i++) {
        int global_idx = lane * r + i;
        int rev_idx = total - 1 - global_idx; // mirrored index
        int src_lane = rev_idx / r;
        int src_local = rev_idx % r;
        Candidate bufRev;
        // Load the mirrored candidate from buf.
        if (src_lane == lane) {
            bufRev = buf[src_local];
        } else {
            bufRev.dist = __shfl_sync(0xffffffff, buf[src_local].dist, src_lane);
            bufRev.idx  = __shfl_sync(0xffffffff, buf[src_local].idx, src_lane);
        }
        // Take the minimum (lower distance) between inter and bufRev.
        Candidate a = inter[i];
        Candidate b = bufRev;
        Candidate m;
        m.dist = (a.dist < b.dist) ? a.dist : b.dist;
        m.idx  = (a.dist < b.dist) ? a.idx  : b.idx;
        merged[i] = m;
    }
    // Sort the merged distributed array.
    bitonicSortWarpDistributed(merged, total, r);
    // Write sorted merged result back into the registers for intermediate result.
    for (int i = 0; i < r; i++) {
        inter[i] = merged[i];
    }
}

// -----------------------------------------------------------------------------
// Global kernel implementing k-NN for 2D points.
// Each warp processes one query point. It loads batches of data points into
// shared memory, computes distances, and uses a per-warp candidate buffer to
// temporarily store new candidates that are closer than the current max distance.
// When the candidate buffer is full (>= k candidates), it is merged with the
// intermediate sorted result in registers using the Bitonic Sort merge scheme.
// After processing, the sorted k nearest neighbors are written to global memory.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count, const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp (32 threads) processes one query.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x & 31;
    int warpId_in_block = threadIdx.x >> 5; // threadIdx.x / 32
    int global_warp_id = (blockIdx.x * (blockDim.x >> 5)) + warpId_in_block;
    if (global_warp_id >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[global_warp_id];

    // Each warp's registers hold k candidates, distributed: r = k/32 per thread.
    int r = k >> 5;  // k is guaranteed to be a power of two between 32 and 1024.
    Candidate reg[32]; // local register buffer (max r = 32)
    for (int i = 0; i < r; i++) {
        reg[i].idx = -1;
        reg[i].dist = FLT_MAX;
    }

    // Declare shared memory:
    // The first part is used for a tile of data points.
    // The second part allocates per-warp candidate buffers.
    extern __shared__ char shared_mem[];
    float2 *shData = (float2*) shared_mem;
    // Number of warps per block.
    int warpsPerBlock = blockDim.x >> 5;
    // Candidate buffer base pointer (each warp gets k Candidate slots).
    Candidate *shBuffer = (Candidate*)(shData + TILE_SIZE);
    // Pointer to this warp's candidate buffer.
    Candidate *warpBuffer = shBuffer + (warpId_in_block * k);

    // Candidate count for this warp (maintained uniformly via warp-synchronous execution).
    int candidate_count = 0;

    // Process data points in batches (tiles) loaded into shared memory.
    for (int batch = 0; batch < data_count; batch += TILE_SIZE) {
        // Determine how many data points are in this tile.
        int tile_count = (batch + TILE_SIZE <= data_count) ? TILE_SIZE : (data_count - batch);
        // Cooperative loading of the tile from global memory.
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            if (i < tile_count) {
                shData[i] = data[batch + i];
            }
        }
        __syncthreads(); // Ensure the tile is fully loaded.

        // Each warp processes the tile using its 32 lanes.
        for (int j = lane; j < tile_count; j += 32) {
            int data_index = batch + j;
            float2 pt = shData[j];
            float dx = pt.x - q.x;
            float dy = pt.y - q.y;
            float dist = dx * dx + dy * dy;

            // Get current worst candidate's distance (max_distance) from the intermediate result.
            // The worst candidate is at global index k-1.
            int owner = (k - 1) / r;        // owner lane for the (k-1)th element.
            int local_idx = (k - 1) % r;
            float max_distance = __shfl_sync(0xffffffff, reg[local_idx].dist, owner);

            bool qualify = (dist < max_distance);
            // Use warp ballot to collect which lanes have a qualifying candidate.
            unsigned int ballot = __ballot_sync(0xffffffff, qualify);
            if (qualify) {
                // Compute the insertion offset for this lane among those that qualify.
                int offset = __popc(ballot & ((1u << lane) - 1));
                int pos = candidate_count + offset;
                // Write the candidate into the per-warp candidate buffer.
                warpBuffer[pos].idx = data_index;
                warpBuffer[pos].dist = dist;
            }
            // Update candidate_count for the warp.
            candidate_count += __popc(ballot);

            // If candidate buffer is full, merge it with the intermediate result.
            if (candidate_count >= k) {
                // Fill any remaining slots with dummy entries.
                for (int fill = candidate_count; fill < k; fill++) {
                    if (lane == 0) {
                        warpBuffer[fill].idx = -1;
                        warpBuffer[fill].dist = FLT_MAX;
                    }
                }
                __syncwarp(0xffffffff);
                // Merge the candidate buffer (now in shBuffer) with the intermediate result.
                // The merge is executed by:
                //   1. Sorting the candidate buffer (swapped into registers later).
                //   2. Merging it with the sorted intermediate result stored in reg.
                //   3. Sorting the merged results.
                mergeBufferAndResult(reg, warpBuffer, k, r);
                candidate_count = 0;
            }
        }
        __syncthreads(); // Ensure all threads have finished processing the tile.
    }

    // After processing all batches, merge any remaining candidates.
    if (candidate_count > 0) {
        for (int fill = candidate_count; fill < k; fill++) {
            if (lane == 0) {
                warpBuffer[fill].idx = -1;
                warpBuffer[fill].dist = FLT_MAX;
            }
        }
        __syncwarp(0xffffffff);
        mergeBufferAndResult(reg, warpBuffer, k, r);
        candidate_count = 0;
    }

    // Write the final sorted k nearest neighbors to global memory.
    // Each thread writes its k/32 consecutive neighbors.
    int query_idx = global_warp_id;
    for (int i = 0; i < r; i++) {
        int global_pos = query_idx * k + (lane * r + i);
        result[global_pos] = std::pair<int, float>(reg[i].idx, reg[i].dist);
    }
}

// -----------------------------------------------------------------------------
// Host C++ interface for k-NN.
// This function launches the CUDA kernel using an appropriate block configuration
// and shared memory size.
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose a block size of 128 threads (i.e., 4 warps per block).
    int threadsPerBlock = 128;
    int warpsPerBlock = threadsPerBlock >> 5; // 128/32 = 4 warps per block.
    // Each warp handles one query, so the number of warps needed equals query_count.
    int numBlocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Compute shared memory requirements per block:
    //  1. Data tile: TILE_SIZE float2 elements.
    //  2. Candidate buffers: one per warp, each of size k.
    size_t shDataBytes  = TILE_SIZE * sizeof(float2);
    size_t shBufferBytes = warpsPerBlock * k * sizeof(Candidate);
    size_t sharedMemSize = shDataBytes + shBufferBytes;

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}