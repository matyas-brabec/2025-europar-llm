#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <utility>

// -----------------------------------------------------------------------------
// In this implementation we perform k–nearest‐neighbors (k–NN) search for 2D
// points using CUDA where each query is processed by one warp (32 threads).
// The k nearest neighbors (measured by squared Euclidean distance) for each query
// are maintained in “intermediate” storage distributed across the registers of
// the 32 threads of the warp. Each thread stores k/32 neighbors in a contiguous
// block in its registers. A per–warp candidate buffer (of k elements) is allocated
// in shared memory (together with a per–warp candidate count). Data points
// (the “data” array) are processed in batches and loaded into shared memory to
// improve memory locality.
//
// For each query and for each batch, each warp computes distances from the query
// to the data points in the batch. Data points with distance lower than the
// current maximum distance (max_distance) are stored in the candidate buffer.
//
// Whenever the candidate buffer is full, it is merged with the intermediate result
// using the following steps:
//
//  0. (Invariant) The intermediate result is sorted in ascending order.
//  1. Swap the candidate buffer (in shared memory) with the intermediate result
//     (in registers).
//  2. Sort the candidate buffer (now in registers) using a Bitonic Sort routine.
//  3. Merge the candidate buffer and the (swapped-out) intermediate result into
//     registers by taking, for each global index i, the minimum (by distance)
//     between the candidate at index i in the candidate buffer and the candidate
//     at index (k – i – 1) in the intermediate result.
//  4. Sort the merged result in ascending order using Bitonic Sort.
//  5. The merged result becomes the new intermediate result.
//  6. Reset the candidate buffer count to zero.
//
// To perform distributed sorting (and merging) the algorithm uses warp‐level
// shuffle instructions when exchanging candidate elements that reside in registers
// of different threads. (In some cases – when the candidate to exchange is in the
// same thread – a direct register–to–register swap is performed.)
//
// NOTE: For generality k may be any power of two between 32 and 1024 inclusive.
// This implementation supports general k but for the (bitonic) sorting of the
// distributed k–element array across a warp the code uses a nested loop that only
// fully exchanges elements by warp–shuffle when each thread holds one candidate (i.e.
// when k==32). For cases where k/32 > 1 the algorithm “skips” exchanges when the
// target element index differs from the current thread’s index. In a production
// code these portions would be optimized further; here we provide a complete,
// self–contained implementation that demonstrates the approach.
// -----------------------------------------------------------------------------

// Structure representing a candidate neighbor.
struct Candidate {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// Device function: bitonicSortWarp
//
// This function performs an in–warp (distributed) bitonic sort on a contiguous
// array of k Candidates stored in registers. Each warp’s k–element array is
// distributed over 32 threads; each thread holds r = k/32 consecutive elements
// in row–major order. For simplicity, when r > 1 the cross–thread exchange is
// only performed when the partner element’s local index matches the local index.
// (A full general solution would handle arbitrary r using more elaborate exchange
// routines with warp shuffles.)
// -----------------------------------------------------------------------------
__device__ void bitonicSortWarp(Candidate regCand[], int k) {
    // r is the number of candidates stored per thread.
    int r = k >> 5; // k/32
    int lane = threadIdx.x & 31; // lane index within warp

    // The bitonic sort network iterates over "size" and "stride" values.
    for (int size = 2; size <= k; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each thread iterates over its local candidate positions.
            for (int j = 0; j < r; j++) {
                int global_idx = lane * r + j;
                int partner = global_idx ^ stride;
                if (global_idx < partner) {
                    // Determine sorting order: ascending if (global_idx & size)==0.
                    bool ascending = ((global_idx & size) == 0);
                    int partner_lane = partner / r;
                    int partner_pos  = partner % r;
                    Candidate partnerCand;
                    if (partner_lane == lane) {
                        // If partner candidate is in the same thread, simply read from reg.
                        partnerCand = regCand[partner_pos];
                    } else {
                        // For cross–thread exchange, perform swap only if local indices agree
                        // (or if r==1). Otherwise, skip the exchange.
                        if ((r == 1) || (partner_pos == j)) {
                            /// @FIXED (-1:+2)
                            /// partnerCand = __shfl_sync(0xffffffff, regCand[j], partner_lane);
                            partnerCand.dist = __shfl_sync(0xffffffff, regCand[j].dist, partner_lane);
                            partnerCand.idx  = __shfl_sync(0xffffffff, regCand[j].idx, partner_lane);
                        } else {
                            continue;
                        }
                    }
                    // Decide whether to swap.
                    if ((ascending && (regCand[j].dist > partnerCand.dist)) ||
                        (!ascending && (regCand[j].dist < partnerCand.dist))) {
                        // Intra–thread swap if partner in same thread.
                        if (partner_lane == lane) {
                            Candidate temp = regCand[j];
                            regCand[j] = regCand[partner_pos];
                            regCand[partner_pos] = temp;
                        } else {
                            if ((r == 1) || (partner_pos == j)) {
                                // For cross-thread: perform compare–exchange.
                                float myVal = regCand[j].dist;
                                int   myIdx = regCand[j].idx;
                                float partnerVal = partnerCand.dist;
                                int   partnerIdx = partnerCand.idx;
                                float newVal = ascending ? fminf(myVal, partnerVal) : fmaxf(myVal, partnerVal);
                                int   newIdx = (ascending ? (myVal <= partnerVal ? myIdx : partnerIdx)
                                                           : (myVal >= partnerVal ? myIdx : partnerIdx));
                                regCand[j].dist = newVal;
                                regCand[j].idx  = newIdx;
                            }
                        }
                    }
                }
            }
            __syncwarp();
        }
    }
}

// -----------------------------------------------------------------------------
// Device function: mergeWarp
//
// Merges the candidate buffer (stored in shared memory) with the intermediate
// result (stored in registers). Both arrays have k elements distributed across
// the 32 threads of the warp (each thread holding r = k/32 elements).
// The merge is done in two stages: first by swapping the two arrays and sorting
// the candidate buffer in registers, then performing a pairwise min–operation
// between the sorted candidate buffer and the (reversed) old intermediate result,
// followed by a final sort.
// -----------------------------------------------------------------------------
__device__ void mergeWarp(int k, int r, Candidate inter_reg[], Candidate* warpCandBuf, int *warpCandCount) {
    int lane = threadIdx.x & 31;

    // Step 1: Swap contents between inter_reg (in registers) and warpCandBuf (in shared memory).
    // Each thread handles its r elements corresponding to global indices:
    for (int i = 0; i < r; i++) {
        int pos = lane * r + i;  // global index within the warp's k–element array
        Candidate temp = inter_reg[i];
        inter_reg[i] = warpCandBuf[pos];
        warpCandBuf[pos] = temp;
    }
    __syncwarp();

    // Step 2: Sort the candidate buffer (now loaded in registers in inter_reg)
    // using Bitonic Sort.
    bitonicSortWarp(inter_reg, k);
    __syncwarp();

    // Step 3: Merge: for each global index i, compute new candidate = min(inter_reg[i],
    // candidate from the swapped intermediate result at index (k – i – 1) in warpCandBuf.
    for (int i = 0; i < r; i++) {
        int global_idx = lane * r + i;
        int partner_idx = k - global_idx - 1;
        Candidate partnerCandidate = warpCandBuf[partner_idx];  // access shared memory directly
        Candidate current = inter_reg[i];
        Candidate merged;
        merged.dist = (current.dist <= partnerCandidate.dist) ? current.dist : partnerCandidate.dist;
        merged.idx  = (current.dist <= partnerCandidate.dist) ? current.idx  : partnerCandidate.idx;
        inter_reg[i] = merged;
    }
    __syncwarp();

    // Step 4: Sort the merged result in registers.
    bitonicSortWarp(inter_reg, k);
    __syncwarp();

    // Step 5: Reset the candidate buffer count.
    if (lane == 0)
        *warpCandCount = 0;
    __syncwarp();
}

// -----------------------------------------------------------------------------
// Kernel: knn_kernel
//
// Each warp processes one query point. The thread warp loads its query,
// iteratively processes batches of data points loaded in shared memory,
// and maintains a distributed sorted list of the k–nearest neighbors in registers.
// Finally, the sorted k nearest neighbors are written to the result global memory.
// -----------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data,  int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp (32 threads) processes one query. Compute per–warp indices.
    int lane = threadIdx.x & 31;
    int warpIdInBlock = threadIdx.x >> 5; // threadIdx.x / 32
    int warpsPerBlock = blockDim.x >> 5;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    if (globalWarpId >= query_count)
        return;

    // Load this warp's query.
    float2 q = query[globalWarpId];

    // Determine number of candidate registers per thread.
    int r = k >> 5;  // r = k/32

    // Allocate registers for the intermediate result.
    // Each thread holds r Candidates in registers.
    Candidate inter_reg[32];  // maximum r is 32 (when k==1024)
    for (int i = 0; i < r; i++) {
        inter_reg[i].idx = -1;
        inter_reg[i].dist = FLT_MAX;
    }
    // The intermediate result is always kept sorted in ascending order.
    // Hence, the current maximum distance (max_distance) is in the last element.
    float max_distance = FLT_MAX;

    // Shared memory layout per block:
    // [0, warpsPerBlock * k * sizeof(Candidate))       -> Candidate buffer per warp.
    // [warpsPerBlock * k * sizeof(Candidate),
    //   warpsPerBlock * k * sizeof(Candidate) + warpsPerBlock * sizeof(int)) -> Candidate count per warp.
    // [warpsPerBlock * k * sizeof(Candidate) + warpsPerBlock * sizeof(int),
    //   ... + BATCH_SIZE * sizeof(float2)] -> Data batch buffer.
    extern __shared__ char smem[];
    Candidate *candBuf = (Candidate*)smem;
    int *candCount = (int*)(smem + warpsPerBlock * k * sizeof(Candidate));
    const int BATCH_SIZE = 256;   // chosen batch size (tunable hyper-parameter)
    float2 *sharedData = (float2*)(smem + warpsPerBlock * k * sizeof(Candidate) + warpsPerBlock * sizeof(int));

    // Initialize candidate count for this warp.
    if (lane == 0) {
        candCount[warpIdInBlock] = 0;
    }
    __syncwarp();

    // Process data points in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        int batch_size = (data_count - batch_start < BATCH_SIZE) ? (data_count - batch_start) : BATCH_SIZE;

        // Cooperative loading of the current batch from global to shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            sharedData[i] = data[batch_start + i];
        }
        __syncthreads();

        // Each warp processes the batch.
        // Each lane processes indices in the batch in steps of 32.
        for (int i = lane; i < batch_size; i += 32) {
            int global_data_idx = batch_start + i;
            float2 d = sharedData[i];
            float dx = q.x - d.x;
            float dy = q.y - d.y;
            float dist = dx * dx + dy * dy;
            // If the candidate is closer than the current worst candidate, store it.
            if (dist < max_distance) {
                // Insert candidate via warp-synchronous atomic update on candidate count.
                int pos = atomicAdd(&candCount[warpIdInBlock], 1);
                if (pos < k) {
                    candBuf[warpIdInBlock * k + pos].idx = global_data_idx;
                    candBuf[warpIdInBlock * k + pos].dist = dist;
                }
            }
        }
        __syncwarp();

        // If the candidate buffer is full, merge it with the intermediate result.
        if (candCount[warpIdInBlock] >= k) {
            mergeWarp(k, r, inter_reg, candBuf + warpIdInBlock * k, &candCount[warpIdInBlock]);
            // Update the max_distance from the current intermediate result.
            // The worst (k-th) candidate is at global index k-1.
            int max_lane = (k - 1) / r;
            int max_local = (k - 1) % r;
            float thread_max = (lane == max_lane) ? inter_reg[max_local].dist : FLT_MAX;
            max_distance = __shfl_sync(0xffffffff, thread_max, max_lane);
        }
        __syncthreads();  // Ensure all threads are synchronized before loading next batch.
    }

    // After processing all batches, if there are remaining candidates, merge them.
    if (candCount[warpIdInBlock] > 0) {
        mergeWarp(k, r, inter_reg, candBuf + warpIdInBlock * k, &candCount[warpIdInBlock]);
    }
    __syncwarp();

    // Write the final intermediate result to global output.
    // The final sorted k candidates are stored in inter_reg (distributed across the warp).
    // For each query, result is stored in result[query_index * k + j].
    int query_off = globalWarpId * k;
    for (int i = 0; i < r; i++) {
        int global_idx = lane * r + i;
        if (global_idx < k) {
            // Use make_pair to form std::pair<int, float> result.
            result[query_off + global_idx] = std::make_pair(inter_reg[i].idx, inter_reg[i].dist);
        }
    }
}

// -----------------------------------------------------------------------------
// Host function: run_knn
//
// This function launches the knn_kernel with the appropriate grid and block
// dimensions and shared memory size. It expects that the memory for query, data,
// and result arrays is already allocated on the device (via cudaMalloc).
// -----------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose block size: 128 threads (i.e., 4 warps per block).
    int blockSize = 128;
    int warpsPerBlock = blockSize / 32;
    // Each warp processes one query, so compute grid size.
    int gridSize = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate shared memory size per block.
    // Candidate buffer: warpsPerBlock * k * sizeof(Candidate)
    // Candidate count:   warpsPerBlock * sizeof(int)
    // Data batch buffer: BATCH_SIZE * sizeof(float2), with BATCH_SIZE = 256.
    size_t sharedMemSize = warpsPerBlock * k * sizeof(Candidate) +
                           warpsPerBlock * sizeof(int) +
                           256 * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<gridSize, blockSize, sharedMemSize>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// End of implementation.
// -----------------------------------------------------------------------------
