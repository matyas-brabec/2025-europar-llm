#include <cuda_runtime.h>
#include <utility>

// This implementation computes k-nearest neighbors (k-NN) for 2D points using a warp-per-query strategy.
// Key design choices:
// - One warp (32 threads) computes all k-NN for one query. We organize blocks with multiple warps.
// - The dataset is processed in tiles that are cached in shared memory. We set the tile size equal to k.
//   This ensures that the union of lane-processed elements equals exactly the tile, which makes merging exact and simple.
// - For each warp (i.e., for each query), we maintain two private shared-memory buffers of size k:
//   one for the "current top-k" (sorted ascending by distance) and one for the "current tile candidates" (to be sorted).
//   After each tile, we sort the candidates with a bitonic sort (in-place in shared memory) and merge the two
//   sorted arrays (top-k and candidates) to produce the new top-k, using a merge-path based parallel merge.
//   To avoid needing a third buffer, each lane first copies its merge segment into registers (local arrays),
//   then writes the merged outputs back to the top-k buffer. This avoids aliasing hazards.
// - The requested k is a power-of-two between 32 and 1024 inclusive, and data_count >= k.
// - The final result for each query is k pairs (index, squared distance), sorted in ascending order of distance.
//
// Performance considerations:
// - The per-tile sort and merge are cooperative across the 32 threads of the warp.
// - Using tile size equal to k keeps shared memory requirements predictable and modest, and guarantees correctness
//   of merges without losing candidates.
// - Block-shared tiles of data are reused by all warps in the block to reduce global memory traffic.
// - This kernel uses only dynamic shared memory and no additional global allocations.
//
// Shared memory usage per block (for WARPS_PER_BLOCK warps):
//   - Data tile: k * sizeof(float2) bytes.
//   - Per-warp buffers: two sets (top-k and candidates) of (k floats + k ints) each.
//     That is 2 * (k * 4 + k * 4) = 16k bytes per warp. For 4 warps: 64k.
//   - Total per block: 8k + 16k * WARPS_PER_BLOCK bytes. For k=1024 and 4 warps: 8KB + 64KB = 72KB.
//
// Notes:
// - Distances are squared L2, no sqrt, tie-breaking is arbitrary.
// - The bitonic sort is implemented cooperatively by 32 threads over up to 1024 elements in shared memory.
// - The merge of two sorted arrays uses the merge-path partitioning for parallelism.
// - We assume std::pair<int, float> is allocated in device memory and accessible for device writes.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Compute squared Euclidean distance between two float2 points.
static __device__ __forceinline__ float squared_distance(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Cooperative in-place bitonic sort for pairs (distance, index) stored in shared memory arrays.
// - dist: pointer to distances in shared memory, length 'len' (power of two).
// - idx:  pointer to indices in shared memory, length 'len'.
// - len:  number of elements to sort (power of two, 32 <= len <= 1024).
// Sorting order: ascending by distance.
static __device__ __forceinline__ void warp_bitonic_sort_pairs(float* dist, int* idx, int len) {
    // Each compare-exchange step processes pairs (i, ixj) with ixj = i ^ j.
    // We loop i across [0, len) with stride WARP_SIZE so 32 threads cooperate on len elements.
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Bitonic sort network
    for (int k = 2; k <= len; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = lane; i < len; i += WARP_SIZE) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    float ai = dist[i];
                    float aj = dist[ixj];
                    int aidx = idx[i];
                    int ajdx = idx[ixj];
                    // If ascending, ensure dist[i] <= dist[ixj]; if descending, ensure dist[i] >= dist[ixj].
                    bool do_swap = (ai > aj) == ascending;
                    if (do_swap) {
                        dist[i]  = aj;
                        idx[i]   = ajdx;
                        dist[ixj]= ai;
                        idx[ixj] = aidx;
                    }
                }
            }
            __syncwarp(); // Synchronize within warp to ensure all compare-swaps at this 'j' are visible before next stage.
        }
    }
}

// Merge-path binary search to locate partition on diagonal 'diag' for merging sorted arrays A and B.
// Returns the number of elements taken from A (i), where j = diag - i are taken from B.
// Preconditions:
// - Arrays A[0..lenA) and B[0..lenB) are sorted ascending.
// - 0 <= diag <= lenA + lenB.
// - Works with sentinels: we use -INF and +INF for bounds.
static __device__ __forceinline__ int merge_path_search(const float* __restrict__ A, int lenA,
                                                        const float* __restrict__ B, int lenB,
                                                        int diag)
{
    int low  = max(0, diag - lenB);
    int high = min(diag, lenA);
    while (low <= high) {
        int i = (low + high) >> 1;
        int j = diag - i;

        float Ai_1 = (i > 0)     ? A[i - 1] : -CUDART_INF_F;
        float Bj_1 = (j > 0)     ? B[j - 1] : -CUDART_INF_F;
        float Ai   = (i < lenA)  ? A[i]     :  CUDART_INF_F;
        float Bj   = (j < lenB)  ? B[j]     :  CUDART_INF_F;

        // If A[i-1] > B[j], then i is too big.
        if (Ai_1 > Bj) {
            high = i - 1;
        }
        // Else if B[j-1] > A[i], then i is too small.
        else if (Bj_1 > Ai) {
            low = i + 1;
        }
        // Else we found valid partition.
        else {
            return i;
        }
    }
    return low;
}

// Cooperative merge of the first 'k' elements from two sorted arrays A and B into Out.
// Partition the first k outputs into WARP_SIZE segments; each lane merges its segment.
// Arrays are in shared memory; to avoid read-write aliasing, each lane first copies its needed
// input segment into registers, then writes the merged segment back to Out.
// - A, idxA: sorted ascending by A (length lenA = k).
// - B, idxB: sorted ascending by B (length lenB = k; entries beyond valid tile are +INF).
// - Out, idxOut: output arrays (length k), may alias A if inputs are fully read to registers first.
// - k: number of outputs to produce (multiple of 32).
static __device__ __forceinline__ void warp_merge_first_k(const float* __restrict__ A, const int* __restrict__ idxA,
                                                          const float* __restrict__ B, const int* __restrict__ idxB,
                                                          float* __restrict__ Out, int* __restrict__ idxOut,
                                                          int k)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int seg  = k >> 5; // k / 32, guaranteed integer since k is power-of-two >= 32.

    const int diag0 = lane * seg;
    const int diag1 = diag0 + seg;

    // Find start and end partitions for this lane's output segment [diag0, diag1)
    int i0 = merge_path_search(A, k, B, k, diag0);
    int j0 = diag0 - i0;
    int i1 = merge_path_search(A, k, B, k, diag1);
    int j1 = diag1 - i1;

    const int cntA = i1 - i0;
    const int cntB = j1 - j0;

    // Copy needed input ranges to registers to avoid aliasing with output writes.
    float regA[32]; int regAi[32];
    float regB[32]; int regBi[32];

    for (int u = 0; u < cntA; ++u) {
        regA[u]  = A[i0 + u];
        regAi[u] = idxA[i0 + u];
    }
    for (int v = 0; v < cntB; ++v) {
        regB[v]  = B[j0 + v];
        regBi[v] = idxB[j0 + v];
    }

    __syncwarp(); // Ensure all lanes have finished reading inputs before any lane writes to Out.

    // Merge to produce seg outputs for this lane.
    int a = 0, b = 0;
    int outPos = diag0;
    while (outPos < diag1) {
        float va = (a < cntA) ? regA[a] : CUDART_INF_F;
        float vb = (b < cntB) ? regB[b] : CUDART_INF_F;
        if (va <= vb) {
            Out[outPos]   = va;
            idxOut[outPos]= regAi[a];
            ++a;
        } else {
            Out[outPos]   = vb;
            idxOut[outPos]= regBi[b];
            ++b;
        }
        ++outPos;
    }

    __syncwarp(); // Make sure all outputs are visible before next tile iteration.
}

// Kernel: one warp processes one query. All warps in a block share the same data tile cached in shared memory.
static __global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                                  const float2* __restrict__ data,  int data_count,
                                  std::pair<int, float>* __restrict__ result, int k)
{
    // Thread/wrap identification
    const int lane          = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_blk = blockDim.x >> 5;
    const int warp_global   = blockIdx.x * warps_per_blk + warp_in_block;
    const bool warp_active  = (warp_global < query_count);

    // Dynamic shared memory layout:
    // [0] tile points (k float2s)
    // Then per-warp regions, each of size:
    //   topk_dist[k], topk_idx[k], cand_dist[k], cand_idx[k]
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_ptr = smem_raw;

    // Data tile cached in shared memory
    float2* smem_tile = reinterpret_cast<float2*>(smem_ptr);
    smem_ptr += sizeof(float2) * k;

    // Per-warp top-k and candidate buffers
    float* smem_topk_dist = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += sizeof(float) * (size_t)warps_per_blk * (size_t)k;

    int* smem_topk_idx = reinterpret_cast<int*>(smem_ptr);
    smem_ptr += sizeof(int) * (size_t)warps_per_blk * (size_t)k;

    float* smem_cand_dist = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += sizeof(float) * (size_t)warps_per_blk * (size_t)k;

    int* smem_cand_idx = reinterpret_cast<int*>(smem_ptr);
    // smem_ptr += sizeof(int) * (size_t)warps_per_blk * (size_t)k; // not needed further

    // Slice per-warp regions
    float* myTopDist = smem_topk_dist + warp_in_block * k;
    int*   myTopIdx  = smem_topk_idx  + warp_in_block * k;
    float* myCandDist= smem_cand_dist + warp_in_block * k;
    int*   myCandIdx = smem_cand_idx  + warp_in_block * k;

    // Prepare query point in registers, broadcasted across warp.
    float2 q;
    if (lane == 0 && warp_active) {
        q = query[warp_global];
    }
    // Broadcast q to all lanes of the warp; inactive warp broadcasts undefined but won't be used.
    q.x = __shfl_sync(0xFFFFFFFFu, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFFu, q.y, 0);

    // Initialize top-k buffers: distances to +INF and indices to -1 (cooperative across lanes).
    // Even if warp is inactive, we can harmlessly initialize its private region.
    for (int i = lane; i < k; i += WARP_SIZE) {
        myTopDist[i] = CUDART_INF_F;
        myTopIdx[i]  = -1;
    }

    // Process the dataset in tiles of size k.
    for (int tile_start = 0; tile_start < data_count; tile_start += k) {
        const int tile_count = min(k, data_count - tile_start);

        // Block-cooperative load of data tile into shared memory.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            smem_tile[i] = data[tile_start + i];
        }
        __syncthreads(); // Ensure tile fully loaded before warps read.

        if (warp_active) {
            // Each warp computes squared distances for the tile and stores them in its candidate buffer.
            // Entries beyond actual tile_count are filled with +INF to simplify sorting/merging.
            for (int t = lane; t < tile_count; t += WARP_SIZE) {
                const float2 p = smem_tile[t];
                const float d = squared_distance(p, q);
                myCandDist[t] = d;
                myCandIdx[t]  = tile_start + t;
            }
            for (int t = lane + tile_count; t < k; t += WARP_SIZE) {
                myCandDist[t] = CUDART_INF_F;
                myCandIdx[t]  = -1;
            }
            __syncwarp(); // Make sure candidate buffer is fully populated.

            // Sort candidate distances (and indices) ascending.
            warp_bitonic_sort_pairs(myCandDist, myCandIdx, k);

            // Merge the first k elements from (myTopDist,myTopIdx) and (myCandDist,myCandIdx) into myTop*.
            // Note: warp_merge_first_k reads inputs into registers first, then writes to myTop*, avoiding aliasing hazards.
            warp_merge_first_k(myTopDist, myTopIdx, myCandDist, myCandIdx, myTopDist, myTopIdx, k);
        }

        __syncthreads(); // Ensure all warps are done using the tile before it is overwritten.
    }

    // Write results to global memory: k nearest neighbors for this query, sorted ascending.
    if (warp_active) {
        for (int j = lane; j < k; j += WARP_SIZE) {
            const int out_idx = warp_global * k + j;
            result[out_idx].first  = myTopIdx[j];
            result[out_idx].second = myTopDist[j];
        }
    }
}

// Host launcher. Chooses launch parameters, configures dynamic shared memory, and launches the kernel.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Hyperparameters: 4 warps per block (128 threads). This balances shared memory usage and occupancy.
    const int WARPS_PER_BLOCK   = 4;
    const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    // Grid: one warp per query.
    const int num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Dynamic shared memory size per block:
    //   tile: k * sizeof(float2)
    //   per-warp: two sets (topk and candidates) of (k floats + k ints)
    // total = 8*k + WARPS_PER_BLOCK * ( (k*4 + k*4) * 2 ) = 8k + 16*k*WARPS_PER_BLOCK
    size_t smem_bytes = (size_t)k * sizeof(float2)
                      + (size_t)WARPS_PER_BLOCK * ( (size_t)k * sizeof(float) + (size_t)k * sizeof(int) ) * 2;

    // Optionally opt-in to larger dynamic shared memory if necessary.
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

    // Launch kernel
    knn_kernel<<<num_blocks, THREADS_PER_BLOCK, smem_bytes>>>(query, query_count, data, data_count, result, k);
}