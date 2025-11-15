#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <utility>

// Simple POD replacement for std::pair<int,float> on device.
// Layout-compatibility is enforced by the static_assert below.
struct PairIntFloat {
    int   first;
    float second;
};

static_assert(sizeof(PairIntFloat) == sizeof(std::pair<int, float>),
              "PairIntFloat must match size of std::pair<int,float>");

constexpr int WARP_SIZE         = 32;
constexpr int WARPS_PER_BLOCK   = 4;   // 4 warps => 128 threads per block
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int TILE_CAPACITY     = 1024; // number of data points cached per block in shared memory

// Warp-level bitonic sort on (distance, index) pairs stored in shared memory.
// - 'dist' and 'idx' are arrays of length 'length' (power of two, up to 1024).
// - All 32 threads in the warp cooperate to sort the array in ascending order of 'dist'.
// Based on the provided serial pseudocode, parallelized over lanes.
__device__ __forceinline__ void bitonic_sort_pairs(float* dist, int* idx, int length)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    for (int k = 2; k <= length; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each lane processes multiple indices i separated by WARP_SIZE
            for (int i = lane; i < length; i += WARP_SIZE) {
                int l = i ^ j;
                if (l > i) {
                    bool ascending = ((i & k) == 0);
                    float di = dist[i];
                    float dl = dist[l];
                    bool swap = ascending ? (di > dl) : (di < dl);
                    if (swap) {
                        dist[i] = dl;
                        dist[l] = di;
                        int ti = idx[i];
                        idx[i] = idx[l];
                        idx[l] = ti;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

// Flush the candidate buffer for a single warp, merging it with the intermediate result.
// - k: number of neighbors
// - cand_dist / cand_idx: candidate buffer (shared memory, size k)
// - res_dist / res_idx:   intermediate result buffer (shared memory, size k), sorted ascending
// - cand_count_ptr:       pointer to shared integer holding candidate count for this warp
// - max_dist_ptr:         pointer to shared float holding current max_distance for this warp
//
// Algorithm (per problem statement):
// 0. Intermediate result is sorted ascending (invariant).
// 1. Sort the buffer (cand_*) in ascending order using Bitonic Sort.
// 2. Merge buffer and intermediate result: for each i, take min of buffer[i] and result[k-1-i].
//    The merged result is a bitonic sequence of length k stored in res_*.
// 3. Sort merged res_* in ascending order using Bitonic Sort, update max_distance to res_[k-1].
__device__ __forceinline__ void warp_flush_buffer(
    int k,
    float* cand_dist, int* cand_idx,
    float* res_dist,  int* res_idx,
    int*  cand_count_ptr,
    float* max_dist_ptr)
{
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    int lane = threadIdx.x & (WARP_SIZE - 1);

    // Load candidate count (same for all lanes)
    int count = 0;
    if (lane == 0) {
        count = *cand_count_ptr;
    }
    count = __shfl_sync(FULL_MASK, count, 0);
    if (count == 0) {
        return; // nothing to merge
    }

    // Pad unused candidate entries with +inf so that we always sort 'k' elements.
    for (int i = lane + count; i < k; i += WARP_SIZE) {
        cand_dist[i] = CUDART_INF_F;
        cand_idx[i]  = -1;
    }
    __syncwarp(FULL_MASK);

    // 1. Sort candidate buffer ascending.
    bitonic_sort_pairs(cand_dist, cand_idx, k);
    __syncwarp(FULL_MASK);

    // 2. Merge candidates with intermediate result into a bitonic sequence in res_*.
    for (int i = lane; i < k; i += WARP_SIZE) {
        int j = k - 1 - i;
        float bd = cand_dist[i];
        float rd = res_dist[j];
        if (bd < rd) {
            res_dist[i] = bd;
            res_idx[i]  = cand_idx[i];
        } else {
            res_dist[i] = rd;
            res_idx[i]  = res_idx[j];
        }
    }
    __syncwarp(FULL_MASK);

    // 3. Sort merged result ascending to update intermediate result.
    bitonic_sort_pairs(res_dist, res_idx, k);
    __syncwarp(FULL_MASK);

    // Update max_distance (distance of k-th nearest neighbor) and reset candidate count.
    if (lane == 0) {
        float md = res_dist[k - 1];
        *max_dist_ptr   = md;
        *cand_count_ptr = 0;
    }
    __syncwarp(FULL_MASK);
}

// CUDA kernel: each warp processes a single query and finds its k nearest neighbors.
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    int k,
    PairIntFloat* __restrict__ result)
{
    extern __shared__ unsigned char smem[];

    // Shared memory layout:
    // [ float2 data_tile[TILE_CAPACITY] ]
    // [ int   cand_idx[WARPS_PER_BLOCK * k] ]
    // [ float cand_dist[WARPS_PER_BLOCK * k] ]
    // [ int   res_idx[WARPS_PER_BLOCK * k] ]
    // [ float res_dist[WARPS_PER_BLOCK * k] ]
    // [ int   cand_count[WARPS_PER_BLOCK] ]
    // [ float max_dist[WARPS_PER_BLOCK] ]

    float2* shared_points = reinterpret_cast<float2*>(smem);
    unsigned char* ptr = smem + TILE_CAPACITY * sizeof(float2);

    int* cand_idx_all   = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(int);

    float* cand_dist_all = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(float);

    int* res_idx_all     = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(int);

    float* res_dist_all  = reinterpret_cast<float*>(ptr);
    ptr += WARPS_PER_BLOCK * k * sizeof(float);

    int* cand_count      = reinterpret_cast<int*>(ptr);
    ptr += WARPS_PER_BLOCK * sizeof(int);

    float* max_dist      = reinterpret_cast<float*>(ptr);
    // End of shared memory region.

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;      // warp index within block [0, WARPS_PER_BLOCK)
    int lane    = tid & (WARP_SIZE - 1);

    int query_index = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    bool valid_query = (query_index < query_count);

    // Pointers to this warp's shared buffers.
    int*   cand_idx  = cand_idx_all  + warp_id * k;
    float* cand_dist = cand_dist_all + warp_id * k;
    int*   res_idx   = res_idx_all   + warp_id * k;
    float* res_dist  = res_dist_all  + warp_id * k;
    int*   cand_count_ptr = &cand_count[warp_id];
    float* max_dist_ptr   = &max_dist[warp_id];

    const unsigned FULL_MASK = 0xFFFFFFFFu;

    // Initialize intermediate result and per-warp state.
    if (valid_query) {
        // Intermediate result: distances = +inf, indices = -1 (sorted ascending by construction).
        for (int i = lane; i < k; i += WARP_SIZE) {
            res_dist[i] = CUDART_INF_F;
            res_idx[i]  = -1;
        }
        if (lane == 0) {
            *cand_count_ptr = 0;
            *max_dist_ptr   = CUDART_INF_F;
        }
    }
    __syncthreads(); // Ensure shared memory initialization does not race with tile loads.

    // Load query point into registers, broadcasted to all lanes of the warp.
    float2 q_point;
    if (valid_query) {
        if (lane == 0) {
            q_point = query[query_index];
        }
        q_point.x = __shfl_sync(FULL_MASK, q_point.x, 0);
        q_point.y = __shfl_sync(FULL_MASK, q_point.y, 0);
    }

    // Process data points in batches cached in shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_CAPACITY) {
        int tile_size = data_count - tile_start;
        if (tile_size > TILE_CAPACITY) {
            tile_size = TILE_CAPACITY;
        }

        // Load a tile of data points into shared memory using the whole block.
        for (int i = tid; i < tile_size; i += blockDim.x) {
            shared_points[i] = data[tile_start + i];
        }
        __syncthreads();

        if (valid_query) {
            // Each warp processes all points in the shared tile for its query.
            for (int idx_in_tile = lane; idx_in_tile < tile_size; idx_in_tile += WARP_SIZE) {
                float2 p = shared_points[idx_in_tile];
                float dx = q_point.x - p.x;
                float dy = q_point.y - p.y;
                float dist_val = dx * dx + dy * dy;

                // Filter by current max_distance (distance of k-th nearest neighbor).
                float current_max = *max_dist_ptr;
                bool is_candidate = (dist_val < current_max);
                int data_index = tile_start + idx_in_tile;

                // Warp-level candidate insertion using atomicAdd on the per-warp counter.
                unsigned candidate_mask = __ballot_sync(FULL_MASK, is_candidate);
                int num_candidates = __popc(candidate_mask);
                if (num_candidates > 0) {
                    // Check if the buffer has enough space; if not, flush it first and re-evaluate.
                    int base_pos = 0;
                    bool need_flush = false;

                    if (lane == 0) {
                        int old_count = *cand_count_ptr;
                        if (old_count + num_candidates > k) {
                            need_flush = true;
                        }
                    }
                    need_flush = __shfl_sync(FULL_MASK, need_flush, 0);
                    if (need_flush) {
                        // Flush current candidates and update max_distance.
                        warp_flush_buffer(
                            k,
                            cand_dist, cand_idx,
                            res_dist,  res_idx,
                            cand_count_ptr,
                            max_dist_ptr);

                        // After flush, re-evaluate candidate against updated max_distance.
                        current_max   = *max_dist_ptr;
                        is_candidate  = (dist_val < current_max);
                        candidate_mask = __ballot_sync(FULL_MASK, is_candidate);
                        num_candidates = __popc(candidate_mask);
                        if (num_candidates == 0) {
                            continue; // no candidates left for this data point
                        }
                    }

                    // Reserve 'num_candidates' slots in the candidate buffer using atomicAdd.
                    if (lane == 0) {
                        base_pos = atomicAdd(cand_count_ptr, num_candidates);
                    }
                    base_pos = __shfl_sync(FULL_MASK, base_pos, 0);

                    if (is_candidate) {
                        // Compute this lane's offset among candidate lanes using prefix popcount.
                        unsigned mask_before = candidate_mask & ((1u << lane) - 1u);
                        int offset = __popc(mask_before);
                        int pos    = base_pos + offset;

                        // Store candidate in shared memory buffer.
                        cand_idx[pos]  = data_index;
                        cand_dist[pos] = dist_val;
                    }
                }
            }
        }

        __syncthreads(); // Ensure tile is not reused before all warps finish processing it.
    }

    // After processing all tiles, flush any remaining candidates.
    if (valid_query) {
        warp_flush_buffer(
            k,
            cand_dist, cand_idx,
            res_dist,  res_idx,
            cand_count_ptr,
            max_dist_ptr);

        // Write final k nearest neighbors (sorted ascending) to global memory.
        int base_out = query_index * k;
        for (int i = lane; i < k; i += WARP_SIZE) {
            int out_idx = base_out + i;
            result[out_idx].first  = res_idx[i];
            result[out_idx].second = res_dist[i];
        }
    }
}

// Host entry point: launches the CUDA kernel.
// - query: device pointer to query points (float2)
// - data: device pointer to data points (float2)
// - result: device pointer to std::pair<int,float>
// All pointers are assumed to be allocated with cudaMalloc on the current device.
void run_knn(const float2* query, int query_count,
             const float2* data,  int data_count,
             std::pair<int, float>* result, int k)
{
    using Pair = std::pair<int, float>;
    PairIntFloat* d_result = reinterpret_cast<PairIntFloat*>(result);

    // Compute shared memory size per block based on k.
    const size_t shared_points_bytes = TILE_CAPACITY * sizeof(float2);
    const size_t per_warp_bytes =
        2 * static_cast<size_t>(k) * sizeof(int)   + // cand_idx + res_idx
        2 * static_cast<size_t>(k) * sizeof(float) + // cand_dist + res_dist
        sizeof(int) +                               // cand_count
        sizeof(float);                              // max_dist

    const size_t smem_per_block = shared_points_bytes +
                                  WARPS_PER_BLOCK * per_warp_bytes;

    // Opt-in to required dynamic shared memory size (for modern GPUs this is supported).
    int device = 0;
    cudaGetDevice(&device);
    int max_optin_smem = 0;
    cudaDeviceGetAttribute(&max_optin_smem,
                           /// @FIXED
                           /// cudaDevAttrMaxDynamicSharedMemoryPerBlockOptin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    int smem_int = static_cast<int>(smem_per_block);
    if (smem_int > max_optin_smem) {
        // Clamp to maximum opt-in size if requested size is too large.
        smem_int = max_optin_smem;
    }
    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_int);

    // Each warp handles one query.
    int warps_needed = query_count;
    int blocks = (warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    dim3 grid(blocks);
    dim3 block(THREADS_PER_BLOCK);

    // Launch kernel. Error checking is left to the caller if desired.
    knn_kernel<<<grid, block, smem_per_block>>>(
        query,
        query_count,
        data,
        data_count,
        k,
        d_result);
}
