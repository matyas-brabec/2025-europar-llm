#include <cuda_runtime.h>
#include <utility>
#include <limits>

// Optimized CUDA implementation of k-NN for 2D points.
// Each query is processed by a single warp (32 threads).
// The kernel processes the data points in batches cached in shared memory.
// For each warp (query), we keep:
//  - An intermediate result of top-k (indices + distances) distributed across the 32 lanes (registers).
//  - A per-warp candidate buffer of size k stored in shared memory (indices + distances).
// Candidates that are not closer than the current worst (k-th) distance in the intermediate result are skipped.
// When the candidate buffer is full (or at the end), we merge it into the intermediate result cooperatively within the warp.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Utility: lane id (0..31)
static __device__ __forceinline__ int lane_id() {
    return threadIdx.x & (WARP_SIZE - 1);
}

// Utility: warp id within block
static __device__ __forceinline__ int warp_id_in_block() {
    return threadIdx.x >> 5;
}

// Utility: recompute local worst (maximum distance) in this lane's portion
static __device__ __forceinline__ void recompute_local_worst(const float* top_d, int L, float& local_worst_val, int& local_worst_pos) {
    float wv = top_d[0];
    int wp = 0;
    for (int i = 1; i < L; ++i) {
        float v = top_d[i];
        if (v > wv) {
            wv = v;
            wp = i;
        }
    }
    local_worst_val = wv;
    local_worst_pos = wp;
}

// Utility: compute global worst (max) across warp, returning worst value and the owner lane+pos
static __device__ __forceinline__ void warp_argmax(float local_val, int local_pos, float& global_val, int& owner_lane, int& owner_pos, unsigned mask) {
    int lane = lane_id();
    float best_val = local_val;
    int best_lane = lane;
    int best_pos = local_pos;

    // Warp reduction for max with arg
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float oth_val = __shfl_down_sync(mask, best_val, offset);
        int   oth_lane = __shfl_down_sync(mask, best_lane, offset);
        int   oth_pos  = __shfl_down_sync(mask, best_pos,  offset);

        if (oth_val > best_val) {
            best_val = oth_val;
            best_lane = oth_lane;
            best_pos = oth_pos;
        }
    }
    // Broadcast from lane 0
    global_val = __shfl_sync(mask, best_val, 0);
    owner_lane = __shfl_sync(mask, best_lane, 0);
    owner_pos  = __shfl_sync(mask, best_pos,  0);
}

// Utility: compute global min (for final ordering) across warp, returning min value and owner lane+pos
static __device__ __forceinline__ void warp_argmin(float local_val, int local_pos, float& global_val, int& owner_lane, int& owner_pos, unsigned mask) {
    int lane = lane_id();
    float best_val = local_val;
    int best_lane = lane;
    int best_pos = local_pos;

    // Warp reduction for min with arg
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float oth_val = __shfl_down_sync(mask, best_val, offset);
        int   oth_lane = __shfl_down_sync(mask, best_lane, offset);
        int   oth_pos  = __shfl_down_sync(mask, best_pos,  offset);

        if (oth_val < best_val) {
            best_val = oth_val;
            best_lane = oth_lane;
            best_pos = oth_pos;
        }
    }
    // Broadcast from lane 0
    global_val = __shfl_sync(mask, best_val, 0);
    owner_lane = __shfl_sync(mask, best_lane, 0);
    owner_pos  = __shfl_sync(mask, best_pos,  0);
}

// Merge the per-warp candidate buffer into the warp's distributed top-k.
// - cand_dists, cand_indices: per-warp candidate arrays in shared memory (size >= count).
// - count: number of candidates currently in buffer (in shared memory counter).
// - top_*: per-lane portion of top-k arrays in registers (size L).
// - local_worst_*: per-lane cached worst info (max of local portion).
// - worst_threshold: updated to new global worst (k-th distance) after merge.
// All threads in the warp call this function cooperatively.
static __device__ __forceinline__ void merge_candidates_into_topk(
    float* cand_dists_base,
    int*   cand_indices_base,
    volatile int* cand_count_ptr,
    float* top_dists, int* top_indices, int L,
    float& local_worst_val, int& local_worst_pos,
    float& worst_threshold,
    unsigned mask)
{
    int count = 0;
    // Lane 0 reads and resets the candidate count
    if (lane_id() == 0) {
        count = *cand_count_ptr;
        *cand_count_ptr = 0;
    }
    // Broadcast count to all lanes
    count = __shfl_sync(mask, count, 0);

    // For each candidate, attempt to insert if it improves the current top-k.
    for (int i = 0; i < count; ++i) {
        float cand_d = cand_dists_base[i];
        int   cand_i = cand_indices_base[i];

        // Compute current global worst (k-th best) and owner of that slot
        float global_worst_val; int owner_lane; int owner_pos;
        warp_argmax(local_worst_val, local_worst_pos, global_worst_val, owner_lane, owner_pos, mask);

        // If candidate is better than current worst, replace it
        if (cand_d < global_worst_val) {
            if (lane_id() == owner_lane) {
                top_dists[owner_pos]  = cand_d;
                top_indices[owner_pos] = cand_i;
                // Update this lane's local worst, as it changed
                recompute_local_worst(top_dists, L, local_worst_val, local_worst_pos);
            }
        }
        // All lanes keep participating; implicit warp-synchronous execution
    }

    // Update worst_threshold for future filtering (global worst after merge)
    float global_worst_val; int owner_lane_dummy; int owner_pos_dummy;
    warp_argmax(local_worst_val, local_worst_pos, global_worst_val, owner_lane_dummy, owner_pos_dummy, mask);
    worst_threshold = global_worst_val;
}

// Kernel implementing warp-per-query k-NN for 2D points.
__global__ void knn_kernel(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    std::pair<int, float>* __restrict__ result,
    int k,
    int tile_points) // number of data points cached per block in shared memory per batch
{
    extern __shared__ unsigned char shared_raw[];
    unsigned char* smem = shared_raw;

    const int lane = lane_id();
    const int warp_in_block = warp_id_in_block();
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int query_idx = blockIdx.x * warps_per_block + warp_in_block;

    if (query_idx >= query_count) return;

    // Lay out shared memory:
    // [ data tile (float2[tile_points]) ][ candidate counts (int[warps_per_block]) ][ cand dists (float[warps_per_block*k]) ][ cand indices (int[warps_per_block*k]) ]
    size_t offset = 0;

    float2* sData = reinterpret_cast<float2*>(smem + offset);
    offset += static_cast<size_t>(tile_points) * sizeof(float2);

    // Align next section to 8 bytes
    offset = (offset + 7) & ~size_t(7);

    int* sCandCounts = reinterpret_cast<int*>(smem + offset);
    offset += static_cast<size_t>(warps_per_block) * sizeof(int);

    // Align to 8
    offset = (offset + 7) & ~size_t(7);

    float* sCandDists = reinterpret_cast<float*>(smem + offset);
    offset += static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * sizeof(float);

    int* sCandIdxs = reinterpret_cast<int*>(smem + offset);
    // offset += warps_per_block * k * sizeof(int); // Not needed further

    // Per-warp view into candidate buffer
    float* myCandDists = sCandDists + static_cast<size_t>(warp_in_block) * static_cast<size_t>(k);
    int*   myCandIdxs  = sCandIdxs  + static_cast<size_t>(warp_in_block) * static_cast<size_t>(k);
    volatile int* myCandCount = sCandCounts + warp_in_block;

    // Initialize candidate count for this warp
    if (lane == 0) {
        *myCandCount = 0;
    }
    __syncwarp();

    // Load the query point and broadcast within the warp
    float qx = 0.0f, qy = 0.0f;
    if (lane == 0) {
        float2 q = query[query_idx];
        qx = q.x; qy = q.y;
    }
    unsigned full_mask = 0xFFFFFFFFu;
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);

    // Each warp maintains its top-k as lane-distributed arrays in registers.
    // L elements per lane; k is guaranteed to be divisible by 32.
    const int L = k / WARP_SIZE;
    float top_dists[32];
    int   top_indices[32];

    // Initialize intermediate result with +inf distance and -1 index
    const float INF = CUDART_INF_F;
    for (int i = 0; i < L; ++i) {
        top_dists[i] = INF;
        top_indices[i] = -1;
    }

    // Local worst for this lane's segment
    float local_worst_val = INF;
    int   local_worst_pos = 0;

    // Global worst threshold used for candidate filtering
    float worst_threshold = INF;

    // Process data in batches loaded into shared memory by the whole block
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        int tile_count = data_count - tile_start;
        if (tile_count > tile_points) tile_count = tile_points;

        // Load the tile cooperatively by all threads in the block
        __syncthreads();
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            sData[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes the tile, computing distances to its query point
        for (int i = lane; i < tile_count; i += WARP_SIZE) {
            float2 p = sData[i];
            float dx = p.x - qx;
            float dy = p.y - qy;
            float dist = dx * dx + dy * dy;

            // Filter by current worst threshold of intermediate result
            int accept = (dist < worst_threshold) ? 1 : 0;

            // Warp-scope append of accepted candidates into per-warp shared buffer
            unsigned mask = __ballot_sync(full_mask, accept != 0);
            int num = __popc(mask);
            if (num > 0) {
                // Ensure capacity: if adding num would overflow, merge existing candidates first
                int base = 0;
                if (lane == 0) {
                    int count = *myCandCount;
                    if (count + num > k) {
                        // Merge all buffered candidates before adding new ones
                        merge_candidates_into_topk(myCandDists, myCandIdxs, myCandCount,
                                                   top_dists, top_indices, L,
                                                   local_worst_val, local_worst_pos,
                                                   worst_threshold, full_mask);
                        count = 0;
                    }
                    base = count;
                    *myCandCount = count + num;
                }
                base = __shfl_sync(full_mask, base, 0);

                if (accept) {
                    // Compute my position among active lanes
                    unsigned lower = mask & ((1u << lane) - 1u);
                    int offset = __popc(lower);
                    int pos = base + offset;
                    myCandDists[pos] = dist;
                    myCandIdxs[pos]  = tile_start + i;
                }

                // If buffer just became full, merge it now
                int need_merge = 0;
                if (lane == 0) {
                    need_merge = (*myCandCount >= k) ? 1 : 0;
                }
                need_merge = __shfl_sync(full_mask, need_merge, 0);
                if (need_merge) {
                    merge_candidates_into_topk(myCandDists, myCandIdxs, myCandCount,
                                               top_dists, top_indices, L,
                                               local_worst_val, local_worst_pos,
                                               worst_threshold, full_mask);
                }
            }
        }
        // Proceed to next tile; leftover candidates (if any) will be merged later
    }

    // After processing all tiles, merge any remaining candidates
    int leftover = 0;
    if (lane == 0) leftover = *myCandCount;
    leftover = __shfl_sync(full_mask, leftover, 0);
    if (leftover > 0) {
        merge_candidates_into_topk(myCandDists, myCandIdxs, myCandCount,
                                   top_dists, top_indices, L,
                                   local_worst_val, local_worst_pos,
                                   worst_threshold, full_mask);
    }

    // Final step: write out the k nearest neighbors sorted by ascending distance.
    // We perform a warp-cooperative selection sort:
    //  - Repeat k times: find the global minimum among remaining entries,
    //    write it to output in order, and mark it as used (set to +inf).
    int out_base = query_idx * k;
    for (int sel = 0; sel < k; ++sel) {
        // Each lane finds its local minimum among its L elements
        float local_min_val = INF;
        int   local_min_pos = -1;
        for (int i = 0; i < L; ++i) {
            float v = top_dists[i];
            if (v < local_min_val) {
                local_min_val = v;
                local_min_pos = i;
            }
        }

        // Global argmin across warp
        float global_min_val; int owner_lane; int owner_pos;
        warp_argmin(local_min_val, local_min_pos, global_min_val, owner_lane, owner_pos, full_mask);

        // Owner lane writes the result and marks the entry as used
        if (lane == owner_lane) {
            int idx = top_indices[owner_pos];
            // Write (index, distance) to result
            result[out_base + sel].first  = idx;
            result[out_base + sel].second = global_min_val;
            // Mark as used for next iteration
            top_dists[owner_pos] = INF;
        }
        // Implicit warp synchronization is sufficient; all lanes participate uniformly
    }
}

// Host function: selects kernel launch configuration and shared memory tile size, then launches the kernel.
void run_knn(const float2* query, int query_count, const float2* data, int data_count, std::pair<int, float>* result, int k)
{
    // Choose threads per block: 8 warps (256 threads) per block.
    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    // Determine an appropriate tile size for data cached in shared memory,
    // subject to the device's dynamic shared memory limit and per-warp candidate buffers.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    size_t max_dyn_shmem = prop.sharedMemPerBlockOptin ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock;

    // Per-block shared memory usage:
    //   - Candidate counts: warps_per_block * sizeof(int)
    //   - Candidate buffers: warps_per_block * k * (sizeof(float) + sizeof(int))
    //   - Data tile: tile_points * sizeof(float2)  <= remaining memory
    const size_t cand_counts_bytes = static_cast<size_t>(warps_per_block) * sizeof(int);
    const size_t cand_buffers_bytes = static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    // Leave some safety margin (optional). Here we use full available memory.
    size_t remaining = (max_dyn_shmem > cand_counts_bytes + cand_buffers_bytes)
                     ? (max_dyn_shmem - cand_counts_bytes - cand_buffers_bytes)
                     : 0;

    // Compute tile_points as large as possible, but at least one block of threads
    int tile_points = 0;
    if (remaining >= sizeof(float2)) {
        tile_points = static_cast<int>(remaining / sizeof(float2));
    }
    // Align tile_points down to a multiple of threads_per_block for coalesced loads
    if (tile_points > 0) {
        tile_points = (tile_points / threads_per_block) * threads_per_block;
    }
    // Ensure a reasonable minimum
    if (tile_points <= 0) {
        // Fallback minimal tile equals one block worth of threads
        tile_points = threads_per_block;
    }

    // Compute final dynamic shared memory size for kernel launch
    size_t shmem_bytes = static_cast<size_t>(tile_points) * sizeof(float2)
                       + cand_counts_bytes
                       + cand_buffers_bytes;

    // Request the needed dynamic shared memory size
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_bytes));

    // Launch configuration
    int total_warps = (query_count + warps_per_block - 1) / warps_per_block;
    dim3 grid(total_warps, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    // Launch the kernel
    knn_kernel<<<grid, block, shmem_bytes>>>(query, query_count, data, data_count, result, k, tile_points);
}
