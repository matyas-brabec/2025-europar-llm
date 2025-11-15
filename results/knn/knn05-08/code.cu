#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// **************************************
// Configuration parameters
// **************************************

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Number of warps per block (one query per warp)
static constexpr int WARPS_PER_BLOCK   = 8;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

// Number of data points loaded per batch into shared memory.
// This value is chosen to balance shared memory usage and global memory traffic.
// For k up to 1024 and WARPS_PER_BLOCK = 8, this fits into the 164 KB shared memory
// limit of A100/H100 GPUs when using dynamic shared memory opt-in.
static constexpr int BATCH_SIZE = 1024;

// **************************************
// Helper types
// **************************************

// Device-side representation of result pairs. It must match std::pair<int,float>.
struct DevicePairIF {
    int   first;
    float second;
};

static_assert(sizeof(DevicePairIF) == sizeof(std::pair<int, float>),
              "DevicePairIF must have the same size as std::pair<int,float>");

// **************************************
// Device-side helper functions
// **************************************

// Warp-wide merge of current intermediate k-NN list with candidate buffer.
// - k:                   number of neighbors to keep
// - warp_id:             warp index within the block [0, WARPS_PER_BLOCK)
// - warp_knn_size:       per-warp current size of intermediate list (<= k)
// - warp_cand_size:      per-warp current number of candidates in buffer (<= k)
// - warp_knn_worst:      per-warp current k-th nearest distance (INF if knn_size < k)
// - my_idx:              per-warp shared memory indices array of length 2*k:
//                        0..k-1   : intermediate results
//                        k..2*k-1 : candidate buffer
// - my_dist:             per-warp shared memory distances array, same layout as my_idx
// - full_mask:           active mask for warp-synchronous operations
__device__ __forceinline__
void warp_merge_knn(const int k,
                    const int warp_id,
                    int* __restrict__ warp_knn_size,
                    int* __restrict__ warp_cand_size,
                    float* __restrict__ warp_knn_worst,
                    int* __restrict__ my_idx,
                    float* __restrict__ my_dist,
                    const unsigned full_mask)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const float INF = CUDART_INF_F;

    int knn_size  = warp_knn_size[warp_id];
    int cand_size = warp_cand_size[warp_id];
    int total     = knn_size + cand_size;

    if (total == 0) {
        // Nothing to merge.
        return;
    }

    // Fill unused slots with +INF so that they sink to the end of the sorted array.
    // First half [0, k): intermediate list.
    // Second half [k, 2k): candidate buffer.
    for (int i = lane; i < k; i += WARP_SIZE) {
        if (i >= knn_size) {
            my_idx[i]  = -1;
            my_dist[i] = INF;
        }
        if (i >= cand_size) {
            my_idx[k + i]  = -1;
            my_dist[k + i] = INF;
        }
    }
    __syncwarp(full_mask);

    // Bitonic sort over 2*k elements (intermediate + candidates), ascending by distance.
    // 2*k is always a power of two in the allowed range of k (k is power of two in [32, 1024]).
    const int n = 2 * k;

    // Standard bitonic sort network.
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Stride-based compare-and-swap across the array.
            for (int tid = lane; tid < n; tid += WARP_SIZE) {
                int ixj = tid ^ stride;
                if (ixj > tid) {
                    float val_tid  = my_dist[tid];
                    float val_ixj  = my_dist[ixj];
                    int   idx_tid  = my_idx[tid];
                    int   idx_ixj  = my_idx[ixj];

                    // Direction of comparison: ascending in even blocks, descending in odd.
                    bool up = ((tid & size) == 0);
                    bool do_swap = up ? (val_tid > val_ixj) : (val_tid < val_ixj);

                    if (do_swap) {
                        my_dist[tid] = val_ixj;
                        my_dist[ixj] = val_tid;
                        my_idx[tid]  = idx_ixj;
                        my_idx[ixj]  = idx_tid;
                    }
                }
            }
            __syncwarp(full_mask);
        }
    }

    // Update intermediate result state.
    int new_knn_size = (total < k) ? total : k;

    if (lane == 0) {
        warp_knn_size[warp_id]  = new_knn_size;
        warp_cand_size[warp_id] = 0;  // candidate buffer is consumed

        if (new_knn_size < k) {
            // We still do not have k neighbors, so we don't prune by distance yet.
            warp_knn_worst[warp_id] = INF;
        } else {
            // k-th nearest neighbor distance after merge.
            warp_knn_worst[warp_id] = my_dist[k - 1];
        }
    }
    __syncwarp(full_mask);
}

// **************************************
// KNN kernel
// **************************************

// Each warp processes a single query point. All warps in a block cooperatively
// load batches of data points into shared memory. For each query, the warp keeps
// an intermediate list of k nearest neighbors and a shared-memory buffer of k
// candidate neighbors. Candidates are merged into the intermediate list using
// a warp-parallel bitonic sort whenever the buffer is nearly full, and once more
// at the end of processing.
__global__
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                DevicePairIF* __restrict__ result,
                int k)
{
    extern __shared__ unsigned char shared_mem[];

    // Layout of dynamic shared memory:
    // [0 .. BATCH_SIZE-1]                          : float2 smem_points[BATCH_SIZE]
    // [next .. next + WARPS_PER_BLOCK*2*k-1]       : int   smem_idx[WARPS_PER_BLOCK * 2 * k]
    // [next .. next + WARPS_PER_BLOCK*2*k-1]       : float smem_dist[WARPS_PER_BLOCK * 2 * k]
    float2* smem_points = reinterpret_cast<float2*>(shared_mem);

    size_t offset = sizeof(float2) * BATCH_SIZE;
    int* smem_idx = reinterpret_cast<int*>(shared_mem + offset);
    offset += sizeof(int) * WARPS_PER_BLOCK * 2 * static_cast<size_t>(k);
    float* smem_dist = reinterpret_cast<float*>(shared_mem + offset);

    // Per-block shared metadata for each warp.
    __shared__ int   warp_knn_size[WARPS_PER_BLOCK];
    __shared__ int   warp_cand_size[WARPS_PER_BLOCK];
    __shared__ float warp_knn_worst[WARPS_PER_BLOCK];

    const int thread_id = threadIdx.x;
    const int warp_id   = thread_id / WARP_SIZE;       // warp index within block
    const int lane      = thread_id & (WARP_SIZE - 1); // lane index within warp
    const unsigned full_mask = 0xFFFFFFFFu;

    const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool active = (warp_global < query_count);

    // Per-warp segments in shared memory for intermediate and candidate lists.
    int*   my_idx  = smem_idx  + warp_id * 2 * static_cast<size_t>(k);
    float* my_dist = smem_dist + warp_id * 2 * static_cast<size_t>(k);

    // Initialize per-warp metadata.
    if (lane == 0) {
        warp_knn_size[warp_id]  = 0;
        warp_cand_size[warp_id] = 0;
        warp_knn_worst[warp_id] = CUDART_INF_F;
    }
    __syncwarp(full_mask);

    // Load query point for this warp and broadcast to all lanes.
    float qx = 0.0f, qy = 0.0f;
    if (active && lane == 0) {
        float2 q = query[warp_global];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);

    const int num_batches = (data_count + BATCH_SIZE - 1) / BATCH_SIZE;

    for (int batch = 0; batch < num_batches; ++batch) {
        const int base       = batch * BATCH_SIZE;
        int       batch_size = data_count - base;
        if (batch_size > BATCH_SIZE) batch_size = BATCH_SIZE;

        // Load current batch of data points into shared memory.
        for (int i = thread_id; i < batch_size; i += blockDim.x) {
            smem_points[i] = data[base + i];
        }
        __syncthreads();

        if (active) {
            // Process this batch for the current query using a single warp.
            for (int g = 0; g < batch_size; g += WARP_SIZE) {

                // If candidate buffer is close to full (within one warp of capacity),
                // merge it with the current intermediate result.
                bool do_merge = false;
                if (lane == 0) {
                    if (warp_cand_size[warp_id] >= k - WARP_SIZE) {
                        do_merge = true;
                    }
                }
                do_merge = __shfl_sync(full_mask, do_merge, 0);
                if (do_merge) {
                    warp_merge_knn(k,
                                   warp_id,
                                   warp_knn_size,
                                   warp_cand_size,
                                   warp_knn_worst,
                                   my_idx,
                                   my_dist,
                                   full_mask);
                }

                const int idx_in_batch = g + lane;
                const bool valid = (idx_in_batch < batch_size);

                float d2          = 0.0f;
                int   data_index  = 0;
                bool  is_candidate = false;

                if (valid) {
                    float2 p = smem_points[idx_in_batch];
                    float dx = p.x - qx;
                    float dy = p.y - qy;
                    d2 = dx * dx + dy * dy;
                    data_index = base + idx_in_batch;

                    const float worst = warp_knn_worst[warp_id];
                    // Skip candidates that are not closer than the current k-th nearest neighbor.
                    is_candidate = (d2 < worst);
                }

                // Warp-level compaction of candidates using ballot and prefix sums.
                const unsigned cand_mask = __ballot_sync(full_mask, is_candidate);
                const int num_new = __popc(cand_mask);

                if (num_new > 0) {
                    int cand_base = 0;
                    if (lane == 0) {
                        cand_base = warp_cand_size[warp_id];
                        warp_cand_size[warp_id] = cand_base + num_new;
                    }
                    cand_base = __shfl_sync(full_mask, cand_base, 0);

                    if (is_candidate) {
                        const unsigned lane_mask = cand_mask & ((1u << lane) - 1u);
                        const int offset = __popc(lane_mask);
                        const int pos = cand_base + offset;

                        // Store candidate in second half of per-warp buffer.
                        my_idx[k + pos]  = data_index;
                        my_dist[k + pos] = d2;
                    }
                }
            } // end for g (groups)
        } // if (active)

        // Ensure all warps in the block are done with this batch before overwriting shared memory.
        __syncthreads();
    } // end for batch

    if (active) {
        // Final merge for leftover candidates, if any.
        bool do_merge_final = (warp_cand_size[warp_id] > 0);
        do_merge_final = __shfl_sync(full_mask, do_merge_final, 0);
        if (do_merge_final) {
            warp_merge_knn(k,
                           warp_id,
                           warp_knn_size,
                           warp_cand_size,
                           warp_knn_worst,
                           my_idx,
                           my_dist,
                           full_mask);
        }

        // Write k nearest neighbors for this query to global memory.
        const int knn_size_final = warp_knn_size[warp_id];
        const int out_base = warp_global * k;

        for (int i = lane; i < k; i += WARP_SIZE) {
            int   idx_val  = -1;
            float dist_val = CUDART_INF_F;

            if (i < knn_size_final) {
                idx_val  = my_idx[i];
                dist_val = my_dist[i];
            }

            result[out_base + i].first  = idx_val;
            result[out_base + i].second = dist_val;
        }
    }
}

// **************************************
// Host-side API
// **************************************

// Runs k-NN search for 2D points on the GPU.
// - query:       device pointer to query points (float2[query_count])
// - query_count: number of query points
// - data:        device pointer to data points (float2[data_count])
// - data_count:  number of data points
// - result:      device pointer to output pairs (std::pair<int,float>[query_count * k])
// - k:           number of nearest neighbors (power of two, 32 <= k <= 1024)
void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Number of blocks: one block processes WARPS_PER_BLOCK queries.
    const int num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // Compute required dynamic shared memory size per block.
    const size_t shmem_bytes =
        sizeof(float2) * BATCH_SIZE +                      // points cache
        sizeof(int)   * WARPS_PER_BLOCK * 2 * size_t(k) +  // indices (intermediate + candidates)
        sizeof(float) * WARPS_PER_BLOCK * 2 * size_t(k);   // distances (intermediate + candidates)

    // Opt-in to the maximum available dynamic shared memory if necessary (for A100/H100).
    int device = 0;
    cudaGetDevice(&device);

    int max_shmem_optin = 0;
    cudaDeviceGetAttribute(&max_shmem_optin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);

    if (shmem_bytes > static_cast<size_t>(max_shmem_optin)) {
        // In typical A100/H100 configurations and the specified k range,
        // this should not happen. If it does, reduce BATCH_SIZE or WARPS_PER_BLOCK.
        // For this implementation, we assume the hardware provides enough shared memory.
    } else {
        cudaFuncSetAttribute(knn_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_shmem_optin);
    }

    // Reinterpret result pointer for device kernel.
    DevicePairIF* device_result = reinterpret_cast<DevicePairIF*>(result);

    dim3 grid(num_blocks);
    dim3 block(THREADS_PER_BLOCK);

    knn_kernel<<<grid, block, shmem_bytes>>>(query,
                                             query_count,
                                             data,
                                             data_count,
                                             device_result,
                                             k);

    // Ensure completion before returning.
    cudaDeviceSynchronize();
}