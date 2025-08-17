#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>
#include <cfloat>
#include <cmath>

// This implementation assigns a single warp (32 threads) to each query. The kernel iterates
// over the data points in large batches cached in shared memory. Each warp maintains:
// - A "private" intermediate result (top-k indices and distances) per query.
// - A per-warp shared-memory candidate buffer of size k (indices and distances).
// The warp processes the cached batch in tiles of warp-size, filters candidates using the
// current threshold (k-th smallest distance in the intermediate result), and appends them
// into the candidate buffer. When the candidate buffer becomes full, it is sorted and
// merged (in parallel using the warp) with the intermediate result. After all batches,
// any remaining candidates are merged as well.
//
// The intermediate result and candidate buffer both use shared memory so that multiple
// threads in a warp can cooperate during sort/merge. No additional device allocations
// are performed. The kernel uses dynamic shared memory sized by the host launcher.
//
// Requirements observed:
// - One warp per query (32 threads).
// - Batch-based processing; each batch is cached by the whole block.
// - Per-query buffer for k candidates in shared memory.
// - When the buffer is full, merge candidates with the intermediate result using the warp.
// - After last batch, merge any remaining candidates.
// - k is a power of two, 32 <= k <= 1024.
//
// Notes on shared memory usage per block (WARPS_PER_BLOCK = 4):
// - Two arrays per warp for intermediate (idx/dist) and two for candidates => 16*k bytes/warp.
// - For k=1024: 16*1024 = 16384 bytes per warp => 65536 bytes for 4 warps.
// - Remaining shared memory is used for the batch cache of float2 points.
// - Dynamic shared memory size is selected by the host to stay within the device limit
//   (opt-in maximum, e.g., ~164KB on A100, higher on H100).

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif

// Utility: lane id and warp id within block
__device__ __forceinline__ int lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
__device__ __forceinline__ int warp_id_in_block() { return threadIdx.x >> 5; }

// Squared Euclidean distance between 2D points
__device__ __forceinline__ float l2_sq_dist(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // fmaf(dx, dx, dy*dy) may fuse multiply-add on capable architectures
    return fmaf(dx, dx, dy * dy);
}

// In-warp bitonic sort on pairs (dist, idx) stored in shared memory.
// 'n' must be a power of two. This sorts ascending by distance.
// All 32 threads in the warp cooperate; the arrays are per-warp disjoint regions in smem.
__device__ void warp_bitonic_sort_pairs(float* dist, int* idx, int n, unsigned mask) {
    int lane = lane_id();

    // Standard bitonic sorting network over indices 0..n-1
    for (int size = 2; size <= n; size <<= 1) {
        // For each stage, perform log2(size) passes with decreasing stride
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncwarp(mask);
            // Each lane processes multiple indices i strided by warp size
            for (int i = lane; i < n; i += WARP_SIZE) {
                int j = i ^ stride;
                if (j > i && j < n) {
                    bool up = ((i & size) == 0); // ascending for this subsequence
                    float di = dist[i];
                    float dj = dist[j];
                    int ii = idx[i];
                    int ij = idx[j];

                    // Compare-and-swap depending on desired order
                    if (up) {
                        if (di > dj) {
                            dist[i] = dj; dist[j] = di;
                            idx[i] = ij; idx[j] = ii;
                        }
                    } else {
                        if (di < dj) {
                            dist[i] = dj; dist[j] = di;
                            idx[i] = ij; idx[j] = ii;
                        }
                    }
                }
            }
        }
    }
    __syncwarp(mask);
}

// Merge-path search to partition the merge of A and B at diagonal 'diag'.
// Both A and B are ascending arrays of length lenA=lenB=k.
// Returns (i, j) such that i+j = diag and the merge path invariants hold.
// This is used to parallelize the merge across warp lanes.
__device__ __forceinline__ int2 merge_path_search(const float* A, int lenA,
                                                  const float* B, int lenB,
                                                  int diag) {
    int low = max(0, diag - lenB);
    int high = min(diag, lenA);

    while (low <= high) {
        int i = (low + high) >> 1;
        int j = diag - i;

        float A_im1 = (i > 0)     ? A[i - 1] : -CUDART_INF_F;
        float B_j   = (j < lenB)  ? B[j]     :  CUDART_INF_F;

        if (A_im1 > B_j) {
            high = i - 1;
        } else {
            float B_jm1 = (j > 0)    ? B[j - 1] : -CUDART_INF_F;
            float A_i   = (i < lenA) ? A[i]     :  CUDART_INF_F;
            if (B_jm1 > A_i) {
                low = i + 1;
            } else {
                low = i;
                break;
            }
        }
    }

    int i = min(max(0, low), lenA);
    int j = diag - i;
    return make_int2(i, j);
}

// Merge the two ascending arrays A and B (length k each) into the first k elements of Out.
// The output is the k smallest elements of the 2k-length union.
// All 32 lanes in the warp cooperate, each lane produces a contiguous subrange.
// Out can alias A or B; the function reads only from A and B, writes to Out.
__device__ void warp_merge_topk_ascending(const float* A_dist, const int* A_idx,
                                          const float* B_dist, const int* B_idx,
                                          float* Out_dist, int* Out_idx,
                                          int k, unsigned mask) {
    int lane = lane_id();
    // Split the first 'k' outputs evenly among warp lanes
    int outputs_per_lane = (k + WARP_SIZE - 1) / WARP_SIZE;
    int out_start = lane * outputs_per_lane;
    int out_end = min(out_start + outputs_per_lane, k);
    if (out_start >= out_end) {
        __syncwarp(mask);
        return;
    }

    // Find (i,j) starting positions for out_start
    int2 start = merge_path_search(A_dist, k, B_dist, k, out_start);
    int i = start.x;
    int j = start.y;
    int out_pos = out_start;

    // Merge until out_end
    while (out_pos < out_end) {
        float a = (i < k) ? A_dist[i] : CUDART_INF_F;
        float b = (j < k) ? B_dist[j] : CUDART_INF_F;

        if (a <= b) {
            Out_dist[out_pos] = a;
            Out_idx[out_pos] = A_idx[i];
            ++i;
        } else {
            Out_dist[out_pos] = b;
            Out_idx[out_pos] = B_idx[j];
            ++j;
        }
        ++out_pos;
    }
    __syncwarp(mask);
}

// Flush the per-warp candidate buffer by sorting it and merging with the intermediate result.
// After merge, the new intermediate result is stored in A (by swapping pointers if needed).
// candCount is reset to 0 and the threshold (k-th distance) is updated.
// A_dist/A_idx and B_dist/B_idx are references (pointers passed by reference) so we can swap them.
__device__ void warp_flush_and_merge(float*& A_dist, int*& A_idx,
                                     float*& B_dist, int*& B_idx,
                                     volatile int* candCountPtr,
                                     volatile float* thrPtr,
                                     int k, unsigned mask) {
    // Read candidate count (lane 0) and broadcast
    int lane = lane_id();
    int candCount = 0;
    if (lane == 0) candCount = *candCountPtr;
    candCount = __shfl_sync(mask, candCount, 0);

    // Pad remaining candidate slots with +inf so we can sort a full power-of-two length
    for (int i = lane + candCount; i < k; i += WARP_SIZE) {
        B_dist[i] = CUDART_INF_F;
        B_idx[i] = -1;
    }
    __syncwarp(mask);

    // Sort candidates ascending by distance
    warp_bitonic_sort_pairs(B_dist, B_idx, k, mask);

    // Merge A and B into the first k outputs placed into B, then swap A<->B
    warp_merge_topk_ascending(A_dist, A_idx, B_dist, B_idx, B_dist, B_idx, k, mask);
    __syncwarp(mask);

    // Swap pointers so that A holds the new intermediate result
    if (lane == 0) {
        float* tmpD = A_dist; A_dist = B_dist; B_dist = tmpD;
        int*   tmpI = A_idx;  A_idx  = B_idx;  B_idx  = tmpI;
    }
    // Broadcast the swapped pointers within the warp
    A_dist = (float*)__shfl_sync(mask, (unsigned long long)A_dist, 0);
    A_idx  = (int*)__shfl_sync(mask, (unsigned long long)A_idx, 0);
    B_dist = (float*)__shfl_sync(mask, (unsigned long long)B_dist, 0);
    B_idx  = (int*)__shfl_sync(mask, (unsigned long long)B_idx, 0);
    __syncwarp(mask);

    // Update threshold (k-th smallest, i.e., last element in ascending array)
    float new_thr = 0.0f;
    if (lane == 0) new_thr = A_dist[k - 1];
    new_thr = __shfl_sync(mask, new_thr, 0);
    if (lane == 0) {
        *thrPtr = new_thr;
        *candCountPtr = 0;
    }
    __syncwarp(mask);
}

// Kernel: one warp per query
__global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                           const float2* __restrict__ data, int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k, int batch_size) {
    extern __shared__ unsigned char smem[];
    const unsigned full_mask = 0xFFFFFFFFu;

    // Dynamic shared memory layout:
    // [0, batch_size * sizeof(float2)) -> cached data points
    // Then per-warp arrays laid out contiguously:
    // - A_idx [WARPS_PER_BLOCK * k]
    // - A_dist[WARPS_PER_BLOCK * k]
    // - B_idx [WARPS_PER_BLOCK * k]
    // - B_dist[WARPS_PER_BLOCK * k]
    // - candCount[WARPS_PER_BLOCK]
    // - thr[WARPS_PER_BLOCK]

    size_t offset = 0;
    auto smem_align = [&](size_t a) {
        offset = (offset + a - 1) & ~(a - 1);
    };
    auto alloc = [&](size_t bytes) -> unsigned char* {
        unsigned char* ptr = smem + offset;
        offset += bytes;
        return ptr;
    };

    smem_align(alignof(float2));
    float2* sh_points = reinterpret_cast<float2*>(alloc((size_t)batch_size * sizeof(float2)));

    smem_align(alignof(int));
    int* sh_A_idx_all = reinterpret_cast<int*>(alloc((size_t)WARPS_PER_BLOCK * k * sizeof(int)));
    smem_align(alignof(float));
    float* sh_A_dist_all = reinterpret_cast<float*>(alloc((size_t)WARPS_PER_BLOCK * k * sizeof(float)));
    smem_align(alignof(int));
    int* sh_B_idx_all = reinterpret_cast<int*>(alloc((size_t)WARPS_PER_BLOCK * k * sizeof(int)));
    smem_align(alignof(float));
    float* sh_B_dist_all = reinterpret_cast<float*>(alloc((size_t)WARPS_PER_BLOCK * k * sizeof(float)));

    smem_align(alignof(int));
    int* sh_candCount = reinterpret_cast<int*>(alloc((size_t)WARPS_PER_BLOCK * sizeof(int)));
    smem_align(alignof(float));
    float* sh_threshold = reinterpret_cast<float*>(alloc((size_t)WARPS_PER_BLOCK * sizeof(float)));

    const int lane = lane_id();
    const int warp = warp_id_in_block();
    const int query_global = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool warp_active = (query_global < query_count);

    // Per-warp slice base offset into shared arrays
    const int warp_offset = warp * k;

    // Pointers to this warp's arrays
    float* A_dist = sh_A_dist_all + warp_offset;
    int*   A_idx  = sh_A_idx_all  + warp_offset;
    float* B_dist = sh_B_dist_all + warp_offset;
    int*   B_idx  = sh_B_idx_all  + warp_offset;

    // Initialize per-warp intermediate result to +inf and indices to -1
    if (warp_active) {
        for (int i = lane; i < k; i += WARP_SIZE) {
            A_dist[i] = CUDART_INF_F;
            A_idx[i]  = -1;
        }
        if (lane == 0) {
            sh_candCount[warp] = 0;
            sh_threshold[warp] = CUDART_INF_F;
        }
    }
    __syncthreads(); // synchronize entire block before starting batch processing

    // Load query point (broadcast within the warp)
    float qx = 0.0f, qy = 0.0f;
    if (warp_active) {
        float2 q = query[query_global];
        qx = q.x;
        qy = q.y;
    }
    // Broadcast qx, qy to ensure all lanes have valid registers even if only lane 0 loaded
    qx = __shfl_sync(full_mask, qx, 0);
    qy = __shfl_sync(full_mask, qy, 0);

    // Process data in batches cached into shared memory
    for (int base = 0; base < data_count; base += batch_size) {
        const int this_batch = min(batch_size, data_count - base);

        // All threads in the block cooperatively load the batch into shared memory
        for (int i = threadIdx.x; i < this_batch; i += blockDim.x) {
            sh_points[i] = data[base + i];
        }
        __syncthreads();

        // Each warp processes the cached points in tiles of 32
        if (warp_active) {
            for (int t = 0; t < this_batch; t += WARP_SIZE) {
                int idx_local = t + lane;
                bool valid = (idx_local < this_batch);

                // Compute squared distance for this lane's point in the tile
                float dist = 0.0f;
                int   gidx = -1;
                if (valid) {
                    float2 p = sh_points[idx_local];
                    dist = l2_sq_dist(p, make_float2(qx, qy));
                    gidx = base + idx_local;
                }

                // A lane's candidate is "pending" until stored into the candidate buffer or discarded
                bool pending = valid;

                // Inner loop: while there are pending candidates for this tile, try to push them
                // into the candidate buffer; if full, flush and merge, then retry with updated threshold.
                while (true) {
                    // Read current threshold
                    float thr = sh_threshold[warp];

                    // Decide which lanes have a qualifying pending candidate
                    bool qualifies = pending && (dist < thr);
                    unsigned mask = __ballot_sync(full_mask, qualifies);
                    int count = __popc(mask);

                    // Load current candidate count (lane 0) and broadcast
                    int candCount = 0;
                    if (lane == 0) candCount = sh_candCount[warp];
                    candCount = __shfl_sync(full_mask, candCount, 0);
                    int free_slots = k - candCount;

                    if (free_slots == 0) {
                        // Candidate buffer is full: flush and merge, then retry
                        warp_flush_and_merge(A_dist, A_idx, B_dist, B_idx,
                                             &sh_candCount[warp], &sh_threshold[warp], k, full_mask);
                        // Continue; threshold and candCount updated
                        continue;
                    }

                    if (count == 0) {
                        // No more candidates from this tile under current threshold
                        break;
                    }

                    // Compute per-lane prefix among qualifiers
                    unsigned lane_mask_lt = (1u << lane) - 1u;
                    int prefix = __popc(mask & lane_mask_lt);

                    if (count <= free_slots) {
                        // All qualifiers fit into the buffer
                        int write_base = candCount;
                        if (qualifies) {
                            int write_pos = write_base + prefix;
                            B_dist[write_pos] = dist;
                            B_idx[write_pos]  = gidx;
                            // Mark as no longer pending (written)
                            pending = false;
                        }
                        if (lane == 0) sh_candCount[warp] = candCount + count;
                        __syncwarp(full_mask);
                        break; // tile consumed under current threshold
                    } else {
                        // Partial fill: write only the first 'free_slots' qualifiers
                        bool push_now = qualifies && (prefix < free_slots);
                        unsigned mask_partial = __ballot_sync(full_mask, push_now);

                        int write_base = candCount;
                        int write_prefix = __popc(mask_partial & lane_mask_lt);

                        if (push_now) {
                            int write_pos = write_base + write_prefix;
                            B_dist[write_pos] = dist;
                            B_idx[write_pos]  = gidx;
                            // Mark as no longer pending (written)
                            pending = false;
                        }
                        if (lane == 0) sh_candCount[warp] = k; // buffer is now full
                        __syncwarp(full_mask);

                        // Flush and merge, then loop to try pushing remaining pending candidates
                        warp_flush_and_merge(A_dist, A_idx, B_dist, B_idx,
                                             &sh_candCount[warp], &sh_threshold[warp], k, full_mask);
                        // Continue: threshold updated, candCount reset to 0
                    }
                } // while pending loop
            } // for each tile in batch
        }

        __syncthreads(); // sync before next batch is loaded
    } // for each batch

    // After processing all batches, flush any remaining candidates
    if (warp_active) {
        int candCount = 0;
        if (lane == 0) candCount = sh_candCount[warp];
        candCount = __shfl_sync(full_mask, candCount, 0);
        if (candCount > 0) {
            warp_flush_and_merge(A_dist, A_idx, B_dist, B_idx,
                                 &sh_candCount[warp], &sh_threshold[warp], k, full_mask);
        }

        // Write the final sorted top-k (ascending distances) to the result array
        // Each lane writes a strided subset for coalescing and to utilize all lanes.
        size_t out_base = static_cast<size_t>(query_global) * static_cast<size_t>(k);
        for (int i = lane; i < k; i += WARP_SIZE) {
            std::pair<int, float> outp;
            outp.first = A_idx[i];
            outp.second = A_dist[i];
            result[out_base + i] = outp;
        }
    }
}

// Host launcher. Chooses block size, grid size, dynamic shared memory size, and batch size.
// The kernel uses WARPS_PER_BLOCK warps per block (compile-time constant).
void run_knn(const float2* query, int query_count,
             const float2* data, int data_count,
             std::pair<int, float>* result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Ensure k is a power of two in [32, 1024]
    // (Assumed by problem statement; no runtime check needed, but keeping a guard.)
    if (k < 32 || k > 1024 || (k & (k - 1)) != 0) return;

    // Device attributes for dynamic shared memory
    int device = 0;
    cudaGetDevice(&device);

    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_smem_optin <= 0) {
        // Fallback to default if opt-in not reported; typical default is 48KB.
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        max_smem_optin = prop.sharedMemPerBlock; // may be conservative
    }

    // Opt-in to the maximum dynamic shared memory size
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_optin);
    // Prefer shared memory carveout if supported (100% shared mem)
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = warps_per_block * WARP_SIZE;

    // Compute per-block shared memory usage for per-warp arrays
    // Two sets (A and B) of (idx + dist) per warp: 2 * k * (4 + 4) bytes per warp => 16*k per warp
    size_t per_warp_bytes = (size_t)k * (sizeof(int) + sizeof(float)) * 2; // 8*k bytes per set
    size_t per_block_arrays = per_warp_bytes * warps_per_block * 2;        // A and B => 16*k*warps bytes

    // Additional small arrays for candCount and threshold per warp
    size_t per_block_small = warps_per_block * (sizeof(int) + sizeof(float)); // negligible but accounted

    // Remaining shared memory goes to the batch cache (float2 points)
    size_t overhead = per_block_arrays + per_block_small;

    // Choose batch size as large as possible within the dynamic shared memory limit
    size_t max_batch_bytes = (overhead < (size_t)max_smem_optin) ? (size_t)max_smem_optin - overhead : 0;
    // At least 1 point per batch; prefer multiples of warp size for better coalescing
    int batch_size = 0;
    if (max_batch_bytes >= sizeof(float2)) {
        batch_size = static_cast<int>(max_batch_bytes / sizeof(float2));
        // Round down to nearest multiple of WARP_SIZE to simplify tile processing
        if (batch_size > WARP_SIZE) batch_size = (batch_size / WARP_SIZE) * WARP_SIZE;
        if (batch_size <= 0) batch_size = WARP_SIZE;
    } else {
        // If no space for caching, set a minimal batch size of 32 to keep algorithmic structure
        // (the kernel will still function; caching array will be small)
        batch_size = WARP_SIZE;
    }

    // Compute actual dynamic shared memory size used for launch
    size_t dyn_smem_bytes = overhead + (size_t)batch_size * sizeof(float2);

    // Grid configuration: one warp per query
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch kernel
    void* args[] = {
        (void*)&query,
        (void*)&query_count,
        (void*)&data,
        (void*)&data_count,
        (void*)&result,
        (void*)&k,
        (void*)&batch_size
    };
    cudaLaunchKernel((const void*)&knn_kernel,
                     dim3(blocks), dim3(threads_per_block),
                     nullptr, (size_t)dyn_smem_bytes, nullptr, args, nullptr);

    // Optional sync or error checking can be performed by the caller as desired.
}