#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// The device-side pair used for all intermediate computations.
// Distance is the primary key for comparisons.
struct PairK {
    float dist;
    int   idx;
};

// Warp size constant
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Utility: Compare-and-swap to enforce ascending or descending order
static __device__ __forceinline__ void cas_pair(PairK &a, PairK &b, bool ascending) {
    // If ascending, swap when a>b; if descending, swap when a<b
    if (ascending) {
        if (a.dist > b.dist) {
            PairK t = a; a = b; b = t;
        }
    } else {
        if (a.dist < b.dist) {
            PairK t = a; a = b; b = t;
        }
    }
}

// In-warp bitonic sort for an array in shared memory.
// - n must be a power of two (here, k is guaranteed to be).
// - All 32 threads in the warp cooperate to sort the entire 'arr' range.
// - Synchronization is warp-level (__syncwarp(mask)).
// The algorithm is the standard, serial bitonic sort lifted to parallel by
// having each lane iterate i in strides of warpSize, and only acting on pairs
// with l > i to avoid double-swaps.
static __device__ void bitonic_sort_warp(PairK* arr, int n, unsigned mask) {
    // Outer loop controls the size of subsequences to be merged (ksize),
    // inner loop controls the distance (j) between compared elements.
    for (int ksize = 2; ksize <= n; ksize <<= 1) {
        for (int j = ksize >> 1; j > 0; j >>= 1) {
            // Each lane handles multiple i's in strides of warp size
            for (int i = threadIdx.x & (WARP_SIZE - 1); i < n; i += WARP_SIZE) {
                int l = i ^ j;
                if (l > i) {
                    bool up = ((i & ksize) == 0);
                    PairK ai = arr[i];
                    PairK al = arr[l];
                    // Compare-and-swap into correct order
                    if (up) {
                        if (ai.dist > al.dist) {
                            arr[i] = al;
                            arr[l] = ai;
                        }
                    } else {
                        if (ai.dist < al.dist) {
                            arr[i] = al;
                            arr[l] = ai;
                        }
                    }
                }
            }
            __syncwarp(mask);
        }
    }
}

// Merge buffer 'cand' (sorted ascending) with 'res' (sorted ascending) into a
// bitonic sequence using pairwise minima, then sort that sequence to update 'res'.
// This follows the required merging procedure in the prompt:
//   1) buffer 'cand' is assumed sorted in ascending order.
//   2) merged[i] = min(cand[i], res[k-1-i])  for i in [0, k-1]
//   3) sort merged in ascending order with bitonic sort to get new 'res'
static __device__ void merge_and_update(PairK* res, PairK* cand, int k, float &max_dist, unsigned mask) {
    // Step 2: form bitonic sequence via pairwise minima between cand[i] and reversed res
    for (int i = threadIdx.x & (WARP_SIZE - 1); i < k; i += WARP_SIZE) {
        PairK a = cand[i];
        PairK b = res[k - 1 - i];
        cand[i] = (a.dist <= b.dist) ? a : b;
    }
    __syncwarp(mask);

    // Step 3: sort the merged bitonic sequence to get the updated result
    bitonic_sort_warp(cand, k, mask);
    __syncwarp(mask);

    // Copy back to res
    for (int i = threadIdx.x & (WARP_SIZE - 1); i < k; i += WARP_SIZE) {
        res[i] = cand[i];
    }
    __syncwarp(mask);

    // Update max_dist to the current k-th neighbor distance and broadcast to the warp
    float new_max = res[k - 1].dist;
    // Broadcast from lane 0
    new_max = __shfl_sync(mask, new_max, 0);
    max_dist = new_max;
}

// Flush the warp-local candidate buffer into the intermediate result.
// - Fills unused slots (if any) up to k with +inf to enable a full-size sort.
// - Sorts the buffer, merges with the current result via pairwise minima, then sorts again.
// - Resets candidate count and unlocks buffer for further accumulation.
// - Updates max_dist to the distance of the current k-th neighbor.
static __device__ void flush_candidates_warp(PairK* res,
                                             PairK* cand,
                                             volatile int* cand_count,
                                             volatile int* cand_lock,
                                             int k,
                                             float &max_dist,
                                             unsigned mask) {
    // Snapshot and clamp the count at k
    int count = *cand_count;
    if (count > k) count = k;

    // Fill remaining slots with sentinel (+inf, idx=-1) to have exactly k items
    for (int i = threadIdx.x & (WARP_SIZE - 1); i < k; i += WARP_SIZE) {
        if (i >= count) {
            cand[i].dist = FLT_MAX;
            cand[i].idx  = -1;
        }
    }
    __syncwarp(mask);

    // Step 1: sort buffer ascending
    bitonic_sort_warp(cand, k, mask);
    __syncwarp(mask);

    // Merge and update intermediate result and threshold
    merge_and_update(res, cand, k, max_dist, mask);
    __syncwarp(mask);

    // Reset candidate count and unlock for further accumulation
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        *cand_count = 0;
        *cand_lock  = 0;
    }
    __syncwarp(mask);
}

// CUDA Kernel: One warp (32 threads) processes exactly one query at a time.
// The block cooperatively tiles the data points into shared memory; each warp
// uses the cached tile to compute distances to its own query and filters them
// by the current max distance (distance of the k-th nearest neighbor).
// Closer candidates are added to a per-warp shared buffer using atomicAdd.
// Whenever the buffer becomes full, it is merged into the intermediate result.
__global__ void knn_warp_kernel(const float2* __restrict__ query,
                                int query_count,
                                const float2* __restrict__ data,
                                int data_count,
                                std::pair<int, float>* __restrict__ result,
                                int k,
                                int tile_points_runtime) {
    extern __shared__ unsigned char smem[];
    unsigned mask = __activemask();

    // Shared memory layout (dynamic, computed at launch-time):
    // [0 .. tile_points_runtime) float2 tile points for cooperative cache
    float2* tile = reinterpret_cast<float2*>(smem);

    // After tile, per-warp arrays for intermediate result and candidate buffer:
    // Layout: [ warp0_res[k], warp0_cand[k], warp1_res[k], warp1_cand[k], ... ]
    int warps_per_block = blockDim.x / WARP_SIZE;
    PairK* smem_pairs_base = reinterpret_cast<PairK*>(tile + tile_points_runtime);

    // After per-warp PairK arrays, we allocate per-warp candidate counters and locks
    int per_warp_pair_slots = 2 * k; // res[k] + cand[k]
    int total_pair_slots = warps_per_block * per_warp_pair_slots;
    int* smem_counts_base = reinterpret_cast<int*>(smem_pairs_base + total_pair_slots);
    int* smem_locks_base  = smem_counts_base + warps_per_block;

    // Warp identifiers
    int lane_id         = threadIdx.x & (WARP_SIZE - 1);
    int warp_in_block   = threadIdx.x >> 5; // threadIdx.x / WARP_SIZE
    int warp_global_id  = blockIdx.x * warps_per_block + warp_in_block;
    int total_warps_grid = gridDim.x * warps_per_block;

    // Every warp processes multiple queries in a grid-stride loop
    for (int q_idx = warp_global_id; q_idx < query_count; q_idx += total_warps_grid) {
        // Pointers into shared memory for this warp
        PairK* res  = smem_pairs_base + warp_in_block * per_warp_pair_slots;
        PairK* cand = res + k;
        volatile int* cand_count = smem_counts_base + warp_in_block;
        volatile int* cand_lock  = smem_locks_base + warp_in_block;

        // Initialize per-warp intermediate result with "+inf" and invalid indices
        for (int i = lane_id; i < k; i += WARP_SIZE) {
            res[i].dist = FLT_MAX;
            res[i].idx  = -1;
        }

        // Initialize candidate counters and lock
        if (lane_id == 0) {
            *cand_count = 0;
            *cand_lock  = 0;
        }
        __syncwarp(mask);

        // Load this warp's query point (broadcast from lane 0)
        float qx = 0.0f, qy = 0.0f;
        if (lane_id == 0) {
            float2 q = query[q_idx];
            qx = q.x; qy = q.y;
        }
        qx = __shfl_sync(mask, qx, 0);
        qy = __shfl_sync(mask, qy, 0);

        // Initialize threshold to "+inf"
        float max_dist = FLT_MAX;

        // Iterate over data in tiles; each tile is loaded into shared memory by the whole block
        for (int base = 0; base < data_count; base += tile_points_runtime) {
            int tile_size = data_count - base;
            if (tile_size > tile_points_runtime) tile_size = tile_points_runtime;

            // Cooperative load of the current tile into shared memory
            for (int t = threadIdx.x; t < tile_size; t += blockDim.x) {
                tile[t] = data[base + t];
            }
            __syncthreads(); // ensure tile is fully populated

            // Each warp processes all points in the tile, lanes stride by warp size
            for (int j = lane_id; j < tile_size; j += WARP_SIZE) {
                float2 p = tile[j];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = fmaf(dx, dx, dy * dy); // squared Euclidean distance

                // Filter by current threshold and only push when not in flushing state
                if (dist < max_dist && *cand_lock == 0) {
                    int pos = atomicAdd((int*)cand_count, 1);
                    if (pos < k) {
                        cand[pos].dist = dist;
                        cand[pos].idx  = base + j;
                    }
                }

                __syncwarp(mask);

                // If buffer is full, initiate a flush
                int need_flush = 0;
                if (lane_id == 0) {
                    need_flush = (*cand_count >= k) ? 1 : 0;
                    if (need_flush) *cand_lock = 1;
                }
                need_flush = __shfl_sync(mask, need_flush, 0);

                if (need_flush) {
                    flush_candidates_warp(res, cand, cand_count, cand_lock, k, max_dist, mask);
                }
            }

            __syncwarp(mask);

            // After finishing the tile, flush any remaining candidates
            int pending = 0;
            if (lane_id == 0) {
                pending = (*cand_count > 0) ? 1 : 0;
                if (pending) *cand_lock = 1;
            }
            pending = __shfl_sync(mask, pending, 0);
            if (pending) {
                flush_candidates_warp(res, cand, cand_count, cand_lock, k, max_dist, mask);
            }

            __syncthreads(); // ensure all warps are done with the tile before loading the next
        }

        // Write out the final sorted k-NN for this query to global memory
        // The result array is laid out so that result[q_idx * k + j] holds the j-th nearest neighbor
        for (int i = lane_id; i < k; i += WARP_SIZE) {
            int out_idx = q_idx * k + i;
            result[out_idx].first  = res[i].idx;
            result[out_idx].second = res[i].dist;
        }
    }
}

// Host-side launcher.
// Chooses reasonable defaults for block size and shared memory sizing.
// - Uses 32-thread warps, one warp per query, multiple warps per block.
// - Computes dynamic shared memory usage as:
//     tile_points * sizeof(float2) + warps_per_block * (2*k*sizeof(PairK) + 2*sizeof(int))
// - Attempts to opt-in to the maximum available shared memory per block if needed.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    // Query device shared memory limits
    int dev = 0;
    cudaGetDevice(&dev);

    int smem_default = 0;
    int smem_optin   = 0;
    cudaDeviceGetAttribute(&smem_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&smem_optin,   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (smem_optin <= 0) smem_optin = smem_default; // fallback

    // Decide warps per block (try larger first, then reduce if memory bound)
    // We'll try up to 4 warps per block for better latency hiding, then fall back.
    int warps_per_block_candidates[] = {4, 3, 2, 1};
    int chosen_warps = 1;
    int chosen_block_threads = 32;
    int chosen_tile_points = 512; // default; will be recalculated to fit shared memory
    size_t chosen_smem_bytes = 0;

    // Attempt to fit within opt-in shared memory limit (we will set the attribute accordingly)
    for (int w : warps_per_block_candidates) {
        // Per-warp memory: 2*k PairK (res + cand) + 2 ints (count + lock)
        size_t per_warp_pairs_bytes = size_t(2 * k) * sizeof(PairK);
        size_t per_warp_ctrl_bytes  = 2 * sizeof(int);
        size_t per_warp_total       = per_warp_pairs_bytes + per_warp_ctrl_bytes;

        // Use as much of smem_optin as possible, but reserve some bytes for safety
        size_t reserved = 0; // no extra reservation
        size_t rem = (smem_optin > reserved) ? (smem_optin - reserved) : 0;

        // Compute max tile points that fit after allocating per-warp buffers
        size_t total_warp_bytes = size_t(w) * per_warp_total;
        if (rem <= total_warp_bytes) {
            // Not enough memory for w warps; try fewer warps
            continue;
        }
        size_t tile_bytes = rem - total_warp_bytes;
        int tile_points = int(tile_bytes / sizeof(float2));
        if (tile_points < 128) {
            // Too small tile; try fewer warps
            continue;
        }

        // Cap tile size to a reasonable upper bound to avoid huge sorts and to keep occupancy
        if (tile_points > 8192) tile_points = 8192;

        // Compute actual shared memory bytes we will request
        size_t smem_bytes = size_t(tile_points) * sizeof(float2) + total_warp_bytes;

        chosen_warps = w;
        chosen_block_threads = w * WARP_SIZE;
        chosen_tile_points = tile_points;
        chosen_smem_bytes = smem_bytes;
        break;
    }

    // If we couldn't fit the configuration using opt-in size, fall back to default and recompute
    if (chosen_smem_bytes == 0) {
        for (int w : warps_per_block_candidates) {
            size_t per_warp_pairs_bytes = size_t(2 * k) * sizeof(PairK);
            size_t per_warp_ctrl_bytes  = 2 * sizeof(int);
            size_t total_warp_bytes     = size_t(w) * (per_warp_pairs_bytes + per_warp_ctrl_bytes);
            if (smem_default <= total_warp_bytes) continue;
            size_t tile_bytes = smem_default - total_warp_bytes;
            int tile_points = int(tile_bytes / sizeof(float2));
            if (tile_points < 128) continue;
            if (tile_points > 4096) tile_points = 4096;
            size_t smem_bytes = size_t(tile_points) * sizeof(float2) + total_warp_bytes;

            chosen_warps = w;
            chosen_block_threads = w * WARP_SIZE;
            chosen_tile_points = tile_points;
            chosen_smem_bytes = smem_bytes;
            break;
        }
    }

    // As a last-resort fallback, ensure there is at least one warp per block
    if (chosen_smem_bytes == 0) {
        chosen_warps = 1;
        chosen_block_threads = WARP_SIZE;
        // Use the largest tile that fits into default shared memory with one warp
        size_t per_warp_pairs_bytes = size_t(2 * k) * sizeof(PairK);
        size_t per_warp_ctrl_bytes  = 2 * sizeof(int);
        size_t total_warp_bytes     = per_warp_pairs_bytes + per_warp_ctrl_bytes;
        size_t tile_bytes = (smem_default > total_warp_bytes) ? (smem_default - total_warp_bytes) : 0;
        int tile_points = int(tile_bytes / sizeof(float2));
        if (tile_points < 32) tile_points = 32;
        if (tile_points > 2048) tile_points = 2048;
        chosen_tile_points = tile_points;
        chosen_smem_bytes = size_t(tile_points) * sizeof(float2) + total_warp_bytes;
    }

    // Opt-in to larger shared memory per block if needed and available
    if ((int)chosen_smem_bytes > smem_default && smem_optin >= (int)chosen_smem_bytes) {
        cudaFuncSetAttribute(knn_warp_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)chosen_smem_bytes);
    }

    // Grid size: assign approximately one warp per query, but also allow grid-stride looping
    int blocks = (query_count + chosen_warps - 1) / chosen_warps;
    if (blocks <= 0) blocks = 1;

    // Launch the kernel
    knn_warp_kernel<<<blocks, chosen_block_threads, chosen_smem_bytes>>>(
        query, query_count, data, data_count, result, k, chosen_tile_points
    );
}