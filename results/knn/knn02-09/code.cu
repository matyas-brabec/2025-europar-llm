#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>
#include <algorithm>
#include <cfloat>

// This implementation assigns one warp (32 threads) per query.
// Each warp maintains its K-best list in shared memory (indices and distances).
// The dataset is processed in tiles cached in shared memory. Each tile is loaded
// cooperatively by the entire block, then each warp scans the tile against its query and
// updates its private K-best list. Updates are done warp-synchronously:
//   - For each group of 32 candidates (one per lane), we repeatedly insert the globally
//     smallest candidate among the warp into the K-best list while it improves the list,
//     recomputing the current worst element ("tau") cooperatively after each insertion.
// The K-best list is maintained unsorted during scanning; after all tiles are processed,
// each warp performs a parallel bitonic sort on its K elements (ascending by distance)
// and writes results to the output array.
// The code uses only registers and shared memory (no extra global allocations).
// It targets large data_count (up to millions) and large query_count (thousands).
// K is a power of two in [32, 1024].

static __device__ __forceinline__ float sq_l2_distance(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // Use FMA to compute dx*dx + dy*dy
    return fmaf(dx, dx, dy * dy);
}

// Warp-wide argmin over 32 lanes: input 'val' and implicit lane id.
// Outputs the minimum value and the lane that owns it, broadcast to all lanes.
static __device__ __forceinline__ void warp_argmin32(float val, int& minLaneOut, float& minValOut) {
    unsigned mask = 0xFFFFFFFFu;
    float v = val;
    int lane = threadIdx.x & 31;
    int id = lane;
    // Tree reduction for minimum value; tie-break with smaller lane id.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float v2 = __shfl_down_sync(mask, v, offset);
        int id2 = __shfl_down_sync(mask, id, offset);
        if (v2 < v || (v2 == v && id2 < id)) {
            v = v2;
            id = id2;
        }
    }
    minValOut = __shfl_sync(mask, v, 0);
    minLaneOut = __shfl_sync(mask, id, 0);
}

// Recompute the current maximum distance (tau) and its position within the per-warp K-best list.
// All 32 lanes of the warp cooperate: each lane scans a consecutive chunk of size (k/32).
static __device__ __forceinline__ void warp_recompute_tau_max(const float* __restrict__ top_dist,
                                                              int k, float& tau_out, int& max_pos_out) {
    const int lane = threadIdx.x & 31;
    const int chunk = k >> 5; // k/32, valid because k is power-of-two >= 32
    const int start = lane * chunk;

    float local_max = -CUDART_INF_F;
    int local_pos = start;
    // Scan this lane's chunk
    #pragma unroll
    for (int i = 0; i < chunk; ++i) {
        float v = top_dist[start + i];
        if (v > local_max) {
            local_max = v;
            local_pos = start + i;
        }
    }

    // Warp-wide argmax over the 32 local maxima; tie-break with larger index.
    unsigned mask = 0xFFFFFFFFu;
    float val = local_max;
    int pos = local_pos;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float v2 = __shfl_down_sync(mask, val, offset);
        int p2 = __shfl_down_sync(mask, pos, offset);
        if (v2 > val || (v2 == val && p2 > pos)) {
            val = v2;
            pos = p2;
        }
    }
    tau_out = __shfl_sync(mask, val, 0);
    max_pos_out = __shfl_sync(mask, pos, 0);
}

// Parallel bitonic sort (ascending by distance) over the per-warp K elements stored in shared memory.
// The 32 lanes cooperatively perform compare-exchange passes on K elements.
// For each stride, pairs (i, i^stride) form disjoint matches; to avoid races, only the owner with i < ixj updates both elements.
static __device__ __forceinline__ void warp_bitonic_sort_topk(float* __restrict__ top_dist,
                                                              int* __restrict__ top_idx, int k) {
    const int lane = threadIdx.x & 31;
    const int chunk = k >> 5; // k/32
    // Outer stage size: 2,4,8,...,k
    for (int size = 2; size <= k; size <<= 1) {
        // Inner stride: size/2, size/4, ..., 1
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int i_begin = lane * chunk;
            int i_end = i_begin + chunk;
            for (int i = i_begin; i < i_end; ++i) {
                int ixj = i ^ stride;
                if (ixj > i && ixj < k) {
                    bool up = ((i & size) == 0); // desired order for this subsequence
                    float di = top_dist[i];
                    float dj = top_dist[ixj];
                    int ii = top_idx[i];
                    int ij = top_idx[ixj];
                    // If up==true: ensure di <= dj; else ensure di >= dj
                    bool doSwap = (di > dj) == up;
                    if (doSwap) {
                        top_dist[i] = dj;
                        top_dist[ixj] = di;
                        top_idx[i] = ij;
                        top_idx[ixj] = ii;
                    }
                }
            }
            __syncwarp();
        }
    }
}

__global__ void knn_kernel(const float2* __restrict__ query, int query_count,
                           const float2* __restrict__ data, int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k, int tile_size)
{
    // Thread and warp identifiers
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;           // 0..warps_per_block-1
    const int warps_per_block = blockDim.x >> 5;
    const int query_idx = blockIdx.x * warps_per_block + warp_in_block;
    const bool warp_active = (query_idx < query_count);

    // Shared memory layout:
    // [ warps_per_block * k ints (indices) ][ warps_per_block * k floats (distances) ][ tile_size float2 points ]
    extern __shared__ unsigned char smem_raw[];
    int* sm_idx = reinterpret_cast<int*>(smem_raw);
    float* sm_dist = reinterpret_cast<float*>(sm_idx + warps_per_block * k);
    float2* sm_tile = reinterpret_cast<float2*>(sm_dist + warps_per_block * k);

    // Pointers to this warp's private K-best buffers in shared memory
    int* top_idx = sm_idx + warp_in_block * k;
    float* top_dist = sm_dist + warp_in_block * k;

    // Initialize K-best buffers for active warps
    if (warp_active) {
        const int chunk = k >> 5; // k/32
        const int start = lane * chunk;
        #pragma unroll
        for (int i = 0; i < chunk; ++i) {
            top_dist[start + i] = CUDART_INF_F;
            top_idx[start + i] = -1;
        }
        __syncwarp();
    }

    // Load the query point and broadcast to all lanes in the warp
    float2 qpt = make_float2(0.f, 0.f);
    if (warp_active) {
        if (lane == 0) qpt = query[query_idx];
        unsigned mask = 0xFFFFFFFFu;
        qpt.x = __shfl_sync(mask, qpt.x, 0);
        qpt.y = __shfl_sync(mask, qpt.y, 0);
    }

    // Current worst distance (tau) and its position in top_k
    float tau = CUDART_INF_F;
    int max_pos = 0;
    if (warp_active) {
        // Compute initial tau and max_pos (will be +inf, but gives a valid position for updates)
        warp_recompute_tau_max(top_dist, k, tau, max_pos);
    }

    // Process the dataset in tiles cached in shared memory
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_size) {
        const int tile_count = min(tile_size, data_count - tile_start);

        // All threads in the block cooperatively load the tile
        for (int t = threadIdx.x; t < tile_count; t += blockDim.x) {
            sm_tile[t] = data[tile_start + t];
        }
        __syncthreads();

        // Each active warp scans the tile with 32 candidates at a time
        if (warp_active) {
            for (int base = 0; base < tile_count; base += 32) {
                const int idx_in_tile = base + lane;
                // Compute candidate distance for this lane (or INF if out-of-range)
                float cand_dist = CUDART_INF_F;
                int cand_gidx = -1;
                if (idx_in_tile < tile_count) {
                    float2 p = sm_tile[idx_in_tile];
                    cand_dist = sq_l2_distance(qpt, p);
                    cand_gidx = tile_start + idx_in_tile;
                }

                // Repeatedly select and insert the smallest candidate among the 32 lanes
                while (true) {
                    float dmin;
                    int minLane;
                    warp_argmin32(cand_dist, minLane, dmin);
                    // If the smallest candidate is not better than tau, none of the 32 are
                    if (!(dmin < tau)) break;

                    // The owning lane inserts its candidate into the K-best at position 'max_pos'
                    if (lane == minLane) {
                        top_dist[max_pos] = dmin;
                        top_idx[max_pos] = cand_gidx;
                        // Mark this lane's candidate as consumed so it won't be selected again
                        cand_dist = CUDART_INF_F;
                    }
                    __syncwarp();

                    // Recompute tau (the current worst in top_k) cooperatively
                    warp_recompute_tau_max(top_dist, k, tau, max_pos);
                    __syncwarp();
                } // while there are improving candidates in this group
            } // for base
            __syncwarp();
        } // if warp_active

        __syncthreads(); // Ensure all warps completed before reusing sm_tile
    } // tile loop

    // Sort the K-best list (ascending by distance) and write results
    if (warp_active) {
        warp_bitonic_sort_topk(top_dist, top_idx, k);
        __syncwarp();

        // Write out sorted results: each lane writes strided positions
        const int out_base = query_idx * k;
        for (int j = lane; j < k; j += 32) {
            // Store (index, distance) pair
            result[out_base + j].first = top_idx[j];
            result[out_base + j].second = top_dist[j];
        }
    }
}

// Host-side launcher. It computes a suitable configuration (warps per block and tile size),
// sets the kernel's dynamic shared memory attribute as needed, and launches.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Prefer 8 warps per block (256 threads), fallback to fewer warps if shared memory is tight.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    int max_dyn_shmem = 0;
    // Hopper/Ampere support opt-in dynamic shared memory sizes; fall back to legacy if needed
    /// @FIXED
    /// cudaDeviceGetAttribute(&max_dyn_shmem, cudaDevAttrMaxDynamicSharedMemoryPerBlockOptin, device);
    cudaDeviceGetAttribute(&max_dyn_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_dyn_shmem == 0) {
        max_dyn_shmem = prop.sharedMemPerBlockOptin ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock;
    }

    // Choose warps per block and tile size to fit into available dynamic shared memory.
    int warps_per_block = 0;
    int threads_per_block = 0;
    int tile_size = 0;

    for (int w = 8; w >= 1; w >>= 1) {
        const size_t knn_bytes = static_cast<size_t>(w) * static_cast<size_t>(k) * (sizeof(int) + sizeof(float));
        int max_tile_points = static_cast<int>((max_dyn_shmem - (int)knn_bytes) / sizeof(float2));
        if (max_tile_points <= 0) continue;

        // Cap the tile size for good cache behavior; enforce a multiple of 32
        int candidate_tile = max_tile_points;
        if (candidate_tile > 4096) candidate_tile = 4096;
        candidate_tile = (candidate_tile / 32) * 32;
        if (candidate_tile <= 0) continue;

        warps_per_block = w;
        threads_per_block = w * 32;
        tile_size = candidate_tile;
        break;
    }

    if (warps_per_block == 0) {
        // Extremely unlikely fallback
        warps_per_block = 1;
        threads_per_block = 32;
        tile_size = 32;
    }

    // Compute required dynamic shared memory size
    const size_t smem_bytes = static_cast<size_t>(warps_per_block) * static_cast<size_t>(k) * (sizeof(int) + sizeof(float))
                            + static_cast<size_t>(tile_size) * sizeof(float2);

    // Allow the kernel to use the requested dynamic shared memory size (up to the device maximum)
    int optin_bytes = static_cast<int>(std::min(smem_bytes, static_cast<size_t>(max_dyn_shmem)));
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin_bytes);

    // Launch configuration
    dim3 block(threads_per_block, 1, 1);
    dim3 grid((query_count + warps_per_block - 1) / warps_per_block, 1, 1);

    // Launch the kernel
    knn_kernel<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k, tile_size);
}
