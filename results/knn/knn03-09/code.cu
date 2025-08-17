#include <cuda_runtime.h>
#include <utility>

// Optimized CUDA k-NN (squared L2) for 2D points.
// - One warp (32 threads) processes one query.
// - The block loads data points in tiles into shared memory.
// - Each lane keeps L = k/32 local best candidates for the current tile in registers.
// - Warps merge the 32 local lists into a sorted buffer in shared memory.
// - Merge the sorted tile buffer with the warp's global best (size k) stored in shared memory.
// - Repeat for all tiles, then write the final k-NN results for each query.
//
// Notes:
// - k is a power of two in [32, 1024], so L is an integer in [1, 32].
// - Shared memory usage per block:
//     tile_points: TILE_POINTS * sizeof(float2)
//     per-warp buffers (dist+idx for each entry): 3 buffers * WARPS_PER_BLOCK * k * (sizeof(float)+sizeof(int))
//   With the parameters below (TILE_POINTS=4096, WARPS_PER_BLOCK=4), the maximum is ~128 KB for k=1024.
//
// Compile with a recent CUDA toolkit. Use cudaFuncSetAttribute to opt-in to large shared memory if necessary.

namespace {
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;       // 4 warps -> 128 threads per block
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int TILE_POINTS = 4096;        // Number of data points cached per block iteration
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
}

// Insert a candidate (dist, idx) into an ascending-sorted fixed-size array (length L).
// If dist is not smaller than the current worst (last entry), it is discarded.
// Arrays top_dist[0..L-1], top_idx[0..L-1] are always kept ascending.
__device__ __forceinline__
void insert_into_top_ascending(float dist, int idx, float* top_dist, int* top_idx, int L)
{
    if (L <= 0) return;
    if (dist >= top_dist[L - 1]) return;

    // Find insertion position by shifting elements to the right.
    int j = L - 1;
    // Move larger elements up to make room for 'dist'
    for (; j > 0 && top_dist[j - 1] > dist; --j) {
        top_dist[j] = top_dist[j - 1];
        top_idx[j]  = top_idx[j - 1];
    }
    top_dist[j] = dist;
    top_idx[j]  = idx;
}

// Merge two ascending-sorted sequences from src into dst.
// left:  src_dist[left_start .. left_start + left_len - 1]
// right: src_dist[right_start .. right_start + right_len - 1]
// Output ascending to dst_dist[dst_start .. dst_start + (left_len+right_len) - 1]
__device__ __forceinline__
void merge_two_sequences_ascending(
    const float* __restrict__ src_dist, const int* __restrict__ src_idx,
    float* __restrict__ dst_dist,       int* __restrict__ dst_idx,
    int left_start, int left_len, int right_start, int right_len, int dst_start)
{
    int i = 0, j = 0, o = 0;
    while (i < left_len && j < right_len) {
        float dl = src_dist[left_start + i];
        float dr = src_dist[right_start + j];
        if (dl <= dr) {
            dst_dist[dst_start + o] = dl;
            dst_idx [dst_start + o] = src_idx[left_start + i];
            ++i; ++o;
        } else {
            dst_dist[dst_start + o] = dr;
            dst_idx [dst_start + o] = src_idx[right_start + j];
            ++j; ++o;
        }
    }
    // Copy remaining items (at most one of the loops executes)
    while (i < left_len) {
        dst_dist[dst_start + o] = src_dist[left_start + i];
        dst_idx [dst_start + o] = src_idx[left_start + i];
        ++i; ++o;
    }
    while (j < right_len) {
        dst_dist[dst_start + o] = src_dist[right_start + j];
        dst_idx [dst_start + o] = src_idx[right_start + j];
        ++j; ++o;
    }
}

// Kernel implementing k-NN for 2D points with squared L2 distances.
__global__ void knn_kernel_2d(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    std::pair<int, float>* __restrict__ result,
    int k)
{
    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_ptr = smem_raw;

    // Layout of shared memory:
    // [ tile_points (float2) | gbest_dist (float) | gbest_idx (int) | tA_dist (float) | tA_idx (int) | tB_dist (float) | tB_idx (int) ]
    float2* tile_points = reinterpret_cast<float2*>(smem_ptr);
    smem_ptr += sizeof(float2) * TILE_POINTS;

    float* gbest_dist_base = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += sizeof(float) * WARPS_PER_BLOCK * k;

    int* gbest_idx_base = reinterpret_cast<int*>(smem_ptr);
    smem_ptr += sizeof(int) * WARPS_PER_BLOCK * k;

    float* tA_dist_base = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += sizeof(float) * WARPS_PER_BLOCK * k;

    int* tA_idx_base = reinterpret_cast<int*>(smem_ptr);
    smem_ptr += sizeof(int) * WARPS_PER_BLOCK * k;

    float* tB_dist_base = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += sizeof(float) * WARPS_PER_BLOCK * k;

    int* tB_idx_base = reinterpret_cast<int*>(smem_ptr);
    // smem_ptr += sizeof(int) * WARPS_PER_BLOCK * k; // not used further

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_in_block = tid >> 5;                        // warp index within block [0..WARPS_PER_BLOCK-1]
    const int warp_global   = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    const bool active       = (warp_global < query_count);
    const int L = k / WARP_SIZE;                               // per-lane capacity

    // Per-warp slices of shared memory arrays
    const int warp_offset = warp_in_block * k;
    float* gbest_dist = gbest_dist_base + warp_offset;
    int*   gbest_idx  = gbest_idx_base  + warp_offset;
    float* tA_dist    = tA_dist_base    + warp_offset;
    int*   tA_idx     = tA_idx_base     + warp_offset;
    float* tB_dist    = tB_dist_base    + warp_offset;
    int*   tB_idx     = tB_idx_base     + warp_offset;

    // Initialize the warp's global best buffers to +inf and -1
    const float INF = CUDART_INF_F;
    for (int i = lane; i < k; i += WARP_SIZE) {
        gbest_dist[i] = INF;
        gbest_idx [i] = -1;
    }
    __syncwarp();

    // Load query point (lane 0 loads from global, broadcast to warp)
    float qx = 0.f, qy = 0.f;
    if (active && lane == 0) {
        float2 q = query[warp_global];
        qx = q.x; qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    // Iterate over data in tiles
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        int tile_n = data_count - base;
        if (tile_n > TILE_POINTS) tile_n = TILE_POINTS;

        // Block-wide load of tile data into shared memory
        for (int t = tid; t < tile_n; t += blockDim.x) {
            tile_points[t] = data[base + t];
        }
        __syncthreads(); // ensure tile is loaded before any warp uses it

        // Each lane initializes its local top-L list in registers
        float local_dist[32];
        int   local_idx [32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i < L) {
                local_dist[i] = INF;
                local_idx [i] = -1;
            }
        }

        // Compute distances for this warp's query vs. all points in the tile;
        // each lane strides over the tile by WARP_SIZE.
        if (active) {
            for (int i = lane; i < tile_n; i += WARP_SIZE) {
                float2 p = tile_points[i];
                float dx = p.x - qx;
                float dy = p.y - qy;
                float d2 = fmaf(dx, dx, dy * dy); // squared L2 distance
                int   idx = base + i;
                insert_into_top_ascending(d2, idx, local_dist, local_idx, L);
            }
        }

        // Store each lane's local top-L into the warp's tile buffer A (concatenate lanes)
        if (active) {
            const int lane_out_start = lane * L;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                if (i < L) {
                    tA_dist[lane_out_start + i] = local_dist[i];
                    tA_idx [lane_out_start + i] = local_idx [i];
                }
            }
        }
        __syncwarp(); // ensure all lanes wrote their L candidates

        // Merge 32 ascending sequences of length L into one ascending sequence of length k.
        // We perform log2(32)=5 merge stages, alternating between tA and tB as src/dst.
        // At stage s: sequence length = L * (1 << s); there are pair_count = 32 >> (s+1) merges.
        if (active) {
            bool src_is_A = true;
            int stages = 0;
            // Handle L possibly equal to 0 (shouldn't happen as k>=32), but guard anyway
            if (L > 0) {
                for (int s = 0; s < 5; ++s) {
                    int seq_len = L << s;
                    int pair_count = 32 >> (s + 1);
                    if (pair_count <= 0) break;

                    if (lane < pair_count) {
                        int left_start  = (lane * 2) * seq_len;
                        int right_start = left_start + seq_len;
                        int dst_start   = left_start;
                        if (src_is_A) {
                            merge_two_sequences_ascending(
                                tA_dist, tA_idx,
                                tB_dist, tB_idx,
                                left_start,  seq_len,
                                right_start, seq_len,
                                dst_start);
                        } else {
                            merge_two_sequences_ascending(
                                tB_dist, tB_idx,
                                tA_dist, tA_idx,
                                left_start,  seq_len,
                                right_start, seq_len,
                                dst_start);
                        }
                    }
                    __syncwarp();
                    src_is_A = !src_is_A;
                    ++stages;
                }
            }

            // After merging, the final sorted tile buffer is in the source of the last stage toggle.
            // If stages is even: src_is_A == true => tA holds final; else tB holds final.
            float* tile_sorted_dist = src_is_A ? tA_dist : tB_dist;
            int*   tile_sorted_idx  = src_is_A ? tA_idx  : tB_idx;

            // Merge the sorted tile candidates (k) with the warp's global best (k),
            // keeping the k smallest. Let lane 0 perform the sequential 2-way merge,
            // then all lanes copy back to gbest in parallel.
            float* merge_dst_dist = src_is_A ? tB_dist : tA_dist; // use the other buffer as temporary destination
            int*   merge_dst_idx  = src_is_A ? tB_idx  : tA_idx;

            if (lane == 0) {
                int a = 0, b = 0, o = 0;
                while (o < k) {
                    float da = gbest_dist[a];
                    float db = tile_sorted_dist[b];
                    // Choose the smaller value, guarding index bounds.
                    bool take_a = (b >= k) || (a < k && da <= db);
                    if (take_a) {
                        merge_dst_dist[o] = da;
                        merge_dst_idx [o] = gbest_idx[a];
                        ++a; ++o;
                    } else {
                        merge_dst_dist[o] = db;
                        merge_dst_idx [o] = tile_sorted_idx[b];
                        ++b; ++o;
                    }
                }
            }
            __syncwarp();

            // Copy merged results back to global best cooperatively (strided by lane)
            for (int i = lane; i < k; i += WARP_SIZE) {
                gbest_dist[i] = merge_dst_dist[i];
                gbest_idx [i] = merge_dst_idx [i];
            }
        }

        __syncthreads(); // ensure all warps finished with the tile before loading the next one
    }

    // Write final results for active queries.
    if (active) {
        const int out_base = warp_global * k;
        for (int i = lane; i < k; i += WARP_SIZE) {
            std::pair<int, float> p;
            p.first  = gbest_idx[i];
            p.second = gbest_dist[i];
            result[out_base + i] = p;
        }
    }
}

// Host API: run_knn
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Grid: one warp per query.
    dim3 block(BLOCK_THREADS, 1, 1);
    dim3 grid((query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    // Dynamic shared memory size:
    // tile: TILE_POINTS * sizeof(float2)
    // per-warp buffers: 3 buffers * WARPS_PER_BLOCK * k * (sizeof(float) + sizeof(int))
    size_t smem_bytes = sizeof(float2) * size_t(TILE_POINTS)
                      + size_t(24) * size_t(WARPS_PER_BLOCK) * size_t(k); // 24 = 6 arrays * 4 bytes each

    // Opt-in larger dynamic shared memory if needed (for A100/H100 this is available).
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Set the kernel attribute to allow the required dynamic shared memory.
    // On recent devices, sharedMemPerBlockOptin typically >= 163840 bytes (160 KB).
    int smem_bytes_int = static_cast<int>(smem_bytes);
    int max_optin = static_cast<int>(prop.sharedMemPerBlockOptin);
    if (smem_bytes_int > max_optin) {
        // If requested exceeds opt-in maximum, clamp to maximum to avoid an error.
        // Kernel launch will fail if smem_bytes > configured limit, so choose parameters accordingly.
        smem_bytes_int = max_optin;
    }
    cudaFuncSetAttribute(
        knn_kernel_2d,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes_int
    );

    // Launch the kernel on the default stream.
    knn_kernel_2d<<<grid, block, smem_bytes>>>(
        query, query_count,
        data,  data_count,
        result, k
    );
}