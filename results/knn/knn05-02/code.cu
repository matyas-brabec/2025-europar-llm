#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// Simple POD type with the same layout as std::pair<int,float>
// so we can safely reinterpret_cast on the host side.
struct DevicePair {
    int   first;
    float second;
};

// Entry stored in per-query buffers (shared memory)
struct KnnEntry {
    float dist;
    int   index;
};

static constexpr int WARP_SIZE          = 32;
static constexpr int WARPS_PER_BLOCK    = 8;      // 8 warps (256 threads) per block
static constexpr int THREADS_PER_BLOCK  = WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int TILE_POINTS        = 2048;   // Number of data points cached per block in shared memory

// Warp-synchronous bitonic sort over 2*k KnnEntry items stored in shared memory.
// The layout is:
//   warp_buf[0 .. k-1]         : current intermediate top-k
//   warp_buf[k .. 2*k-1]       : candidate buffer (only cand_count entries valid, rest filled with +INF)
// After the call, the first k entries of warp_buf contain the k smallest distances (ascending order).
// The remaining entries are not used until the candidate buffer is filled again.
__device__ __forceinline__
void bitonic_merge_topk_candidates(KnnEntry* warp_buf, int k, int cand_count, int lane_id)
{
    unsigned full_mask = 0xffffffffu;

    KnnEntry* topk = warp_buf;
    KnnEntry* cand = warp_buf + k;

    // Fill unused candidate slots with +INF so that we always sort exactly 2*k elements.
    // Only threads in this warp touch these locations.
    for (int i = lane_id + cand_count; i < k; i += WARP_SIZE) {
        cand[i].dist  = CUDART_INF_F;
        cand[i].index = -1;
    }
    __syncwarp(full_mask);

    const int total = 2 * k;        // total number of elements to sort (power of two because k is power of two)
    // Standard in-place bitonic sort network over total elements.
    for (int size = 2; size <= total; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int idx = lane_id; idx < total; idx += WARP_SIZE) {
                int ixj = idx ^ stride;
                if (ixj > idx) {
                    bool ascending = ((idx & size) == 0);
                    KnnEntry a = warp_buf[idx];
                    KnnEntry b = warp_buf[ixj];

                    bool swap = ascending ? (a.dist > b.dist) : (a.dist < b.dist);
                    if (swap) {
                        warp_buf[idx] = b;
                        warp_buf[ixj] = a;
                    }
                }
            }
            __syncwarp(full_mask);
        }
    }

    // After sorting, topk[0..k-1] contain the k smallest distances in ascending order.
    // Caller will update its threshold (max_topk_dist) from topk[k-1].dist.
}

// Each warp processes exactly one query point.
// The block cooperatively loads data points in tiles into shared memory.
__global__
void knn_kernel_2d(const float2* __restrict__ query,
                   int                       query_count,
                   const float2* __restrict__ data,
                   int                       data_count,
                   DevicePair* __restrict__  result,
                   int                       k)
{
    extern __shared__ unsigned char shared_raw[];
    float2*  sh_points = reinterpret_cast<float2*>(shared_raw);
    KnnEntry* sh_entries = reinterpret_cast<KnnEntry*>(sh_points + TILE_POINTS);

    const int lane_id        = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id        = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    const int query_idx      = global_warp_id;       // one warp per query
    const bool active        = (query_idx < query_count);

    const unsigned full_mask = 0xffffffffu;

    // Per-warp shared-memory region for intermediate top-k and candidate buffer.
    // Layout for this warp:
    //   warp_buf[0 .. k-1]       : current top-k
    //   warp_buf[k .. 2*k-1]     : candidate buffer
    KnnEntry* warp_buf = sh_entries + warp_id * (2 * k);
    KnnEntry* topk     = warp_buf;
    KnnEntry* cand     = warp_buf + k;

    // Initialize per-warp top-k and candidate buffer.
    if (active) {
        for (int i = lane_id; i < 2 * k; i += WARP_SIZE) {
            warp_buf[i].dist  = CUDART_INF_F;
            warp_buf[i].index = -1;
        }
    }
    // All threads must participate in block-level barrier.
    __syncthreads();

    // Load query point for this warp, broadcast via shuffle.
    float2 q;
    if (active) {
        if (lane_id == 0) {
            q = query[query_idx];
        }
        q.x = __shfl_sync(full_mask, q.x, 0);
        q.y = __shfl_sync(full_mask, q.y, 0);
    }

    float max_topk_dist = CUDART_INF_F; // current k-th (worst) distance in top-k
    int   cand_count    = 0;            // number of valid candidates in cand[0 .. cand_count-1]

    // Process data in tiles cached in shared memory.
    for (int base = 0; base < data_count; base += TILE_POINTS) {
        int tile_size = data_count - base;
        if (tile_size > TILE_POINTS) tile_size = TILE_POINTS;

        // Cooperative load of this tile into shared memory.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            sh_points[i] = data[base + i];
        }
        __syncthreads();

        if (active) {
            // Each warp processes the cached tile for its own query point.
            for (int i = lane_id; i < tile_size; i += WARP_SIZE) {
                float2 pt = sh_points[i];
                float dx = pt.x - q.x;
                float dy = pt.y - q.y;
                float dist = dx * dx + dy * dy;
                int   data_index = base + i;

                // Try to insert this point into the candidate buffer (if closer than threshold).
                // If candidate buffer would overflow, merge it with current top-k, update threshold,
                // and re-evaluate this point.
                while (true) {
                    bool is_candidate = (dist < max_topk_dist);
                    unsigned mask = __ballot_sync(full_mask, is_candidate);
                    int num_new = __popc(mask);

                    if (num_new == 0) {
                        // No thread in this warp wants to insert this point; done.
                        break;
                    }

                    // If buffer would overflow, merge current candidates with top-k first.
                    if (cand_count + num_new > k) {
                        bitonic_merge_topk_candidates(warp_buf, k, cand_count, lane_id);

                        // Update threshold from the k-th (worst) neighbor after merge.
                        float new_thresh = CUDART_INF_F;
                        if (lane_id == 0) {
                            new_thresh = topk[k - 1].dist;
                        }
                        max_topk_dist = __shfl_sync(full_mask, new_thresh, 0);

                        cand_count = 0;
                        continue; // Re-evaluate this point with the new threshold.
                    }

                    // We can insert all new candidates into the buffer.
                    int prefix = __popc(mask & ((1u << lane_id) - 1));
                    if (is_candidate) {
                        int pos = cand_count + prefix;
                        cand[pos].dist  = dist;
                        cand[pos].index = data_index;
                    }
                    if (lane_id == 0) {
                        cand_count += num_new;
                    }
                    cand_count = __shfl_sync(full_mask, cand_count, 0);
                    break;
                }
            }
        }

        __syncthreads();
    }

    // After processing all tiles, merge any remaining candidates with top-k.
    if (active) {
        if (cand_count > 0) {
            bitonic_merge_topk_candidates(warp_buf, k, cand_count, lane_id);
            float new_thresh = CUDART_INF_F;
            if (lane_id == 0) {
                new_thresh = topk[k - 1].dist;
            }
            max_topk_dist = __shfl_sync(full_mask, new_thresh, 0);
        }

        // Write final top-k results for this query to global memory.
        // topk[0..k-1] is sorted by increasing distance.
        for (int i = lane_id; i < k; i += WARP_SIZE) {
            int out_idx = query_idx * k + i;
            result[out_idx].first  = topk[i].index;
            result[out_idx].second = topk[i].dist;
        }
    }
}

// Host-side interface.
// All pointers are assumed to be device pointers obtained via cudaMalloc.
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Reinterpret std::pair<int,float> as a POD struct with identical layout.
    DevicePair *d_result = reinterpret_cast<DevicePair*>(result);

    dim3 block_dim(THREADS_PER_BLOCK);
    int  num_blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid_dim(num_blocks);

    // Shared memory size:
    //   - TILE_POINTS float2 points for the data tile
    //   - 2 * k KnnEntry per warp (top-k + candidate buffer)
    size_t shared_bytes =
        sizeof(float2) * TILE_POINTS +
        sizeof(KnnEntry) * 2 * k * WARPS_PER_BLOCK;

    // Opt-in to the required amount of dynamic shared memory.
    cudaFuncSetAttribute(
        knn_kernel_2d,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));

    knn_kernel_2d<<<grid_dim, block_dim, shared_bytes>>>(
        query, query_count,
        data, data_count,
        d_result,
        k);

    // Ensure completion before returning.
    cudaDeviceSynchronize();
}