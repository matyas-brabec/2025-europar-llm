#include <cuda_runtime.h>
#include <utility>

// Optimized k-NN (k-nearest neighbors) for 2D points with squared Euclidean distances.
// Strategy:
// - One CUDA block handles one query point.
// - The block streams through all data points in tiles.
// - Each tile, threads compute distances and collect "survivors" whose distances are <= current kth distance (threshold).
// - Survivors are merged with the current top-k using a shared-memory bitonic sort on (top_k + survivors) elements.
// - Only when there are survivors do we perform a merge; otherwise, we skip sorting to minimize overhead.
// - k is a power of two between 32 and 1024 inclusive, data_count >= k. Arrays are in device memory.
// - No extra device memory is allocated; only shared memory is used.
//
// Notes on performance/tuning for A100/H100:
// - BLOCK_DIM = 256 and ITEMS_PER_THREAD = 4 produce TILE_SIZE = 1024, which keeps the sort size bounded at <= 2048 elements,
//   yielding a 16KB shared buffer (distances + indices), allowing high occupancy with 96KB per SM.
// - Warp-aggregated atomics are used to pack survivors efficiently into the shared buffer.
// - Sorting is bitonic in shared memory, padded to the next power-of-two with +INF sentinels as needed.

static inline int next_pow2_host(int x) {
    // Assumes x > 0
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

__device__ inline int next_pow2_device(int x) {
    // Assumes x > 0
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

// Warp-aggregated append: packs items from a warp into a shared buffer using one atomicAdd per warp.
// Returns the write position for this thread if pred is true, or -1 otherwise.
__device__ __forceinline__ int warp_aggregated_append(bool pred, int* counter) {
    unsigned int mask = __ballot_sync(0xffffffffu, pred);
    int count = __popc(mask);
    int lane = threadIdx.x & 31;
    int pos = -1;
    if (count) {
        int leader = __ffs(mask) - 1; // first active lane in warp
        int base = 0;
        if (lane == leader) {
            base = atomicAdd(counter, count);
        }
        // Broadcast base to all lanes in the warp that are active in this ballot
        base = __shfl_sync(mask, base, leader);
        int prefix = __popc(mask & ((1u << lane) - 1));
        if (pred) {
            pos = base + prefix;
        }
    }
    return pos;
}

// In-place bitonic sort on pairs (vals, idx) of length Npow2 (power-of-two).
// We assume that vals[nActive..Npow2-1] are padded with +INF and idx with -1 so that sorting correctness holds.
// Sort ascending by vals.
__device__ inline void bitonic_sort_pairs(float* vals, int* idx, int nPow2) {
    // Classic bitonic sort network operating entirely in shared memory.
    for (int size = 2; size <= nPow2; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < nPow2; i += blockDim.x) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    bool up = ((i & size) == 0);
                    float vi = vals[i];
                    float vj = vals[ixj];
                    int ii = idx[i];
                    int ij = idx[ixj];
                    // For ascending sort in the "up" region:
                    // - if up == true, ensure vals[i] <= vals[ixj]
                    // - if up == false, ensure vals[i] >= vals[ixj]
                    bool do_swap = (up ? (vi > vj) : (vi < vj));
                    if (do_swap) {
                        vals[i] = vj;
                        vals[ixj] = vi;
                        idx[i] = ij;
                        idx[ixj] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<int BLOCK_DIM, int ITEMS_PER_THREAD>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           std::pair<int, float>* __restrict__ result,
                           int k) {
    // Shared memory layout (dynamic):
    // [0 .. SMEM_ELEMS-1] floats for distances
    // [SMEM_ELEMS .. 2*SMEM_ELEMS-1] ints for indices
    extern __shared__ unsigned char smem[];
    // Capacity of shared arrays in elements = next_pow2(k + TILE_SIZE)
    const int TILE_SIZE = BLOCK_DIM * ITEMS_PER_THREAD;
    const int SMEM_ELEMS = next_pow2_device(k + TILE_SIZE);

    float* sh_dists = reinterpret_cast<float*>(smem);
    int* sh_index = reinterpret_cast<int*>(sh_dists + SMEM_ELEMS);

    __shared__ int topk_len;       // number of valid items currently in top-k (<= k)
    __shared__ float threshold;    // current k-th smallest distance (worst in top-k); +INF if topk_len < k

    const int qid = blockIdx.x;
    if (qid >= query_count) return;

    // Load query into registers
    float2 q = query[qid];
    const float qx = q.x;
    const float qy = q.y;

    // Initialize top-k state
    if (threadIdx.x == 0) {
        topk_len = 0;
        threshold = CUDART_INF_F; // No threshold until we have at least k items
    }
    __syncthreads();

    // Stream across the dataset in tiles
    for (int base = 0; base < data_count; base += TILE_SIZE) {
        // Determine the bounds of this tile
        int tile_end = base + TILE_SIZE;
        if (tile_end > data_count) tile_end = data_count;

        // Reset survivors counter
        __shared__ int survivors_count;
        if (threadIdx.x == 0) survivors_count = 0;
        __syncthreads();

        // Snapshot current top-k state for use within this tile
        const int curr_topk_len = topk_len;
        const float curr_threshold = threshold;

        // Process the tile: each thread handles multiple items with a stride of BLOCK_DIM
        for (int idx = base + threadIdx.x; idx < tile_end; idx += BLOCK_DIM) {
            // Load point and compute squared L2 distance
            float2 p = data[idx];
            float dx = p.x - qx;
            float dy = p.y - qy;
            float dist = fmaf(dy, dy, dx * dx);

            // Selection predicate:
            // - If we don't have k items yet, keep everything to fill the top-k.
            // - Otherwise, keep only candidates not worse than current threshold.
            bool keep = (curr_topk_len < k) || (dist <= curr_threshold);

            // Warp-aggregated append into survivors region [curr_topk_len .. curr_topk_len + survivors_count)
            int pos = warp_aggregated_append(keep, &survivors_count);
            if (pos >= 0) {
                int write_idx = curr_topk_len + pos;
                sh_dists[write_idx] = dist;
                sh_index[write_idx] = idx;
            }
        }
        __syncthreads();

        // If there are survivors, merge them with current top-k using bitonic sort
        int ns = topk_len + survivors_count; // total active candidates to consider
        if (ns > 0) {
            // For the first tile(s), when topk_len == 0, survivors_count == tile size; this builds initial top-k.
            // Pad up to next power-of-two for the sort with +INF sentinels.
            int nPow2 = next_pow2_device(ns);
            for (int i = threadIdx.x + ns; i < nPow2; i += BLOCK_DIM) {
                sh_dists[i] = CUDART_INF_F;
                sh_index[i] = -1;
            }
            __syncthreads();

            // Note: The existing top-k (if any) must be at sh_{dists,index}[0..topk_len-1] and already sorted ascending.
            // Survivors are at [topk_len .. ns-1]. Sorting the full [0..nPow2-1] yields a merged ascending order.
            bitonic_sort_pairs(sh_dists, sh_index, nPow2);

            // Update top-k (keep only first k items, which are the smallest)
            if (threadIdx.x == 0) {
                topk_len = (ns < k ? ns : k);
                threshold = sh_dists[topk_len - 1]; // kth smallest distance
            }
        }
        __syncthreads();
        // After this point, the first topk_len entries of sh_dists/sh_index hold the block's current top-k in ascending order.
    }

    // Write final top-k for this query to the result array (ascending order).
    // Each element result[qid * k + j] is {index, distance} of j-th nearest neighbor.
    // Parallelize the write across the block's threads.
    const int out_base = qid * k;
    for (int j = threadIdx.x; j < k; j += BLOCK_DIM) {
        result[out_base + j].first = sh_index[j];
        result[out_base + j].second = sh_dists[j];
    }
}

// Host interface as requested.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Tunable parameters selected for A100/H100-class GPUs:
    // - 256 threads per block
    // - 4 items per thread per tile (TILE_SIZE = 1024)
    // This keeps the shared memory footprint modest (<= 16KB for worst nPow2=2048),
    // allows high occupancy, and bounds merge/sort cost.
    constexpr int BLOCK_DIM = 256;
    constexpr int ITEMS_PER_THREAD = 4;
    constexpr int TILE_SIZE = BLOCK_DIM * ITEMS_PER_THREAD;

    // Dynamic shared memory size in bytes:
    // We need storage for N elements where N is the next power-of-two of (k + TILE_SIZE).
    // Each element has a float distance and an int index.
    const int smem_elems = next_pow2_host(k + TILE_SIZE);
    const size_t smem_bytes = static_cast<size_t>(smem_elems) * (sizeof(float) + sizeof(int));

    dim3 block(BLOCK_DIM);
    dim3 grid(query_count);

    // Launch the kernel. No additional device memory is allocated.
    knn_kernel<BLOCK_DIM, ITEMS_PER_THREAD><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k);

    // Note: This function is intentionally non-blocking. Synchronization and error checks (if desired)
    // should be performed by the caller.
}