#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <utility>

// Optimized k-NN (top-k) for 2D points using a single block per query.
// Strategy:
// - Each block processes one query and streams through the entire dataset in tiles.
// - For each tile, threads compute distances and cooperatively sort the tile using
//   CUB's BlockRadixSort (sorting key-value pairs: distance, index).
// - Only the smallest k entries from the current tile are retained (stored in shared memory).
// - These k entries are merged with the running global top-k for the query by sorting
//   the union of size up to 2k using another BlockRadixSort; the smallest k are kept.
// - This approach is correct and efficient: the global top-k must belong to the union
//   of the per-tile top-k sets (any element not in a tile's top-k has at least k
//   elements in the same tile that are smaller, so it cannot belong to the global top-k).
// - All work is done in-place with registers, static shared memory (for CUB temp storage),
//   and dynamic shared memory (for only 2k entries), without any device allocations.
//
// Assumptions (as per problem statement):
// - k is a power of two in [32, 1024].
// - data_count >= k.
// - Inputs are large enough to benefit from GPU parallelism.
// - Arrays are allocated by cudaMalloc; result is an array of std::pair<int, float>.
//
// Tuning choices:
// - BLOCK_THREADS = 256: good occupancy and memory coalescing on A100/H100.
// - TILE_ITEMS_PER_THREAD = 8: per tile we sort 256*8 = 2048 items, a good balance
//   between shared memory usage and sort overhead.
// - MERGE_ITEMS_PER_THREAD = 8: capacity for merging up to 2k = 2048 items when k=1024.
//
// Notes on memory usage per block:
// - Dynamic shared memory: 2 * k * (sizeof(float) + sizeof(int)) = 8*k bytes (<= 8KB).
// - Static shared memory: CUB BlockRadixSort temp storage (reused via a union).
// - Fits within default per-block shared memory limits on A100/H100 without extra attributes.

template <int BLOCK_THREADS, int TILE_ITEMS_PER_THREAD, int MERGE_ITEMS_PER_THREAD>
__global__ void knn2d_kernel(const float2* __restrict__ query,
                             int query_count,
                             const float2* __restrict__ data,
                             int data_count,
                             std::pair<int, float>* __restrict__ result,
                             int k)
{
    using KeyT = float;
    using ValT = int;

    // Block-wide radix sort types for the tile sort and the merge sort.
    using TileSort  = cub::BlockRadixSort<KeyT, BLOCK_THREADS, TILE_ITEMS_PER_THREAD, ValT>;
    using MergeSort = cub::BlockRadixSort<KeyT, BLOCK_THREADS, MERGE_ITEMS_PER_THREAD, ValT>;

    // Reuse the same static shared memory region for the two sorts via a union.
    __shared__ union {
        typename TileSort::TempStorage  tile_sort;
        typename MergeSort::TempStorage merge_sort;
    } sort_storage;

    // Dynamic shared memory layout:
    // [ topk_keys (k) | topk_vals (k) | tile_k_keys (k) | tile_k_vals (k) ]
    extern __shared__ unsigned char smem[];
    unsigned char* sptr = smem;

    KeyT* s_topk_keys = reinterpret_cast<KeyT*>(sptr);       sptr += sizeof(KeyT) * k;
    ValT* s_topk_vals = reinterpret_cast<ValT*>(sptr);       sptr += sizeof(ValT) * k;
    KeyT* s_tilek_keys = reinterpret_cast<KeyT*>(sptr);      sptr += sizeof(KeyT) * k;
    ValT* s_tilek_vals = reinterpret_cast<ValT*>(sptr);      // end

    const int qid = blockIdx.x;
    if (qid >= query_count) return;

    // Load the query point once per block into shared memory to avoid redundant global loads.
    __shared__ float2 q_sh;
    if (threadIdx.x == 0) {
        q_sh = query[qid];
    }
    __syncthreads();

    const float qx = q_sh.x;
    const float qy = q_sh.y;

    const int TILE_SIZE = BLOCK_THREADS * TILE_ITEMS_PER_THREAD;
    const KeyT INF = CUDART_INF_F;

    int cur_k_count = 0; // number of valid entries in s_topk_* arrays

    // Stream through data in tiles of TILE_SIZE elements.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {

        const int tile_valid = min(TILE_SIZE, data_count - tile_start);

        // Load distances and indices for this tile into per-thread registers.
        KeyT keys_tile[TILE_ITEMS_PER_THREAD];
        ValT vals_tile[TILE_ITEMS_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < TILE_ITEMS_PER_THREAD; ++i) {
            int idx = tile_start + threadIdx.x + i * BLOCK_THREADS;
            if (idx < data_count) {
                float2 p = data[idx];
                float dx = p.x - qx;
                float dy = p.y - qy;
                // Squared L2 distance
                KeyT d2 = fmaf(dy, dy, dx * dx);
                keys_tile[i] = d2;
                vals_tile[i] = idx;
            } else {
                keys_tile[i] = INF; // pad with +inf to keep processing simple
                vals_tile[i] = -1;
            }
        }

        // Sort the tile (blocked-to-blocked): after this, thread t owns the items in
        // global sorted positions [t*IPT, t*IPT + IPT).
        TileSort(sort_storage.tile_sort).SortBlockedToBlocked(keys_tile, vals_tile);
        __syncthreads(); // ensure sort_storage reuse safety

        // Keep only the smallest tile_k entries from this tile.
        const int tile_k = (tile_valid < k) ? tile_valid : k;

        // Write the smallest tile_k items from registers to shared memory buffer s_tilek_*.
        // Each thread writes the portion of [0, tile_k) that it owns after the blocked sort.
        {
            const int base = threadIdx.x * TILE_ITEMS_PER_THREAD;
            int count = tile_k - base;
            if (count > TILE_ITEMS_PER_THREAD) count = TILE_ITEMS_PER_THREAD;
            if (count < 0) count = 0;

            #pragma unroll
            for (int i = 0; i < count; ++i) {
                s_tilek_keys[base + i] = keys_tile[i];
                s_tilek_vals[base + i] = vals_tile[i];
            }
        }
        __syncthreads();

        // Merge current top-k and this tile's top-k by sorting their union (size <= 2k).
        // Build the union in per-thread registers (blocked layout), padding with +inf as needed.
        KeyT keys_merge[MERGE_ITEMS_PER_THREAD];
        ValT vals_merge[MERGE_ITEMS_PER_THREAD];

        const int union_total = cur_k_count + tile_k; // <= 2k by construction

        #pragma unroll
        for (int j = 0; j < MERGE_ITEMS_PER_THREAD; ++j) {
            int pos = threadIdx.x * MERGE_ITEMS_PER_THREAD + j;
            if (pos < cur_k_count) {
                keys_merge[j] = s_topk_keys[pos];
                vals_merge[j] = s_topk_vals[pos];
            } else if (pos < union_total) {
                int tpos = pos - cur_k_count;
                keys_merge[j] = s_tilek_keys[tpos];
                vals_merge[j] = s_tilek_vals[tpos];
            } else {
                keys_merge[j] = INF;
                vals_merge[j] = -1;
            }
        }

        // Sort the union and keep the smallest k into s_topk_*.
        MergeSort(sort_storage.merge_sort).SortBlockedToBlocked(keys_merge, vals_merge);
        __syncthreads(); // ensure sort_storage reuse safety

        #pragma unroll
        for (int j = 0; j < MERGE_ITEMS_PER_THREAD; ++j) {
            int pos = threadIdx.x * MERGE_ITEMS_PER_THREAD + j;
            if (pos < k) {
                s_topk_keys[pos] = keys_merge[j];
                s_topk_vals[pos] = vals_merge[j];
            }
        }
        __syncthreads();

        cur_k_count = (union_total < k) ? union_total : k;
    }

    // Store results for this query in ascending distance order.
    // Each thread writes a striped subset of the k results.
    std::pair<int, float>* out = result + static_cast<size_t>(qid) * static_cast<size_t>(k);
    for (int j = threadIdx.x; j < k; j += BLOCK_THREADS) {
        // Write fields individually to avoid invoking any std::pair constructors on device.
        out[j].first  = s_topk_vals[j];
        out[j].second = s_topk_keys[j];
    }
}

// Host interface
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Tuned launch configuration for A100/H100:
    // - 256 threads per block
    // - Each block processes one query
    // - Dynamic shared memory sized to hold 2*k (distance,index) pairs.
    constexpr int BLOCK_THREADS = 256;
    constexpr int TILE_ITEMS_PER_THREAD = 8;   // 256*8 = 2048 elements per tile
    constexpr int MERGE_ITEMS_PER_THREAD = 8;  // 256*8 = 2048 capacity for up to 2k elements

    dim3 block(BLOCK_THREADS);
    dim3 grid(query_count);

    // Dynamic shared memory needed per block:
    // top-k (k) + tile-k (k), each having (float distance + int index).
    size_t smem_bytes = static_cast<size_t>(2) * static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    knn2d_kernel<BLOCK_THREADS, TILE_ITEMS_PER_THREAD, MERGE_ITEMS_PER_THREAD>
        <<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k);

    // Optionally, synchronize or check for errors in production code.
    // cudaDeviceSynchronize();
    // cudaGetLastError();
}