#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <utility>

// Optimized k-NN for 2D points using block-wide radix sort and tile-wise reduction.
// Each block processes one query point and streams over the dataset in tiles.
// For each tile, we compute all distances, perform a block-wide sort to get the tile's top-k,
// then merge with the running block top-k using another block-wide sort on the union (2k items).
// This approach guarantees correctness: a globally top-k element must belong to its tile's top-k.
// It uses only on-chip (shared) memory and no auxiliary device allocations.
//
// Assumptions:
// - data_count >= k
// - k is a power of two in [32, 1024], but the algorithm doesn't rely on power-of-two specifically.
// - query_count is typically thousands, data_count typically millions.
// - result is an array of std::pair<int,float> in device memory.
//
// Hyperparameters:
// - BLOCK_THREADS: number of threads per block.
// - TILE_ITEMS_PER_THREAD: number of elements each thread loads per tile, giving a tile size of
//   TILE_ITEMS_PER_THREAD * BLOCK_THREADS. Must be >= k to guarantee the tile-wide top-k includes
//   the global top-k candidates.
// - MERGE_ITEMS_PER_THREAD: items per thread for merging 2k candidates (block top-k and tile top-k).
//   Must satisfy MERGE_ITEMS_PER_THREAD * BLOCK_THREADS >= 2 * KMAX.
//
// Choices below target Ampere/Hopper GPUs. They give tiles of 2048 items per block and
// 2k merge capacity up to k=1024.

#ifndef KNN_BLOCK_THREADS
#define KNN_BLOCK_THREADS 128
#endif

#ifndef KNN_TILE_ITEMS_PER_THREAD
#define KNN_TILE_ITEMS_PER_THREAD 16
#endif

#ifndef KNN_KMAX
#define KNN_KMAX 1024
#endif

// For union(2k) sort, with k up to 1024 and BLOCK_THREADS=128, we need at least 2048 capacity.
// 128 * 16 = 2048, so MERGE_ITEMS_PER_THREAD = 16 is sufficient.
#ifndef KNN_MERGE_ITEMS_PER_THREAD
#define KNN_MERGE_ITEMS_PER_THREAD 16
#endif

static_assert(KNN_BLOCK_THREADS > 0 && (KNN_BLOCK_THREADS % 32) == 0, "BLOCK_THREADS must be a positive multiple of warp size.");
static_assert(KNN_TILE_ITEMS_PER_THREAD > 0, "TILE_ITEMS_PER_THREAD must be positive.");
static_assert(KNN_MERGE_ITEMS_PER_THREAD > 0, "MERGE_ITEMS_PER_THREAD must be positive.");
static_assert(KNN_BLOCK_THREADS * KNN_TILE_ITEMS_PER_THREAD >= KNN_KMAX, "Tile capacity must be >= KMAX.");
static_assert(KNN_BLOCK_THREADS * KNN_MERGE_ITEMS_PER_THREAD >= 2 * KNN_KMAX, "Merge capacity must be >= 2*KMAX.");

__device__ __forceinline__ float sq_l2_distance_2d(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    // FMA helps reduce latency and improve precision slightly.
    return fmaf(dx, dx, dy * dy);
}

template <typename T>
__device__ __forceinline__ size_t align_up(size_t offset) {
    const size_t align = alignof(T);
    return (offset + (align - 1)) & ~(align - 1);
}

__global__ void knn2d_kernel(const float2* __restrict__ query,
                             int query_count,
                             const float2* __restrict__ data,
                             int data_count,
                             std::pair<int, float>* __restrict__ result,
                             int k)
{
    // One block per query
    const int qid = blockIdx.x;
    if (qid >= query_count) return;

    constexpr int BLOCK_THREADS = KNN_BLOCK_THREADS;
    constexpr int TILE_ITEMS_PER_THREAD = KNN_TILE_ITEMS_PER_THREAD;
    constexpr int TILE_ITEMS = BLOCK_THREADS * TILE_ITEMS_PER_THREAD;

    constexpr int KMAX = KNN_KMAX;

    constexpr int MERGE_ITEMS_PER_THREAD = KNN_MERGE_ITEMS_PER_THREAD;
    constexpr int MERGE_ITEMS = BLOCK_THREADS * MERGE_ITEMS_PER_THREAD; // Must cover up to 2*KMAX

    // CUB block radix sort types
    using TileBlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, TILE_ITEMS_PER_THREAD, int>;
    using MergeBlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, MERGE_ITEMS_PER_THREAD, int>;

    // Dynamic shared memory layout
    extern __shared__ unsigned char smem_raw[];
    size_t offset = 0;

    offset = align_up<typename TileBlockSort::TempStorage>(offset);
    auto* tile_sort_storage = reinterpret_cast<typename TileBlockSort::TempStorage*>(smem_raw + offset);
    offset += sizeof(typename TileBlockSort::TempStorage);

    offset = align_up<typename MergeBlockSort::TempStorage>(offset);
    auto* merge_sort_storage = reinterpret_cast<typename MergeBlockSort::TempStorage*>(smem_raw + offset);
    offset += sizeof(typename MergeBlockSort::TempStorage);

    // Shared arrays for current block top-k and current tile top-k
    offset = align_up<float>(offset);
    float* sh_topk_dist = reinterpret_cast<float*>(smem_raw + offset);
    offset += KMAX * sizeof(float);

    float* sh_tile_topk_dist = reinterpret_cast<float*>(smem_raw + offset);
    offset += KMAX * sizeof(float);

    offset = align_up<int>(offset);
    int* sh_topk_idx = reinterpret_cast<int*>(smem_raw + offset);
    offset += KMAX * sizeof(int);

    int* sh_tile_topk_idx = reinterpret_cast<int*>(smem_raw + offset);
    offset += KMAX * sizeof(int);

    // Load query point once into registers
    const float2 q = query[qid];
    const float INF = CUDART_INF_F;

    // Local storage for CUB sorts
    float tile_keys[TILE_ITEMS_PER_THREAD];
    int   tile_vals[TILE_ITEMS_PER_THREAD];

    float merge_keys[MERGE_ITEMS_PER_THREAD];
    int   merge_vals[MERGE_ITEMS_PER_THREAD];

    // Iterate over dataset in tiles
    for (int tile_base = 0, tile_iter = 0; tile_base < data_count; tile_base += TILE_ITEMS, ++tile_iter) {
        const int tile_count = min(TILE_ITEMS, data_count - tile_base);
        const int keep_count = min(k, tile_count); // we only need top-k from this tile

        // Load and compute distances for this tile in a striped pattern for perfect coalescing.
        // Each thread processes TILE_ITEMS_PER_THREAD items.
        #pragma unroll
        for (int it = 0; it < TILE_ITEMS_PER_THREAD; ++it) {
            const int pos_in_tile = it * BLOCK_THREADS + threadIdx.x;
            const int idx = tile_base + pos_in_tile;
            if (pos_in_tile < tile_count) {
                const float2 d = data[idx];
                const float dist2 = sq_l2_distance_2d(d, q);
                tile_keys[it] = dist2;
                tile_vals[it] = idx;
            } else {
                // Pad with +inf so padded items sort to the end
                tile_keys[it] = INF;
                tile_vals[it] = -1;
            }
        }

        // Sort the entire tile across the block by distance (ascending)
        TileBlockSort(*tile_sort_storage).Sort(tile_keys, tile_vals);

        // After sort, items are in "striped" order: global_rank = it * BLOCK_THREADS + threadIdx.x
        // Extract first keep_count items into the shared tile top-k buffers.
        #pragma unroll
        for (int it = 0; it < TILE_ITEMS_PER_THREAD; ++it) {
            const int global_rank = it * BLOCK_THREADS + threadIdx.x;
            if (global_rank < keep_count) {
                sh_tile_topk_dist[global_rank] = tile_keys[it];
                sh_tile_topk_idx[global_rank]  = tile_vals[it];
            }
        }

        // For ranks [keep_count, k) fill padding with +inf to keep fixed-size merge
        for (int r = threadIdx.x; r < k; r += BLOCK_THREADS) {
            if (r >= keep_count) {
                sh_tile_topk_dist[r] = INF;
                sh_tile_topk_idx[r]  = -1;
            }
        }
        __syncthreads();

        if (tile_iter == 0) {
            // First tile: initialize block top-k with tile top-k
            for (int r = threadIdx.x; r < k; r += BLOCK_THREADS) {
                sh_topk_dist[r] = sh_tile_topk_dist[r];
                sh_topk_idx[r]  = sh_tile_topk_idx[r];
            }
            __syncthreads();
        } else {
            // Merge current block top-k and tile top-k (2k items) and keep the best k.
            #pragma unroll
            for (int it = 0; it < MERGE_ITEMS_PER_THREAD; ++it) {
                const int global_rank = it * BLOCK_THREADS + threadIdx.x;
                if (global_rank < 2 * k) {
                    if (global_rank < k) {
                        merge_keys[it] = sh_topk_dist[global_rank];
                        merge_vals[it] = sh_topk_idx[global_rank];
                    } else {
                        const int r2 = global_rank - k;
                        merge_keys[it] = sh_tile_topk_dist[r2];
                        merge_vals[it] = sh_tile_topk_idx[r2];
                    }
                } else {
                    merge_keys[it] = INF;
                    merge_vals[it] = -1;
                }
            }

            MergeBlockSort(*merge_sort_storage).Sort(merge_keys, merge_vals);

            // Write back the first k items into block top-k
            #pragma unroll
            for (int it = 0; it < MERGE_ITEMS_PER_THREAD; ++it) {
                const int global_rank = it * BLOCK_THREADS + threadIdx.x;
                if (global_rank < k) {
                    sh_topk_dist[global_rank] = merge_keys[it];
                    sh_topk_idx[global_rank]  = merge_vals[it];
                }
            }
            __syncthreads();
        }
    }

    // Store final results for this query in row-major: result[qid * k + j]
    for (int j = threadIdx.x; j < k; j += BLOCK_THREADS) {
        const int out_idx = qid * k + j;
        // Write directly to std::pair in device memory
        result[out_idx].first  = sh_topk_idx[j];
        result[out_idx].second = sh_topk_dist[j];
    }
}

// Host-side driver
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    // Configure launch
    constexpr int BLOCK_THREADS = KNN_BLOCK_THREADS;
    constexpr int TILE_ITEMS_PER_THREAD = KNN_TILE_ITEMS_PER_THREAD;
    constexpr int MERGE_ITEMS_PER_THREAD = KNN_MERGE_ITEMS_PER_THREAD;
    constexpr int KMAX = KNN_KMAX;

    // Sanity caps on k
    if (k <= 0) return;
    if (k > KMAX) k = KMAX;

    // Compute dynamic shared memory requirement
    using TileBlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, TILE_ITEMS_PER_THREAD, int>;
    using MergeBlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, MERGE_ITEMS_PER_THREAD, int>;

    size_t smem_bytes = 0;
    auto align_up_host = [](size_t offset, size_t alignment) -> size_t {
        return (offset + (alignment - 1)) & ~(alignment - 1);
    };

    smem_bytes = align_up_host(smem_bytes, alignof(typename TileBlockSort::TempStorage));
    smem_bytes += sizeof(typename TileBlockSort::TempStorage);

    smem_bytes = align_up_host(smem_bytes, alignof(typename MergeBlockSort::TempStorage));
    smem_bytes += sizeof(typename MergeBlockSort::TempStorage);

    smem_bytes = align_up_host(smem_bytes, alignof(float));
    smem_bytes += KMAX * sizeof(float); // sh_topk_dist
    smem_bytes += KMAX * sizeof(float); // sh_tile_topk_dist

    smem_bytes = align_up_host(smem_bytes, alignof(int));
    smem_bytes += KMAX * sizeof(int);   // sh_topk_idx
    smem_bytes += KMAX * sizeof(int);   // sh_tile_topk_idx

    // Increase max dynamic shared memory size attribute if needed
    cudaFuncSetAttribute(knn2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));

    dim3 block(BLOCK_THREADS);
    dim3 grid(query_count);

    // Launch kernel
    knn2d_kernel<<<grid, block, smem_bytes>>>(query, query_count, data, data_count, result, k);
}