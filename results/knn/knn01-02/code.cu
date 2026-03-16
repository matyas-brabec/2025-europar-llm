#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace {

// Chosen for the prompt's stated regime (query_count in the low-thousands, data_count in
// the millions) and for modern data-center GPUs:
//
// * 256 threads/block  -> 8 warps/block, so 8 queries are processed together.
// * 512 data points/tile -> each thread loads exactly 2 float2 values into shared memory.
// * Worst-case shared memory:
//       8 warps * 1024 heap entries * (4B dist + 4B idx) = 64 KiB dynamic
//       512 x-coordinates + 512 y-coordinates            =  4 KiB static
//   Total ~= 68 KiB, which still allows multiple resident blocks on A100/H100 once the
//   kernel opts into the larger shared-memory carveout.
constexpr int kWarpSize          = 32;
constexpr int kBlockThreads      = 256;
constexpr int kQueriesPerBlock   = kBlockThreads / kWarpSize;   // 8 queries/block
constexpr int kTilePoints        = 512;                         // 2 loads/thread
constexpr int kRoundsPerTile     = kTilePoints / kWarpSize;     // 16 warp rounds/tile
constexpr int kTileLoadsPerThread= kTilePoints / kBlockThreads; // 2
constexpr int kMaxK              = 1024;

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be warp-aligned.");
static_assert(kTilePoints % kWarpSize == 0, "Tile size must be warp-aligned.");
static_assert(kTilePoints % kBlockThreads == 0, "Tile size must be an integer number of per-thread loads.");

// Device-side POD mirror of std::pair<int,float>.
// The public interface is fixed to std::pair<int,float>, but device code only needs raw
// field stores. Using a POD mirror avoids depending on std::pair constructors/operators
// being device-callable, while preserving the same binary size/layout in practice.
struct PairIF {
    int   first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>),
              "Unexpected std::pair<int,float> layout.");

// Per-warp max-heap helpers. The heap stores the current exact k best candidates seen so
// far for one query. The root (index 0) is the current kth-smallest distance threshold.
__device__ __forceinline__ void heap_swap(float* dist, int* idx, int a, int b) {
    const float td = dist[a];
    dist[a] = dist[b];
    dist[b] = td;

    const int ti = idx[a];
    idx[a] = idx[b];
    idx[b] = ti;
}

__device__ __forceinline__ void sift_down_max_heap(float* dist, int* idx, int root, int count) {
    while (true) {
        int child = (root << 1) + 1;
        if (child >= count) break;

        int largest = child;
        const int right = child + 1;
        if (right < count && dist[right] > dist[child]) {
            largest = right;
        }

        if (dist[root] >= dist[largest]) break;

        heap_swap(dist, idx, root, largest);
        root = largest;
    }
}

__device__ __forceinline__ void build_max_heap(float* dist, int* idx, int count) {
    for (int i = (count >> 1) - 1; i >= 0; --i) {
        sift_down_max_heap(dist, idx, i, count);
    }
}

__device__ __forceinline__ void replace_heap_root(float* dist, int* idx, int count,
                                                  float new_dist, int new_idx) {
    dist[0] = new_dist;
    idx[0]  = new_idx;
    sift_down_max_heap(dist, idx, 0, count);
}

// Standard in-place heapsort on a max-heap. Result is ascending by distance, which is
// exactly the output order required by result[i * k + j] == j-th nearest neighbor.
__device__ __forceinline__ void heap_sort_ascending(float* dist, int* idx, int count) {
    for (int end = count - 1; end > 0; --end) {
        heap_swap(dist, idx, 0, end);
        sift_down_max_heap(dist, idx, 0, end);
    }
}

} // namespace

// Exact k-NN kernel for 2D points:
//
//   * One warp owns one query for the entire scan.
//   * A block therefore handles 8 queries at once.
//   * Data is streamed through a shared-memory tile so each global load is reused by
//     8 queries before eviction.
//   * Each warp maintains its own exact top-k max-heap in shared memory.
//   * After the first k points initialize the heap, later points only incur heap work
//     when their squared distance is below the current kth-best threshold.
//
// This is exact, needs no extra device allocations, and is tuned for large batched scans.
__global__ __launch_bounds__(256, 2)
void knn_kernel_2d_euclidean(const float2* __restrict__ query,
                             int query_count,
                             const float2* __restrict__ data,
                             int data_count,
                             PairIF* __restrict__ result,
                             int k) {
    constexpr unsigned kFullMask = 0xffffffffu;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int qid     = static_cast<int>(blockIdx.x) * kQueriesPerBlock + warp_id;
    const bool active = (qid < query_count);

    // Dynamic shared memory layout:
    //   [0, kQueriesPerBlock * k) floats : heap distances
    //   [same size] ints                 : heap indices
    extern __shared__ unsigned char smem[];
    float* const all_heap_dist = reinterpret_cast<float*>(smem);
    int*   const all_heap_idx  = reinterpret_cast<int*>(all_heap_dist + kQueriesPerBlock * k);

    float* const heap_dist = all_heap_dist + warp_id * k;
    int*   const heap_idx  = all_heap_idx  + warp_id * k;

    // Shared-memory tile in SoA form. This avoids the 2-way bank conflicts that a naive
    // float2 AoS tile would create for warp-wide shared-memory reads.
    __shared__ float tile_x[kTilePoints];
    __shared__ float tile_y[kTilePoints];

    // One query point per warp; lane 0 loads it and broadcasts to the warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active) {
        float2 q = make_float2(0.0f, 0.0f);
        if (lane == 0) {
            q = query[qid];
        }
        qx = __shfl_sync(kFullMask, q.x, 0);
        qy = __shfl_sync(kFullMask, q.y, 0);
    }

    // k is guaranteed to be a multiple of 32, so the initial fill phase reaches exactly
    // k entries on a warp-round boundary.
    int   local_count = 0;
    float kth_dist    = CUDART_INF_F; // valid in lane 0 once the heap is built

    for (int base = 0; base < data_count; base += kTilePoints) {
        int tile_count = data_count - base;
        if (tile_count > kTilePoints) tile_count = kTilePoints;

        // Cooperative global->shared staging. Each thread loads exactly two float2 values
        // on full tiles, which maps well to the chosen 512-point tile size.
        #pragma unroll
        for (int load = 0; load < kTileLoadsPerThread; ++load) {
            const int t = tid + load * kBlockThreads;
            if (t < tile_count) {
                const float2 p = data[base + t];
                tile_x[t] = p.x;
                tile_y[t] = p.y;
            }
        }

        __syncthreads();

        if (active) {
            const unsigned lane_mask_lt = (1u << lane) - 1u;

            // Fixed 16 rounds per tile; the valid predicate handles the tail tile.
            #pragma unroll
            for (int round = 0; round < kRoundsPerTile; ++round) {
                const int t     = (round << 5) + lane;
                const bool valid = (t < tile_count);

                float dist = 0.0f;
                int   idx  = base + t;

                if (valid) {
                    // Squared L2 distance; no sqrt because ordering is unchanged.
                    const float dx = tile_x[t] - qx;
                    const float dy = tile_y[t] - qy;
                    dist = fmaf(dx, dx, dy * dy);
                }

                if (local_count < k) {
                    // Heap fill phase: append the next valid distances contiguously.
                    const unsigned valid_mask = __ballot_sync(kFullMask, valid);
                    const int prefix = __popc(valid_mask & lane_mask_lt);

                    if (valid) {
                        heap_dist[local_count + prefix] = dist;
                        heap_idx [local_count + prefix] = idx;
                    }

                    // Needed before lane 0 can build the heap from the just-written batch.
                    __syncwarp(kFullMask);

                    local_count += __popc(valid_mask);

                    if (local_count == k) {
                        if (lane == 0) {
                            build_max_heap(heap_dist, heap_idx, k);
                            kth_dist = heap_dist[0];
                        }
                        __syncwarp(kFullMask);
                    }
                } else {
                    // Steady state: only candidates below the current kth-best threshold
                    // can possibly enter the exact top-k.
                    const float threshold = __shfl_sync(kFullMask, kth_dist, 0);
                    const bool hit = valid && (dist < threshold);
                    unsigned hit_mask = __ballot_sync(kFullMask, hit);

                    // The loop count is warp-uniform because hit_mask is warp-uniform, so
                    // the shuffle calls are legal. Lane 0 performs the heap updates.
                    while (hit_mask) {
                        const int src = __ffs(hit_mask) - 1;
                        const float cand_dist = __shfl_sync(kFullMask, dist, src);
                        const int   cand_idx  = __shfl_sync(kFullMask, idx,  src);

                        if (lane == 0 && cand_dist < heap_dist[0]) {
                            replace_heap_root(heap_dist, heap_idx, k, cand_dist, cand_idx);
                        }

                        hit_mask &= hit_mask - 1;
                    }

                    if (lane == 0) {
                        kth_dist = heap_dist[0];
                    }
                    __syncwarp(kFullMask);
                }
            }
        }

        // Protect the tile before the next global->shared load overwrites it.
        __syncthreads();
    }

    if (active) {
        if (lane == 0) {
            heap_sort_ascending(heap_dist, heap_idx, k);
        }
        __syncwarp(kFullMask);

        PairIF* const out = result + static_cast<std::size_t>(qid) * static_cast<std::size_t>(k);
        for (int j = lane; j < k; j += kWarpSize) {
            out[j].first  = heap_idx[j];
            out[j].second = heap_dist[j];
        }
    }
}

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Request enough dynamic shared memory for the worst-case heap footprint and ask the
    // driver to maximize the shared-memory carveout. This matches the target hardware in
    // the prompt (A100/H100-class parts).
    const int max_dynamic_smem =
        kQueriesPerBlock * kMaxK * (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(int)));
    (void)cudaFuncSetAttribute(knn_kernel_2d_euclidean,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               max_dynamic_smem);
    (void)cudaFuncSetAttribute(knn_kernel_2d_euclidean,
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               100);

    const std::size_t dynamic_smem =
        static_cast<std::size_t>(kQueriesPerBlock) *
        static_cast<std::size_t>(k) *
        (sizeof(float) + sizeof(int));

    const int blocks = (query_count + kQueriesPerBlock - 1) / kQueriesPerBlock;

    // Intentionally asynchronous, matching normal CUDA kernel-launch semantics.
    knn_kernel_2d_euclidean<<<blocks, kBlockThreads, dynamic_smem>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<PairIF*>(result),
        k
    );
}