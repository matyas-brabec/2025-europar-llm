#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace {

// Exact brute-force k-NN specialized for:
//   * 2D float points
//   * squared Euclidean distance
//   * k in [32, 1024], power of two
//
// Main optimization strategy:
//
// 1) One warp handles one query.
//    This keeps the query point in registers and lets us use warp ballot/shuffle
//    to compact candidates without any atomics.
//
// 2) One block handles 8 queries (8 warps / 256 threads).
//    The block streams the data set in 256-point tiles and loads each tile from
//    global memory exactly once into shared memory, then reuses it across 8 queries.
//    This is the key bandwidth optimization for the "thousands of queries" regime.
//
// 3) Each warp maintains an exact max-heap of size k in shared memory.
//    The root holds the current worst accepted neighbor, i.e. the pruning threshold.
//
// 4) Exact tile-level pruning:
//    For a tile, every lane compares against a snapshot of the current threshold.
//    Any point with distance >= snapshot threshold can never enter the final top-k
//    because the threshold only decreases as more points are processed.
//    Therefore, only points below the snapshot threshold are compacted into a
//    per-warp candidate buffer, and lane 0 merges them into the heap exactly.
//
// 5) No additional device allocations.
//    All temporary state lives in dynamic shared memory.

constexpr int kWarpSize        = 32;
constexpr int kWarpsPerBlock   = 8;
constexpr int kBlockThreads    = kWarpSize * kWarpsPerBlock;  // 256 threads
constexpr int kDataTilePoints  = kBlockThreads;               // 256 points / tile
constexpr int kTileRounds      = kDataTilePoints / kWarpSize; // 8 rounds of 32 points
constexpr int kMaxK            = 1024;
constexpr unsigned kFullMask   = 0xffffffffu;

static_assert(kDataTilePoints % kWarpSize == 0, "Tile size must be a multiple of warp size.");

struct alignas(8) Neighbor {
    float dist;
    int   idx;
};
static_assert(sizeof(Neighbor) == 8, "Neighbor must remain compact (8 bytes).");

// The public API uses std::pair<int, float>. The caller also guarantees the buffer
// comes from cudaMalloc, i.e. raw device storage. To avoid depending on device-side
// std::pair support, the kernel writes through an ABI-compatible POD view.
struct result_pair_t {
    int   first;
    float second;
};
static_assert(sizeof(result_pair_t)  == sizeof(std::pair<int, float>), "std::pair<int,float> ABI mismatch.");
static_assert(alignof(result_pair_t) == alignof(std::pair<int, float>), "std::pair<int,float> ABI mismatch.");

// Worst-case dynamic shared memory needed at k=1024.
constexpr std::size_t kMaxDynamicSharedBytes =
    kDataTilePoints * sizeof(float2) +
    static_cast<std::size_t>(kWarpsPerBlock) *
        static_cast<std::size_t>(kMaxK + kDataTilePoints) * sizeof(Neighbor);

__device__ __forceinline__ float sq_l2(const float2 p, const float qx, const float qy) {
    const float dx = p.x - qx;
    const float dy = p.y - qy;
    return fmaf(dx, dx, dy * dy);
}

__device__ __forceinline__ void swap_neighbor(Neighbor* a, Neighbor* b) {
    const Neighbor tmp = *a;
    *a = *b;
    *b = tmp;
}

__device__ __forceinline__ void sift_down(Neighbor* heap, int root, const int n) {
    while (true) {
        int child = (root << 1) + 1;
        if (child >= n) break;

        int largest = child;
        const int right = child + 1;
        if (right < n && heap[right].dist > heap[child].dist) {
            largest = right;
        }

        if (heap[root].dist >= heap[largest].dist) {
            break;
        }

        swap_neighbor(heap + root, heap + largest);
        root = largest;
    }
}

__device__ __forceinline__ void build_max_heap(Neighbor* heap, const int n) {
    for (int i = (n >> 1) - 1; i >= 0; --i) {
        sift_down(heap, i, n);
    }
}

__device__ __forceinline__ void maybe_replace_root(Neighbor* heap, const int n, const Neighbor cand) {
    if (cand.dist < heap[0].dist) {
        heap[0] = cand;
        sift_down(heap, 0, n);
    }
}

// Standard heapsort on a max-heap leaves the array in ascending order.
__device__ __forceinline__ void sort_heap_ascending(Neighbor* heap, const int n) {
    for (int end = n - 1; end > 0; --end) {
        swap_neighbor(heap + 0, heap + end);
        sift_down(heap, 0, end);
    }
}

__global__ __launch_bounds__(kBlockThreads, 2)
void knn_kernel(const float2* __restrict__ query,
                int query_count,
                const float2* __restrict__ data,
                int data_count,
                result_pair_t* __restrict__ result,
                int k) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & (kWarpSize - 1);
    const int qid     = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool active_query = (qid < query_count);

    // One query point per warp. Lane 0 loads, then broadcasts.
    float qx = 0.0f;
    float qy = 0.0f;
    if (active_query && lane == 0) {
        const float2 q = query[qid];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    // Shared memory layout:
    //   [ data tile:          kDataTilePoints * float2 ]
    //   [ per-warp heaps:     kWarpsPerBlock * k * Neighbor ]
    //   [ per-warp cand bufs: kWarpsPerBlock * kDataTilePoints * Neighbor ]
    extern __shared__ __align__(8) unsigned char smem_raw[];
    float2*  s_data = reinterpret_cast<float2*>(smem_raw);
    Neighbor* s_heap = reinterpret_cast<Neighbor*>(s_data + kDataTilePoints);
    Neighbor* s_cand = s_heap + static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(k);

    Neighbor* warp_heap = s_heap + static_cast<std::size_t>(warp_id) * static_cast<std::size_t>(k);
    Neighbor* warp_cand = s_cand + static_cast<std::size_t>(warp_id) * static_cast<std::size_t>(kDataTilePoints);

    bool heap_ready = false;
    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    for (int base = 0; base < data_count; base += kDataTilePoints) {
        // Load one 256-point tile once per block and reuse it across 8 queries.
        const int data_idx = base + threadIdx.x;
        if (data_idx < data_count) {
            s_data[threadIdx.x] = data[data_idx];
        }
        __syncthreads();

        const int remaining = data_count - base;
        const int valid = (remaining < kDataTilePoints) ? remaining : kDataTilePoints;

        if (active_query) {
            if (!heap_ready) {
                // Initialization:
                //   - the first k points go directly into the heap array
                //   - if k falls inside this tile (small k), the tile tail is also
                //     used as a warm-up sample and merged immediately
                #pragma unroll
                for (int round = 0; round < kTileRounds; ++round) {
                    const int local = round * kWarpSize + lane;
                    if (local < valid) {
                        const int idx = base + local;
                        Neighbor n;
                        n.dist = sq_l2(s_data[local], qx, qy);
                        n.idx  = idx;

                        if (idx < k) {
                            warp_heap[idx] = n;
                        } else if (base < k) {
                            // Only the tile that crosses the initialization boundary
                            // can contribute here. Storing at idx-k keeps the buffer
                            // dense from 0..extra_count-1.
                            warp_cand[idx - k] = n;
                        }
                    }
                }
                __syncwarp();

                if (base + valid >= k) {
                    if (lane == 0) {
                        build_max_heap(warp_heap, k);

                        // extra_count > 0 only when k lies inside this tile
                        // (i.e. k < 256 in the requested range).
                        const int extra_count = base + valid - k;
                        for (int i = 0; i < extra_count; ++i) {
                            maybe_replace_root(warp_heap, k, warp_cand[i]);
                        }
                    }
                    heap_ready = true;
                }
            } else {
                // Snapshot the current worst accepted distance. This threshold is exact
                // for all points processed so far. Filtering against it for the whole tile
                // is also exact because the threshold can only decrease after processing
                // more points.
                const float threshold =
                    __shfl_sync(kFullMask, (lane == 0) ? warp_heap[0].dist : 0.0f, 0);

                // Warp-private compaction into warp_cand[].
                // Only lane 0 tracks the running count; no atomics are needed because
                // the whole buffer is private to this warp.
                int warp_count = 0;

                #pragma unroll
                for (int round = 0; round < kTileRounds; ++round) {
                    const int local = round * kWarpSize + lane;

                    Neighbor n;
                    bool keep = false;
                    if (local < valid) {
                        const float d = sq_l2(s_data[local], qx, qy);
                        keep = (d < threshold);
                        n.dist = d;
                        n.idx  = base + local;
                    }

                    const unsigned mask = __ballot_sync(kFullMask, keep);
                    if (mask) {
                        const int count = __popc(mask);

                        int offset = 0;
                        if (lane == 0) {
                            offset = warp_count;
                            warp_count += count;
                        }
                        offset = __shfl_sync(kFullMask, offset, 0);

                        if (keep) {
                            const int rank = __popc(mask & lane_mask_lt);
                            warp_cand[offset + rank] = n;
                        }
                    }
                }
                __syncwarp();

                // Exact heap maintenance is serialized in lane 0. This is cheap in practice
                // because only a tiny fraction of distances survive the threshold test once
                // the heap has warmed up.
                if (lane == 0) {
                    for (int i = 0; i < warp_count; ++i) {
                        maybe_replace_root(warp_heap, k, warp_cand[i]);
                    }
                }
            }
        }

        // Ensure all warps are done with the current shared tile before overwriting it.
        __syncthreads();
    }

    if (active_query && lane == 0) {
        sort_heap_ascending(warp_heap, k);

        result_pair_t* out =
            result + static_cast<std::size_t>(qid) * static_cast<std::size_t>(k);

        for (int i = 0; i < k; ++i) {
            out[i].first  = warp_heap[i].idx;
            out[i].second = warp_heap[i].dist;
        }
    }
}

} // namespace

void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Needed because the worst case (k=1024) uses ~84 KiB of dynamic shared memory,
    // which is above the legacy 48 KiB default. The target hardware in the prompt
    // supports this opt-in size.
    cudaFuncSetAttribute(
        knn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kMaxDynamicSharedBytes));

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    const std::size_t shared_bytes =
        kDataTilePoints * sizeof(float2) +
        static_cast<std::size_t>(kWarpsPerBlock) *
            static_cast<std::size_t>(k + kDataTilePoints) * sizeof(Neighbor);

    knn_kernel<<<blocks, kBlockThreads, shared_bytes>>>(
        query,
        query_count,
        data,
        data_count,
        reinterpret_cast<result_pair_t*>(result),
        k);
}