#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace {

// 256 data points per tile = 2 KiB of staged data.
// This is large enough to amortize synchronization, yet small enough that
// stale-threshold false positives remain limited and shared-memory use stays low.
constexpr int TILE_POINTS = 256;
constexpr int TILE_ITERS  = TILE_POINTS / 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// Internal heap node. We keep the distance first because the max-heap compares
// only on distance; the index is carried alongside it.
struct alignas(8) Neighbor {
    float dist;
    int   idx;
};

// Device code avoids constructing std::pair directly because standard-library
// constructors/operators are not guaranteed to be __device__-callable.
// We therefore write through an ABI-compatible POD view. CUDA-supported host
// toolchains lay out pair<int,float> as two consecutive 32-bit fields.
struct ResultPair {
    int   first;
    float second;
};

static_assert(TILE_POINTS % 32 == 0, "Tile size must be a whole number of warps.");
static_assert(sizeof(Neighbor) == 8, "Unexpected Neighbor size.");
static_assert(sizeof(ResultPair) == sizeof(std::pair<int, float>),
              "Unexpected std::pair<int,float> size.");
static_assert(alignof(ResultPair) <= alignof(std::pair<int, float>),
              "Unexpected std::pair<int,float> alignment.");

constexpr std::size_t align_up(std::size_t x, std::size_t a) {
    return (x + a - 1) & ~(a - 1);
}

__device__ __forceinline__ float sq_l2(const float qx, const float qy, const float2 p) {
    const float dx = p.x - qx;
    const float dy = p.y - qy;
    return fmaf(dx, dx, dy * dy);
}

// Standard in-place sift-down for a max-heap stored in shared memory.
__device__ __forceinline__ void heap_sift_down(Neighbor* heap, int n, int root) {
    Neighbor v = heap[root];
    int child = (root << 1) + 1;

    while (child < n) {
        int max_child = child;
        const int right = child + 1;
        if (right < n && heap[right].dist > heap[child].dist) {
            max_child = right;
        }
        if (heap[max_child].dist <= v.dist) {
            break;
        }
        heap[root] = heap[max_child];
        root = max_child;
        child = (root << 1) + 1;
    }
    heap[root] = v;
}

__device__ __forceinline__ void heap_build_max(Neighbor* heap, int n) {
    for (int i = (n >> 1) - 1; i >= 0; --i) {
        heap_sift_down(heap, n, i);
    }
}

__device__ __forceinline__ void heap_replace_root(Neighbor* heap, int n, const Neighbor candidate) {
    heap[0] = candidate;
    heap_sift_down(heap, n, 0);
}

// Exact brute-force 2D k-NN.
//
// Mapping:
//   - one warp owns one query;
//   - a block batches WARPS_PER_BLOCK queries;
//   - the block stages TILE_POINTS data points into shared memory once;
//   - every query-warp reuses that tile.
//
// Exact top-k maintenance:
//   - lane 0 of each warp maintains an exact max-heap of size k in shared memory;
//   - the heap root is the current k-th best distance threshold;
//   - while scanning a tile, the warp only records ballot masks for points with
//     d < threshold_snapshot;
//   - lane 0 revisits only those marked points and inserts them exactly into the heap.
//
// This is exact because the threshold only ever decreases: a point with
// d >= threshold_snapshot cannot enter the final top-k later.
template<int WARPS_PER_BLOCK>
__global__ void knn_kernel(const float2* __restrict__ query,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           ResultPair* __restrict__ result,
                           int k) {
    extern __shared__ unsigned char smem[];

    std::size_t off = 0;

    off = align_up(off, alignof(float2));
    float2* tile = reinterpret_cast<float2*>(smem + off);
    off += TILE_POINTS * sizeof(float2);

    off = align_up(off, alignof(Neighbor));
    Neighbor* heaps = reinterpret_cast<Neighbor*>(smem + off);
    off += static_cast<std::size_t>(WARPS_PER_BLOCK) *
           static_cast<std::size_t>(k) * sizeof(Neighbor);

    off = align_up(off, alignof(unsigned int));
    unsigned int* qual_masks = reinterpret_cast<unsigned int*>(smem + off);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp;
    const bool active = query_idx < query_count;

    float qx = 0.0f;
    float qy = 0.0f;
    if (active && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(FULL_MASK, qx, 0);
    qy = __shfl_sync(FULL_MASK, qy, 0);

    Neighbor* const my_heap = heaps + static_cast<std::size_t>(warp) * static_cast<std::size_t>(k);
    unsigned int* const my_masks = qual_masks + warp * TILE_ITERS;

    // Initialize the exact top-k state from the first k data points.
    if (active) {
        for (int i = lane; i < k; i += 32) {
            const float2 p = data[i];
            my_heap[i] = Neighbor{sq_l2(qx, qy, p), i};
        }
    }
    __syncwarp(FULL_MASK);

    float my_threshold = CUDART_INF_F;
    if (active && lane == 0) {
        heap_build_max(my_heap, k);
        my_threshold = my_heap[0].dist;
    }

    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    for (int base = k; base < data_count; base += TILE_POINTS) {
        // Cooperative staging of one data tile.
        #pragma unroll
        for (int t = threadIdx.x; t < TILE_POINTS; t += THREADS) {
            const int data_idx = base + t;
            if (data_idx < data_count) {
                tile[t] = data[data_idx];
            }
        }
        __syncthreads();

        if (active) {
            // Only lane 0 owns the authoritative threshold register, so every
            // tile starts by broadcasting it to the full warp.
            const float thr = __shfl_sync(FULL_MASK, my_threshold, 0);

            // Squared distances are non-negative. Once the current k-th best is
            // exactly zero, no future point can improve the heap.
            if (thr > 0.0f) {
                int tile_count = data_count - base;
                if (tile_count > TILE_POINTS) {
                    tile_count = TILE_POINTS;
                }

                // One ballot mask per 32-point slice of the tile.
                #pragma unroll
                for (int iter = 0; iter < TILE_ITERS; ++iter) {
                    const int local = (iter << 5) + lane;
                    bool qualifies = false;

                    if (local < tile_count) {
                        const float2 p = tile[local];
                        qualifies = (sq_l2(qx, qy, p) < thr);
                    }

                    const unsigned int mask = __ballot_sync(FULL_MASK, qualifies);
                    if (lane == 0) {
                        my_masks[iter] = mask;
                    }
                }

                __syncwarp(FULL_MASK);

                if (lane == 0) {
                    float root = my_threshold;

                    // Revisit only candidates marked by the ballots. The root is
                    // refreshed after every successful insertion, so false
                    // positives caused by the stale tile-wide threshold are
                    // filtered here exactly.
                    #pragma unroll
                    for (int iter = 0; iter < TILE_ITERS; ++iter) {
                        unsigned int mask = my_masks[iter];
                        const int base_local = iter << 5;

                        while (mask) {
                            const int bit = __ffs(static_cast<int>(mask)) - 1;
                            const int local = base_local + bit;
                            const float2 p = tile[local];

                            const Neighbor cand{sq_l2(qx, qy, p), base + local};

                            if (cand.dist < root) {
                                heap_replace_root(my_heap, k, cand);
                                root = my_heap[0].dist;

                                // Early stop inside this tile when no point can
                                // possibly improve the heap anymore.
                                if (root == 0.0f) {
                                    break;
                                }
                            }

                            mask &= (mask - 1u);
                        }

                        if (root == 0.0f) {
                            break;
                        }
                    }

                    my_threshold = root;
                }
            }
        }

        // Ensure every warp has finished using the current tile before it is overwritten.
        __syncthreads();
    }

    // Final heap sort. Because this is a max-heap, repeatedly popping the root
    // and writing from the end yields ascending distance order in result[].
    if (active && lane == 0) {
        ResultPair* const out = result +
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(k);

        int heap_size = k;
        for (int pos = k - 1; pos >= 0; --pos) {
            const Neighbor top = my_heap[0];
            out[pos].first  = top.idx;
            out[pos].second = top.dist;

            --heap_size;
            if (heap_size > 0) {
                my_heap[0] = my_heap[heap_size];
                heap_sift_down(my_heap, heap_size, 0);
            }
        }
    }
}

template<int WARPS_PER_BLOCK>
std::size_t shared_bytes_for(int k) {
    std::size_t off = 0;

    off = align_up(off, alignof(float2));
    off += TILE_POINTS * sizeof(float2);

    off = align_up(off, alignof(Neighbor));
    off += static_cast<std::size_t>(WARPS_PER_BLOCK) *
           static_cast<std::size_t>(k) * sizeof(Neighbor);

    off = align_up(off, alignof(unsigned int));
    off += static_cast<std::size_t>(WARPS_PER_BLOCK) *
           static_cast<std::size_t>(TILE_ITERS) * sizeof(unsigned int);

    return off;
}

// Pick the largest query batch (one warp/query) that:
//   1) fits in per-block shared memory, and
//   2) still leaves at least one block per SM in the grid.
// Larger batches are preferable because each staged data tile is then reused by
// more queries, which is the dominant optimization for this bandwidth-heavy kernel.
int choose_warps_per_block(int query_count, int sm_count, std::size_t max_shared_per_block, int k) {
    if (((query_count + 31) / 32) >= sm_count && shared_bytes_for<32>(k) <= max_shared_per_block) return 32;
    if (((query_count + 15) / 16) >= sm_count && shared_bytes_for<16>(k) <= max_shared_per_block) return 16;
    if (((query_count +  7) /  8) >= sm_count && shared_bytes_for<8 >(k) <= max_shared_per_block) return 8;
    if (((query_count +  3) /  4) >= sm_count && shared_bytes_for<4 >(k) <= max_shared_per_block) return 4;
    if (((query_count +  1) /  2) >= sm_count && shared_bytes_for<2 >(k) <= max_shared_per_block) return 2;
    return 1;
}

template<int WARPS_PER_BLOCK>
void launch_knn_kernel(const float2* query,
                       int query_count,
                       const float2* data,
                       int data_count,
                       std::pair<int, float>* result,
                       int k) {
    const std::size_t shmem = shared_bytes_for<WARPS_PER_BLOCK>(k);

    // Opt in to larger dynamic shared-memory allocations when needed.
    if (shmem > (48u * 1024u)) {
        cudaFuncSetAttribute(knn_kernel<WARPS_PER_BLOCK>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shmem));
    }

    // This kernel is deliberately shared-memory heavy; prefer the shared carveout.
    cudaFuncSetAttribute(knn_kernel<WARPS_PER_BLOCK>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    const int blocks = (query_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    // See the ResultPair comment above for the ABI assumption behind this cast.
    ResultPair* const result_view = reinterpret_cast<ResultPair*>(result);

    knn_kernel<WARPS_PER_BLOCK><<<blocks, WARPS_PER_BLOCK * 32, shmem>>>(
        query, query_count, data, data_count, result_view, k);
}

} // anonymous namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        device = 0;
    }

    int sm_count = 1;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sm_count < 1) {
        sm_count = 1;
    }

    int max_shared_regular = 48 * 1024;
    cudaDeviceGetAttribute(&max_shared_regular, cudaDevAttrMaxSharedMemoryPerBlock, device);

    int max_shared_optin = max_shared_regular;
    const cudaError_t optin_status =
        cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (optin_status != cudaSuccess || max_shared_optin < max_shared_regular) {
        max_shared_optin = max_shared_regular;
    }

    const int warps_per_block =
        choose_warps_per_block(query_count,
                               sm_count,
                               static_cast<std::size_t>(max_shared_optin),
                               k);

    // The interface has no stream argument, so this intentionally uses CUDA's
    // normal asynchronous launch semantics on the current default stream.
    switch (warps_per_block) {
        case 32: launch_knn_kernel<32>(query, query_count, data, data_count, result, k); break;
        case 16: launch_knn_kernel<16>(query, query_count, data, data_count, result, k); break;
        case 8:  launch_knn_kernel<8 >(query, query_count, data, data_count, result, k); break;
        case 4:  launch_knn_kernel<4 >(query, query_count, data, data_count, result, k); break;
        case 2:  launch_knn_kernel<2 >(query, query_count, data, data_count, result, k); break;
        default: launch_knn_kernel<1 >(query, query_count, data, data_count, result, k); break;
    }
}