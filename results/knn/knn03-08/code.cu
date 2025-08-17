#include <cuda_runtime.h>
#include <utility>

// Efficient k-NN in 2D using one warp per query, block-level shared-memory tiling of the dataset,
// and a per-warp max-heap of size k maintained in shared memory.
// - Each block cooperatively loads a tile of data points into shared memory.
// - Each warp processes one query and iterates over the tile computing distances.
// - Within a warp, each lane computes distances for a strided subset of tile points.
// - Candidate insertions into the per-warp heap are handled warp-synchronously by lane 0.
// - The heap is a max-heap to allow fast pruning using the current worst (root) distance.
// - At the end, the heap is heap-sorted to ascending order and written to the output.
// The implementation assumes modern NVIDIA GPUs (Ampere/Hopper) and uses warp shuffles/ballots.

static inline __device__ void heap_swap(float &ad, int &ai, float &bd, int &bi) {
    float td = ad; ad = bd; bd = td;
    int   ti = ai; ai = bi; bi = ti;
}

static inline __device__ void heap_bubble_up_max(float* __restrict__ dist, int* __restrict__ idx, int pos) {
    // Bubble-up to maintain max-heap property (parent >= child)
    int curr = pos;
    while (curr > 0) {
        int parent = (curr - 1) >> 1;
        if (dist[parent] >= dist[curr]) break;
        heap_swap(dist[parent], idx[parent], dist[curr], idx[curr]);
        curr = parent;
    }
}

static inline __device__ void heapify_down_max(float* __restrict__ dist, int* __restrict__ idx, int size) {
    // Heapify-down from root for a max-heap
    int curr = 0;
    while (true) {
        int left  = (curr << 1) + 1;
        int right = left + 1;
        if (left >= size) break;
        int largest = left;
        if (right < size && dist[right] > dist[left]) largest = right;
        if (dist[curr] >= dist[largest]) break;
        heap_swap(dist[curr], idx[curr], dist[largest], idx[largest]);
        curr = largest;
    }
}

__global__ void knn_warp_kernel(const float2* __restrict__ query,
                                int query_count,
                                const float2* __restrict__ data,
                                int data_count,
                                int k,
                                int tile_points,
                                std::pair<int, float>* __restrict__ result)
{
    extern __shared__ unsigned char smem[];
    // Layout of shared memory:
    // [data tile: tile_points * sizeof(float2)] [per-warp heaps: W*k floats][per-warp heaps: W*k ints]
    float2* s_data = reinterpret_cast<float2*>(smem);

    const int warpsPerBlock = blockDim.x >> 5;
    const int warpIdInBlock = threadIdx.x >> 5;
    const int lane          = threadIdx.x & 31;
    const unsigned fullMask = __activemask();

    // Heaps base pointers after the tile region
    size_t tile_bytes = static_cast<size_t>(tile_points) * sizeof(float2);
    unsigned char* heaps_base = smem + tile_bytes;

    float* s_heap_dist = reinterpret_cast<float*>(heaps_base);
    int*   s_heap_idx  = reinterpret_cast<int*>(s_heap_dist + static_cast<size_t>(warpsPerBlock) * k);

    // Pointers to this warp's heap region
    float* heap_dist = s_heap_dist + static_cast<size_t>(warpIdInBlock) * k;
    int*   heap_idx  = s_heap_idx  + static_cast<size_t>(warpIdInBlock) * k;

    const int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    const bool warp_active = (globalWarpId < query_count);

    float2 q;
    if (warp_active) {
        q = query[globalWarpId];
    }

    // Per-warp heap size tracked by lane 0
    int heap_size = 0;

    // Process dataset in tiles; the whole block loads tiles cooperatively.
    for (int tileStart = 0; tileStart < data_count; tileStart += tile_points) {
        const int nTile = min(tile_points, data_count - tileStart);

        // Cooperative load of data tile into shared memory
        for (int i = threadIdx.x; i < nTile; i += blockDim.x) {
            s_data[i] = data[tileStart + i];
        }
        __syncthreads();

        if (warp_active) {
            // Iterate over tile entries, striding by warp size for coalesced reuse of the tile cache
            for (int i = lane; i < nTile; i += warpSize) {
                float2 p = s_data[i];
                float dx = q.x - p.x;
                float dy = q.y - p.y;
                float dist = fmaf(dx, dx, dy * dy); // squared L2

                int idx = tileStart + i;

                // Lane 0 broadcasts current threshold (root of heap) to the warp.
                // For heap_size < k, threshold is +inf so all candidates are considered.
                float thr = CUDART_INF_F;
                int hs = 0;
                if (lane == 0) {
                    hs = heap_size;
                    if (hs >= k) {
                        thr = heap_dist[0];
                    }
                }
                thr = __shfl_sync(fullMask, thr, 0);
                hs  = __shfl_sync(fullMask, hs, 0);

                // Warp-wide ballot to find which lanes have candidates worth considering
                unsigned mask = __ballot_sync(fullMask, (hs < k) || (dist < thr));

                // Lane 0 sequentially processes the set of candidate lanes for this micro-batch.
                // This reduces insertion work when many candidates are pruned by the current threshold.
                while (mask) {
                    int srcLane = __ffs(mask) - 1;
                    float cand_d = __shfl_sync(fullMask, dist, srcLane);
                    int   cand_i = __shfl_sync(fullMask, idx,  srcLane);

                    if (lane == 0) {
                        if (heap_size < k) {
                            int pos = heap_size++;
                            heap_dist[pos] = cand_d;
                            heap_idx[pos]  = cand_i;
                            heap_bubble_up_max(heap_dist, heap_idx, pos);
                        } else if (cand_d < heap_dist[0]) {
                            heap_dist[0] = cand_d;
                            heap_idx[0]  = cand_i;
                            heapify_down_max(heap_dist, heap_idx, k);
                        }
                    }

                    // Clear processed bit
                    mask &= (mask - 1);
                }
            }
        }

        __syncthreads(); // Ensure the tile buffer can be reused safely
    }

    // Finalize: per-warp heap sort to ascending order and store results
    if (warp_active && lane == 0) {
        // Ensure heap is fully built; since data_count >= k, heap_size should be k.
        // But we guard anyway.
        int n = heap_size;
        if (n > 1) {
            // Standard heapsort over a max-heap yields ascending order.
            for (int i = n - 1; i > 0; --i) {
                heap_swap(heap_dist[0], heap_idx[0], heap_dist[i], heap_idx[i]);
                // Restore heap property on range [0, i)
                int curr = 0;
                while (true) {
                    int left  = (curr << 1) + 1;
                    int right = left + 1;
                    if (left >= i) break;
                    int largest = left;
                    if (right < i && heap_dist[right] > heap_dist[left]) largest = right;
                    if (heap_dist[curr] >= heap_dist[largest]) break;
                    heap_swap(heap_dist[curr], heap_idx[curr], heap_dist[largest], heap_idx[largest]);
                    curr = largest;
                }
            }
        }

        // If, for any reason, heap_size < k (shouldn't happen per assumptions), pad with infinities/invalid indices.
        // Then write results.
        std::pair<int, float>* out = result + static_cast<size_t>(globalWarpId) * k;
        int outN = min(n, k);
        for (int j = 0; j < outN; ++j) {
            out[j].first  = heap_idx[j];
            out[j].second = heap_dist[j];
        }
        for (int j = outN; j < k; ++j) {
            out[j].first  = -1;
            out[j].second = CUDART_INF_F;
        }
    }
}

// Host-side launcher.
// This function configures the kernel launch parameters, selects the number of warps per block,
// and the tile size for shared-memory caching. It also opts the kernel into using the maximum
// available dynamic shared memory when supported (Ampere/Hopper).
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    int dev = 0;
    cudaGetDevice(&dev);

    int maxSmemDefault = 0;
    int maxSmemOptin   = 0;
    cudaDeviceGetAttribute(&maxSmemDefault, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&maxSmemOptin,   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    // Attempt to allow the kernel to use the opt-in maximum dynamic shared memory.
    if (maxSmemOptin > 0) {
        cudaFuncSetAttribute(knn_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmemOptin);
    }

    // Use the largest available shared memory capacity for dynamic allocation.
    size_t smemLimit = (maxSmemOptin > 0) ? static_cast<size_t>(maxSmemOptin)
                                          : static_cast<size_t>(maxSmemDefault);

    // Per-warp heap bytes (dist + idx)
    const size_t heapBytesPerWarp = static_cast<size_t>(k) * (sizeof(float) + sizeof(int));

    // Choose warps per block (W). Try up to 8 warps/block. Reduce if shared memory is tight.
    int chosenWarps = 0;
    int tile_points = 0;

    for (int W = 8; W >= 1; --W) {
        size_t heapsBytes = heapBytesPerWarp * static_cast<size_t>(W);
        if (heapsBytes >= smemLimit) continue; // need room for at least one data point tile entry

        // Allocate the remaining shared memory to the data tile.
        size_t remaining = smemLimit - heapsBytes;
        int tp = static_cast<int>(remaining / sizeof(float2));
        if (tp >= 1) {
            // Prefer larger tiles but keep reasonable minimum; no explicit upper cap needed.
            chosenWarps = W;
            tile_points = tp;
            break;
        }
    }

    // Fallback safety (should not trigger on A100/H100); ensure at least one warp and tile point.
    if (chosenWarps == 0) {
        chosenWarps = 1;
        tile_points = 1;
    }

    const int threadsPerBlock = chosenWarps * 32;
    const dim3 block(threadsPerBlock, 1, 1);
    const dim3 grid((query_count + chosenWarps - 1) / chosenWarps, 1, 1);

    // Dynamic shared memory size: tile bytes + heap bytes for all warps in block
    size_t dynamicSmemBytes = static_cast<size_t>(tile_points) * sizeof(float2)
                            + heapBytesPerWarp * static_cast<size_t>(chosenWarps);

    knn_warp_kernel<<<grid, block, dynamicSmemBytes>>>(query, query_count, data, data_count, k, tile_points, result);
}