#include <cuda_runtime.h>
#include <utility>

// This file implements a CUDA kernel for k-nearest neighbors (k-NN) of 2D points,
// optimized for modern NVIDIA data center GPUs (A100/H100).
//
// Key design points:
// - One warp (32 threads) processes one query point.
// - Data points are processed in tiles cached in shared memory to minimize global memory traffic.
// - Each warp maintains a private max-heap of size k (in shared memory) holding the current k nearest neighbors.
//   The heap keeps the largest (worst) distance at the root so new candidates only trigger updates when they beat the current worst.
// - Warp threads cooperatively compute distances; new candidates are inserted into the heap by lane 0 using warp shuffles.
// - After processing all data, the heap is popped to produce k nearest neighbors in ascending order of distance.
// - The kernel uses dynamic shared memory sized at launch based on k and the chosen tile size.
// - The result is written as std::pair<int, float> but we avoid using std facilities in device code by reinterpreting it as a POD struct.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// POD pair with the same field layout as std::pair<int, float> (first, second).
// We avoid using std::pair operations in device code and only write raw fields.
struct PairIF {
    int first;
    float second;
};

static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "PairIF must match std::pair<int, float> size");

// Max-heap helper functions (lane 0 only).
// The heap stores distances (float) and their corresponding indices (int) in parallel arrays.
// Invariants: For a max-heap, parent >= children. The root at index 0 is the largest distance.

// Sift-up during insertion to maintain max-heap property.
__device__ __forceinline__
void heap_sift_up(float* __restrict__ dist, int* __restrict__ idx, int pos, float val, int id)
{
    while (pos > 0) {
        int parent = (pos - 1) >> 1;
        float parentVal = dist[parent];
        if (parentVal >= val) break;
        dist[pos] = parentVal;
        idx[pos] = idx[parent];
        pos = parent;
    }
    dist[pos] = val;
    idx[pos] = id;
}

// Insert a (distance, index) pair into the heap of current size 'size' (size < k).
__device__ __forceinline__
void heap_insert_max(float* __restrict__ dist, int* __restrict__ idx, int size, float val, int id)
{
    heap_sift_up(dist, idx, size, val, id);
}

// Sift-down during root replacement to maintain max-heap property.
__device__ __forceinline__
void heap_sift_down(float* __restrict__ dist, int* __restrict__ idx, int size, float val, int id)
{
    int pos = 0;
    int child = 1;
    while (child < size) {
        int right = child + 1;
        int largestChild = (right < size && dist[right] > dist[child]) ? right : child;
        float lcVal = dist[largestChild];
        if (lcVal <= val) break;
        dist[pos] = lcVal;
        idx[pos] = idx[largestChild];
        pos = largestChild;
        child = (pos << 1) + 1;
    }
    dist[pos] = val;
    idx[pos] = id;
}

// Replace the root (largest) value with a new smaller value and restore heap property.
__device__ __forceinline__
void heap_replace_root(float* __restrict__ dist, int* __restrict__ idx, int size, float val, int id)
{
    // Precondition: val < dist[0] and size > 0
    heap_sift_down(dist, idx, size, val, id);
}

__global__ void knn2d_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    int k,
    PairIF* __restrict__ result,
    int tile_points)
{
    // Dynamic shared memory layout:
    // [ float2 s_data[tile_points] ][ float s_heapDist[WARPS_PER_BLOCK * k] ][ int s_heapIdx[WARPS_PER_BLOCK * k] ]
    extern __shared__ unsigned char smem[];
    float2* s_data = reinterpret_cast<float2*>(smem);

    const int threads_per_block = blockDim.x;
    const int warps_per_block = threads_per_block / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x >> 5;
    const int warp_global = blockIdx.x * warps_per_block + warp;

    // Compute shared memory offsets for per-warp heaps.
    size_t off = static_cast<size_t>(tile_points) * sizeof(float2);
    // Align to 8-byte boundary to avoid misaligned accesses.
    off = (off + 7) & ~size_t(7);

    float* s_heapDist_base = reinterpret_cast<float*>(smem + off);
    int*   s_heapIdx_base  = reinterpret_cast<int*>(s_heapDist_base + warps_per_block * k);

    float* s_heapDist = s_heapDist_base + warp * k;
    int*   s_heapIdx  = s_heapIdx_base  + warp * k;

    // All warps must participate in __syncthreads, even if they have no query to process.
    // So we guard computations but not barriers.
    const bool warp_has_query = (warp_global < query_count);

    // Initialize per-warp heap size and worst distance (tau).
    int heapSize = 0;
    float tau = __int_as_float(0x7f800000); // +inf

    // Each warp's query point loaded once.
    float2 q = make_float2(0.0f, 0.0f);
    if (warp_has_query) {
        q = query[warp_global];
    }

    // Process data in tiles cached in shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        int tile_count = data_count - tile_start;
        if (tile_count > tile_points) tile_count = tile_points;

        // Load tile into shared memory cooperatively by the block.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            s_data[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes the tile for its query.
        if (warp_has_query) {
            // Iterate over points in the tile, each lane handles a strided subset.
            for (int i = lane; i < tile_count; i += WARP_SIZE) {
                float2 p = s_data[i];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                // Fused multiply-adds to compute squared distance.
                float dist = fmaf(dx, dx, dy * dy);
                int gidx = tile_start + i;

                // Broadcast current tau and heapSize from lane 0 to all lanes so that each lane can do a quick precheck.
                unsigned mask_full = __activemask();
                float tau_bcast = __shfl_sync(mask_full, tau, 0);
                int heapSize_bcast = __shfl_sync(mask_full, heapSize, 0);

                // Precheck to reduce number of candidates reaching the heap:
                // - If heap is not yet full, we must insert.
                // - Else insert only if dist < current worst (tau).
                int needs_insert = (heapSize_bcast < k) || (dist < tau_bcast);

                unsigned insert_mask = __ballot_sync(mask_full, needs_insert);

                // Process all lanes that need insertion. Lane 0 performs heap updates.
                while (insert_mask) {
                    int lead_lane = __ffs(insert_mask) - 1;
                    float dist_lead = __shfl_sync(mask_full, dist, lead_lane);
                    int   idx_lead  = __shfl_sync(mask_full, gidx, lead_lane);

                    if (lane == 0) {
                        if (heapSize < k) {
                            // Insert directly until heap is full.
                            heap_insert_max(s_heapDist, s_heapIdx, heapSize, dist_lead, idx_lead);
                            heapSize++;
                            // Update tau (worst distance) when heap becomes full.
                            if (heapSize == k) {
                                tau = s_heapDist[0];
                            }
                        } else if (dist_lead < s_heapDist[0]) {
                            // Replace root with better candidate and fix heap.
                            heap_replace_root(s_heapDist, s_heapIdx, heapSize, dist_lead, idx_lead);
                            tau = s_heapDist[0];
                        }
                    }
                    insert_mask &= (insert_mask - 1);
                }
            }
        }

        __syncthreads(); // Ensure all warps finished reading s_data before the next tile is loaded.
    }

    // After processing all data, write out results sorted in ascending order of distance.
    if (warp_has_query && lane == 0) {
        // Pop max-heap repeatedly into output from the end to the beginning.
        int qbase = warp_global * k;
        int size = heapSize; // Should be equal to k (since data_count >= k).
        // If for any reason heapSize < k (e.g., pathological), fill the rest with +inf/-1.
        while (size > 0) {
            // Extract max (largest distance).
            float topd = s_heapDist[0];
            int   topi = s_heapIdx[0];

            // Move last to root and sift down.
            int last = size - 1;
            float lastd = s_heapDist[last];
            int   lasti = s_heapIdx[last];
            size--;
            if (size > 0) {
                heap_sift_down(s_heapDist, s_heapIdx, size, lastd, lasti);
            }

            // Place the extracted element at the end of the sorted range.
            int out_pos = qbase + size; // as size decreases, out_pos goes from qbase + k - 1 down to qbase
            // Writing as POD PairIF ensures device-side compatibility.
            result[out_pos].first = topi;
            result[out_pos].second = topd;
        }

        // If heapSize < k (shouldn't happen given data_count >= k), pad remaining entries with invalid index and +inf.
        for (int r = heapSize; r < k; ++r) {
            int out_pos = qbase + r;
            result[out_pos].first = -1;
            result[out_pos].second = __int_as_float(0x7f800000); // +inf
        }
    }
}

// Host function to launch the kernel.
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) return;

    const int threads_per_block = 256; // 8 warps per block for good balance of occupancy and shared memory usage.
    const int warps_per_block = threads_per_block / WARP_SIZE;
    const int grid_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Determine a tile size that fits into available dynamic shared memory along with per-warp heaps.
    int device = 0;
    cudaGetDevice(&device);

    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_smem_optin == 0) {
        // Fallback to default maximum if opt-in attribute is unavailable.
        cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlock, device);
    }

    size_t per_warp_heap_bytes = static_cast<size_t>(k) * (sizeof(float) + sizeof(int));
    size_t heaps_bytes = static_cast<size_t>(warps_per_block) * per_warp_heap_bytes;

    // Choose the largest tile_points that fits in shared memory budget.
    // Reserve some safety margin (e.g., 1024 bytes) for alignment overhead.
    size_t safety = 1024;
    int tile_points = 0;
    if (max_smem_optin > static_cast<int>(heaps_bytes + safety)) {
        size_t available = static_cast<size_t>(max_smem_optin) - heaps_bytes - safety;
        tile_points = static_cast<int>(available / sizeof(float2));
    }
    // Ensure tile_points is sensible and aligned for coalesced loads.
    if (tile_points <= 0) tile_points = 1024; // minimal fallback
    // Align down to a multiple of threads_per_block to make loads coalesced across block.
    tile_points = (tile_points / threads_per_block) * threads_per_block;
    if (tile_points <= 0) tile_points = threads_per_block; // at least one block stride

    // Compute dynamic shared memory size.
    size_t smem_bytes = static_cast<size_t>(tile_points) * sizeof(float2) + heaps_bytes;
    // Try to set the kernel attribute to allow this much dynamic shared memory.
    cudaFuncSetAttribute(knn2d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

    // Reinterpret the result pointer as a POD to avoid std::pair device code usage.
    PairIF* result_pod = reinterpret_cast<PairIF*>(result);

    // Launch the kernel.
    knn2d_kernel<<<grid_blocks, threads_per_block, smem_bytes>>>(query, query_count, data, data_count, k, result_pod, tile_points);

    // Note: Error checking (cudaGetLastError/cudaDeviceSynchronize) is intentionally omitted as per instruction
    // to keep the code focused on the core kernel logic and to avoid extra host-side behavior.
}