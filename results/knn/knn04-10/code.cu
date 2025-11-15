#include <cuda_runtime.h>
#include <utility>

// Maximum k supported by this implementation (power of two, up to 1024 as per problem statement)
static constexpr int MAX_K = 1024;
static constexpr int WARPS_PER_BLOCK = 8;      // 8 warps per block -> 256 threads
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
static constexpr int TILE_SIZE = 1024;         // Number of data points cached in shared memory per iteration

// CUDA kernel implementing k-NN for 2D points.
// Each warp processes a single query point and maintains a private max-heap of size k
// in shared memory to store the current k nearest neighbors (distance and index).
// Data points are processed in tiles, each tile cached in shared memory for reuse.
template<int TILE_POINTS, int WARPS_PER_BLK>
__global__ void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    std::pair<int, float>* __restrict__ result,
    int k)
{
    // Layout of dynamic shared memory:
    // [ float2 s_data[TILE_POINTS] ]
    // [ float warp_heap_dists[WARPS_PER_BLK * MAX_K] ]
    // [ int   warp_heap_idxs [WARPS_PER_BLK * MAX_K] ]
    // [ int   warp_heap_sizes[WARPS_PER_BLK] ]
    extern __shared__ unsigned char shared[];
    unsigned char* smem_ptr = shared;

    // Shared-memory tile of data points
    float2* s_data = reinterpret_cast<float2*>(smem_ptr);
    smem_ptr += TILE_POINTS * sizeof(float2);

    // Per-warp heaps: distances and indices
    float* s_heap_dists = reinterpret_cast<float*>(smem_ptr);
    smem_ptr += WARPS_PER_BLK * MAX_K * sizeof(float);

    int* s_heap_idxs = reinterpret_cast<int*>(smem_ptr);
    smem_ptr += WARPS_PER_BLK * MAX_K * sizeof(int);

    // Per-warp heap sizes
    int* s_heap_sizes = reinterpret_cast<int*>(smem_ptr);
    // smem_ptr += WARPS_PER_BLK * sizeof(int); // not needed further

    const int lane_id        = threadIdx.x & 31;          // thread index within warp [0,31]
    const int warp_in_block  = threadIdx.x >> 5;          // warp index within block
    const int warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; // global warp index

    if (warp_global_id >= query_count)
        return; // No query to process for this warp

    // Per-warp heap pointers
    float* heap_dists = s_heap_dists + warp_in_block * MAX_K;
    int*   heap_idxs  = s_heap_idxs  + warp_in_block * MAX_K;
    int&   heap_size  = s_heap_sizes[warp_in_block];

    // Initialize per-warp heap size once
    if (lane_id == 0) {
        heap_size = 0;
    }
    __syncwarp();

    // Load the query point for this warp and broadcast to all lanes
    float2 q;
    if (lane_id == 0) {
        q = query[warp_global_id];
    }
    q.x = __shfl_sync(0xFFFFFFFFu, q.x, 0);
    q.y = __shfl_sync(0xFFFFFFFFu, q.y, 0);

    // Iterate over data points in tiles
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_POINTS) {
        int tile_size = TILE_POINTS;
        if (tile_start + tile_size > data_count) {
            tile_size = data_count - tile_start;
        }

        // Load tile into shared memory cooperatively by the whole block
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            s_data[i] = data[tile_start + i];
        }
        __syncthreads(); // Ensure tile is fully cached before any warp uses it

        // Process the tile: each warp processes all points in the tile.
        // We step in batches of 32 points so that each lane handles one candidate,
        // and the warp leader (lane 0) integrates these 32 candidates into the
        // per-warp max-heap via warp shuffles.
        for (int j_base = 0; j_base < tile_size; j_base += warpSize) {
            int j = j_base + lane_id;

            float dist = CUDART_INF_F;
            int   data_idx = -1;

            if (j < tile_size) {
                float2 p = s_data[j];
                float dx = q.x - p.x;
                float dy = q.y - p.y;
                // Squared Euclidean distance
                dist = dx * dx + dy * dy;
                data_idx = tile_start + j;
            }

            // Active lanes in this batch (those with a valid candidate)
            unsigned int active_mask = __ballot_sync(0xFFFFFFFFu, j < tile_size);
            if (active_mask == 0)
                continue; // No candidates in this batch for this warp

            // Warp leader integrates all candidates in this batch into the heap.
            if (lane_id == 0) {
                int local_heap_size = heap_size;

                // Iterate over all active lanes in this batch
                while (active_mask) {
                    int src_lane = __ffs(active_mask) - 1; // index of a lane with a candidate

                    float c_dist = __shfl_sync(0xFFFFFFFFu, dist, src_lane);
                    int   c_idx  = __shfl_sync(0xFFFFFFFFu, data_idx, src_lane);

                    // Max-heap insertion / update:
                    // - While heap not full: push and sift-up.
                    // - When full: if candidate better than root, replace root and sift-down.
                    if (local_heap_size < k) {
                        // Insert new element at the end and sift-up
                        int i = local_heap_size++;
                        while (i > 0) {
                            int parent = (i - 1) >> 1;
                            float parent_val = heap_dists[parent];
                            if (parent_val >= c_dist)
                                break;
                            heap_dists[i] = parent_val;
                            heap_idxs[i]  = heap_idxs[parent];
                            i = parent;
                        }
                        heap_dists[i] = c_dist;
                        heap_idxs[i]  = c_idx;
                    } else {
                        // Heap is full; candidate must be better than current worst (root)
                        float root_dist = heap_dists[0];
                        if (c_dist < root_dist) {
                            // Replace root and sift-down
                            int i = 0;
                            int size = local_heap_size;
                            int left = 1;
                            while (left < size) {
                                int right = left + 1;
                                int largest = left;
                                float largest_val = heap_dists[left];
                                if (right < size) {
                                    float right_val = heap_dists[right];
                                    if (right_val > largest_val) {
                                        largest = right;
                                        largest_val = right_val;
                                    }
                                }
                                if (largest_val <= c_dist)
                                    break;
                                heap_dists[i] = largest_val;
                                heap_idxs[i]  = heap_idxs[largest];
                                i = largest;
                                left = (i << 1) + 1;
                            }
                            heap_dists[i] = c_dist;
                            heap_idxs[i]  = c_idx;
                        }
                    }

                    // Clear this lane from the active mask and continue with next
                    active_mask &= active_mask - 1;
                }

                heap_size = local_heap_size;
            }

            // Synchronize warp before next batch so all lanes see updated heap state
            __syncwarp();
        }

        __syncthreads(); // Ensure all warps finish using this tile before loading a new one
    }

    // At this point, heap_dists/heap_idxs contain a max-heap of size k with the k nearest neighbors
    // (max-heap: root is the worst among the k best).
    // We now convert the heap into an array sorted by ascending distance using in-place heapsort.
    __syncwarp();
    if (lane_id == 0) {
        int n = heap_size; // should be equal to k since data_count >= k
        // Heapsort on max-heap to obtain ascending order in-place
        for (int i = n - 1; i > 0; --i) {
            // Swap root (current maximum) with position i
            float tmp_dist = heap_dists[0];
            int   tmp_idx  = heap_idxs[0];
            heap_dists[0] = heap_dists[i];
            heap_idxs[0]  = heap_idxs[i];
            heap_dists[i] = tmp_dist;
            heap_idxs[i]  = tmp_idx;

            // Sift-down from root within reduced heap [0, i)
            int size = i;
            int parent = 0;
            while (true) {
                int left = (parent << 1) + 1;
                if (left >= size)
                    break;
                int right = left + 1;
                int largest = left;
                float largest_val = heap_dists[left];
                if (right < size) {
                    float right_val = heap_dists[right];
                    if (right_val > largest_val) {
                        largest = right;
                        largest_val = right_val;
                    }
                }
                if (largest_val <= heap_dists[parent])
                    break;
                float pd = heap_dists[parent];
                int   pi = heap_idxs[parent];
                heap_dists[parent] = heap_dists[largest];
                heap_idxs[parent]  = heap_idxs[largest];
                heap_dists[largest] = pd;
                heap_idxs[largest]  = pi;
                parent = largest;
            }
        }
    }
    __syncwarp();

    // Now heap_dists[0..k-1] is sorted in ascending order of distance.
    // Write out results cooperatively: each lane handles a strided subset of neighbors.
    const int base_out = warp_global_id * k;
    for (int i = lane_id; i < k; i += warpSize) {
        int   nn_idx = heap_idxs[i];
        float nn_dst = heap_dists[i];
        result[base_out + i].first  = nn_idx;
        result[base_out + i].second = nn_dst;
    }
}

// Host function interface.
// query, data, and result are assumed to be device pointers allocated with cudaMalloc.
// query_count: number of query points
// data_count : number of data points
// k          : number of nearest neighbors (power of two between 32 and 1024)
void run_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k)
{
    // Guard against unsupported k at runtime
    if (k <= 0 || k > MAX_K) {
        // For production code, you might want to handle this case differently
        return;
    }

    const int warps_per_block = WARPS_PER_BLOCK;
    const int threads_per_block = THREADS_PER_BLOCK;

    // Each warp processes one query
    int total_warps = query_count;
    int num_blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    // Dynamic shared memory size
    size_t shared_bytes =
        sizeof(float2) * TILE_SIZE +                         // s_data
        warps_per_block * MAX_K * sizeof(float) +            // s_heap_dists
        warps_per_block * MAX_K * sizeof(int)   +            // s_heap_idxs
        warps_per_block * sizeof(int);                       // s_heap_sizes

    knn_kernel<TILE_SIZE, WARPS_PER_BLOCK>
        <<<num_blocks, threads_per_block, shared_bytes>>>(query, query_count, data, data_count, result, k);
}
