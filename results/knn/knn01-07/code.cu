#include <cuda_runtime.h>
#include <cfloat>

// This implementation assigns one CUDA block to each query point.
// It scans the dataset in tiles of size k (where k is a power-of-two between 32 and 1024).
// For each tile, it computes distances to the query and sorts the tile's distances
// in ascending order using a block-wide bitonic sort in shared memory. It maintains
// a block-local top-k list (also sorted ascending) in shared memory and merges the
// sorted tile with the current top-k using a linear two-pointer merge (O(k)).
// This yields the k smallest distances across all tiles after the full scan.
// It avoids any global memory allocations and uses only per-block shared memory.

struct PairIF {
    int first;   // index in 'data'
    float second; // squared distance
};

// Block-wide bitonic sort in shared memory for K elements.
// Sorts keys[] ascending and permutes vals[] accordingly.
// Requires exactly K threads per block.
template<int K>
__device__ inline void bitonic_sort_shared(float* keys, int* vals) {
    // Standard bitonic sort network
    unsigned int tid = threadIdx.x;
    // Outer loop controls the size of the sequences to merge
    for (unsigned int size = 2; size <= (unsigned int)K; size <<= 1) {
        // Inner loop controls the stride of the compare-exchange
        for (unsigned int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            unsigned int partner = tid ^ stride;
            if (partner > tid) {
                bool up = ((tid & size) == 0); // sort direction for this pair
                float key_i = keys[tid];
                float key_p = keys[partner];
                int val_i = vals[tid];
                int val_p = vals[partner];
                // Decide if we need to swap
                bool do_swap = up ? (key_i > key_p) : (key_i < key_p);
                if (do_swap) {
                    keys[tid] = key_p;
                    keys[partner] = key_i;
                    vals[tid] = val_p;
                    vals[partner] = val_i;
                }
            }
        }
    }
    __syncthreads();
}

template<int K>
__global__ void knn2d_kernel(const float2* __restrict__ query,
                             int query_count,
                             const float2* __restrict__ data,
                             int data_count,
                             PairIF* __restrict__ result) {
    // Each block handles one query
    int qid = blockIdx.x;
    if (qid >= query_count) return;

    // Shared memory layout:
    // [0 .. K-1]   tile_dist (float)
    // [K .. 2K-1]  tile_idx  (int)
    // [2K .. 3K-1] topk_dist (float)
    // [3K .. 4K-1] topk_idx  (int)
    // [4K .. 5K-1] merge_dist (float)
    // [5K .. 6K-1] merge_idx  (int)
    extern __shared__ unsigned char smem_raw[];
    float* tile_dist  = reinterpret_cast<float*>(smem_raw);
    int*   tile_idx   = reinterpret_cast<int*>(tile_dist + K);
    float* topk_dist  = reinterpret_cast<float*>(tile_idx + K);
    int*   topk_idx   = reinterpret_cast<int*>(topk_dist + K);
    float* merge_dist = reinterpret_cast<float*>(topk_idx + K);
    int*   merge_idx  = reinterpret_cast<int*>(merge_dist + K);

    const int tid = threadIdx.x;

    // Load query into registers
    float2 q = query[qid];

    // Initialize top-k with +infinity distances and invalid indices
    if (tid < K) {
        topk_dist[tid] = CUDART_INF_F;
        topk_idx[tid] = -1;
    }
    __syncthreads();

    // Process dataset in tiles of K points
    const int tiles = (data_count + K - 1) / K;
    for (int t = 0; t < tiles; ++t) {
        int data_idx = t * K + tid;

        // Compute squared Euclidean distance for this thread's data element
        if (data_idx < data_count) {
            float2 p = data[data_idx];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            // Use FMA to accumulate dx*dx + dy*dy with improved throughput
            float dist = fmaf(dy, dy, dx * dx);
            tile_dist[tid] = dist;
            tile_idx[tid] = data_idx;
        } else {
            // Pad out-of-range elements with +inf so they sort to the end
            tile_dist[tid] = CUDART_INF_F;
            tile_idx[tid] = -1;
        }
        __syncthreads();

        // Sort tile distances ascending
        bitonic_sort_shared<K>(tile_dist, tile_idx);

        // Merge sorted tile (tile_dist/tile_idx) with current top-k (topk_dist/topk_idx)
        if (tid == 0) {
            int i = 0; // index into tile_dist
            int j = 0; // index into topk_dist
            for (int out = 0; out < K; ++out) {
                // Take the smaller of the two current candidates
                if (tile_dist[i] <= topk_dist[j]) {
                    merge_dist[out] = tile_dist[i];
                    merge_idx[out] = tile_idx[i];
                    // Advance in tile; i < K always true because tile is padded with +inf
                    ++i;
                } else {
                    merge_dist[out] = topk_dist[j];
                    merge_idx[out] = topk_idx[j];
                    // Advance in top-k; j < K always true because top-k is filled with +inf initially
                    ++j;
                }
            }
            // Copy merged result back into top-k buffers
            for (int x = 0; x < K; ++x) {
                topk_dist[x] = merge_dist[x];
                topk_idx[x] = merge_idx[x];
            }
        }
        __syncthreads();
    }

    // Write results: k nearest neighbors for this query, in ascending distance order
    if (tid < K) {
        int out_idx = qid * K + tid;
        result[out_idx].first  = topk_idx[tid];
        result[out_idx].second = topk_dist[tid];
    }
}

// Host-side launcher: selects a specialized kernel based on k for optimal performance.
// k is guaranteed to be a power-of-two between 32 and 1024 inclusive.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    if (query_count <= 0) return;
    if (k < 32 || k > 1024) return; // Basic guard; k is assumed valid per problem statement

    // Use a type with the same layout as std::pair<int, float> for device I/O
    PairIF* d_out = reinterpret_cast<PairIF*>(result);

    dim3 grid(query_count);
    size_t smem_bytes = 0;

    // Launch one specialized kernel per supported k to benefit from compile-time unrolling
    switch (k) {
        case 32: {
            dim3 block(32);
            smem_bytes = 6 * 32 * sizeof(int); // 3*float + 3*int == 6*4 bytes per element
            // Correct the byte count: 6 arrays * 32 elements * 4 bytes = 768 bytes
            smem_bytes = 6u * 32u * sizeof(int);
            knn2d_kernel<32><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, d_out);
            break;
        }
        case 64: {
            dim3 block(64);
            smem_bytes = 6u * 64u * sizeof(int);
            knn2d_kernel<64><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, d_out);
            break;
        }
        case 128: {
            dim3 block(128);
            smem_bytes = 6u * 128u * sizeof(int);
            knn2d_kernel<128><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, d_out);
            break;
        }
        case 256: {
            dim3 block(256);
            smem_bytes = 6u * 256u * sizeof(int);
            knn2d_kernel<256><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, d_out);
            break;
        }
        case 512: {
            dim3 block(512);
            smem_bytes = 6u * 512u * sizeof(int);
            knn2d_kernel<512><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, d_out);
            break;
        }
        case 1024: {
            dim3 block(1024);
            smem_bytes = 6u * 1024u * sizeof(int);
            knn2d_kernel<1024><<<grid, block, smem_bytes>>>(query, query_count, data, data_count, d_out);
            break;
        }
        default:
            // Should not happen given constraints; fall back to nearest lower power-of-two if needed
            // but we return to avoid producing incorrect results.
            return;
    }

    // Ensure kernel completion before returning (synchronous behavior)
    cudaDeviceSynchronize();
}