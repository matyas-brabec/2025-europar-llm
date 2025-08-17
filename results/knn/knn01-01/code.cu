#include <cuda_runtime.h>
#include <math_constants.h>
#include <utility>

// This implementation assigns one thread block per query point.
// Each block scans the entire data set in tiles of size k and maintains the current
// top-k nearest neighbors in shared memory. For each tile, we merge the current top-k
// with the k new candidates by performing an in-place bitonic sort on 2k elements
// in shared memory and then keep the first k elements as the updated top-k.
// This approach avoids any additional global memory allocations and keeps the
// memory footprint and synchronization confined within a block.
//
// Notes:
// - k is assumed to be a power of two in [32, 1024] (as per problem statement).
// - The distance metric is squared Euclidean distance.
// - Shared memory required per block: 2*k*(sizeof(float)+sizeof(int)) bytes (<= 16KB for k=1024).
// - threads per block: 256 (chosen as a good default for A100/H100).
//
// Performance considerations:
// - The kernel is memory-bandwidth bound for large datasets; the selection cost per tile is kept
//   moderate by sorting 2k elements per tile (bitonic sort).
// - Memory loads are coalesced as threads within a block read contiguous data indices.
// - The union arrays are maintained in shared memory to minimize global traffic.

struct PairIF {
    int   first;
    float second;
};

// In-place bitonic sort of N pairs (keys = dists ascending, values = idxs) in shared memory.
// N must be a power of two. Sorting order: ascending by distance; ties broken by index to be deterministic.
__device__ inline void bitonic_sort_pairs_asc(float* dists, int* idxs, int N) {
    // Standard bitonic sort network operating on shared memory arrays.
    // Each thread processes multiple elements in strides of blockDim.x.
    for (int size = 2; size <= N; size <<= 1) {
        // Sort in ascending order when ((i & size) == 0), descending otherwise.
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    bool ascending = ((i & size) == 0);
                    float di = dists[i];
                    float dj = dists[ixj];
                    int   ii = idxs[i];
                    int   ij = idxs[ixj];

                    // Compare by distance first, then by index to break ties deterministically.
                    bool should_swap;
                    if (ascending) {
                        should_swap = (di > dj) || ((di == dj) && (ii > ij));
                    } else {
                        should_swap = (di < dj) || ((di == dj) && (ii < ij));
                    }
                    if (should_swap) {
                        dists[i]  = dj;
                        dists[ixj] = di;
                        idxs[i]   = ij;
                        idxs[ixj]  = ii;
                    }
                }
            }
        }
    }
    __syncthreads();
}

__global__ void knn2d_kernel(const float2* __restrict__ query,
                              int query_count,
                              const float2* __restrict__ data,
                              int data_count,
                              PairIF* __restrict__ result,
                              int k)
{
    int qid = blockIdx.x;
    if (qid >= query_count) return;

    // Shared memory layout:
    // [0 .. 2k-1] float union_dists
    // [0 .. 2k-1] int   union_idxs
    extern __shared__ unsigned char smem[];
    float* union_dists = reinterpret_cast<float*>(smem);
    int*   union_idxs  = reinterpret_cast<int*>(union_dists + 2 * k);

    const int two_k = 2 * k;

    // Load query point into registers
    float2 q = query[qid];

    // Initialize the union arrays to +inf and -1.
    // Lower half [0..k-1] will hold the current best k results (initialized to +inf).
    // Upper half [k..2k-1] will be filled with the current tile's candidates.
    for (int i = threadIdx.x; i < two_k; i += blockDim.x) {
        union_dists[i] = CUDART_INF_F;
        union_idxs[i]  = -1;
    }
    __syncthreads();

    // Process the dataset in tiles of size k
    for (int base = 0; base < data_count; base += k) {
        const int tile_count = min(k, data_count - base);

        // Fill the upper half of union arrays with distances for the current tile.
        // For positions beyond tile_count, fill with +inf to keep N=2k consistent.
        for (int t = threadIdx.x; t < k; t += blockDim.x) {
            if (t < tile_count) {
                int idx = base + t;
                float2 p = data[idx];
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                // Squared L2 distance; use FMA to improve throughput/precision slightly.
                float d = __fmaf_rn(dx, dx, dy * dy);
                union_dists[k + t] = d;
                union_idxs[k + t]  = idx;
            } else {
                union_dists[k + t] = CUDART_INF_F;
                union_idxs[k + t]  = -1;
            }
        }
        __syncthreads();

        // Sort 2k elements; the smallest k will end up in positions [0..k-1]
        bitonic_sort_pairs_asc(union_dists, union_idxs, two_k);
        // No need to move/copy: union[0..k-1] now holds the updated best k,
        // union[k..2k-1] can be overwritten by the next tile.
    }

    // Write out the final top-k neighbors for this query
    // The output array is row-major: result[qid * k + j] corresponds to the j-th NN.
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        PairIF r;
        r.first  = union_idxs[j];
        r.second = union_dists[j];
        result[static_cast<size_t>(qid) * static_cast<size_t>(k) + static_cast<size_t>(j)] = r;
    }
}

// Host-side launcher
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Ensure our device-side PairIF layout matches std::pair<int,float>
    static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "PairIF must match std::pair<int,float> layout");

    const int threads_per_block = 256;  // Good default for A100/H100
    const int blocks = query_count;     // One block per query

    // Shared memory: 2k floats + 2k ints
    const size_t smem_bytes = static_cast<size_t>(2 * k) * (sizeof(float) + sizeof(int));

    PairIF* result_alias = reinterpret_cast<PairIF*>(result);

    knn2d_kernel<<<blocks, threads_per_block, smem_bytes>>>(query, query_count, data, data_count, result_alias, k);

    // Optional synchronization to ensure completion before returning.
    // cudaDeviceSynchronize();
}