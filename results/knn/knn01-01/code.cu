#include <cuda_runtime.h>
#include <float.h>
#include <utility>

/*
 * Highly optimized brute-force k-NN for 2D points (float2) on modern NVIDIA GPUs.
 *
 * Algorithm (per query point):
 *  - Process the data set in fixed-size chunks (CHUNK_SIZE points per chunk).
 *  - For each chunk:
 *      * All threads in the block compute squared Euclidean distances to the query.
 *      * Distances and indices for the whole chunk are stored in shared memory.
 *      * A parallel bitonic sort is executed on the chunk in shared memory
 *        to sort all distances in ascending order.
 *      * The best local_k = min(k, chunk_size) neighbors from this chunk
 *        (i.e., the first local_k entries after sort) are merged with the
 *        current global top-k list for this query, which is stored in shared
 *        memory and kept sorted.
 *  - After all chunks are processed, the global top-k for each query is written
 *    to the result array in ascending order of distance.
 *
 * Properties:
 *  - No dynamic device memory allocation is performed.
 *  - All heavy work (distance computation and per-chunk selection) is parallelized.
 *  - Merging partial results between chunks is O(k) and done by a single thread.
 *  - k is supported up to MAX_K (1024) and can vary at runtime (power of two in [32,1024]).
 *  - Data layout:
 *      query:  float2[query_count]
 *      data:   float2[data_count]
 *      result: std::pair<int,float>[query_count * k]
 */

constexpr int BLOCK_SIZE  = 256;   // Threads per block; tuned for modern GPUs
constexpr int MAX_K       = 1024;  // Maximum supported k (assumed by problem)
constexpr int CHUNK_SIZE  = 2048;  // Number of data points processed per chunk (power of two)

static_assert((CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0,
              "CHUNK_SIZE must be a power of two for bitonic sort.");

/*
 * In-place bitonic sort on shared memory arrays of length N (power-of-two).
 *
 * - dist: array of distances to sort (ascending).
 * - idx:  parallel array of indices to permute along with dist.
 * - N:    compile-time constant equal to CHUNK_SIZE.
 *
 * The implementation is the standard bitonic sort network, parallelized over
 * blockDim.x threads. Each thread processes multiple indices (stride blockDim.x).
 */
template <int N>
__device__ __forceinline__ void bitonic_sort_shared(float *dist, int *idx)
{
    // Outer loop: size of subsequences being merged (k)
    for (int k = 2; k <= N; k <<= 1) {
        // Inner loop: distance between elements to compare (j)
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each thread handles multiple indices: tid, tid + blockDim.x, ...
            for (int tid = threadIdx.x; tid < N; tid += blockDim.x) {
                int ixj = tid ^ j;  // Partner index for comparison
                if (ixj > tid) {
                    float di = dist[tid];
                    float dj = dist[ixj];
                    int   ii = idx[tid];
                    int   ij = idx[ixj];

                    bool ascending = ((tid & k) == 0);
                    // For ascending sequences, keep smaller at lower index.
                    // For descending sequences, keep larger at lower index.
                    bool do_swap = ascending ? (di > dj) : (di < dj);

                    if (do_swap) {
                        dist[tid] = dj;
                        dist[ixj] = di;
                        idx[tid]  = ij;
                        idx[ixj]  = ii;
                    }
                }
            }
            __syncthreads();
        }
    }
}

/*
 * Kernel: compute k nearest neighbors for 2D points.
 *
 * Mapping:
 *  - One block handles one query point.
 *  - All threads within the block cooperate on scanning the full data set,
 *    chunk by chunk.
 */
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           int k,
                           std::pair<int, float> * __restrict__ result)
{
    int qid = blockIdx.x;
    if (qid >= query_count) {
        return;
    }

    // Shared memory for per-block (per-query) computations.
    __shared__ float  s_dist[CHUNK_SIZE];   // Distances for current chunk
    __shared__ int    s_idx[CHUNK_SIZE];    // Corresponding data indices

    __shared__ float  topk_dist[MAX_K];     // Global top-k distances for this query (sorted)
    __shared__ int    topk_idx[MAX_K];      // Global top-k indices for this query (sorted)

    __shared__ float  temp_dist[MAX_K];     // Temporary buffer for merging
    __shared__ int    temp_idx[MAX_K];      // Temporary buffer for merging

    __shared__ int    s_current_k;          // Current number of valid entries in topk arrays
    __shared__ float2 s_query;              // Query point for this block

    // Initialize query and current_k once per block
    if (threadIdx.x == 0) {
        s_query     = query[qid];
        s_current_k = 0;
    }
    __syncthreads();

    float qx = s_query.x;
    float qy = s_query.y;

    int num_chunks = (data_count + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Process data in chunks
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int base = chunk * CHUNK_SIZE;
        int chunk_size = data_count - base;
        if (chunk_size > CHUNK_SIZE) {
            chunk_size = CHUNK_SIZE;
        }

        // Compute distances for this chunk in parallel
        for (int offset = threadIdx.x; offset < CHUNK_SIZE; offset += blockDim.x) {
            int idx_global = base + offset;
            if (idx_global < data_count) {
                float2 p = data[idx_global];
                float dx = p.x - qx;
                float dy = p.y - qy;
                s_dist[offset] = dx * dx + dy * dy;  // squared Euclidean distance
                s_idx[offset]  = idx_global;
            } else {
                // Out-of-range entries are filled with +INF so they sort to the end
                s_dist[offset] = FLT_MAX;
                s_idx[offset]  = -1;
            }
        }
        __syncthreads();

        // Sort the entire chunk (CHUNK_SIZE elements) by distance ascending
        bitonic_sort_shared<CHUNK_SIZE>(s_dist, s_idx);
        __syncthreads();

        // Thread 0 merges this chunk's best local_k neighbors with the global top-k
        if (threadIdx.x == 0) {
            int cur_k   = s_current_k;
            int local_k = k;
            if (local_k > chunk_size) {
                local_k = chunk_size;
            }

            // Merge two sorted lists:
            //  - topk_dist[0..cur_k)
            //  - s_dist[0..local_k)
            // into temp_dist[0..new_k), keeping at most k elements.
            int i = 0;      // index in global top-k
            int j = 0;      // index in chunk best
            int t = 0;      // index in temp

            // Merge until one list is exhausted or we have k elements
            while (t < k && i < cur_k && j < local_k) {
                if (topk_dist[i] <= s_dist[j]) {
                    temp_dist[t] = topk_dist[i];
                    temp_idx[t]  = topk_idx[i];
                    ++i;
                } else {
                    temp_dist[t] = s_dist[j];
                    temp_idx[t]  = s_idx[j];
                    ++j;
                }
                ++t;
            }

            // Copy any remaining elements from global top-k
            while (t < k && i < cur_k) {
                temp_dist[t] = topk_dist[i];
                temp_idx[t]  = topk_idx[i];
                ++i;
                ++t;
            }

            // Copy any remaining elements from chunk's best
            while (t < k && j < local_k) {
                temp_dist[t] = s_dist[j];
                temp_idx[t]  = s_idx[j];
                ++j;
                ++t;
            }

            // t is the new current_k (never exceeds k)
            s_current_k = t;

            // Copy merged result back into topk arrays
            for (int m = 0; m < t; ++m) {
                topk_dist[m] = temp_dist[m];
                topk_idx[m]  = temp_idx[m];
            }
        }
        __syncthreads();
    }

    // Write final top-k results for this query to global memory.
    // topk_dist/topk_idx are sorted ascending by distance.
    int final_k = s_current_k;  // Should be exactly k since data_count >= k
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        int out_idx = qid * k + j;
        if (j < final_k) {
            result[out_idx].first  = topk_idx[j];
            result[out_idx].second = topk_dist[j];
        } else {
            // Safety fallback; should not occur when data_count >= k.
            result[out_idx].first  = -1;
            result[out_idx].second = FLT_MAX;
        }
    }
}

/*
 * Host entry point.
 *
 * Parameters:
 *  - query:       device pointer to array of query_count float2 points
 *  - query_count: number of queries
 *  - data:        device pointer to array of data_count float2 points
 *  - data_count:  number of data points
 *  - result:      device pointer to array of size query_count * k of std::pair<int,float>
 *  - k:           number of nearest neighbors to find (power of two in [32,1024])
 *
 * Assumptions:
 *  - All pointers are device pointers allocated with cudaMalloc.
 *  - data_count >= k.
 *  - query_count, data_count are large enough to benefit from GPU parallelism.
 */
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0) {
        return;
    }

    // Clamp k to supported maximum; per problem statement, this should never trigger.
    if (k > MAX_K) {
        k = MAX_K;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(query_count);  // One block per query

    knn_kernel<<<grid, block>>>(query, query_count, data, data_count, k, result);

    // Synchronize to ensure results are ready when run_knn returns.
    // For maximum performance in more complex applications, the caller
    // may choose to remove this synchronize and manage streams explicitly.
    cudaDeviceSynchronize();
}