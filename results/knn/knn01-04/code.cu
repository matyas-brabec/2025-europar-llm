#include <cuda_runtime.h>
#include <cfloat>
#include <utility>  // for std::pair

// -----------------------------------------------------------------------------
// This kernel performs k-nearest neighbors search for a single 2D query point.
// Each block is responsible for one query point. Inside each block, a fixed
// number of threads cooperatively scan all data points to compute the
// k smallest squared Euclidean distances. Each thread processes a strided
// portion from the global "data" array and maintains a small, local candidate
// list (of fixed size THREAD_CAND) of best neighbors seen in its portion.
// Then, all threads write their local candidate lists into shared memory;
// a bitonic sort is performed in shared memory to sort all candidates by
// distance (ascending order). Finally, the best k candidates (i.e., the first
// k elements of the sorted list) are written out to global memory.
// -----------------------------------------------------------------------------
//
// Hyper-parameter: number of candidates maintained per thread.
// Chosen as a fixed constant (8) such that (blockDim.x * THREAD_CAND) is always
// at least k. (For worst-case k = 1024 and typical blockDim.x = 256, we have
// 256*8 = 2048 candidates to merge.)
//
#define THREAD_CAND 8

// -----------------------------------------------------------------------------
// Device kernel: each block processes one query point.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float>* result, int k)
{
    // Each block handles one query point.
    int qid = blockIdx.x;
    if (qid >= query_count) return;

    // Load the query point from global memory.
    float2 qPoint = query[qid];

    // Number of threads in the block.
    const int tN = blockDim.x;

    // Each thread maintains a local candidate list (of length THREAD_CAND)
    // storing the index and squared distance.
    int localIdx[THREAD_CAND];
    float localDist[THREAD_CAND];

    // Initialize local candidate list with "worst" values.
    for (int i = 0; i < THREAD_CAND; i++) {
        localDist[i] = FLT_MAX;
        localIdx[i] = -1;
    }

    // Each thread loops over a chunk of the data points in a strided manner.
    // For each data point in its portion, compute the squared Euclidean distance.
    // Then, if the distance is smaller than the worst one in its local candidate
    // list, update the list.
    for (int i = threadIdx.x; i < data_count; i += tN) {
        float2 dPoint = data[i];
        float dx = dPoint.x - qPoint.x;
        float dy = dPoint.y - qPoint.y;
        float dist = dx * dx + dy * dy;

        // Linear search of the current local candidate list for the worst candidate.
        int worstIdx = 0;
        float worstVal = localDist[0];
        #pragma unroll
        for (int j = 1; j < THREAD_CAND; j++) {
            float val = localDist[j];
            if (val > worstVal) {
                worstVal = val;
                worstIdx = j;
            }
        }
        // If the new distance is better than the worst stored, update.
        if (dist < worstVal) {
            localDist[worstIdx] = dist;
            localIdx[worstIdx] = i;
        }
    }

    // -------------------------------------------------------------------------
    // Now, each thread has a local candidate list of THREAD_CAND elements.
    // We need to merge the results from all threads.
    //
    // We use shared memory to hold all candidate pairs from the block.
    // Total number of candidates is: totalCand = tN * THREAD_CAND.
    // We allocate shared memory dynamically. The layout is as follows:
    // [  int s_idx[totalCand] | float s_dist[totalCand] ]
    //
    int totalCand = tN * THREAD_CAND;
    extern __shared__ char smem[];
    int *s_idx = (int*) smem;
    float *s_dist = (float*) (s_idx + totalCand);

    // Each thread copies its local candidate list to shared memory.
    int offset = threadIdx.x * THREAD_CAND;
    #pragma unroll
    for (int j = 0; j < THREAD_CAND; j++) {
        s_idx[offset + j]  = localIdx[j];
        s_dist[offset + j] = localDist[j];
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // Perform an in-block bitonic sort on the shared candidates.
    // Sort order: ascending by distance.
    // We assume totalCand is a power of two. For example, with tN = 256 and
    // THREAD_CAND = 8, totalCand = 2048.
    //
    // Bitonic sort algorithm:
    //  for (size = 2; size <= totalCand; size *= 2)
    //    for (stride = size/2; stride > 0; stride /= 2)
    //      for each index i in 0 .. totalCand-1 (each handled by some thread):
    //          int ixj = i ^ stride;
    //          if (ixj > i) then
    //              if ( (i & size)==0 ? (s_dist[i] > s_dist[ixj])
    //                                   : (s_dist[i] < s_dist[ixj]) )
    //                  swap(s_dist[i], s_dist[ixj]); swap(s_idx[i], s_idx[ixj]);
    //
    for (int size = 2; size <= totalCand; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            // Each thread processes multiple indices with stride equal to tN.
            for (int i = threadIdx.x; i < totalCand; i += tN) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    // Determine the sort direction: ascending if (i & size) == 0.
                    bool ascending = ((i & size) == 0);
                    // Compare and swap based on ascending order.
                    if (ascending) {
                        if (s_dist[i] > s_dist[ixj]) {
                            float tmpD = s_dist[i];
                            s_dist[i] = s_dist[ixj];
                            s_dist[ixj] = tmpD;

                            int tmpIdx = s_idx[i];
                            s_idx[i] = s_idx[ixj];
                            s_idx[ixj] = tmpIdx;
                        }
                    } else {
                        if (s_dist[i] < s_dist[ixj]) {
                            float tmpD = s_dist[i];
                            s_dist[i] = s_dist[ixj];
                            s_dist[ixj] = tmpD;

                            int tmpIdx = s_idx[i];
                            s_idx[i] = s_idx[ixj];
                            s_idx[ixj] = tmpIdx;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // -------------------------------------------------------------------------
    // After the bitonic sort the first k elements in shared memory correspond
    // to the k smallest distances.
    // Write out the sorted k results to the output global array.
    // Each result is a pair: (data index, squared distance).
    for (int i = threadIdx.x; i < k; i += tN) {
        // Write result for query 'qid' at result offset (qid * k + i)
        result[qid * k + i] = std::make_pair(s_idx[i], s_dist[i]);
    }
    // End of kernel.
}

// -----------------------------------------------------------------------------
// Host interface: run_knn
//
// Parameters:
//   query       - pointer to queries (each a float2)
//   query_count - number of query points
//   data        - pointer to dataset points (each a float2)
//   data_count  - number of data points (>= k)
//   result      - pointer to output array where each query gets k results;
//                 result[i*k + j] corresponds to j-th nearest neighbor for query[i].
//   k           - number of nearest neighbors to retrieve (power of two between 32 and 1024)
// -----------------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k)
{
    // Choose block size; here we use 256 threads per block.
    const int blockSize = 256;
    // One query point is processed per block.
    int gridSize = query_count;

    // Calculate the amount of shared memory required per block.
    // We need to allocate shared memory for totalCand candidate indices and distances.
    // totalCand = blockSize * THREAD_CAND.
    // Shared memory bytes = totalCand * (sizeof(int) + sizeof(float))
    int totalCand = blockSize * THREAD_CAND;
    int sharedMemBytes = totalCand * (sizeof(int) + sizeof(float));

    // Launch the kernel.
    knn_kernel<<<gridSize, blockSize, sharedMemBytes>>>(query, query_count,
                                                        data, data_count,
                                                        result, k);
    // It is assumed that the caller checks for errors and synchronizes as needed.
}
