#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>

// Define macros for shared‐memory batch size and merge interval.
// BATCH_SIZE: number of data points loaded per batch from global memory.
// MERGE_INTERVAL: number of data points processed within a batch before performing a warp‐level merge.
#define BATCH_SIZE 1024
#define MERGE_INTERVAL 128

// Define a simple pair structure for the result.
struct Pair {
    int first;    // index of the neighbor in the data set
    float second; // squared Euclidean distance to the query point
};

//---------------------------------------------------------------------------
// Device kernel: each warp (32 threads) processes one query point and
// computes its k nearest neighbors. The k best candidates are maintained
// in registers distributed evenly among the 32 lanes (each holds k_per_thread = k/32 candidates).
// The global data set is processed in batches loaded into shared memory.
// Every MERGE_INTERVAL candidates processed from shared memory, the warp
// merges its distributed top-k arrays into a globally-sorted list (via bitonic sort)
// using warp-local shared memory buffers.
/// @FIXED
/// extern "C" __global__
__global__
void knn_kernel(const float2 *query, int query_count,
                const float2 *data, int data_count,
/// @FIXED
///                 int k, Pair *result,
                int k, Pair *result)
                // dynamic shared memory: used for per-warp merge buffers.
                // Layout: first (numWarpsPerBlock * k) floats for distances,
                // then (numWarpsPerBlock * k) ints for indices.
/// @FIXED (-1:0)
///                 char *dynShMem)
{
    // Compute the warp id within the block and the lane id within the warp.
    const int warpSize = 32;
    int lane   = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    // Total number of warps per block.
    int warpsPerBlock = blockDim.x / warpSize;

    // Global warp id (each warp processes one query).
    int globalWarpId = warpId + warpsPerBlock * blockIdx.x;
    if (globalWarpId >= query_count) {
        return; // no query to process for this warp
    }

    // Load the query point for this warp.
    float2 q = query[globalWarpId];

    // Compute how many candidates each thread in the warp will hold.
    // (Assume k is always a power of two and k >= 32).
    int k_per_thread = k / warpSize;

    // Each thread maintains its own local sorted (ascending order) list of k_per_thread candidate distances and indices.
    // The list is maintained in registers.
    float localTopk[1024/32];  // maximum k_per_thread is 1024/32 = 32.
    int   localTopkIdx[1024/32];
    // Initialize with "infinite" distances and invalid indices.
    for (int i = 0; i < k_per_thread; i++){
        localTopk[i] = FLT_MAX;
        localTopkIdx[i] = -1;
    }

    // Declare a block-level shared memory for the current batch of data points.
    __shared__ float2 s_batch[BATCH_SIZE];
    // Pointer to the dynamic shared memory used for warp merge buffers.
    // Partition dynamic shmem: first part for distances, second for indices.
    // Each warp gets a contiguous block of k floats and k ints.
    /// @FIXED (-0:+1)
    extern __shared__ char dynShMem[];
    float *warpMergeDist = (float *) dynShMem;
    int   *warpMergeIdx  = (int *) (dynShMem + warpsPerBlock * k * sizeof(float));

    // Process the entire data set in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {

        // Compute number of points in this batch.
        int batch_size = ((batch_start + BATCH_SIZE) <= data_count) ? BATCH_SIZE : (data_count - batch_start);

        // Each thread in the block cooperatively loads data points for this batch into shared memory.
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            s_batch[i] = data[batch_start + i];
        }
        __syncthreads();

        // Process the batch in chunks for periodic merge.
        for (int chunk_start = 0; chunk_start < batch_size; chunk_start += MERGE_INTERVAL) {
            int chunk_end = chunk_start + MERGE_INTERVAL;
            if (chunk_end > batch_size) {
                chunk_end = batch_size;
            }

            // Each warp processes the current chunk: distribute indices among lanes.
            for (int j = chunk_start + lane; j < chunk_end; j += warpSize) {
                // Load the candidate data point from shared memory.
                float2 pt = s_batch[j];
                // Compute squared Euclidean distance.
                float dx = q.x - pt.x;
                float dy = q.y - pt.y;
                float dist = dx * dx + dy * dy;
                int dataIndex = batch_start + j; // global index

                // Check if candidate is promising compared to the worst candidate in this thread's list.
                if (dist < localTopk[k_per_thread - 1]) {
                    // Insertion sort into the local list (which is sorted in ascending order).
                    int pos = k_per_thread - 1;
                    // Shift larger elements down.
                    while (pos > 0 && dist < localTopk[pos - 1]) {
                        localTopk[pos] = localTopk[pos - 1];
                        localTopkIdx[pos] = localTopkIdx[pos - 1];
                        pos--;
                    }
                    localTopk[pos] = dist;
                    localTopkIdx[pos] = dataIndex;
                }
            } // end loop over current chunk

            // ----- Warp-level merge of the distributed top-k structure -----
            // Each warp has 32 threads, each with k_per_thread candidates.
            // They collectively form an array of k candidates. We merge them
            // into a globally sorted list (ascending order) using a bitonic sort.

            // Compute pointer for this warp's merge buffer.
            float *myWarpDist = warpMergeDist + warpId * k;
            int   *myWarpIdx  = warpMergeIdx  + warpId * k;

            // Each thread writes its local list to the merge buffer.
            int base = lane * k_per_thread;
            for (int i = 0; i < k_per_thread; i++) {
                myWarpDist[base + i] = localTopk[i];
                myWarpIdx[base + i]  = localTopkIdx[i];
            }
            // Use warp-level synchronization.
            __syncwarp();

            // Have one thread (lane 0) in the warp perform an in-place bitonic sort
            // on the merged array of k elements.
            if (lane == 0) {
                // Bitonic sort: k is a power-of-two.
                for (int size = 2; size <= k; size <<= 1) {
                    for (int stride = size >> 1; stride > 0; stride >>= 1) {
                        for (int i = 0; i < k; i++) {
                            int ixj = i ^ stride;
                            if (ixj > i) {
                                // Determine sorting direction.
                                bool ascending = ((i & size) == 0);
                                if (ascending) {
                                    if (myWarpDist[i] > myWarpDist[ixj]) {
                                        float temp   = myWarpDist[i];
                                        myWarpDist[i] = myWarpDist[ixj];
                                        myWarpDist[ixj] = temp;
                                        int tempIdx  = myWarpIdx[i];
                                        myWarpIdx[i] = myWarpIdx[ixj];
                                        myWarpIdx[ixj] = tempIdx;
                                    }
                                } else {
                                    if (myWarpDist[i] < myWarpDist[ixj]) {
                                        float temp   = myWarpDist[i];
                                        myWarpDist[i] = myWarpDist[ixj];
                                        myWarpDist[ixj] = temp;
                                        int tempIdx  = myWarpIdx[i];
                                        myWarpIdx[i] = myWarpIdx[ixj];
                                        myWarpIdx[ixj] = tempIdx;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncwarp();
            // Redistribute the merged sorted results back to the per-thread local arrays.
            for (int i = 0; i < k_per_thread; i++) {
                localTopk[i] = myWarpDist[lane * k_per_thread + i];
                localTopkIdx[i] = myWarpIdx[lane * k_per_thread + i];
            }
            __syncwarp();
            // End of warp-level merge for this chunk.
        } // end chunk loop over current batch

        __syncthreads();
        // End processing of current batch.
    } // end batch loop over entire data set

    // ----- Final merge: merge the distributed top-k data one more time to ensure sorted order -----
    {
        float *myWarpDist = warpMergeDist + warpId * k;
        int   *myWarpIdx  = warpMergeIdx  + warpId * k;
        int base = lane * k_per_thread;
        for (int i = 0; i < k_per_thread; i++) {
            myWarpDist[base + i] = localTopk[i];
            myWarpIdx[base + i]  = localTopkIdx[i];
        }
        __syncwarp();
        if (lane == 0) {
            for (int size = 2; size <= k; size <<= 1) {
                for (int stride = size >> 1; stride > 0; stride >>= 1) {
                    for (int i = 0; i < k; i++) {
                        int ixj = i ^ stride;
                        if (ixj > i) {
                            bool ascending = ((i & size) == 0);
                            if (ascending) {
                                if (myWarpDist[i] > myWarpDist[ixj]) {
                                    float temp   = myWarpDist[i];
                                    myWarpDist[i] = myWarpDist[ixj];
                                    myWarpDist[ixj] = temp;
                                    int tempIdx  = myWarpIdx[i];
                                    myWarpIdx[i] = myWarpIdx[ixj];
                                    myWarpIdx[ixj] = tempIdx;
                                }
                            } else {
                                if (myWarpDist[i] < myWarpDist[ixj]) {
                                    float temp   = myWarpDist[i];
                                    myWarpDist[i] = myWarpDist[ixj];
                                    myWarpDist[ixj] = temp;
                                    int tempIdx  = myWarpIdx[i];
                                    myWarpIdx[i] = myWarpIdx[ixj];
                                    myWarpIdx[ixj] = tempIdx;
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncwarp();
    }

    // ----- Write the final sorted k nearest neighbors to global memory. -----
    // The final merged array for this warp is in the warp merge buffer.
    int final_offset = globalWarpId * k;
    {
        float *myWarpDist = warpMergeDist + warpId * k;
        int   *myWarpIdx  = warpMergeIdx  + warpId * k;
        // Each thread writes its portion.
        for (int i = 0; i < k_per_thread; i++) {
            int idx = lane * k_per_thread + i;
            if (idx < k) {
                result[final_offset + idx].first = myWarpIdx[idx];
                result[final_offset + idx].second = myWarpDist[idx];
            }
        }
    }
}

//---------------------------------------------------------------------------
// Host function: run_knn
// This function sets up the grid and block dimensions, calculates the required
// dynamic shared memory size, and launches the k-NN kernel. It assumes that the
// pointers 'query', 'data', and 'result' have been allocated on the device via cudaMalloc.
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
/// @FIXED
///                         Pair *result, int k)
                        std::pair<int, float> *result, int k)
{
    // Choose number of threads per block.
    // We use 128 threads per block (i.e., 4 warps per block).
    int threadsPerBlock = 128;
    const int warpSize = 32;
    int warpsPerBlock = threadsPerBlock / warpSize;

    // Each warp processes one query.
    // Total number of warps needed = query_count.
    int totalWarps = query_count;
    // Compute number of blocks required.
    int blocks = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    // Compute dynamic shared memory size per block.
    // We need storage for the per-warp merge buffers.
    // Each warp requires space for k floats and k ints.
    // Total bytes = (warpsPerBlock * k * sizeof(float)) + (warpsPerBlock * k * sizeof(int)).
    size_t dshmemSize = warpsPerBlock * k * (sizeof(float) + sizeof(int));

    // Launch the kernel.
    /// @FIXED
    /// knn_kernel<<<blocks, threadsPerBlock, dshmemSize>>>(query, query_count, data, data_count, k, result, (char*)nullptr);
    knn_kernel<<<blocks, threadsPerBlock, dshmemSize>>>(query, query_count, data, data_count, k, reinterpret_cast<Pair*>(result));
    // Note: We pass a null pointer for the dynamic shmem parameter in the kernel launch.
    // The pointer will be provided automatically by the CUDA runtime using the dynamically allocated shmem.

    // (In an actual production code, you may want to check cudaGetLastError and synchronize.)
}
