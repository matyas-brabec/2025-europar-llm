#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <utility>

// In this implementation we assume that each query (2D point) is processed by one warp (32 threads).
// The kernel maintains a private (register‐stored) intermediate result ("IR") of the query's best k candidates,
// and for each query a candidate buffer stored in shared memory is used to accumulate new candidates (from batched data)
// whose squared distances are lower than the current maximum (i.e. k-th smallest) distance.
// When the candidate buffer gets full (its count reaches k), it is merged with the IR.
// The merging is done cooperatively by the warp – here we let lane 0 perform the merge serially and then broadcast the new threshold.
// The input data (the data points array) is processed in batches: each batch is loaded into shared memory by the whole block,
// and then each warp processes the batch. We use atomicAdd on per‐warp candidate buffer counters.
// Note: This implementation uses a simple insertion-sort merge in lane0 for candidate-buffer sorting and a two‐array merge.
// It is tuned for modern NVIDIA GPUs and H100/A100 (with recent CUDA toolkit).

// Define the batch size (number of data points loaded per block iteration).
// We choose a moderate batch size to control shared memory usage.
#define BATCH_SIZE 128

// Kernel to process queries for k-NN.
// Each warp (32 threads) processes one query.
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result,
                           int k)
{
    // Each warp processes one query: obtain warp id and lane id.
    const int warpSizeVal = 32;
    int warpId = threadIdx.x / warpSizeVal;
    int lane = threadIdx.x % warpSizeVal;
    int warpsPerBlock = blockDim.x / warpSizeVal;
    // Global query index for this warp.
    int query_idx = blockIdx.x * warpsPerBlock + warpId;
    if (query_idx >= query_count)
        return;

    // Load query point (each warp has one query).
    float2 q = query[query_idx];

    // Shared memory layout for each thread block:
    // [0, warpsPerBlock * k * sizeof(pair<int,float>) )       : candidate buffers (one per warp)
    // [warpsPerBlock * k * sizeof(pair<int,float>), warpsPerBlock * k * sizeof(pair<int,float>) + warpsPerBlock * sizeof(int))
    //                                                            : candidate counts (one int per warp)
    // [ ... next, warpsPerBlock * (2 * k) * sizeof(pair<int,float>) )
    //                                                            : merge scratch area (one per warp, size = 2*k candidates)
    // [ ... next, BATCH_SIZE * sizeof(float2) ]                 : shared buffer for a batch of data points.
    extern __shared__ char smem[];

    // Ptr to candidate buffer.
    std::pair<int, float>* candBuffer = (std::pair<int, float>*)smem;
    // Ptr to candidate counts; immediately following candidate buffers.
    int* candCount = (int*)(smem + warpsPerBlock * k * sizeof(std::pair<int, float>));
    // Ptr to merge scratch area; immediately following candidate counts.
    std::pair<int, float>* mergeScratch = (std::pair<int, float>*)
        (smem + warpsPerBlock * k * sizeof(std::pair<int, float>) + warpsPerBlock * sizeof(int));
    // Ptr to shared data batch buffer; following merge scratch.
    float2* sharedData = (float2*)(smem + warpsPerBlock * k * sizeof(std::pair<int, float>) +
                                   warpsPerBlock * sizeof(int) +
                                   warpsPerBlock * (2 * k) * sizeof(std::pair<int, float>));

    // Per-warp pointers in shared memory.
    std::pair<int, float>* myCandBuffer = &candBuffer[warpId * k];
    int* myCandCount = &candCount[warpId];
    std::pair<int, float>* myMergeScratch = &mergeScratch[warpId * (2 * k)];

    // Initialize candidate buffer count (only one thread in the warp does it).
    if (lane == 0) {
        *myCandCount = 0;
    }
    __syncwarp();

    // We maintain the intermediate result (IR) in registers.
    // Since k can be up to 1024, and k is determined at runtime, we allocate fixed‐size arrays (max size 1024).
    // Only lane 0 holds the full IR; other threads keep only the threshold (currentMax) updated via shuffles.
    // IR is kept sorted in ascending order (so the kth candidate, i.e. IR[k-1], holds the current maximum).
    const int maxK = 1024; // maximum allowed k
    // Declare arrays in registers; we assume k <= maxK.
    __shared__ int dummy; // used only for control flow, not for storing IR in shared mem.
    // Only lane 0 will store the full IR in its local arrays.
    int IR_idx[maxK];
    float IR_dist[maxK];

    // Initialize IR with dummy values: index = -1 and distance = FLT_MAX.
    if (lane == 0) {
        for (int i = 0; i < k; i++) {
            IR_idx[i] = -1;
            IR_dist[i] = FLT_MAX;
        }
    }
    // Broadcast the current threshold (max distance in IR) from lane 0.
    float currentMax = __shfl_sync(0xFFFFFFFF, (lane == 0 ? IR_dist[k - 1] : 0.0f), 0);

    // Process input data in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE)
    {
        int batch_count = (batch_start + BATCH_SIZE <= data_count) ? BATCH_SIZE : (data_count - batch_start);

        // Each thread in the block loads one or more data points into shared memory.
        for (int i = threadIdx.x; i < batch_count; i += blockDim.x)
        {
            sharedData[i] = data[batch_start + i];
        }
        __syncthreads();

        // Each warp iterates over the data points in the shared batch.
        for (int i = 0; i < batch_count; i++)
        {
            // All lanes compute the squared Euclidean distance for this data point.
            float2 p = sharedData[i];
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float dist = dx * dx + dy * dy;

            // If the new candidate is promising (distance less than current threshold)
            if (dist < currentMax)
            {
                // Insert candidate into our warp's candidate buffer using atomicAdd.
                int pos = atomicAdd(myCandCount, 1);
                // Only store if within candidate buffer capacity (k elements).
                if (pos < k)
                {
                    myCandBuffer[pos] = std::pair<int, float>(batch_start + i, dist);
                }
            }

            // If candidate buffer is full (or overfull), merge it with the intermediate result.
            if (*myCandCount >= k)
            {
                // Let lane 0 perform the merging.
                if (lane == 0)
                {
                    int m = *myCandCount;
                    // We merge at most k candidates from the candidate buffer.
                    if (m > k)
                        m = k;

                    // --- Step 1: Sort the candidate buffer (unsorted) from index 0 to m-1 ---
                    // Copy candidate buffer to merge scratch area.
                    for (int j = 0; j < m; j++) {
                        myMergeScratch[j] = myCandBuffer[j];
                    }
                    // Simple insertion sort on merge scratch.
                    for (int j = 1; j < m; j++) {
                        std::pair<int, float> key = myMergeScratch[j];
                        int l = j - 1;
                        while (l >= 0 && myMergeScratch[l].second > key.second) {
                            myMergeScratch[l + 1] = myMergeScratch[l];
                            l--;
                        }
                        myMergeScratch[l + 1] = key;
                    }

                    // --- Step 2: Merge sorted candidate array (length m) with IR (length k) ---
                    // IR is maintained sorted in ascending order.
                    // We'll perform a two‐array merge to form a new sorted array of size (k + m),
                    // then keep the first k entries.
                    int i_ir = 0, i_cand = 0, out = 0;
                    // Temporary buffer (allocated on lane0's stack) for merged result.
                    std::pair<int, float> merged[2048]; // max 2*k (k<=1024) elements.
                    while (out < k && (i_ir < k || i_cand < m)) {
                        std::pair<int, float> cand_ir = (i_ir < k) ? std::pair<int, float>(IR_idx[i_ir], IR_dist[i_ir])
                                                                  : std::pair<int, float>(-1, FLT_MAX);
                        std::pair<int, float> cand_buf = (i_cand < m) ? myMergeScratch[i_cand]
                                                                    : std::pair<int, float>(-1, FLT_MAX);
                        if (cand_ir.second <= cand_buf.second) {
                            merged[out++] = cand_ir;
                            i_ir++;
                        }
                        else {
                            merged[out++] = cand_buf;
                            i_cand++;
                        }
                    }
                    // Update IR with merged result.
                    for (int j = 0; j < k; j++) {
                        IR_idx[j] = merged[j].first;
                        IR_dist[j] = merged[j].second;
                    }
                    // Update the threshold.
                    currentMax = IR_dist[k - 1];
                    // Reset candidate buffer count.
                    *myCandCount = 0;
                }
                // All lanes in the warp synchronize to ensure merge completion.
                __syncwarp();
                // Broadcast updated threshold.
                currentMax = __shfl_sync(0xFFFFFFFF, currentMax, 0);
            } // end merge check
        } // end batch loop over sharedData
        __syncthreads(); // Ensure block synchronization before next batch load.
    } // end loop over data batches

    // After all batches, merge any remaining candidates (if candidate buffer is non-empty).
    if (*myCandCount > 0)
    {
        if (lane == 0)
        {
            int m = *myCandCount;
            if (m > k)
                m = k;
            // Copy remaining candidates to merge scratch.
            for (int j = 0; j < m; j++) {
                myMergeScratch[j] = myCandBuffer[j];
            }
            // Insertion sort on merge scratch.
            for (int j = 1; j < m; j++) {
                std::pair<int, float> key = myMergeScratch[j];
                int l = j - 1;
                while (l >= 0 && myMergeScratch[l].second > key.second) {
                    myMergeScratch[l + 1] = myMergeScratch[l];
                    l--;
                }
                myMergeScratch[l + 1] = key;
            }
            // Merge sorted remaining candidates with IR.
            int i_ir = 0, i_cand = 0, out = 0;
            std::pair<int, float> merged[2048];
            while (out < k && (i_ir < k || i_cand < m)) {
                std::pair<int, float> cand_ir = (i_ir < k) ? std::pair<int, float>(IR_idx[i_ir], IR_dist[i_ir])
                                                          : std::pair<int, float>(-1, FLT_MAX);
                std::pair<int, float> cand_buf = (i_cand < m) ? myMergeScratch[i_cand]
                                                             : std::pair<int, float>(-1, FLT_MAX);
                if (cand_ir.second <= cand_buf.second) {
                    merged[out++] = cand_ir;
                    i_ir++;
                }
                else {
                    merged[out++] = cand_buf;
                    i_cand++;
                }
            }
            // Update IR and threshold.
            for (int j = 0; j < k; j++) {
                IR_idx[j] = merged[j].first;
                IR_dist[j] = merged[j].second;
            }
            currentMax = IR_dist[k - 1];
            *myCandCount = 0;
        }
        __syncwarp();
        currentMax = __shfl_sync(0xFFFFFFFF, currentMax, 0);
    }

    // Write out the final k-NN result for this query.
    // Only lane 0 writes the complete sorted IR (k nearest neighbors) to global memory.
    if (lane == 0) {
        // The j-th nearest neighbor for query 'query_idx' is stored at result[query_idx*k + j].
        for (int j = 0; j < k; j++) {
            result[query_idx * k + j] = std::pair<int, float>(IR_idx[j], IR_dist[j]);
        }
    }
}

// Host interface for running k-NN on 2D points. All device arrays are assumed to have been allocated via cudaMalloc.
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Choose block configuration:
    // We assign one warp (32 threads) per query.
    // To increase occupancy we group multiple queries per block.
    // Here we choose blockDim.x = 256 (i.e. 8 warps per block).
    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock / 32;
    // Compute grid size to cover all queries.
    int numWarpsNeeded = (query_count + 0 - 1) / 1;  // one warp per query
    int numBlocks = (numWarpsNeeded + warpsPerBlock - 1) / warpsPerBlock;

    // Compute required dynamic shared memory size.
    // Layout:
    //   candidate buffers: warpsPerBlock * k * sizeof(std::pair<int, float>)
    //   candidate counts: warpsPerBlock * sizeof(int)
    //   merge scratch: warpsPerBlock * (2 * k) * sizeof(std::pair<int,float>)
    //   data batch buffer: BATCH_SIZE * sizeof(float2)
    size_t shCandidates = warpsPerBlock * k * sizeof(std::pair<int, float>);
    size_t shCandCounts = warpsPerBlock * sizeof(int);
    size_t shMergeScratch = warpsPerBlock * (2 * k) * sizeof(std::pair<int, float>);
    size_t shDataBatch = BATCH_SIZE * sizeof(float2);
    size_t shMemTotal = shCandidates + shCandCounts + shMergeScratch + shDataBatch;

    // Launch the kernel.
    knn_kernel<<<numBlocks, threadsPerBlock, shMemTotal>>>(query, query_count, data, data_count, result, k);
    cudaDeviceSynchronize();
}