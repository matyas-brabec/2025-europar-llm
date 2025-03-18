#include <cuda_runtime.h>
#include <utility>   // for std::pair

// In our k-NN kernel we use squared Euclidean distances.
// We define a structure to hold a candidate neighbor.
struct Candidate {
    int idx;
    float dist;
};

// -----------------------------------------------------------------------------
// Define algorithm hyper-parameters.
// BLOCK_WARPS: number of warps per block processing queries.
// BLOCK_SIZE: total threads per block = BLOCK_WARPS * warpSize (32).
// BATCH_SIZE: number of data points loaded per batch.
#define BLOCK_WARPS    4
#define BLOCK_SIZE     (BLOCK_WARPS * 32)
#define BATCH_SIZE     256

// A very large number used as infinity.
#define INF_DIST 1e30f

// -----------------------------------------------------------------------------
// The merge_warp function merges the current private best results of a query
// (stored in registers per warp, distributed in a strided layout)
// with the candidate buffer (stored in shared memory and unsorted).
// The merged output (the k best candidates, sorted in ascending order by distance)
// is written to a temporary shared workspace (merge_out) for the warp,
// and then the private best registers are updated with it.
//
// The private best array is distributed among the 32 lanes:
// each thread holds L = k/32 elements such that the global best array element
// with index (i*32 + lane) is held in a register of lane 'lane' at index i.
// Note that k is guaranteed to be a power-of-two between 32 and 1024, so L is an integer.
// This function is called by all lanes in the warp; however, only lane 0 does the actual
// merging work (using a simple insertion sort for candidate buffer and a two‐pointer merge),
// and then the result is broadcast to all lanes.
__device__ inline void merge_warp(
    int k,         // number of neighbors (length of best array)
    int warp_lane, // lane ID within the warp (0..31)
    int L,         // number of elements per lane in best (k/32)
    Candidate best_local[],       // private best array in registers (distributed among lanes)
    Candidate* warp_candidate_buf, // pointer to this warp's candidate buffer (length k) in shared memory
    int* warp_candidate_count,     // pointer to this warp's candidate count (in shared memory)
    Candidate* merge_out           // pointer to this warp's merge workspace (length k) in shared memory
)
{
    // Use full mask for warp-wide intrinsics.
    const unsigned full_mask = 0xffffffff;

    // Only lane 0 performs the merge.
    if (warp_lane == 0) {
        // Temporary arrays allocated in local memory (maximum size 1024 is acceptable since k<=1024).
        Candidate temp_best[1024];  // will hold the current best array in contiguous layout.
        Candidate temp_buff[1024];  // will hold candidate buffer sorted in ascending order.

        // Copy the distributed best (held in registers) into a contiguous temporary array.
        // The global index for an element is: global_index = i*32 + lane, where i from 0 to L-1.
        // Lane 0 obtains each element by using a warp shuffle: the owner of global index 'i'
        // is (i % 32) and the element is in position (i/32) of that thread's private array.
        for (int i = 0; i < k; i++) {
            int owner = i % 32;      // lane that owns the i-th element.
            int pos = i / 32;        // local position in that lane.
            // Use __shfl_sync to obtain the fields from the proper lane.
            int best_idx = __shfl_sync(full_mask, best_local[pos].idx, owner);
            float best_dist = __shfl_sync(full_mask, best_local[pos].dist, owner);
            temp_best[i].idx = best_idx;
            temp_best[i].dist = best_dist;
        }

        // Read the candidate buffer count.
        int m = *warp_candidate_count;
        // Copy m candidate buffer entries into temp_buff.
        for (int i = 0; i < m; i++) {
            temp_buff[i] = warp_candidate_buf[i];
        }
        // Pad the rest (if m < k) with dummy candidates having INF_DIST.
        for (int i = m; i < k; i++) {
            temp_buff[i].idx = -1;
            temp_buff[i].dist = INF_DIST;
        }

        // Sort temp_buff (candidate buffer) in ascending order by distance using insertion sort.
        for (int i = 1; i < k; i++) {
            Candidate key = temp_buff[i];
            int j = i - 1;
            while (j >= 0 && temp_buff[j].dist > key.dist) {
                temp_buff[j+1] = temp_buff[j];
                j--;
            }
            temp_buff[j+1] = key;
        }

        // Now merge the two sorted arrays (temp_best and temp_buff) of length k each,
        // taking the first k smallest elements.
        Candidate merged[1024];  // temporary merged output, size k.
        int i_idx = 0, j_idx = 0;
        for (int cnt = 0; cnt < k; cnt++) {
            if(i_idx < k && j_idx < k) {
                if (temp_best[i_idx].dist <= temp_buff[j_idx].dist) {
                    merged[cnt] = temp_best[i_idx];
                    i_idx++;
                } else {
                    merged[cnt] = temp_buff[j_idx];
                    j_idx++;
                }
            } else if(i_idx < k) {
                merged[cnt] = temp_best[i_idx];
                i_idx++;
            } else {
                merged[cnt] = temp_buff[j_idx];
                j_idx++;
            }
        }
        // Write the merged best array into the merge workspace (contiguous).
        for (int i = 0; i < k; i++) {
            merge_out[i] = merged[i];
        }
        // Reset the candidate buffer count.
        *warp_candidate_count = 0;
    }
    // Synchronize all lanes in the warp.
    __syncwarp(full_mask);
    // Now, each lane loads from the merge workspace into its private best array.
    // We use the same distributed layout: element with global index = lane + i*32.
    for (int i = 0; i < L; i++) {
        int index = warp_lane + i * 32;
        if (index < k) {
            Candidate tmp = merge_out[index];
            best_local[i].idx = tmp.idx;
            best_local[i].dist = tmp.dist;
        }
    }
    __syncwarp(full_mask);
}

// -----------------------------------------------------------------------------
// Kernel implementing k-NN for 2D points.
// Each warp (32 threads) processes one query. It maintains a private best (k nearest)
// list in registers (distributed among lanes) and a candidate buffer in shared memory.
// Data points are processed in batches loaded into shared memory.
__global__ void knn_kernel(
    const float2 *query, int query_count,
    const float2 *data, int data_count,
    std::pair<int, float> *result,  // Output: result for each query is stored contiguously.
    int k)
{
    // Determine warp and lane identifiers.
    int warp_id_in_block = threadIdx.x / 32;  // 0 .. (blockDim.x/32 - 1)
    int lane = threadIdx.x % 32;               // 0 .. 31
    int warps_per_block = blockDim.x / 32;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    if (global_warp_id >= query_count) return; // out-of-range queries

    // Shared memory layout:
    // [0, BATCH_SIZE*sizeof(float2)) : shared_data array for data batch (float2[BATCH_SIZE])
    // [BATCH_SIZE*sizeof(float2), BATCH_SIZE*sizeof(float2) + (BLOCK_WARPS*k*sizeof(Candidate)) )
    //    : candidate buffer array (Candidate[BLOCK_WARPS*k])
    // Next, an int array for candidate counts: (BLOCK_WARPS * sizeof(int))
    // Next, merge workspace for best: Candidate[BLOCK_WARPS*k]
    extern __shared__ char smem[];
    float2 *shared_data = (float2*) smem;
    Candidate *candidate_buf = (Candidate*)(smem + BATCH_SIZE * sizeof(float2));
    int *candidate_count = (int*)(smem + BATCH_SIZE * sizeof(float2) + (BLOCK_WARPS * k * sizeof(Candidate)));
    Candidate *merge_out = (Candidate*)(smem + BATCH_SIZE * sizeof(float2) + (BLOCK_WARPS * k * sizeof(Candidate)) + (BLOCK_WARPS * sizeof(int)));

    // Each warp gets its candidate buffer and candidate count.
    Candidate *warp_candidate_buf = candidate_buf + warp_id_in_block * k;
    int *warp_candidate_count = candidate_count + warp_id_in_block;
    if (lane == 0) {
        *warp_candidate_count = 0;
    }
    __syncwarp();

    // Load query point corresponding to this warp.
    float2 q = query[global_warp_id];

    // Each warp maintains its intermediate best k neighbors in private registers.
    // They are stored in a distributed, strided layout.
    int L = k / 32;   // number of elements per lane.
    Candidate best_local[32];  // maximum L is 1024/32 = 32.
#pragma unroll
    for (int i = 0; i < L; i++) {
        best_local[i].idx = -1;
        best_local[i].dist = INF_DIST;
    }

    // Process input data in batches.
    for (int batch_start = 0; batch_start < data_count; batch_start += BATCH_SIZE) {
        int batch_size = (data_count - batch_start < BATCH_SIZE) ? (data_count - batch_start) : BATCH_SIZE;
        // Cooperative loading of current batch of data into shared memory.
        for (int i = threadIdx.x; i < BATCH_SIZE; i += blockDim.x) {
            if (i < batch_size) {
                shared_data[i] = data[batch_start + i];
            }
        }
        __syncthreads();

        // Each warp processes the shared data batch.
        // Each lane processes indices in shared_data with stride 32.
        for (int i = lane; i < batch_size; i += 32) {
            float2 pt = shared_data[i];
            float dx = pt.x - q.x;
            float dy = pt.y - q.y;
            float dist = dx * dx + dy * dy;

            // Get the current worst distance among the stored best neighbors.
            // The worst (largest) neighbor is at global index k-1.
            int worst_owner = (k - 1) % 32;
            int worst_pos = (k - 1) / 32;
            float worst_dist = __shfl_sync(0xffffffff, best_local[worst_pos].dist, worst_owner);

            // If the candidate distance is less than the current worst, it qualifies.
            if (dist < worst_dist) {
                Candidate cand;
                cand.idx = batch_start + i;  // global index of candidate data point.
                cand.dist = dist;
                // Atomically append the candidate into the candidate buffer.
                int pos = atomicAdd(warp_candidate_count, 1);
                if (pos < k) {
                    warp_candidate_buf[pos] = cand;
                }
                // If this insertion fills the candidate buffer, perform a merge.
                if (pos == k - 1) {
                    merge_warp(k, lane, L, best_local, warp_candidate_buf, warp_candidate_count,
                               merge_out + warp_id_in_block * k);
                }
            }
        }
        __syncwarp();
        __syncthreads();

        // At the end of the batch, if any candidates are in the buffer, merge them.
        if (lane == 0 && *warp_candidate_count > 0) {
            merge_warp(k, lane, L, best_local, warp_candidate_buf, warp_candidate_count,
                       merge_out + warp_id_in_block * k);
        }
        __syncwarp();
        __syncthreads();
    } // end for data batches

    // After all batches, do a final merge to gather the best results
    // into a contiguous sorted best array (in merge workspace).
    // We call merge_warp unconditionally to ensure that merge_out holds the last merged best.
    merge_warp(k, lane, L, best_local, warp_candidate_buf, warp_candidate_count,
               merge_out + warp_id_in_block * k);
    __syncwarp();

    // Now, the merge workspace for this warp (merge_out[warp_id_in_block*k ... warp_id_in_block*k + k-1])
    // contains the k best candidates in contiguous sorted order (ascending by distance).
    // Write these results to global memory in row-major order.
    Candidate* my_merge = merge_out + warp_id_in_block * k;
    for (int i = lane; i < k; i += 32) {
        int out_idx = global_warp_id * k + i;
        result[out_idx] = std::pair<int, float>( my_merge[i].idx, my_merge[i].dist );
    }
    __syncwarp();
}

// -----------------------------------------------------------------------------
// Host function interface for running k-NN.
// The arrays 'query', 'data', and 'result' are allocated with cudaMalloc.
//
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Compute grid dimensions.
    int warps_per_block = BLOCK_WARPS;
    int threads_per_block = BLOCK_SIZE; // BLOCK_WARPS * 32
    int total_warps = query_count;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    // Calculate shared memory size (in bytes):
    //   - shared_data: BATCH_SIZE * sizeof(float2)
    //   - candidate_buf: (BLOCK_WARPS * k * sizeof(Candidate))
    //   - candidate_count: (BLOCK_WARPS * sizeof(int))
    //   - merge workspace: (BLOCK_WARPS * k * sizeof(Candidate))
    size_t shared_mem_size = 0;
    shared_mem_size += BATCH_SIZE * sizeof(float2);
    shared_mem_size += BLOCK_WARPS * k * sizeof(Candidate);
    shared_mem_size += BLOCK_WARPS * sizeof(int);
    shared_mem_size += BLOCK_WARPS * k * sizeof(Candidate);

    // Launch the kernel.
    knn_kernel<<<blocks, threads_per_block, shared_mem_size>>>(query, query_count, data, data_count, result, k);
    // Note: error checking and stream synchronization should be added as needed.
}