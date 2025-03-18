#include <cuda_runtime.h>
#include <float.h>
#include <utility>

//------------------------------------------------------------------------------
// Constants and hyper‐parameters
//------------------------------------------------------------------------------
#define TILE_SIZE 1024       // Number of data points loaded per tile (in shared memory)
#define LOCAL_CAND 4         // Number of candidate proposals stored per warp thread per tile
#define WARP_SIZE 32         // Warp size (fixed)
#define MAX_K 1024           // Maximum value for k (upper‐bound for per‐warp candidate list)

//------------------------------------------------------------------------------
// Device inline function to insert a candidate into a sorted candidate list.
// The candidate list is stored in arrays cand_idx[] and cand_dist[] of length 'size'.
// It is maintained in descending order (i.e. element 0 is the worst, largest distance).
// If the new candidate (new_idx,new_dist) is better (i.e. new_dist is smaller than cand_dist[0])
// then it replaces the worst candidate and “bubbles” upward until the order is restored.
//------------------------------------------------------------------------------
__device__ inline void candidate_insert(int *cand_idx, float *cand_dist, int size, int new_idx, float new_dist) {
    // Only proceed if the new candidate is better than the current worst.
    if(new_dist >= cand_dist[0])
        return;
    // Write the new candidate into the worst slot.
    cand_idx[0] = new_idx;
    cand_dist[0] = new_dist;
    // Bubble the new candidate upward to maintain descending order.
    int i = 0;
    while(i < size - 1 && cand_dist[i] < cand_dist[i+1]) {
        // Swap candidate at index i with candidate at i+1.
        int temp_idx = cand_idx[i];
        float temp_dist = cand_dist[i];
        cand_idx[i] = cand_idx[i+1];
        cand_dist[i] = cand_dist[i+1];
        cand_idx[i+1] = temp_idx;
        cand_dist[i+1] = temp_dist;
        i++;
    }
}

//------------------------------------------------------------------------------
// CUDA kernel implementing k-NN for 2D points.
// Each warp processes one query point and computes its k nearest neighbors.
// The algorithm processes the data points in batches (tiles) cached in shared memory.
// Each warp maintains a private candidate list (of size k) in lane 0 registers,
// and each thread in the warp accumulates a small list of candidate proposals (LOCAL_CAND entries)
// from its portion of the tile. At the end of each tile iteration the proposals are merged
// (via warp shuffles) into the global candidate list.
//------------------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k) {
    // Each warp is responsible for one query.
    int warp_id = (blockIdx.x * (blockDim.x / WARP_SIZE)) + (threadIdx.x / WARP_SIZE);
    int lane    = threadIdx.x % WARP_SIZE;
    if (warp_id >= query_count) return;

    //---------------------------------------------------------------------------
    // Each warp loads the query point and broadcasts it to all lanes.
    //---------------------------------------------------------------------------
    float2 q;
    if (lane == 0)
        q = query[warp_id];
    // Broadcast query coordinates from lane 0 to all warp lanes.
    q.x = __shfl_sync(0xffffffff, q.x, 0);
    q.y = __shfl_sync(0xffffffff, q.y, 0);

    //---------------------------------------------------------------------------
    // Each warp thread allocates room for local candidate proposals from the tile.
    // These arrays hold up to LOCAL_CAND proposals computed by each thread.
    //---------------------------------------------------------------------------
    int local_cand_idx[LOCAL_CAND];
    float local_cand_dist[LOCAL_CAND];
#pragma unroll
    for (int i = 0; i < LOCAL_CAND; i++) {
        local_cand_idx[i] = -1;
        local_cand_dist[i] = FLT_MAX;
    }

    //---------------------------------------------------------------------------
    // The warp's global candidate list for the query is maintained only by lane 0.
    // It is stored as an array of k pairs (index, distance) in registers (allocated as local arrays).
    // The list is maintained in descending order so that element 0 has the highest (worst) distance.
    // Initially, all entries are set to ( -1, FLT_MAX ).
    //---------------------------------------------------------------------------
    int global_cand_idx[MAX_K];
    float global_cand_dist[MAX_K];
    if(lane == 0) {
        for (int i = 0; i < k; i++) {
            global_cand_idx[i] = -1;
            global_cand_dist[i] = FLT_MAX;
        }
    }
    // A variable to hold the current threshold (the worst distance in the candidate list).
    // It will be broadcast to all lanes in the warp.
    float cur_thresh = FLT_MAX;

    //---------------------------------------------------------------------------
    // Process the data points in tiles cached in shared memory.
    // All threads in the block cooperatively load a tile of data points.
    //---------------------------------------------------------------------------
    extern __shared__ float2 tile_data[]; // Shared memory array for data tile.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        // Determine the number of points in this tile.
        int tile_size = TILE_SIZE;
        if (tile_start + tile_size > data_count)
            tile_size = data_count - tile_start;
        // Each thread in the block loads one or more data points into shared memory.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile_data[i] = data[tile_start + i];
        }
        __syncthreads();  // Ensure the tile is fully loaded.

        //-----------------------------------------------------------------------
        // Broadcast the current global threshold from lane 0.
        // (global candidate list is maintained by lane 0; its worst candidate is at index 0.)
        //-----------------------------------------------------------------------
        if(lane == 0)
            cur_thresh = global_cand_dist[0];
        cur_thresh = __shfl_sync(0xffffffff, cur_thresh, 0);

        //-----------------------------------------------------------------------
        // Each warp processes its portion of the tile.
        // Each thread in the warp iterates over tile indices starting from its lane id
        // and striding by WARP_SIZE.
        //-----------------------------------------------------------------------
        for (int i = lane; i < tile_size; i += WARP_SIZE) {
            float2 dpt = tile_data[i];
            float dx = q.x - dpt.x;
            float dy = q.y - dpt.y;
            float dist = dx * dx + dy * dy;
            // If this distance is better than the current threshold, try to insert as a local proposal.
            if (dist < cur_thresh) {
                candidate_insert(local_cand_idx, local_cand_dist, LOCAL_CAND, tile_start + i, dist);
            }
        }
        __syncwarp();  // Ensure all warp threads have finished processing the tile.

        //-----------------------------------------------------------------------
        // Merge the local candidate proposals from all warp lanes into the global candidate list.
        // Only lane 0 performs the merge using warp shuffles.
        //-----------------------------------------------------------------------
        if (lane == 0) {
            // For each lane in the warp...
            for (int src = 0; src < WARP_SIZE; src++) {
                // For each proposal in the local candidate array of that lane...
                for (int j = 0; j < LOCAL_CAND; j++) {
                    int prop_idx = __shfl_sync(0xffffffff, local_cand_idx[j], src);
                    float prop_dist = __shfl_sync(0xffffffff, local_cand_dist[j], src);
                    // If the proposal is valid and better than the current worst...
                    if (prop_idx != -1 && prop_dist < global_cand_dist[0]) {
                        candidate_insert(global_cand_idx, global_cand_dist, k, prop_idx, prop_dist);
                    }
                }
            }
        }
        __syncwarp();  // Ensure merge is complete.

        //-----------------------------------------------------------------------
        // Update the threshold for the next tile iteration.
        //-----------------------------------------------------------------------
        if(lane == 0)
            cur_thresh = global_cand_dist[0];
        cur_thresh = __shfl_sync(0xffffffff, cur_thresh, 0);

        //-----------------------------------------------------------------------
        // Reset the local candidate proposals for the next tile.
        //-----------------------------------------------------------------------
#pragma unroll
        for (int i = 0; i < LOCAL_CAND; i++) {
            local_cand_idx[i] = -1;
            local_cand_dist[i] = FLT_MAX;
        }
        // A block-level synchronization ensures that all threads are ready for the next tile load.
        __syncthreads();
    } // End of tile loop

    //---------------------------------------------------------------------------
    // After processing all tiles the global candidate list (in lane 0) holds the k best candidates.
    // The list is stored in descending order (worst candidate at index 0, best at index k-1).
    // Output the results in ascending order (best first).
    //---------------------------------------------------------------------------
    if (lane == 0) {
        for (int i = 0; i < k; i++) {
            int outIndex = warp_id * k + i;
            // Reverse the order: best candidate (smallest distance) comes first.
            int nn_idx = global_cand_idx[k - 1 - i];
            float nn_dist = global_cand_dist[k - 1 - i];
            result[outIndex] = std::pair<int, float>(nn_idx, nn_dist);
        }
    }
}

//------------------------------------------------------------------------------
// Host function to launch the k-NN kernel.
// It chooses a block size such that each warp (32 threads) processes one query.
// The data points are processed in tiles cached in shared memory.
//------------------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k) {
    // Select a block size: e.g., 128 threads (i.e. 4 warps per block).
    int threads_per_block = 128;
    int warps_per_block = threads_per_block / WARP_SIZE;
    // Number of blocks needed so that each warp gets one query.
    int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;
    // Allocate shared memory for one tile of data points.
    size_t shared_mem_bytes = TILE_SIZE * sizeof(float2);

    // Launch the kernel.
    knn_kernel<<<num_blocks, threads_per_block, shared_mem_bytes>>>(query, query_count,
                                                                    data, data_count,
                                                                    result, k);
    cudaDeviceSynchronize();
}