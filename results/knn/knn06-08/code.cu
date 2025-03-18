#include <cuda_runtime.h>
#include <float.h>
#include <utility>

//---------------------------------------------------------------------
// This CUDA code implements a k–nearest neighbors (k-NN) algorithm for
// 2D points in Euclidean space. For each query point the k nearest data 
// points (using squared Euclidean distances) are computed in parallel.
// 
// The algorithm assigns one warp (32 threads) per query. Each warp 
// maintains a private “intermediate result” (the current best k candidates)
// distributed over its 32 threads (each holding k/32 candidates in registers)
// and also uses a per–query candidate buffer located in shared memory.
// Data points are processed in batches (tiles) which are first loaded into
// shared memory by the block. Each warp then computes distances from its 
// query to the tile points, and if a computed distance is below the current 
// threshold (max_distance), the candidate is added to the candidate buffer 
// using an atomic operation. When the buffer becomes full, it is merged 
// (via a warp–cooperative bitonic sort over a merged array) with the 
// intermediate result. At the end a final merge is performed (if needed) 
// and the sorted list of k nearest neighbors is written to global memory.
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// Tunable parameters
//---------------------------------------------------------------------
#define DATA_TILE_SIZE 1024     // number of data points loaded per tile
#define WARP_SIZE 32            // warp width
#define BLOCK_WARPS 4           // number of warps per block
#define BLOCK_SIZE (BLOCK_WARPS * WARP_SIZE)

//---------------------------------------------------------------------
// The Candidate struct holds a data point index and its distance.
//---------------------------------------------------------------------
struct Candidate {
    int idx;
    float dist;
};

//---------------------------------------------------------------------
// A simple inline swap function for Candidate elements.
//---------------------------------------------------------------------
__device__ inline void swap_candidate(Candidate &a, Candidate &b) {
    Candidate tmp = a;
    a = b;
    b = tmp;
}

//---------------------------------------------------------------------
// Bitonic sort: This device function sorts an array "data" of length 
// "length" in ascending order (by dist). It is intended to be executed 
// cooperatively by WARP_SIZE threads (one warp) so that each thread 
// works on indices of the array in a strided manner.
//---------------------------------------------------------------------
__device__ void bitonic_sort(Candidate* data, int length) {
    // Use the calling thread's lane within a warp.
    unsigned int lane = threadIdx.x % WARP_SIZE;
    for (int size = 2; size <= length; size *= 2) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each warp thread processes indices in a strided loop.
            for (int i = lane; i < length; i += WARP_SIZE) {
                int j = i ^ stride;
                if (j > i && j < length) {
                    // In the current bitonic sub–sequence, decide on ascending order.
                    bool ascending = ((i & size) == 0);
                    Candidate a = data[i];
                    Candidate b = data[j];
                    if ((a.dist > b.dist) == ascending) {
                        data[i] = b;
                        data[j] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

//---------------------------------------------------------------------
// mergeWarpCandidates:
// This function is called by a warp when its candidate buffer (in shared
// memory) has become full (or at the end, if nonempty). It merges the 
// warp's private intermediate result (distributed across registers as an 
// array local_knn[]) with the candidate buffer (of capacity k) stored in 
// shared memory. The two sets are combined into a temporary array of 
// size 2*k (in shared memory), sorted using bitonic_sort, and the 
// best k candidates are then redistributed back to local_knn.
//---------------------------------------------------------------------
__device__ void mergeWarpCandidates(Candidate local_knn[], int local_len,
                                    Candidate* warp_candidates, int* warp_candidate_count,
                                    Candidate* merge_buffer, int k)
{
    unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int warpId = threadIdx.x / WARP_SIZE;  // warp index within block
    int candidate_count = warp_candidate_count[warpId];
    int merge_base = warpId * (2 * k);  // starting index for this warp in merge_buffer

    //--- 1. Copy the current intermediate result from registers into merge_buffer[0 .. k-1] ---
    for (int i = 0; i < local_len; i++) {
        int pos = i * WARP_SIZE + lane;
        if (pos < k)
            merge_buffer[merge_base + pos] = local_knn[i];
    }

    //--- 2. Copy the candidate buffer (from shared memory) into merge_buffer[k .. 2*k-1] ---
    int candidate_base = warpId * k;
    for (int i = lane; i < k; i += WARP_SIZE) {
        if (i < candidate_count) {
            merge_buffer[merge_base + k + i] = warp_candidates[candidate_base + i];
        } else {
            // Pad with a dummy candidate (FLT_MAX ensures it will sink)
            merge_buffer[merge_base + k + i].idx = -1;
            merge_buffer[merge_base + k + i].dist = FLT_MAX;
        }
    }
    __syncwarp();

    //--- 3. Sort the merged 2*k array using bitonic_sort. ---
    bitonic_sort(merge_buffer + merge_base, 2 * k);
    __syncwarp();

    //--- 4. Redistribute the best k candidates (first k elements) back to local_knn ---
    for (int i = 0; i < local_len; i++) {
        int pos = i * WARP_SIZE + lane;
        if (pos < k)
            local_knn[i] = merge_buffer[merge_base + pos];
    }
    __syncwarp();

    //--- 5. Reset the candidate counter for this warp (only one thread does it) ---
    if (lane == 0) {
        warp_candidate_count[warpId] = 0;
    }
    __syncwarp();
}

//---------------------------------------------------------------------
// knn_kernel:
// For each query (one per warp) the kernel processes the input data in 
// tiles. It computes the squared Euclidean distance from the query to each 
// data point (in the tile) and if the distance is smaller than the current 
// threshold (max_distance from the intermediate result), the candidate is 
// saved into the candidate buffer using atomicAdd. When full, the buffer is 
// merged into the private (register–resident) intermediate result. At the 
// end, a final merge is done if needed and the sorted k nearest neighbors 
// are written to the output.
//---------------------------------------------------------------------
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp processes one query.
    unsigned int lane    = threadIdx.x % WARP_SIZE;
    unsigned int warpId  = threadIdx.x / WARP_SIZE;  // warp index within block
    unsigned int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warpId;
    if (global_warp_id >= (unsigned)query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[global_warp_id];

    // Each warp maintains an intermediate result (best k candidates) 
    // distributed over its lanes. Given k is a power-of-two and WARP_SIZE=32,
    // each thread holds local_len = k/32 candidates.
    int local_len = k / WARP_SIZE;
    Candidate local_knn[32]; // Maximum local_len is k/32; k<=1024 gives local_len<=32.
    for (int i = 0; i < local_len; i++) {
        local_knn[i].idx = -1;
        local_knn[i].dist = FLT_MAX;
    }
    // Initially, the worst (max) distance is FLT_MAX.
    float warp_max = FLT_MAX;

    //--- Shared Memory Layout ---//
    // The block–shared memory is partitioned as follows:
    // 1) Tile data: float2 tile_data[DATA_TILE_SIZE]
    // 2) Candidate buffers: Candidate warp_candidates[BLOCK_WARPS * k]
    // 3) Candidate counts: int warp_candidate_count[BLOCK_WARPS]
    // 4) Merge buffers: Candidate merge_buffer[BLOCK_WARPS * 2 * k]
    extern __shared__ char smem[];
    float2 *tile_data = (float2*)smem;
    Candidate *warp_candidates = (Candidate*)(tile_data + DATA_TILE_SIZE);
    int *warp_candidate_count = (int*)(warp_candidates + BLOCK_WARPS * k);
    Candidate *merge_buffer = (Candidate*)(warp_candidate_count + BLOCK_WARPS);
    // Initialize candidate count for this warp (only lane0 of each warp does it)
    if (lane == 0) {
        warp_candidate_count[warpId] = 0;
    }
    __syncthreads();

    //--- Process the data in batches (tiles) ---
    for (int tile_start = 0; tile_start < data_count; tile_start += DATA_TILE_SIZE) {
        int tile_size = DATA_TILE_SIZE;
        if (tile_start + tile_size > data_count) {
            tile_size = data_count - tile_start;
        }
        // Load this tile of data into shared memory.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            tile_data[i] = data[tile_start + i];
        }
        __syncthreads(); // Ensure the tile is loaded.

        // Each warp processes the tile: each of its lanes strides over tile_data.
        for (int j = lane; j < tile_size; j += WARP_SIZE) {
            float2 p = tile_data[j];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dist = dx * dx + dy * dy;
            // Only consider the candidate if its distance is less than current warp_max.
            if (dist < warp_max) {
                // Use atomicAdd to reserve a slot in this warp's candidate buffer.
                int pos = atomicAdd(&warp_candidate_count[warpId], 1);
                if (pos < k) {
                    warp_candidates[warpId * k + pos].idx  = tile_start + j;
                    warp_candidates[warpId * k + pos].dist = dist;
                }
                // If pos >= k then the candidate is ignored.
            }
        }
        __syncwarp(); // Synchronize the warp threads.

        // If the candidate buffer is full, merge it with the intermediate result.
        if (warp_candidate_count[warpId] >= k) {
            mergeWarpCandidates(local_knn, local_len,
                                warp_candidates + warpId * k,
                                warp_candidate_count,
                                merge_buffer + warpId * (2 * k), k);
            __syncwarp();
            // Update warp_max from the new intermediate result.
            // The worst candidate is at global position (k-1) in the sorted ordering.
            Candidate worst;
            // Each candidate in the full sorted intermediate result is distributed:
            // position = i * WARP_SIZE + lane, so the (k-1)th candidate
            // is held by the lane equal to ((k-1) % WARP_SIZE).
            if (((k - 1) % WARP_SIZE) == lane) {
                worst = local_knn[(k - 1) / WARP_SIZE];
            }
            worst.dist = __shfl_sync(0xFFFFFFFF, worst.dist, (k - 1) % WARP_SIZE);
            warp_max = worst.dist;
        }
        __syncthreads(); // Block–sync before loading next tile.
    }

    //--- Final merge: if any candidates remain in candidate buffer, merge them.
    if (warp_candidate_count[warpId] > 0) {
        mergeWarpCandidates(local_knn, local_len,
                            warp_candidates + warpId * k,
                            warp_candidate_count,
                            merge_buffer + warpId * (2 * k), k);
        __syncwarp();
        Candidate worst;
        if (((k - 1) % WARP_SIZE) == lane) {
            worst = local_knn[(k - 1) / WARP_SIZE];
        }
        worst.dist = __shfl_sync(0xFFFFFFFF, worst.dist, (k - 1) % WARP_SIZE);
        warp_max = worst.dist;
    }

    //--- Write the final sorted k nearest neighbors to global memory ---
    // The sorted neighbors are distributed: the overall ordering is defined such that
    // the candidate at global index j is in local_knn[j/32] on lane (j % 32).
    int global_result_base = global_warp_id * k;
    for (int i = 0; i < local_len; i++) {
        int pos = i * WARP_SIZE + lane;
        if (pos < k) {
            result[global_result_base + pos] = std::pair<int, float>(local_knn[i].idx, local_knn[i].dist);
        }
    }
}

//---------------------------------------------------------------------
// run_knn:
// Host function that serves as the C++ interface. It launches the k-NN 
// kernel with appropriate grid/block dimensions and calculates the required 
// shared memory size. It assumes that query, data, and result arrays are 
// allocated on the device (using cudaMalloc).
//---------------------------------------------------------------------
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Set block and grid dimensions.
    // Each warp processes one query.
    dim3 blockDim(BLOCK_SIZE);
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    dim3 gridDim((query_count + warps_per_block - 1) / warps_per_block);

    // Compute required shared memory size:
    //   - DATA_TILE_SIZE * sizeof(float2) for tile_data.
    //   - (warps_per_block * k) * sizeof(Candidate) for candidate buffers.
    //   - (warps_per_block) * sizeof(int) for candidate count array.
    //   - (warps_per_block * 2 * k) * sizeof(Candidate) for merge buffers.
    size_t shared_mem_size = 0;
    shared_mem_size += DATA_TILE_SIZE * sizeof(float2);
    shared_mem_size += warps_per_block * k * sizeof(Candidate);
    shared_mem_size += warps_per_block * sizeof(int);
    shared_mem_size += warps_per_block * 2 * k * sizeof(Candidate);

    // Launch kernel.
    knn_kernel<<<gridDim, blockDim, shared_mem_size>>>(query, query_count, data, data_count, result, k);
}