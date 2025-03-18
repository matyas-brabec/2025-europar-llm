// This complete CUDA C++ source code implements an optimized k‑nearest neighbors (k‑NN)
// search for 2D points using a modern NVIDIA GPU (e.g., A100, H100). It assumes that the
// three device pointers (query, data, and result) have been allocated with cudaMalloc.
// Each query (a float2) must compute its k nearest data points (stored as std::pair<int, float>)
// using a single warp (32 threads). The data points are processed iteratively in user‐defined
// tiles that are cached in shared memory. Each warp’s 32 threads cooperatively compute
// distances between the query and the tile data, and each thread maintains a “local” buffer
// (in registers) of best candidate neighbors from its portion of the data. After processing
// all tiles, lane 0 of each warp gathers the candidates from all 32 lanes, sorts them, and writes
// out the top k nearest neighbors (sorted in ascending order by squared distance).
//
// k is assumed to be a power‐of‐two between 32 and 1024 (inclusive).
//
// Compile with the latest CUDA toolkit and a recent C++ compiler.

#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Define tile size for shared memory (number of data points loaded per batch).
// This value is chosen for good occupancy and shared memory use.
#define TILE_SIZE 256
#define WARP_SIZE 32

// __device__ function for computing squared Euclidean distance between two float2 points.
__device__ inline float sqrEuclidean(const float2 a, const float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// The CUDA kernel that computes k-NN for 2D points.
// Each warp (32 threads) processes one query point.
// The kernel loads data points in tiles into shared memory for faster access.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           std::pair<int, float> * __restrict__ result,
                           int k)
{
    // Determine the warp index (each warp processes one query).
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    int warp_id = blockIdx.x * warpsPerBlock + (threadIdx.x / WARP_SIZE);
    int lane_id = threadIdx.x % WARP_SIZE;

    // Check that this warp corresponds to a valid query.
    if (warp_id >= query_count)
        return;

    // Load the query point (all threads in the warp read the same point).
    float2 q = query[warp_id];

    // Each warp will ultimately output k neighbors.
    // We let each thread in the warp maintain a local candidate buffer.
    // We choose the buffer size per thread to be "local_buf_size" = 2*(k/32).
    // (Since k is a power-of-two between 32 and 1024, k/32 is an integer.)
    const int cand_per_lane = k / WARP_SIZE;          // minimum candidates per lane for global k
    const int local_buf_size = 2 * cand_per_lane;       // local buffer size for each thread (allows extra storage)

    // Each thread’s buffer is stored in registers.
    // The arrays are allocated to the maximum possible size (64) since k<=1024 => cand_per_lane<=32.
    float local_dist[64];
    int   local_idx[64];

    // Initialize the local candidate buffer with INF distances.
    for (int i = 0; i < local_buf_size; i++) {
        local_dist[i] = FLT_MAX;
        local_idx[i] = -1;
    }

    // Shared memory tile for caching a batch of data points.
    __shared__ float2 s_data[TILE_SIZE];

    // Process the data in tiles.
    for (int tile_offset = 0; tile_offset < data_count; tile_offset += TILE_SIZE) {
        // Determine number of data points to load in this tile.
        int tile_len = TILE_SIZE;
        if (tile_offset + TILE_SIZE > data_count)
            tile_len = data_count - tile_offset;

        // Each thread in the block cooperatively loads one or more data points.
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            s_data[i] = data[tile_offset + i];
        }
        __syncthreads(); // Ensure the tile is loaded before processing.

        // Each warp processes the tile: each thread iterates over a subset of indices in s_data.
        for (int i = lane_id; i < tile_len; i += WARP_SIZE) {
            float2 p = s_data[i];
            float d = sqrEuclidean(q, p);
            int global_idx = tile_offset + i;

            // Update the local candidate buffer using a simple insertion sort.
            // The buffer is maintained in sorted order (ascending by distance),
            // so that local_dist[local_buf_size-1] holds the worst (largest) candidate.
            if (d < local_dist[local_buf_size - 1]) {
                int pos = local_buf_size - 1;
                // Shift candidate entries to make room for the new candidate.
                while (pos > 0 && d < local_dist[pos - 1]) {
                    local_dist[pos] = local_dist[pos - 1];
                    local_idx[pos] = local_idx[pos - 1];
                    pos--;
                }
                local_dist[pos] = d;
                local_idx[pos] = global_idx;
            }
        }
        __syncthreads(); // Ensure all warps finish processing this tile.
    } // End for each tile

    // After processing all tiles, each thread in the warp has a sorted local candidate buffer.
    // Now merge the 32 (warpSize) local buffers (each of size local_buf_size) to get the global k nearest.
    // For simplicity, we let lane 0 perform the merge.
    if (lane_id == 0) {
        // Define a temporary structure to hold a candidate.
        struct Candidate {
            int idx;
            float dist;
        };

        // Total candidates from the warp: union_size = 32 * local_buf_size.
        const int union_size = WARP_SIZE * local_buf_size;

        // Allocate a temporary array to hold the merged candidates.
        Candidate union_cand[2048];  // Maximum size when local_buf_size==64 (32*64 = 2048).

        // For each lane in the warp and for each position in its local candidate buffer,
        // use warp shuffle to collect the candidate into union_cand.
        for (int src_lane = 0; src_lane < WARP_SIZE; src_lane++) {
            for (int j = 0; j < local_buf_size; j++) {
                // __shfl_sync broadcasts the j-th element from lane 'src_lane' in the warp.
                float cand_d = __shfl_sync(0xFFFFFFFF, local_dist[j], src_lane);
                int   cand_i = __shfl_sync(0xFFFFFFFF, local_idx[j], src_lane);
                union_cand[src_lane * local_buf_size + j].dist = cand_d;
                union_cand[src_lane * local_buf_size + j].idx = cand_i;
            }
        }

        // Now, union_cand array contains union_size candidates.
        // We need to select the k best (i.e. smallest distances) from these.
        // Since union_size is at most 2*k, a simple insertion sort is efficient.
        for (int i = 1; i < union_size; i++) {
            Candidate key = union_cand[i];
            int j = i - 1;
            while (j >= 0 && union_cand[j].dist > key.dist) {
                union_cand[j + 1] = union_cand[j];
                j--;
            }
            union_cand[j + 1] = key;
        }

        // Write the first k sorted candidates (nearest neighbors) to the output.
        int out_base = warp_id * k;
        for (int i = 0; i < k; i++) {
            result[out_base + i] = std::make_pair(union_cand[i].idx, union_cand[i].dist);
        }
    }
}

// Host function that launches the k-NN kernel.
// 'query'       : pointer to device array of query points (float2)
// 'query_count' : number of query points
// 'data'        : pointer to device array of data points (float2)
// 'data_count'  : number of data points (>= k)
// 'result'      : pointer to device array for results (std::pair<int, float>)
// 'k'           : number of nearest neighbors to find (power-of-2 between 32 and 1024)
/// @FIXED
/// extern "C" void run_knn(const float2 *query, int query_count,
void run_knn(const float2 *query, int query_count,
                        const float2 *data, int data_count,
                        std::pair<int, float> *result, int k)
{
    // Choose number of threads per block. Here we use 128 threads (4 warps per block).
    const int THREADS_PER_BLOCK = 128;
    // Each warp processes one query; so each block processes (THREADS_PER_BLOCK/32) queries.
    int warpsPerBlock = THREADS_PER_BLOCK / WARP_SIZE;
    int queryPerBlock = warpsPerBlock;
    // Compute number of blocks required.
    int blocks = (query_count + queryPerBlock - 1) / queryPerBlock;

    // Launch the kernel.
    knn_kernel<<<blocks, THREADS_PER_BLOCK>>>(query, query_count, data, data_count, result, k);

    // Optionally, check for launch errors (omitted here for brevity).
    // cudaDeviceSynchronize(); // Uncomment if host synchronization is needed.
}
