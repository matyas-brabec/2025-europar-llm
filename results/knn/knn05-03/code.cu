// kNN CUDA implementation for 2D points with per-query warp processing.
// Each warp computes k nearest neighbors for one query point.
// The algorithm processes data points in shared-memory batches and
// maintains a per-warp intermediate top-k set plus a per-warp candidate
// buffer (both stored in shared memory). When the candidate buffer fills,
// it is merged with the intermediate result using a warp-wide bitonic sort.

#include <cuda_runtime.h>
#include <utility>
#include <limits>

// Tunable parameters.
static constexpr int WARP_SIZE          = 32;
static constexpr int WARPS_PER_BLOCK    = 4;    // 4 warps = 128 threads per block
static constexpr int TILE_SIZE          = 512;  // Number of data points cached per block

// Simple struct with the same layout as std::pair<int,float> for device use.
struct Neighbor {
    int   index;  // index of the data point
    float dist;   // squared Euclidean distance
};

static_assert(sizeof(Neighbor) == sizeof(std::pair<int,float>),
              "Neighbor must match std::pair<int,float> size");

// Warp-level bitonic sort on a contiguous array of length n (power of two).
// Sorts in ascending order of dist[] while keeping idx[] in sync.
__device__ __forceinline__
void warp_bitonic_sort(float *dist, int *idx, int n, int lane_id)
{
    // Standard bitonic sort network.
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            // Each lane processes multiple indices spaced by WARP_SIZE.
            for (int i = lane_id; i < n; i += WARP_SIZE) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    bool ascending = ((i & size) == 0);
                    float di = dist[i];
                    float dj = dist[ixj];
                    // Compare-and-swap based on desired direction.
                    if ((di > dj) == ascending) {
                        dist[i]  = dj;
                        dist[ixj] = di;
                        int ti   = idx[i];
                        idx[i]   = idx[ixj];
                        idx[ixj] = ti;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// Merge the current intermediate top-k results with up to k candidates
// from the candidate buffer for one warp.
//
// Layout in shared memory for this warp:
//   dist[0 .. k-1]     : current best k distances
//   dist[k .. 2k-1]    : candidate distances (first cand_count valid, rest unused)
//   idx[0 .. k-1]      : best indices
//   idx[k .. 2k-1]     : candidate indices
//
// Unused candidate slots [k + cand_count .. 2k-1] are filled with +INF so that
// a bitonic sort over 2k elements yields the correct top-k in positions [0..k-1].
__device__ __forceinline__
void warp_merge_buffer(float *all_dist,
                       int   *all_idx,
                       int    k,
                       int    warp_id,
                       int    lane_id,
                       int   &cand_count,
                       float &worst_dist)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const float INF = CUDART_INF_F;

    const int warp_offset = warp_id * (2 * k);
    float *dist = all_dist + warp_offset;
    int   *idx  = all_idx  + warp_offset;

    // Fill unused candidate slots with INF / -1.
    int missing = k - cand_count;
    for (int i = lane_id; i < missing; i += WARP_SIZE) {
        int pos = k + cand_count + i;
        dist[pos] = INF;
        idx[pos]  = -1;
    }
    __syncwarp();

    // Bitonic sort over 2k elements (ascending by distance).
    int n = 2 * k;
    warp_bitonic_sort(dist, idx, n, lane_id);

    // After sort: top-k smallest distances are at dist[0..k-1].
    float new_worst = worst_dist;
    if (lane_id == 0) {
        new_worst = dist[k - 1];  // k-th nearest distance
        cand_count = 0;
    }

    new_worst  = __shfl_sync(FULL_MASK, new_worst, 0);
    cand_count = __shfl_sync(FULL_MASK, cand_count, 0);
    worst_dist = new_worst;
}

// Kernel implementing k-NN for 2D points using one warp per query.
__global__
void knn_kernel(const float2 * __restrict__ query,
                int                       query_count,
                const float2 * __restrict__ data,
                int                       data_count,
                Neighbor * __restrict__   result,
                int                       k)
{
    extern __shared__ unsigned char shared_mem[];

    // Shared memory layout:
    // [0 .. TILE_SIZE-1] float2: cached data points
    // followed by per-warp arrays:
    //   float dist[WARPS_PER_BLOCK][2*k]
    //   int   idx [WARPS_PER_BLOCK][2*k]
    float2 *sh_data = reinterpret_cast<float2*>(shared_mem);
    size_t offset   = TILE_SIZE * sizeof(float2);

    // Align to 4-byte boundary for float/int
    offset = (offset + 3) & ~size_t(3);

    float *sh_dist_all = reinterpret_cast<float*>(shared_mem + offset);
    offset += static_cast<size_t>(blockDim.x / WARP_SIZE) * (2 * static_cast<size_t>(k)) * sizeof(float);

    // Align again for int
    offset = (offset + 3) & ~size_t(3);
    int *sh_idx_all = reinterpret_cast<int*>(shared_mem + offset);

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int lane_id  = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const unsigned FULL_MASK = 0xffffffffu;

    const int query_global_id = blockIdx.x * warps_per_block + warp_id;
    const bool active_warp = (query_global_id < query_count);

    // Pointer to this warp's 2k-element distance and index arrays in shared memory.
    const int warp_offset = warp_id * (2 * k);
    float *dist = sh_dist_all + warp_offset;
    int   *idx  = sh_idx_all  + warp_offset;

    // Load query point and broadcast to all lanes in the warp.
    float2 q = make_float2(0.0f, 0.0f);
    if (active_warp) {
        if (lane_id == 0) {
            q = query[query_global_id];
        }
        q.x = __shfl_sync(FULL_MASK, q.x, 0);
        q.y = __shfl_sync(FULL_MASK, q.y, 0);
    }

    // Initialize intermediate top-k (best) array: set distances to +INF and indices to -1.
    if (active_warp) {
        const float INF = CUDART_INF_F;
        // dist[0..k-1] is the current best-k; initialize in parallel.
        for (int i = lane_id; i < k; i += WARP_SIZE) {
            dist[i] = INF;
            idx[i]  = -1;
        }
        // Candidate region [k..2k-1] is left uninitialized; will be filled as needed.
    }

    // Ensure all threads have initialized shared state before proceeding.
    __syncthreads();

    // Per-warp state: candidate count and worst (k-th) distance in current best set.
    int   cand_count = 0;
    float worst_dist = CUDART_INF_F;

    // Process the data points in tiles that are cached in shared memory.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE) {
        const int tile_size = min(TILE_SIZE, data_count - tile_start);

        // Load this tile of data points into shared memory cooperatively by the whole block.
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            sh_data[i] = data[tile_start + i];
        }
        __syncthreads();

        if (active_warp) {
            // Each warp processes the points in the tile in groups of WARP_SIZE.
            for (int base = 0; base < tile_size; base += WARP_SIZE) {
                const int idx_in_tile = base + lane_id;
                bool in_range = (idx_in_tile < tile_size);

                // Compute distance for this lane's point (if in range).
                float dist_val = CUDART_INF_F;
                int   data_idx = -1;
                if (in_range) {
                    float2 p = sh_data[idx_in_tile];
                    float dx = p.x - q.x;
                    float dy = p.y - q.y;
                    dist_val = dx * dx + dy * dy;
                    data_idx = tile_start + idx_in_tile;  // global index
                }

                // Determine whether this lane's point is a candidate based on current worst_dist.
                bool pending = in_range && (dist_val < worst_dist);

                // Insert candidates into the per-warp candidate buffer, possibly
                // merging with the intermediate results when the buffer fills.
                while (true) {
                    unsigned pending_mask = __ballot_sync(FULL_MASK, pending);
                    int pending_count = __popc(pending_mask);
                    if (pending_count == 0)
                        break;

                    // Compute how many free slots remain in the candidate buffer.
                    int slots_left = 0;
                    if (lane_id == 0) {
                        slots_left = k - cand_count;
                    }
                    slots_left = __shfl_sync(FULL_MASK, slots_left, 0);

                    if (slots_left == 0) {
                        // Candidate buffer is full and there are pending candidates.
                        // Merge buffer with current best-k set.
                        warp_merge_buffer(sh_dist_all, sh_idx_all, k,
                                          warp_id, lane_id, cand_count, worst_dist);
                        // Re-evaluate whether this lane's candidate is still valid
                        // under the tightened worst_dist.
                        pending = in_range && (dist_val < worst_dist);
                        continue;
                    }

                    // Number of candidates to insert in this iteration:
                    int take = 0;
                    if (lane_id == 0) {
                        take = (pending_count < slots_left) ? pending_count : slots_left;
                    }
                    take = __shfl_sync(FULL_MASK, take, 0);

                    // Compute local rank among pending lanes.
                    int local_rank = __popc(pending_mask & ((1u << lane_id) - 1u));
                    bool will_insert = pending && (local_rank < take);

                    // Reserve space in the candidate buffer.
                    int insert_base = 0;
                    if (lane_id == 0) {
                        insert_base = cand_count;
                        cand_count += take;
                    }
                    insert_base = __shfl_sync(FULL_MASK, insert_base, 0);
                    cand_count  = __shfl_sync(FULL_MASK, cand_count, 0);

                    if (will_insert) {
                        int pos = k + insert_base + local_rank;  // candidate region
                        dist[pos] = dist_val;
                        idx[pos]  = data_idx;
                        pending   = false;
                    }
                    // If pending_count > take, some lanes remain pending and the loop continues.
                }
            }
        }

        __syncthreads();  // Ensure all warps are done with this tile before loading the next one.
    }

    // After processing all tiles, merge any remaining candidates.
    if (active_warp && cand_count > 0) {
        warp_merge_buffer(sh_dist_all, sh_idx_all, k,
                          warp_id, lane_id, cand_count, worst_dist);
    }

    // Write the final k nearest neighbors for this query to global memory.
    if (active_warp) {
        const int out_base = query_global_id * k;
        for (int j = lane_id; j < k; j += WARP_SIZE) {
            Neighbor n;
            n.index = idx[j];
            n.dist  = dist[j];
            result[out_base + j] = n;
        }
    }
}

// Host interface function.
void run_knn(const float2 *query,
             int           query_count,
             const float2 *data,
             int           data_count,
             std::pair<int, float> *result,
             int           k)
{
    if (query_count <= 0 || data_count <= 0 || k <= 0)
        return;

    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;
    const int warps_per_block   = WARPS_PER_BLOCK;

    // Grid dimension: one warp per query.
    int num_blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Shared memory size per block:
    //   TILE_SIZE * sizeof(float2) for cached data +
    //   WARPS_PER_BLOCK * 2 * k * (sizeof(float) + sizeof(int)) for per-warp buffers.
    size_t shared_bytes =
        TILE_SIZE * sizeof(float2) +
        static_cast<size_t>(WARPS_PER_BLOCK) *
        (2 * static_cast<size_t>(k)) *
        (sizeof(float) + sizeof(int));

    Neighbor *d_result = reinterpret_cast<Neighbor*>(result);

    knn_kernel<<<num_blocks, threads_per_block, shared_bytes>>>(
        query, query_count,
        data, data_count,
        d_result,
        k
    );

    // Optional: synchronize or check errors could be added here if desired.
}