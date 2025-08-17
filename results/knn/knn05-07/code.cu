#include <cuda_runtime.h>
#include <utility>
#include <cfloat>

// Optimized CUDA implementation of k-NN (squared Euclidean distance) for 2D points.
// Design decisions, invariants, and algorithmic steps are thoroughly documented inline.
//
// High-level summary:
//
// - One warp (32 threads) processes one query point.
// - The block loads the data points into shared memory in batches ("tiles") so that
//   all warps in the block can reuse the cached points to compute distances.
// - Each warp maintains two private shared-memory arrays of size k for its query:
//     (a) "best"  (intermediate top-k results) at combined[0..k-1].
//     (b) "cands" (candidate buffer)           at combined[k..2k-1].
//   So per warp we allocate combined[2*k] elements (distance,index).
// - When the candidate buffer fills (contains k elements), the warp merges "best"
//   and "cands" by sorting the 2k elements in-place using a warp-parallel bitonic sort
//   over shared memory, then keeps the first k as the new "best" and resets "cands".
// - After the last batch, any remaining candidates are merged.
// - The kernel does not allocate device memory; only uses shared memory (dynamic).
// - k is a power of two in [32, 1024]. We rely on 2k also being a power-of-two,
//   which is required by the bitonic network.
// - We choose 8 warps per block (256 threads) by default, appropriate for A100/H100.
//   Dynamic shared memory usage is computed at launch to fit device limits and k.
//
// Memory layout in dynamic shared memory per block:
//
//   [ tile float2 array of size tile_capacity ]
//   [ combined arrays per warp (warp_count * 2*k elements of PairDfIdx) ]
//   [ candidate counters per warp (warp_count ints) ]
//
// Where PairDfIdx is 8 bytes (float distance, int index).
//
// Notes on synchronization:
// - __syncthreads() is used to coordinate block-wide tile loads.
// - Within a warp, __syncwarp() is used to coordinate updates to per-warp shared arrays.
// - Warp-level ballot/popc/shfl are used to: compact accepted candidates into the buffer,
//   assign write positions without atomics, and coordinate merging when the buffer is full.
//
// The output array 'result' uses std::pair<int,float>, which we treat as a POD-like
// pair of two 4-byte fields (index, squared distance). We map it to a simple struct
// PairIF with identical layout and write to it from device code.

static_assert(sizeof(std::pair<int,float>) == 8, "std::pair<int,float> is expected to be 8 bytes");

struct PairIF { int first; float second; }; // must match std::pair<int,float> layout

// Internal pair type holding (distance, index) for sorting by distance.
struct PairDfIdx {
    float dist;
    int   idx;
};

// Return lane id within the warp
__device__ __forceinline__ int lane_id() {
    return threadIdx.x & 31;
}

// Warp-scope bitonic sort (ascending by dist) for an array in shared memory.
// - 'arr' points to the warp-private array of length 'N' located in shared memory.
// - 'N' must be a power of two. We use N = 2*k with k in [32, 1024], so N in [64, 2048].
// - Only the calling warp operates on 'arr'. Multiple warps can call this on disjoint regions.
__device__ __forceinline__ void warp_bitonic_sort_shared_ascending(PairDfIdx* arr, int N) {
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = lane_id();

    // Classic bitonic network using the "ixj = i ^ j" formulation.
    for (int k = 2; k <= N; k <<= 1) {
        // For each merge stage, j starts at k/2 and halves to 1
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each lane processes multiple indices 'i' spaced by warp size
            for (int i = lane; i < N; i += warpSize) {
                int ixj = i ^ j;
                if (ixj > i) {
                    // Determine sort direction for this sequence
                    bool ascending = ((i & k) == 0);
                    float ai = arr[i].dist;
                    float aj = arr[ixj].dist;

                    // Compare and swap based on direction
                    bool swap_needed = ascending ? (ai > aj) : (ai < aj);
                    if (swap_needed) {
                        PairDfIdx tmp = arr[i];
                        arr[i] = arr[ixj];
                        arr[ixj] = tmp;
                    }
                }
            }
            __syncwarp(FULL_MASK);
        }
    }
}

// Merge "best" (arr[0..k-1]) with "cands" (arr[k..k+cand_count-1]) and keep top-k (smallest distances).
// This sorts arr[0..2k-1] ascending by distance, after padding unused candidates with +INF.
// After return: arr[0..k-1] holds the new best, and cand_count is reset to 0.
__device__ __forceinline__ void warp_merge_topk(PairDfIdx* arr_combined, int k, int warp_local_id, int* s_cand_count) {
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = lane_id();

    int cand_count = 0;
    if (lane == 0) cand_count = s_cand_count[warp_local_id];
    cand_count = __shfl_sync(FULL_MASK, cand_count, 0);

    // Pad the unused portion of the candidate buffer with +INF so that sorting is well-defined.
    for (int i = lane + cand_count; i < k; i += warpSize) {
        arr_combined[k + i].dist = FLT_MAX;
        arr_combined[k + i].idx  = -1;
    }
    __syncwarp(FULL_MASK);

    // Sort the 2*k elements; ascending order by distance
    warp_bitonic_sort_shared_ascending(arr_combined, 2 * k);
    __syncwarp(FULL_MASK);

    // Reset candidate counter
    if (lane == 0) s_cand_count[warp_local_id] = 0;
    __syncwarp(FULL_MASK);
}

// Attempt to append a candidate (dist, idx) to the warp's candidate buffer (arr_combined[k..2k-1]).
// If the buffer is full, perform a merge and then possibly re-evaluate the candidate against
// the tightened threshold before retrying. This function is warp-synchronous: all lanes call it
// with their own (maybe inactive) candidate. It internally compacts active lanes and assigns
// write offsets without atomics.
//
// Arguments:
//   arr_combined      -> warp's shared memory region of size 2*k: [0..k-1]=best, [k..2k-1]=cands
//   k                 -> top-k
//   warp_local_id     -> warp index within block
//   s_cand_count      -> per-warp candidate counts in shared memory
//   take              -> whether this lane's candidate initially qualifies (dist < current threshold)
//   dist, idx         -> candidate distance and index
__device__ __forceinline__ void warp_append_candidate_with_merge(
    PairDfIdx* arr_combined, int k, int warp_local_id, int* s_cand_count,
    bool take, float dist, int idx)
{
    const unsigned FULL_MASK = 0xffffffffu;
    const int lane = lane_id();

    unsigned active = __ballot_sync(FULL_MASK, take);
    while (active) {
        // Remaining capacity in the candidate buffer
        int ccount = 0;
        if (lane == 0) ccount = s_cand_count[warp_local_id];
        ccount = __shfl_sync(FULL_MASK, ccount, 0);
        int remain = k - ccount;

        if (remain <= 0) {
            // Buffer full: merge best and cands to tighten threshold
            warp_merge_topk(arr_combined, k, warp_local_id, s_cand_count);
            // New threshold equals the k-th element's distance after merge (best[k-1])
            float new_thr = arr_combined[k - 1].dist;
            // Re-evaluate candidates that weren't inserted yet
            bool still_take = take && (dist < new_thr);
            active = __ballot_sync(FULL_MASK, still_take);
            take = still_take;
            continue;
        }

        // Number of active lanes wanting to write; take as many as we can this round
        int active_count = __popc(active);
        int takeN = (active_count < remain) ? active_count : remain;

        // Position among active lanes for this lane
        int pos = __popc(active & ((1u << lane) - 1));

        // Base write index in candidate buffer
        int base = 0;
        if (lane == 0) base = s_cand_count[warp_local_id];
        base = __shfl_sync(FULL_MASK, base, 0);

        // Write candidates (only first 'takeN' active lanes write)
        if (take && pos < takeN) {
            arr_combined[k + base + pos].dist = dist;
            arr_combined[k + base + pos].idx  = idx;
        }
        __syncwarp(FULL_MASK);

        // Update count
        if (lane == 0) s_cand_count[warp_local_id] = base + takeN;
        __syncwarp(FULL_MASK);

        // If some active lanes weren't written, keep them for the next iteration
        if (active_count > takeN) {
            bool keep = take && (pos >= takeN);
            active = __ballot_sync(FULL_MASK, keep);
            take = keep;
        } else {
            break;
        }
    }
}

// Main kernel. Each warp handles one query. The block processes a set of queries equal to the number of warps per block.
__global__ void knn_kernel_2d(
    const float2* __restrict__ query, int query_count,
    const float2* __restrict__ data,  int data_count,
    PairIF* __restrict__ result, int k, int tile_capacity)
{
    const unsigned FULL_MASK = 0xffffffffu;

    const int lane = lane_id();
    const int warp_local_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int query_idx = blockIdx.x * warps_per_block + warp_local_id;
    if (query_idx >= query_count)
        return;

    // Dynamic shared memory layout:
    // [ tile float2 array of length tile_capacity ]
    // [ per-warp combined arrays (warps_per_block * 2*k of PairDfIdx) ]
    // [ per-warp candidate counts (warps_per_block ints) ]
    extern __shared__ unsigned char smem_raw[];
    float2* tile = reinterpret_cast<float2*>(smem_raw);

    // Pointer to per-warp combined arrays
    PairDfIdx* combined_all = reinterpret_cast<PairDfIdx*>(tile + tile_capacity);
    // Pointer to per-warp candidate counters
    int* s_cand_count = reinterpret_cast<int*>(combined_all + (warps_per_block * 2 * k));

    // This warp's combined array: [0..k-1]=best, [k..2k-1]=cands
    PairDfIdx* combined = combined_all + warp_local_id * (2 * k);
    PairDfIdx* best     = combined;         // convenience alias
    // PairDfIdx* cands = combined + k;     // alias (not strictly necessary as we index combined[k+...])

    // Load the query point; lane 0 loads from global, broadcasts to others
    float2 q;
    if (lane == 0) q = query[query_idx];
    q.x = __shfl_sync(FULL_MASK, q.x, 0);
    q.y = __shfl_sync(FULL_MASK, q.y, 0);

    // Initialize best array with +INF and indices -1
    for (int i = lane; i < k; i += warpSize) {
        best[i].dist = FLT_MAX;
        best[i].idx  = -1;
    }
    // Initialize candidate counter
    if (lane == 0) s_cand_count[warp_local_id] = 0;
    __syncwarp(FULL_MASK);

    // Process data in tiles
    for (int base = 0; base < data_count; base += tile_capacity) {
        int tile_n = data_count - base;
        if (tile_n > tile_capacity) tile_n = tile_capacity;

        // Load tile into shared memory (all threads in block cooperate)
        for (int i = threadIdx.x; i < tile_n; i += blockDim.x) {
            tile[i] = data[base + i];
        }
        __syncthreads(); // ensure tile is fully loaded

        // Each warp computes distances from its query to all points in the tile
        // and feeds candidates into its shared candidate buffer.
        // We skip candidates that are not closer than the current k-th best distance.
        for (int t = lane; t < tile_n; t += warpSize) {
            float2 p = tile[t];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dist = dx * dx + dy * dy;

            // Read current threshold (k-th best distance); since 'best' remains sorted ascending
            // after each merge, threshold is best[k-1].dist. It may tighten when merges occur.
            float thr = best[k - 1].dist;

            bool take = (dist < thr);
            int  idx  = base + t;

            // Attempt to append this candidate; if buffer is full, it will merge and retry if needed
            warp_append_candidate_with_merge(combined, k, warp_local_id, s_cand_count, take, dist, idx);
        }

        __syncwarp(FULL_MASK);
        __syncthreads(); // allow the block to reuse smem for the next tile
    }

    // After last batch: merge any remaining candidates
    int final_count = 0;
    if (lane == 0) final_count = s_cand_count[warp_local_id];
    final_count = __shfl_sync(FULL_MASK, final_count, 0);
    if (final_count > 0) {
        warp_merge_topk(combined, k, warp_local_id, s_cand_count);
    }
    __syncwarp(FULL_MASK);

    // Write out results for this query: best[0..k-1] are sorted ascending by distance.
    // Row-major storage: for query i, result[i*k + j] holds the j-th nearest neighbor.
    for (int i = lane; i < k; i += warpSize) {
        PairIF out;
        out.first  = best[i].idx;
        out.second = best[i].dist; // squared distance
        result[query_idx * k + i] = out;
    }
}

// Host interface: run_knn wrapper as requested. It configures launch parameters and dynamic shared memory.
// - query: device pointer to float2 query points
// - query_count: number of query points
// - data: device pointer to float2 data points
// - data_count: number of data points
// - result: device pointer to std::pair<int,float> output array of size query_count * k
// - k: power of two, 32..1024
//
// We choose 8 warps per block (256 threads). Dynamic shared memory is sized as:
//   tile_capacity * sizeof(float2) +
//   warps_per_block * (2 * k * sizeof(PairDfIdx)) +
//   warps_per_block * sizeof(int)
//
// tile_capacity is chosen to fit within the device's opt-in maximum shared memory per block.
// We set the kernel's max dynamic shared memory attribute accordingly.
void run_knn(const float2 *query, int query_count,
             const float2 *data,  int data_count,
             std::pair<int, float> *result, int k)
{
    // Basic launch config
    const int warps_per_block = 8;                         // 8 warps -> 256 threads per block
    const int threads_per_block = warps_per_block * 32;

    // Query how much dynamic shared memory we can use (opt-in)
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    // Max dynamic shared memory we can request for this kernel
    int maxDynSmem = prop.sharedMemPerBlockOptin ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock;

    // Per-warp shared memory for combined arrays (2*k) of PairDfIdx (8 bytes each) and one int counter
    const size_t bytes_per_pair = sizeof(PairDfIdx); // 8 bytes
    const size_t per_warp_combined_bytes = 2ull * static_cast<size_t>(k) * bytes_per_pair;
    const size_t per_warp_counter_bytes  = sizeof(int);
    const size_t block_combined_bytes = warps_per_block * (per_warp_combined_bytes + per_warp_counter_bytes);

    // Choose tile_capacity to fit within maxDynSmem
    int tile_capacity = 0;
    if (maxDynSmem > static_cast<int>(block_combined_bytes)) {
        size_t remaining = maxDynSmem - block_combined_bytes;
        tile_capacity = static_cast<int>(remaining / sizeof(float2));
    } else {
        tile_capacity = 0;
    }

    // Ensure at least one element in the tile (avoid zero shared tile)
    if (tile_capacity < 1) tile_capacity = 1;

    // To avoid excessively large tiles, optionally cap tile size (e.g., 8192 points)
    const int tile_cap_upper = 8192;
    if (tile_capacity > tile_cap_upper) tile_capacity = tile_cap_upper;

    // Compute total dynamic shared memory in bytes we will request
    const size_t dynamic_smem_bytes =
        static_cast<size_t>(tile_capacity) * sizeof(float2) + block_combined_bytes;

    // Set the kernel attribute to allow this much dynamic shared memory (clamped to device maximum)
    int attr_value = static_cast<int>(dynamic_smem_bytes);
    if (attr_value > maxDynSmem) attr_value = maxDynSmem;
    cudaFuncSetAttribute(knn_kernel_2d, cudaFuncAttributeMaxDynamicSharedMemorySize, attr_value);

    // Prefer larger shared memory carveout if supported (helps for smem-heavy kernels)
    cudaFuncSetAttribute(knn_kernel_2d, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    // Grid size: one warp per query; each block has 'warps_per_block' warps
    const int total_warps = (query_count + 0) / 1; // just for clarity
    const int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch kernel
    PairIF* result_as_pairif = reinterpret_cast<PairIF*>(result);
    knn_kernel_2d<<<blocks, threads_per_block, dynamic_smem_bytes>>>(
        query, query_count, data, data_count, result_as_pairif, k, tile_capacity);

    // The caller is assumed to handle stream synchronization or cudaDeviceSynchronize as needed.
}