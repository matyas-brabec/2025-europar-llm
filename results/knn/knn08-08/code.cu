#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Optimized CUDA k-NN (squared L2) for 2D points using one warp (32 threads) per query.
// The implementation follows the specification:
// - One warp processes one query.
// - The intermediate result of size k is stored in registers, distributed so that each thread
//   keeps k/32 consecutive elements.
// - Each warp has a candidate buffer of size k in shared memory, plus a per-warp count.
// - Data points are processed in block-wide shared-memory tiles.
// - Candidates with distance < max_distance are buffered using warp ballot/prefix sums.
// - When the candidate buffer is full, it is merged with the intermediate result using:
//   1) Swap (bring buffer into registers, push current register result into shared)
//   2) Bitonic sort of the buffer (registers) ascending
//   3) Bitonic "min-of-pairs" merge into a bitonic sequence
//   4) Bitonic sort again to obtain the updated sorted intermediate result.
//
// Notes:
// - k is a power of two in [32, 1024]. We use warpSize=32, so items_per_thread = k/32 is a power of two in [1, 32].
// - Bitonic sort stages are implemented with warp shuffles for cross-lane exchanges and register swaps for intra-lane exchanges.
// - The output array is std::pair<int,float> (index, distance). We write to it using a POD type with identical layout.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// POD type matching std::pair<int,float> layout (first=index, second=distance).
struct PairIO {
    int   first;
    float second;
};

// Internal pair representation (distance first) for comparisons.
struct IdxDist {
    float d;
    int   idx;
};

static __device__ __forceinline__ void swap_pair(float &a, int &ai, float &b, int &bi) {
    float ta = a;  int tai = ai;
    a = b;         ai = bi;
    b = ta;        bi = tai;
}

// Warp-wide bitonic sort of N = items_per_thread * 32 elements distributed so that
// each lane holds items_per_thread consecutive elements in arrays d[0..items_per_thread-1], idx[...], ascending order.
// This implementation follows the classic bitonic network:
// for (size=2; size<=N; size*=2)
//   for (stride=size/2; stride>0; stride/=2)
//     compare-exchange i with i^stride in direction up = ((i & size) == 0)
//
// Cross-lane exchanges (stride >= items_per_thread) use __shfl_xor_sync with delta = stride / items_per_thread.
// Intra-lane exchanges (stride < items_per_thread) swap registers directly.
static __device__ __forceinline__
void warp_bitonic_sort(float d[], int idx[], int items_per_thread, int N)
{
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // size: current bitonic sequence size
    for (int size = 2; size <= N; size <<= 1) {
        // stride: current distance between compared indices
        for (int stride = size >> 1; stride > 0; stride >>= 1) {

            if (stride < items_per_thread) {
                // Intra-thread compare-exchange: partner index within the same lane's register array.
                // Process each pair exactly once by requiring partner > j.
                for (int j = 0; j < items_per_thread; ++j) {
                    int partner = j ^ stride;
                    if (partner > j) {
                        int i_index = lane * items_per_thread + j;
                        bool up = ((i_index & size) == 0);

                        float a = d[j];       int ai = idx[j];
                        float b = d[partner]; int bi = idx[partner];

                        // Compare a and b, keep ascending order if up, descending otherwise.
                        bool comp = (a > b);
                        if (up ? comp : !comp) {
                            // swap
                            d[j] = b;       idx[j] = bi;
                            d[partner] = a; idx[partner] = ai;
                        }
                    }
                }
            } else {
                // Cross-thread compare-exchange: partner in different lane with the same local register index.
                // delta is the XOR distance in lane space.
                int delta = stride / items_per_thread;

                for (int j = 0; j < items_per_thread; ++j) {
                    int i_index = lane * items_per_thread + j;
                    bool up = ((i_index & size) == 0);

                    float self_d = d[j];
                    int   self_i = idx[j];

                    float other_d = __shfl_xor_sync(full_mask, self_d, delta);
                    int   other_i = __shfl_xor_sync(full_mask, self_i, delta);

                    bool comp = (self_d > other_d);
                    if (up ? comp : !comp) {
                        d[j] = other_d;
                        idx[j] = other_i;
                    }
                }
            }
        }
    }
}

// Merge the full candidate buffer (size k) in shared memory with the current intermediate result in registers.
// - Load buffer (k items) into regs (bufD, bufI) and overwrite shared buffer with current registers (swap).
// - Sort buffer ascending (bitonic).
// - Min-of-pairs with shared array at mirrored indices to form a bitonic sequence in registers.
// - Sort registers ascending to get the updated intermediate result.
// On exit, regD/regI contain the updated sorted top-k, and returns the updated max_distance.
static __device__ __forceinline__
float merge_full_buffer_with_registers(IdxDist* __restrict__ warp_buf, // shared memory buffer of size k for this warp
                                       float regD[], int regI[],
                                       int items_per_thread, int k)
{
    const unsigned full_mask = 0xFFFFFFFFu;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Step 1: Swap - bring buffer into registers and push current registers into shared memory.
    float bufD[32];
    int   bufI[32];

    for (int j = 0; j < items_per_thread; ++j) {
        int g = lane * items_per_thread + j;
        IdxDist v = warp_buf[g];
        bufD[j] = v.d;
        bufI[j] = v.idx;

        // Store current intermediate result into shared memory at the same positions (swap).
        warp_buf[g].d = regD[j];
        warp_buf[g].idx = regI[j];
    }
    __syncwarp(); // ensure the swap is visible within the warp

    // Step 2: Sort the buffer (in registers) ascending.
    warp_bitonic_sort(bufD, bufI, items_per_thread, k);

    // Step 3: Pairwise min with mirrored positions from the intermediate result in shared memory.
    // After swap, warp_buf[] holds the previous intermediate result (sorted ascending).
    for (int j = 0; j < items_per_thread; ++j) {
        int i = lane * items_per_thread + j;
        int oi = k - 1 - i; // mirrored index
        IdxDist other = warp_buf[oi];
        float a = bufD[j];
        int   ai = bufI[j];
        if (other.d < a) {
            bufD[j] = other.d;
            bufI[j] = other.idx;
        }
    }

    // Step 4: Sort the merged sequence (in registers) ascending.
    warp_bitonic_sort(bufD, bufI, items_per_thread, k);

    // Copy back to the register-based intermediate result.
    for (int j = 0; j < items_per_thread; ++j) {
        regD[j] = bufD[j];
        regI[j] = bufI[j];
    }

    // Update max_distance (the last element, global index k-1, is held by lane 31 at local index items_per_thread-1).
    float last_local = regD[items_per_thread - 1];
    float max_distance = __shfl_sync(full_mask, last_local, WARP_SIZE - 1);
    return max_distance;
}

// Kernel implementing the k-NN with the described warp-centric method.
__global__ void knn_kernel(const float2* __restrict__ queries,
                           int query_count,
                           const float2* __restrict__ data,
                           int data_count,
                           PairIO* __restrict__ out, // same layout as std::pair<int,float>
                           int k,
                           int tile_points)
{
    extern __shared__ unsigned char smem_raw[];
    // Shared memory layout per block:
    // [0, tile_points) float2 cached data tile
    // Then, per warp: candidate buffer of k IdxDist, and a single int count per warp
    float2* tile = reinterpret_cast<float2*>(smem_raw);
    unsigned char* ptr = smem_raw + sizeof(float2) * tile_points;

    const int warps_per_block = blockDim.x / WARP_SIZE;
    IdxDist* warp_buf_base = reinterpret_cast<IdxDist*>(ptr);
    int* warp_count_base = reinterpret_cast<int*>(warp_buf_base + warps_per_block * k);

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const bool warp_active = (warp_global < query_count);
    const unsigned full_mask = 0xFFFFFFFFu;

    // Slice this warp's candidate buffer and count in shared memory.
    IdxDist* warp_buf = warp_buf_base + warp_in_block * k;
    int* warp_count = warp_count_base + warp_in_block;

    // Initialize per-warp candidate count to 0.
    if (lane == 0) {
        *warp_count = 0;
    }
    __syncwarp();

    // Each warp will hold k/32 items in registers.
    const int items_per_thread = k / WARP_SIZE;

    // Register-resident intermediate result (sorted ascending). Initialize with +inf.
    float regD[32];
    int   regI[32];
    if (warp_active) {
        for (int j = 0; j < items_per_thread; ++j) {
            regD[j] = CUDART_INF_F;
            regI[j] = -1;
        }
    }

    // Load this warp's query point and broadcast within the warp.
    float2 q = make_float2(0.0f, 0.0f);
    if (warp_active) {
        if (lane == 0) {
            q = queries[warp_global];
        }
        q.x = __shfl_sync(full_mask, q.x, 0);
        q.y = __shfl_sync(full_mask, q.y, 0);
    }

    // Current max_distance (k-th best) for filtering. Initially +inf.
    float max_distance = CUDART_INF_F;

    // Iterate over the dataset in shared-memory tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += tile_points) {
        int tile_count = data_count - tile_start;
        if (tile_count > tile_points) tile_count = tile_points;

        // Block-wide load of the current tile into shared memory.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            tile[i] = data[tile_start + i];
        }
        __syncthreads();

        // Each warp processes the cached tile against its query point.
        if (warp_active) {
            for (int base = 0; base < tile_count; base += WARP_SIZE) {
                int idx_in_tile = base + lane;
                bool valid = (idx_in_tile < tile_count);

                float2 p = valid ? tile[idx_in_tile] : make_float2(0.0f, 0.0f);
                float dx = p.x - q.x;
                float dy = p.y - q.y;
                float dist = dx * dx + dy * dy;
                int   gidx = tile_start + idx_in_tile;

                // Filter by current max_distance.
                bool is_candidate = valid && (dist < max_distance);
                unsigned cand_mask = __ballot_sync(full_mask, is_candidate);
                int n_cand = __popc(cand_mask);

                if (n_cand > 0) {
                    // Current buffer fill and capacity.
                    int base_pos = 0;
                    int capacity = 0;

                    if (lane == 0) {
                        int cur = *warp_count;
                        capacity = k - cur;
                        int add = (n_cand < capacity) ? n_cand : capacity;
                        *warp_count = cur + add;
                        base_pos = cur;
                    }
                    base_pos = __shfl_sync(full_mask, base_pos, 0);
                    capacity = __shfl_sync(full_mask, capacity, 0);

                    int pos_in_mask = __popc(cand_mask & ((1u << lane) - 1u));
                    bool fits_now = is_candidate && (pos_in_mask < capacity);

                    if (fits_now) {
                        int wpos = base_pos + pos_in_mask;
                        warp_buf[wpos].d = dist;
                        warp_buf[wpos].idx = gidx;
                    }

                    // If the buffer is (now) full, merge it with the register-resident result.
                    if (n_cand > capacity) {
                        // Complete filling made count reach k -> merge.
                        __syncwarp();

                        // Before merging, ensure the buffer has exactly k items (it is full by construction).
                        max_distance = merge_full_buffer_with_registers(warp_buf, regD, regI, items_per_thread, k);

                        // Reset candidate count (buffer emptied by merge).
                        if (lane == 0) *warp_count = 0;
                        __syncwarp();

                        // Handle leftover candidates from this batch (those beyond 'capacity'), re-filtered by updated max_distance.
                        bool leftover = is_candidate && (pos_in_mask >= capacity) && (dist < max_distance);
                        unsigned left_mask = __ballot_sync(full_mask, leftover);
                        int n_left = __popc(left_mask);

                        if (n_left > 0) {
                            int lpos = __popc(left_mask & ((1u << lane) - 1u));
                            if (leftover) {
                                warp_buf[lpos].d = dist;
                                warp_buf[lpos].idx = gidx;
                            }
                            if (lane == 0) *warp_count = n_left;
                        }
                        __syncwarp();
                    }
                }
            }
        }

        __syncthreads(); // Ensure all warps finished using this tile before loading the next one.
    }

    // After processing all tiles, merge any remaining candidates in the buffer.
    if (warp_active) {
        int remain = 0;
        if (lane == 0) remain = *warp_count;
        remain = __shfl_sync(full_mask, remain, 0);

        if (remain > 0) {
            // Pad the buffer to k with +inf so that we can run the full merge.
            for (int i = lane; i < k; i += WARP_SIZE) {
                if (i >= remain) {
                    warp_buf[i].d = CUDART_INF_F;
                    warp_buf[i].idx = -1;
                }
            }
            __syncwarp();

            max_distance = merge_full_buffer_with_registers(warp_buf, regD, regI, items_per_thread, k);

            // Reset count (not strictly necessary after final merge).
            if (lane == 0) *warp_count = 0;
            __syncwarp();
        }

        // Store the final sorted k nearest neighbors to output.
        // Each thread writes its items_per_thread consecutive results.
        int base_out = warp_global * k + lane * items_per_thread;
        for (int j = 0; j < items_per_thread; ++j) {
            PairIO w;
            w.first = regI[j];
            w.second = regD[j];
            out[base_out + j] = w;
        }
    }
}

// Host-side launcher conforming to the requested interface.
/// @FIXED
/// extern "C"

void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Choose a reasonable number of warps per block.
    // 8 warps (256 threads) generally provides good occupancy while leaving sufficient shared memory headroom.
    int warps_per_block = 8;
    int threads_per_block = warps_per_block * WARP_SIZE;

    // Query device shared memory capacity (opt-in).
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Try to use the maximum opt-in shared memory per block if available; fall back to default.
    int shmem_limit = prop.sharedMemPerBlockOptin ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock;

    // Per-warp candidate buffer bytes: k * sizeof(IdxDist) (8 bytes per entry).
    size_t per_warp_bytes = static_cast<size_t>(k) * sizeof(IdxDist) + sizeof(int);
    size_t per_block_warp_bytes = static_cast<size_t>(warps_per_block) * per_warp_bytes;

    // Compute tile size (number of float2) based on remaining shared memory.
    // Ensure tile_points is a multiple of threads_per_block for coalesced loads.
    size_t bytes_for_tile = 0;
    // Leave a small safety margin to avoid over-allocating dynamic smem due to alignment.
    const size_t safety_margin = 0;

    if (per_block_warp_bytes + safety_margin < static_cast<size_t>(shmem_limit)) {
        bytes_for_tile = static_cast<size_t>(shmem_limit) - per_block_warp_bytes - safety_margin;
    } else {
        // If buffers already exceed shmem_limit with 8 warps, reduce warps_per_block to fit.
        // This shouldn't happen for k <= 1024 on modern data center GPUs, but we handle it defensively.
        warps_per_block = 4;
        threads_per_block = warps_per_block * WARP_SIZE;
        per_block_warp_bytes = static_cast<size_t>(warps_per_block) * per_warp_bytes;
        if (per_block_warp_bytes < static_cast<size_t>(shmem_limit)) {
            bytes_for_tile = static_cast<size_t>(shmem_limit) - per_block_warp_bytes - safety_margin;
        } else {
            // As a last resort, set the tile to fit at least one blockDim.x worth of points.
            bytes_for_tile = 0;
        }
    }

    // Ensure the tile can hold at least blockDim.x points (one per thread) if possible.
    size_t tile_points = 0;
    if (bytes_for_tile >= sizeof(float2)) {
        tile_points = bytes_for_tile / sizeof(float2);
        // Make tile_points a multiple of threads_per_block for easy coalesced loads.
        if (tile_points < static_cast<size_t>(threads_per_block)) {
            tile_points = threads_per_block;
        } else {
            tile_points = (tile_points / threads_per_block) * threads_per_block;
        }
    } else {
        // Minimal tile as multiple of threads_per_block.
        tile_points = threads_per_block;
    }

    // Dynamic shared memory size for the kernel launch.
    size_t dynamic_smem_size = tile_points * sizeof(float2) + per_block_warp_bytes;

    // Allow the kernel to use the required dynamic shared memory size (opt-in).
    cudaFuncSetAttribute(knn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dynamic_smem_size);

    // Compute grid size: one warp per query.
    int blocks = (query_count + warps_per_block - 1) / warps_per_block;

    // Launch kernel.
    // Reinterpret the output pointer as PairIO*, which matches std::pair<int,float> layout.
    PairIO* out_ptr = reinterpret_cast<PairIO*>(result);

    knn_kernel<<<blocks, threads_per_block, dynamic_smem_size>>>(
        query, query_count, data, data_count, out_ptr, k, (int)tile_points
    );

    // Optional: synchronize here if the caller expects the results immediately.
    // cudaDeviceSynchronize();
}
