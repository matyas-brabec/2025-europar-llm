#include <cuda_runtime.h>
#include <cuda.h>
#include <utility>

// This implementation computes k-NN (squared Euclidean distances) for 2D points using a warp-per-query strategy.
// - Each warp (32 threads) handles a single query point.
// - The data points are processed in shared-memory tiles loaded cooperatively by the entire block.
// - Each warp maintains its private top-k candidate set distributed across lanes (k/32 per lane).
// - A warp-wide selection replaces the current global worst candidate when a better one is found.
// - Finally, each warp writes its k candidates to shared memory and performs an in-place bitonic sort,
//   then writes the sorted k nearest neighbors to the output.
//
// Notes on data layout:
// - Input arrays 'query' and 'data' hold float2 points (x, y) in row-major order.
// - Output array 'result' holds std::pair<int, float> where 'first' is index in 'data' and 'second' is squared distance.
//
// Performance considerations for modern NVIDIA GPUs (A100/H100):
// - 256 threads per block (8 warps) by default provides good occupancy while leaving sufficient shared memory.
// - Dynamic shared memory is sized at runtime as max(per-warp top-k buffer, preferred tile buffer).
// - The kernel uses warp intrinsics (__reduce_max_sync, __ballot_sync, __shfl_sync) for efficient warp communication.

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Device-side pair compatible with std::pair<int, float> (assumed sequential layout: int first; float second).
struct PairIF {
    int   first;
    float second;
};

// Validate that our device-side pair matches the host-side std::pair<int,float> layout.
static_assert(sizeof(PairIF) == sizeof(std::pair<int, float>), "PairIF must match std::pair<int,float> size");
static_assert(alignof(PairIF) == alignof(std::pair<int, float>), "PairIF must match std::pair<int,float> alignment");

__device__ __forceinline__ float sq_l2_dist2d(const float2 a, const float2 b) {
    // Compute squared Euclidean distance using FMA for better throughput.
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return __fmaf_rn(dx, dx, dy * dy);
}

// Kernel implementing warp-per-query k-NN over 2D points.
// - query:    [query_count] float2
// - data:     [data_count]  float2
// - result:   [query_count * k] PairIF
// - k:        power of two in [32, 1024], and data_count >= k
// - warpsPerBlock: number of warps per block (threads per block = 32 * warpsPerBlock)
// - tilePoints: number of data points per shared-memory tile (block-wide)
__global__ void knn_kernel_2d_warp(const float2* __restrict__ query,
                                   int query_count,
                                   const float2* __restrict__ data,
                                   int data_count,
                                   PairIF* __restrict__ result,
                                   int k,
                                   int warpsPerBlock,
                                   int tilePoints)
{
    extern __shared__ unsigned char smem_bytes[];
    // Shared memory region, first used as a tile buffer (float2), later as per-warp PairIF buffer for sorting.
    float2* tile = reinterpret_cast<float2*>(smem_bytes);

    const int lane           = threadIdx.x & (WARP_SIZE - 1);
    const int warpIdInBlock  = threadIdx.x >> 5;
    const int globalWarpId   = blockIdx.x * warpsPerBlock + warpIdInBlock;
    const bool hasQuery      = (globalWarpId < query_count);

    // Broadcast the query point to the whole warp (one load per warp).
    float qx = 0.0f, qy = 0.0f;
    if (hasQuery) {
        float2 q0 = (lane == 0) ? query[globalWarpId] : make_float2(0.0f, 0.0f);
        unsigned fullMask = 0xFFFFFFFFu;
        qx = __shfl_sync(fullMask, q0.x, 0);
        qy = __shfl_sync(fullMask, q0.y, 0);
    }

    // Each lane maintains k/32 entries.
    const int perLane = k / WARP_SIZE; // k is guaranteed power of two and >= 32.
    float topDist[WARP_SIZE];  // up to 32
    int   topIdx [WARP_SIZE];  // up to 32

    #pragma unroll
    for (int i = 0; i < WARP_SIZE; ++i) {
        if (i < perLane) {
            topDist[i] = CUDART_INF_F;
            topIdx[i]  = -1;
        }
    }

    // Helper to recompute this lane's local worst among its perLane entries.
    auto recompute_local_worst = [&](float& worstVal, int& worstPos) {
        if (!hasQuery) {
            worstVal = -CUDART_INF_F;
            worstPos = -1;
            return;
        }
        float wv = -CUDART_INF_F;
        int wp   = 0;
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            if (i < perLane) {
                float d = topDist[i];
                if (d > wv) { wv = d; wp = i; }
            }
        }
        worstVal = wv;
        worstPos = wp;
    };

    float localWorstVal;
    int   localWorstPos;
    recompute_local_worst(localWorstVal, localWorstPos);

    // Process the data in tiles cached in shared memory by the entire block.
    for (int base = 0; base < data_count; base += tilePoints) {
        const int remaining = data_count - base;
        const int tileCount = (remaining < tilePoints) ? remaining : tilePoints;

        // Cooperative load of the tile by all threads in the block.
        for (int i = threadIdx.x; i < tileCount; i += blockDim.x) {
            tile[i] = data[base + i];
        }
        __syncthreads();

        if (hasQuery) {
            // Each warp loops through the tile, each lane visiting strided elements.
            for (int j = lane; j < tileCount; j += WARP_SIZE) {
                float2 p = tile[j];
                // Compute squared distance without sqrt
                float dx = p.x - qx;
                float dy = p.y - qy;
                float dist = __fmaf_rn(dx, dx, dy * dy);
                unsigned fullMask = 0xFFFFFFFFu;

                // Compute the current global worst value across the warp (max of lane-local worsts).
                // Note: __reduce_max_sync returns the maximum across the active warp lanes.
                /// @FIXED
                /// float worstVal = __reduce_max_sync(fullMask, localWorstVal);
                float worstVal = __reduce_max_sync(fullMask, reinterpret_cast<unsigned&>(localWorstVal)); // unsigned and positive float reinterpretations follow the same ordering

                // Quickly reject if the candidate is not better than the global worst.
                if (dist < worstVal) {
                    // Which lane holds that global worst? Use ballot on equality to worstVal and pick highest lane id.
                    unsigned eqMask = __ballot_sync(fullMask, localWorstVal == worstVal);
                    int worstLane = 31 - __clz(eqMask); // choose the highest set bit (tie-breaker)
                    // Get the position (index within that lane) to replace.
                    int worstPos = __shfl_sync(fullMask, localWorstPos, worstLane);

                    if (lane == worstLane) {
                        topDist[worstPos] = dist;
                        topIdx[worstPos]  = base + j;
                        // Update this lane's local worst for subsequent iterations.
                        recompute_local_worst(localWorstVal, localWorstPos);
                    }
                    // Other lanes keep their local worst cached; it's still valid.
                }
            }
        }
        __syncthreads();
    }

    // Reuse shared memory as per-warp PairIF buffer, then sort using bitonic sort (ascending by distance).
    PairIF* pairbuf = reinterpret_cast<PairIF*>(smem_bytes);

    if (hasQuery) {
        PairIF* myBuf = pairbuf + warpIdInBlock * k;

        // Scatter the per-lane entries into a contiguous buffer [0..k).
        // Map lane-local index i in [0..perLane) to position pos = i * WARP_SIZE + lane.
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            if (i < perLane) {
                int pos = i * WARP_SIZE + lane;
                myBuf[pos].first  = topIdx[i];
                myBuf[pos].second = topDist[i];
            }
        }
        __syncwarp();

        // Bitonic sort network for length k (k is power-of-two).
        // Each thread processes multiple indices strided by warp size.
        for (int size = 2; size <= k; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = lane; i < k; i += WARP_SIZE) {
                    int partner = i ^ stride;
                    if (partner > i) {
                        bool up = ((i & size) == 0);
                        PairIF a = myBuf[i];
                        PairIF b = myBuf[partner];
                        // Compare by distance (second)
                        bool swap = up ? (a.second > b.second) : (a.second < b.second);
                        if (swap) {
                            myBuf[i]       = b;
                            myBuf[partner] = a;
                        }
                    }
                }
                __syncwarp();
            }
        }

        // Write sorted results to global memory.
        PairIF* out = result + static_cast<size_t>(globalWarpId) * static_cast<size_t>(k);
        for (int i = lane; i < k; i += WARP_SIZE) {
            out[i] = myBuf[i];
        }
    }
}

// Host interface.
// The arrays query, data, and result are assumed to be device allocations (cudaMalloc).
// - query:  device pointer to query_count float2 points
// - data:   device pointer to data_count float2 points
// - result: device pointer to query_count * k pairs (std::pair<int,float>)
// - k:      power of two in [32, 1024], and data_count >= k
void run_knn(const float2 *query,
             int query_count,
             const float2 *data,
             int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Choose the number of warps per block based on shared memory availability and k.
    int device = 0;
    cudaGetDevice(&device);

    int maxOptinShared = 0;
    cudaDeviceGetAttribute(&maxOptinShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (maxOptinShared <= 0) {
        // Fallback to default per-block shared memory size if opt-in is unavailable.
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        maxOptinShared = prop.sharedMemPerBlock; // Typically 48KB or 64KB on many devices
    }

    // Start with a preferred number of warps per block.
    int warpsPerBlock = 8; // 8 warps => 256 threads per block (good default on A100/H100)
    // Ensure the per-warp top-k buffer fits into available dynamic shared memory.
    size_t perBlockTopKBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) * sizeof(PairIF);
    if (perBlockTopKBytes > static_cast<size_t>(maxOptinShared)) {
        warpsPerBlock = static_cast<int>(maxOptinShared / (static_cast<size_t>(k) * sizeof(PairIF)));
        if (warpsPerBlock < 1) warpsPerBlock = 1;
        perBlockTopKBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(k) * sizeof(PairIF);
    }

    // Preferred tile size: aim for ~64 KiB of float2 points if possible.
    size_t preferredTileBytes = 64 * 1024; // 64KB
    // Total dynamic shared memory is the maximum of per-warp top-k buffer and preferred tile buffer.
    size_t dynSmemBytes = perBlockTopKBytes > preferredTileBytes ? perBlockTopKBytes : preferredTileBytes;
    // Cap to the device's opt-in limit.
    if (dynSmemBytes > static_cast<size_t>(maxOptinShared)) {
        dynSmemBytes = static_cast<size_t>(maxOptinShared);
    }

    // Configure kernel attribute to allow dynamic shared memory up to dynSmemBytes.
    cudaFuncSetAttribute(knn_kernel_2d_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynSmemBytes));

    // Compute tilePoints from the chosen dynamic shared memory size (shared memory is reused).
    int tilePoints = static_cast<int>(dynSmemBytes / sizeof(float2));
    if (tilePoints < WARP_SIZE) tilePoints = WARP_SIZE; // Ensure minimal useful tile size.

    // Configure grid and block.
    const int threadsPerBlock = warpsPerBlock * WARP_SIZE;
    const int blocks = (query_count + warpsPerBlock - 1) / warpsPerBlock;

    // Launch the kernel.
    PairIF* result_if = reinterpret_cast<PairIF*>(result);
    knn_kernel_2d_warp<<<blocks, threadsPerBlock, dynSmemBytes>>>(
        query, query_count, data, data_count, result_if, k, warpsPerBlock, tilePoints
    );

    // Optional: synchronize or leave to caller. Here we ensure completion.
    cudaDeviceSynchronize();
}
