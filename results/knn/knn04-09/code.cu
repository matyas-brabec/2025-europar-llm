#include <cuda_runtime.h>
#include <utility>
#include <cstddef>

namespace {

// One warp computes one query, and each block processes 8 queries concurrently.
// This keeps the block size at 256 threads, which is a sweet spot on A100/H100
// while maximizing reuse of each staged data tile across multiple queries.
constexpr int kWarpSize        = 32;
constexpr int kWarpsPerBlock   = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr unsigned kFullMask   = 0xffffffffu;

// The result ordering only needs to be consistent, not stable, so we impose a
// total order by sorting by (distance, index).  Distances are squared L2 and are
// therefore non-negative; for non-negative IEEE-754 floats, the raw bit pattern
// is monotone under unsigned comparison.  We exploit that by packing:
//
//   key[63:32] = raw float bits of distance
//   key[31: 0] = data index
//
// Then a single 64-bit unsigned compare implements the desired ordering.
using packed_t = unsigned long long;
constexpr unsigned kInvalidIndex = 0x7fffffffu;
constexpr packed_t kSentinel =
    (packed_t{0x7f800000u} << 32) | packed_t{kInvalidIndex};  // (+inf, invalid)

using result_pair_t = std::pair<int, float>;

static_assert(sizeof(float2)  == sizeof(packed_t), "float2 must be 8 bytes");
static_assert(sizeof(packed_t) == 8,               "packed_t must be 8 bytes");

// Runtime shared-memory limits are queried once per call so that the launcher can
// pick larger tiles on GPUs with larger per-block/per-SM shared memory (e.g. H100)
// while remaining A100-safe.
struct DeviceSharedCaps {
    int max_block_optin = 0;
    int max_sm_shared   = 0;
};

inline DeviceSharedCaps query_device_shared_caps() {
    DeviceSharedCaps caps{};
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&caps.max_block_optin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    cudaDeviceGetAttribute(&caps.max_sm_shared,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                           device);
    return caps;
}

template <int K, int TILE_POINTS>
constexpr int shared_bytes() {
    // Shared layout:
    //   [ TILE_POINTS * float2 ] [ kWarpsPerBlock * (2*K) * packed_t ]
    // Each warp owns:
    //   - top-K sorted array in slots [0, K)
    //   - K-sized unsorted append buffer in slots [K, 2K)
    return TILE_POINTS * static_cast<int>(sizeof(float2)) +
           kWarpsPerBlock * 2 * K * static_cast<int>(sizeof(packed_t));
}

template <int K, int TILE_POINTS>
inline bool supports_shared(const DeviceSharedCaps& caps) {
    constexpr int bytes = shared_bytes<K, TILE_POINTS>();
    return bytes <= caps.max_block_optin;
}

template <int K, int TILE_POINTS>
inline bool supports_two_blocks_per_sm(const DeviceSharedCaps& caps) {
    constexpr int bytes = shared_bytes<K, TILE_POINTS>();
    return bytes <= caps.max_block_optin &&
           (2ll * static_cast<long long>(bytes)) <=
               static_cast<long long>(caps.max_sm_shared);
}

__device__ __forceinline__ unsigned lane_mask_lt() {
    unsigned mask;
    asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ packed_t pack_candidate(const float dist,
                                                   const int   idx) {
    return (static_cast<packed_t>(__float_as_uint(dist)) << 32) |
           static_cast<packed_t>(static_cast<unsigned>(idx));
}

__device__ __forceinline__ float unpack_distance(const packed_t value) {
    return __uint_as_float(static_cast<unsigned>(value >> 32));
}

__device__ __forceinline__ int unpack_index(const packed_t value) {
    return static_cast<int>(static_cast<unsigned>(value));
}

__device__ __forceinline__ packed_t make_candidate(const float qx,
                                                   const float qy,
                                                   const float2 p,
                                                   const int idx) {
    const float dx   = qx - p.x;
    const float dy   = qy - p.y;
    const float dist = fmaf(dx, dx, dy * dy);  // squared Euclidean distance
    return pack_candidate(dist, idx);
}

template <int N>
__device__ __forceinline__ void warp_bitonic_sort(packed_t* values) {
    static_assert((N & (N - 1)) == 0, "bitonic sort length must be a power of two");
    static_assert(N >= kWarpSize,      "bitonic sort length must be at least one warp");

    const int lane = threadIdx.x & (kWarpSize - 1);

    // Generic power-of-two bitonic sort executed by a single warp over shared memory.
    // Each lane owns the positions lane, lane+32, lane+64, ...
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll 1
            for (int pos = lane; pos < N; pos += kWarpSize) {
                const int partner = pos ^ stride;
                if (partner > pos) {
                    const bool up = ((pos & size) == 0);
                    const packed_t a = values[pos];
                    const packed_t b = values[partner];
                    if ((up && a > b) || (!up && a < b)) {
                        values[pos]     = b;
                        values[partner] = a;
                    }
                }
            }
            __syncwarp(kFullMask);
        }
    }
}

template <int K>
__device__ __forceinline__ packed_t warp_read_cutoff(const packed_t* knn) {
    packed_t cutoff = kSentinel;
    if ((threadIdx.x & (kWarpSize - 1)) == 0) {
        cutoff = knn[K - 1];
    }
    return __shfl_sync(kFullMask, cutoff, 0);
}

template <int K>
__device__ __forceinline__ void flush_buffer(packed_t* knn, const int buf_count) {
    // knn[0, K)   : current sorted top-K
    // knn[K, 2K)  : unsorted buffer with buf_count valid entries
    //
    // Flush strategy:
    //   1) Pad the unused buffer suffix with sentinels.
    //   2) Sort only the buffer (size K), not the whole 2K state.
    //   3) Merge sorted top-K and sorted buffer to keep the smallest K keys.
    //
    // Sorting only the buffer keeps maintenance cheaper than re-sorting all 2K
    // elements every time the buffer fills.
    const int lane = threadIdx.x & (kWarpSize - 1);
    packed_t* const buffer = knn + K;

    for (int pos = buf_count + lane; pos < K; pos += kWarpSize) {
        buffer[pos] = kSentinel;
    }
    __syncwarp(kFullMask);

    warp_bitonic_sort<K>(buffer);
    __syncwarp(kFullMask);

    // The merge is done by lane 0 only.  This is acceptable because the costly
    // part is the warp-wide sort, and buffering makes flushes infrequent once
    // the cutoff stabilizes.  The merge itself is linear and in-place.
    if (lane == 0) {
        int i = K - 1;       // end of top-K range [0, K)
        int j = 2 * K - 1;   // end of buffer range [K, 2K)

        // Discard the largest K keys from the union of the two sorted arrays.
        // The remaining prefixes contain exactly the smallest K keys.
        for (int discard = 0; discard < K; ++discard) {
            if (i < 0) {
                --j;
            } else if (j < K) {
                --i;
            } else if (knn[j] >= knn[i]) {
                --j;
            } else {
                --i;
            }
        }

        // Backward in-place merge of the kept prefixes into knn[0, K).
        int out = K - 1;
        while (out >= 0) {
            if (j < K) {
                knn[out--] = knn[i--];
            } else if (i < 0) {
                knn[out--] = knn[j--];
            } else if (knn[j] >= knn[i]) {
                knn[out--] = knn[j--];
            } else {
                knn[out--] = knn[i--];
            }
        }
    }

    __syncwarp(kFullMask);
}

template <int K, int TILE_POINTS>
__global__ __launch_bounds__(kThreadsPerBlock)
void knn_kernel(const float2* __restrict__ query,
                const int query_count,
                const float2* __restrict__ data,
                const int data_count,
                result_pair_t* __restrict__ result) {
    static_assert((K & (K - 1)) == 0, "K must be a power of two");
    static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024]");
    static_assert((TILE_POINTS % kWarpSize) == 0, "Tile size must be warp-aligned");
    static_assert(TILE_POINTS >= K, "Chosen tile must hold at least one initial top-K");

    // Shared memory is addressed in 8-byte units because both float2 and packed_t
    // are 8 bytes.  The first TILE_POINTS slots hold the staged data tile; the rest
    // hold per-warp top-K + buffer state.
    extern __shared__ packed_t smem[];
    float2* const sh_data   = reinterpret_cast<float2*>(smem);
    packed_t* const sh_knn  = smem + TILE_POINTS;

    const int lane    = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int query_idx = blockIdx.x * kWarpsPerBlock + warp_id;
    const bool valid = (query_idx < query_count);

    // Per-warp private K-NN state in shared memory.
    packed_t* const knn = sh_knn + warp_id * (2 * K);

    // One query point is loaded once by lane 0 and broadcast to the whole warp.
    float qx = 0.0f;
    float qy = 0.0f;
    if (valid && lane == 0) {
        const float2 q = query[query_idx];
        qx = q.x;
        qy = q.y;
    }
    qx = __shfl_sync(kFullMask, qx, 0);
    qy = __shfl_sync(kFullMask, qy, 0);

    const unsigned lt_mask  = lane_mask_lt();
    const unsigned lane_bit = 1u << lane;

    int filled    = 0;          // number of valid entries currently in knn[0, K)
    int buf_count = 0;          // number of valid entries currently in knn[K, 2K)
    packed_t cutoff = kSentinel;

    for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS) {
        const int remaining  = data_count - tile_base;
        const int tile_count = (remaining < TILE_POINTS) ? remaining : TILE_POINTS;

        // Stage the current data tile once per block; all 8 warps will reuse it.
        for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
            sh_data[i] = data[tile_base + i];
        }
        __syncthreads();

        if (valid) {
            for (int chunk = 0; chunk < tile_count; chunk += kWarpSize) {
                const int offset = chunk + lane;

                packed_t candidate = kSentinel;
                if (offset < tile_count) {
                    candidate = make_candidate(qx, qy, sh_data[offset], tile_base + offset);
                }

                if (filled < K) {
                    // Initial fill: the first K candidates are written directly,
                    // then sorted once to establish the initial cutoff.
                    knn[filled + lane] = candidate;
                    filled += kWarpSize;

                    if (filled == K) {
                        __syncwarp(kFullMask);
                        warp_bitonic_sort<K>(knn);
                        cutoff = warp_read_cutoff<K>(knn);
                    }
                    continue;
                }

                // Admission test against the current cutoff.  Warp ballot + prefix
                // compacts all admitted candidates from this 32-point chunk into
                // contiguous buffer slots in one shot.
                unsigned mask = __ballot_sync(kFullMask, candidate < cutoff);
                int count = __popc(mask);

                if (count != 0) {
                    // If the current admissions would overflow the K-sized buffer,
                    // flush first, then re-test against the tighter cutoff.  This
                    // prevents overflow and reduces future buffer pressure.
                    if (buf_count != 0 && (buf_count + count) > K) {
                        flush_buffer<K>(knn, buf_count);
                        buf_count = 0;
                        cutoff = warp_read_cutoff<K>(knn);

                        mask  = __ballot_sync(kFullMask, candidate < cutoff);
                        count = __popc(mask);
                    }

                    if (count != 0) {
                        const int rank = __popc(mask & lt_mask);
                        if (mask & lane_bit) {
                            knn[K + buf_count + rank] = candidate;
                        }
                        buf_count += count;

                        if (buf_count == K) {
                            flush_buffer<K>(knn, buf_count);
                            buf_count = 0;
                            cutoff = warp_read_cutoff<K>(knn);
                        }
                    }
                }
            }
        }

        // Ensure no warp overwrites the staged tile before all warps in the block
        // are done consuming it.
        __syncthreads();
    }

    if (valid) {
        if (buf_count != 0) {
            flush_buffer<K>(knn, buf_count);
        }

        // Write the final sorted top-K to global memory.  We write std::pair members
        // directly to avoid relying on device-qualified std::pair constructors or
        // assignments; only raw member stores are needed here.
        const std::size_t out_base =
            static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

        for (int pos = lane; pos < K; pos += kWarpSize) {
            result[out_base + pos].first  = unpack_index(knn[pos]);
            result[out_base + pos].second = unpack_distance(knn[pos]);  // squared distance
        }
    }
}

template <int K, int TILE_POINTS>
inline void launch_knn_specialized(const float2* query,
                                  const int query_count,
                                  const float2* data,
                                  const int data_count,
                                  result_pair_t* result) {
    constexpr int shmem = shared_bytes<K, TILE_POINTS>();

    // The largest specialization used here is the H100-optimized K=1024,
    // TILE_POINTS=8192 path at 196608 bytes of dynamic shared memory.
    static_assert(shmem <= 196608, "specialization exceeds intended shared-memory budget");

    // Modern A100/H100-class GPUs require explicit opt-in for large dynamic shared
    // memory.  Prefer the shared-memory carveout because this kernel intentionally
    // uses shared memory as its main bandwidth-amortization mechanism.
    cudaFuncSetAttribute(knn_kernel<K, TILE_POINTS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem);
    cudaFuncSetAttribute(knn_kernel<K, TILE_POINTS>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    const int blocks = (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock;

    // Default-stream launch is intentionally asynchronous, matching standard CUDA
    // semantics.  The caller can synchronize externally if needed.
    knn_kernel<K, TILE_POINTS><<<blocks, kThreadsPerBlock, shmem>>>(
        query, query_count, data, data_count, result);
}

} // namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Tile sizes are chosen to balance:
    //   - reuse of the staged data tile across 8 concurrent query warps,
    //   - shared memory consumed by the per-warp top-K + buffer state,
    //   - occupancy on A100/H100.
    //
    // The launcher queries shared-memory limits at runtime so that larger tiles
    // can be used on devices that support them (notably H100) while preserving
    // A100-friendly fallbacks.
    const DeviceSharedCaps caps = query_device_shared_caps();

    switch (k) {
        case 32:
            // 8192-point tile keeps the block around 68 KiB shared.
            launch_knn_specialized<32, 8192>(query, query_count, data, data_count, result);
            break;

        case 64:
            // 8192-point tile keeps the block around 72 KiB shared.
            launch_knn_specialized<64, 8192>(query, query_count, data, data_count, result);
            break;

        case 128:
            // 8192-point tile keeps the block around 80 KiB shared.
            launch_knn_specialized<128, 8192>(query, query_count, data, data_count, result);
            break;

        case 256:
            // A100-friendly default: 4096-point tile (64 KiB/block).
            // H100-friendly upgrade: 8192-point tile (96 KiB/block) while still
            // allowing two resident blocks per SM.
            if (supports_two_blocks_per_sm<256, 8192>(caps)) {
                launch_knn_specialized<256, 8192>(query, query_count, data, data_count, result);
            } else {
                launch_knn_specialized<256, 4096>(query, query_count, data, data_count, result);
            }
            break;

        case 512:
            // A100-friendly default: 2048-point tile (80 KiB/block).
            // H100-friendly upgrade: 4096-point tile (96 KiB/block) while still
            // allowing two resident blocks per SM.
            if (supports_two_blocks_per_sm<512, 4096>(caps)) {
                launch_knn_specialized<512, 4096>(query, query_count, data, data_count, result);
            } else {
                launch_knn_specialized<512, 2048>(query, query_count, data, data_count, result);
            }
            break;

        case 1024:
            // The K=1024 per-warp state is large enough that this path is dominated
            // by shared-memory capacity rather than thread count.  We therefore keep
            // 8 queries per block for maximal tile reuse and simply spend the extra
            // shared-memory headroom on the largest tile the device can support.
            if (supports_shared<1024, 8192>(caps)) {
                launch_knn_specialized<1024, 8192>(query, query_count, data, data_count, result);
            } else {
                launch_knn_specialized<1024, 4096>(query, query_count, data, data_count, result);
            }
            break;

        default:
            // Per problem statement, k is always a valid power of two in [32, 1024].
            break;
    }
}