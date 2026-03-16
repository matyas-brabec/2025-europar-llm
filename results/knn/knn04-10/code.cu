#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace {

// Warp-centric exact k-NN for 2D points.
// Design choices tuned for modern datacenter GPUs (A100/H100):
// - One warp computes one query.
// - A 256-thread block (8 warps) cooperatively stages 256 data points into shared memory.
//   This is the largest practical block size that still leaves enough shared memory for the
//   per-warp top-k state when K == 1024.
// - Each warp keeps its private top-k as a sorted array of 64-bit keys in shared memory.
//   Key layout: [high 32 bits = squared-distance float bits][low 32 bits = data index].
//   Since squared distances are non-negative, unsigned integer ordering of the float bit pattern
//   matches the numeric ordering of the distance; the low 32 bits then break ties by index.
// - For each tile, only candidates strictly better than the current threshold are compacted.
//   Survivors are sorted in place and merged with the current top-k via a warp-parallel
//   merge-path partition. This updates K/32 output positions per lane at once.
// - No auxiliary device memory is allocated; all temporary state lives in dynamic shared memory.

constexpr int kWarpSize      = 32;
constexpr int kBlockThreads  = 256;
constexpr int kWarpsPerBlock = kBlockThreads / kWarpSize;
constexpr int kTilePoints    = kBlockThreads;
constexpr int kTileIters     = kTilePoints / kWarpSize;
constexpr unsigned kFullMask = 0xffffffffu;

// Sentinel key larger than any valid finite-distance key and also larger than any real +inf key,
// because the low 32 bits are set to the maximal index value.
constexpr uint64_t kInfKey = 0x7f800000ffffffffULL;

// The public API uses std::pair<int, float>. Device code writes through a trivial POD view with
// the same expected 8-byte {int, float} payload. This avoids depending on device annotations
// in the host STL implementation while keeping the externally visible interface unchanged.
struct ResultPair {
    int   first;
    float second;
};

static_assert(sizeof(float2) == sizeof(uint64_t), "float2 is expected to be 8 bytes.");
static_assert(sizeof(ResultPair) == 8, "ResultPair is expected to be 8 bytes.");
static_assert(offsetof(ResultPair, second) == sizeof(int), "Unexpected ResultPair layout.");
static_assert(std::is_trivially_copyable<ResultPair>::value, "ResultPair must be trivially copyable.");
static_assert(std::is_standard_layout<ResultPair>::value, "ResultPair must be standard layout.");
static_assert(sizeof(std::pair<int, float>) == sizeof(ResultPair),
              "Unexpected std::pair<int, float> size; this implementation expects 8 bytes.");

// Dynamic shared-memory footprint:
//   tile:      256 x float2
//   per warp:  top[K] + tmp[K] + cand[256], all as uint64_t keys
template <int K>
constexpr std::size_t shared_mem_bytes() {
    static_assert(K >= 32 && K <= 1024 && ((K & (K - 1)) == 0),
                  "K must be a power of two in [32, 1024].");
    static_assert((K % kWarpSize) == 0, "K must be a multiple of warp size.");
    return sizeof(float2) * kTilePoints +
           sizeof(uint64_t) * (2 * kWarpsPerBlock * K + kWarpsPerBlock * kTilePoints);
}

// A100 supports 163840 B/block of dynamic shared memory; H100 supports more.
static_assert(shared_mem_bytes<1024>() <= 163840,
              "The chosen block/tile configuration must fit the A100 shared-memory budget.");

__device__ __forceinline__ uint64_t pack_key_bits(uint32_t dist_bits, uint32_t idx) {
    return (static_cast<uint64_t>(dist_bits) << 32) | static_cast<uint64_t>(idx);
}

__device__ __forceinline__ float unpack_dist(uint64_t key) {
    return __uint_as_float(static_cast<uint32_t>(key >> 32));
}

__device__ __forceinline__ int unpack_index(uint64_t key) {
    return static_cast<int>(static_cast<uint32_t>(key));
}

// In-warp bitonic sort of N keys stored in warp-private shared memory.
// N is always a power of two and at most 1024.
template <int N>
__device__ __forceinline__ void bitonic_sort_shared(uint64_t* keys) {
    static_assert((N & (N - 1)) == 0, "Bitonic sort size must be a power of two.");

    const int lane = threadIdx.x & (kWarpSize - 1);

    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int i = lane; i < N; i += kWarpSize) {
                const int j = i ^ stride;
                if (j > i) {
                    const uint64_t a = keys[i];
                    const uint64_t b = keys[j];
                    const bool ascending = ((i & size) == 0);
                    const bool do_swap  = ascending ? (a > b) : (a < b);

                    if (do_swap) {
                        keys[i] = b;
                        keys[j] = a;
                    }
                }
            }
            __syncwarp();
        }
    }
}

__device__ __forceinline__ void sort_candidates_shared(uint64_t* cand, int count_pow2) {
    switch (count_pow2) {
        case 32:  bitonic_sort_shared<32>(cand);  break;
        case 64:  bitonic_sort_shared<64>(cand);  break;
        case 128: bitonic_sort_shared<128>(cand); break;
        default:  bitonic_sort_shared<256>(cand); break;
    }
}

// Merge-path partition for two sorted arrays A and B.
// Returns the number of items taken from A in the first "diag" outputs of A merge B.
__device__ __forceinline__ int merge_path_partition(
    const uint64_t* A, int A_count,
    const uint64_t* B, int B_count,
    int diag) {

    int low = diag - B_count;
    if (low < 0) low = 0;

    int high = (diag < A_count) ? diag : A_count;

    while (low <= high) {
        const int a = (low + high) >> 1;
        const int b = diag - a;

        const bool move_left  = (a > 0)        && (b < B_count) && (B[b]     < A[a - 1]);
        const bool move_right = (b > 0)        && (a < A_count) && (A[a]     < B[b - 1]);

        if (move_left) {
            high = a - 1;
        } else if (move_right) {
            low = a + 1;
        } else {
            return a;
        }
    }

    return low;
}

// Warp-parallel merge of the current sorted top-k and a sorted candidate list.
// Each lane writes K/32 consecutive outputs.
template <int K>
__device__ __forceinline__ void merge_topk_shared(
    const uint64_t* top,
    const uint64_t* cand,
    int cand_count,
    uint64_t* out) {

    constexpr int kItemsPerLane = K / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int out_begin = lane * kItemsPerLane;

    int a = merge_path_partition(top, K, cand, cand_count, out_begin);
    int b = out_begin - a;

#pragma unroll
    for (int n = 0; n < kItemsPerLane; ++n) {
        const uint64_t a_key = (a < K)          ? top[a]  : kInfKey;
        const uint64_t b_key = (b < cand_count) ? cand[b] : kInfKey;
        const bool take_a = (a_key <= b_key);

        out[out_begin + n] = take_a ? a_key : b_key;
        a += static_cast<int>(take_a);
        b += static_cast<int>(!take_a);
    }
}

// Process a subset [begin_p, valid_count) of the current shared-memory tile for one warp/query.
// The loop always executes a fixed 8 iterations (one per 32-point subgroup in a 256-point tile),
// which keeps all lanes converged at each ballot even for partial tiles and shifted begin_p.
template <int K>
__device__ __forceinline__ void process_tile_candidates(
    const float2* tile,
    int tile_base_index,
    int begin_p,
    int valid_count,
    float qx,
    float qy,
    uint64_t*& top,
    uint64_t*& tmp,
    uint64_t* cand,
    uint32_t& threshold_bits,
    uint32_t& threshold_idx) {

    const int lane = threadIdx.x & (kWarpSize - 1);
    const unsigned lane_mask_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    int cand_count = 0;

#pragma unroll
    for (int g = 0; g < kTileIters; ++g) {
        const int p = begin_p + g * kWarpSize + lane;

        bool pass = false;
        uint64_t key = 0;

        if (p < valid_count) {
            const float2 pt = tile[p];
            const float dx = qx - pt.x;
            const float dy = qy - pt.y;
            const float dist = fmaf(dx, dx, dy * dy);

            const uint32_t dist_bits = __float_as_uint(dist);
            const uint32_t idx = static_cast<uint32_t>(tile_base_index + p);

            pass = (dist_bits < threshold_bits) ||
                   ((dist_bits == threshold_bits) && (idx < threshold_idx));

            if (pass) {
                key = pack_key_bits(dist_bits, idx);
            }
        }

        const unsigned mask = __ballot_sync(kFullMask, pass);
        if (pass) {
            const int local_rank = __popc(mask & lane_mask_lt);
            cand[cand_count + local_rank] = key;
        }
        cand_count += __popc(mask);
    }

    __syncwarp();

    if (cand_count == 0) {
        return;
    }

    int merge_count = cand_count;

    if (cand_count > 1) {
        int cand_n = 32;
        if (cand_count > 32) {
            cand_n = (cand_count <= 64) ? 64 : ((cand_count <= 128) ? 128 : 256);
        }

        for (int i = lane + cand_count; i < cand_n; i += kWarpSize) {
            cand[i] = kInfKey;
        }
        __syncwarp();

        sort_candidates_shared(cand, cand_n);
        merge_count = cand_n;
    }

    merge_topk_shared<K>(top, cand, merge_count, tmp);
    __syncwarp();

    uint64_t* old_top = top;
    top = tmp;
    tmp = old_top;

    const uint64_t threshold_key = top[K - 1];
    threshold_bits = static_cast<uint32_t>(threshold_key >> 32);
    threshold_idx  = static_cast<uint32_t>(threshold_key);
}

template <int K>
__global__ __launch_bounds__(kBlockThreads, 1)
void knn_kernel(
    const float2* __restrict__ query,
    int query_count,
    const float2* __restrict__ data,
    int data_count,
    ResultPair* __restrict__ result) {

    constexpr int kMainBase = (K < kTilePoints) ? kTilePoints : K;

    extern __shared__ __align__(8) uint64_t smem_u64[];

    // Shared-memory layout:
    //   [0 .. tile-1]              : float2 tile[256]
    //   [tile .. tile+warps*K-1]   : top keys
    //   [.. next warps*K ..]       : tmp keys
    //   [.. next warps*256 ..]     : candidate keys
    float2*  sh_tile   = reinterpret_cast<float2*>(smem_u64);
    uint64_t* top_base = reinterpret_cast<uint64_t*>(sh_tile + kTilePoints);
    uint64_t* tmp_base = top_base + kWarpsPerBlock * K;
    uint64_t* cand_base = tmp_base + kWarpsPerBlock * K;

    const int lane    = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x >> 5;
    const int qid     = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_id;
    const bool active = (qid < query_count);

    float2 q;
    q.x = 0.0f;
    q.y = 0.0f;
    if (lane == 0 && active) {
        q = query[qid];
    }

    const float qx = __shfl_sync(kFullMask, q.x, 0);
    const float qy = __shfl_sync(kFullMask, q.y, 0);

    uint64_t* top  = top_base  + warp_id * K;
    uint64_t* tmp  = tmp_base  + warp_id * K;
    uint64_t* cand = cand_base + warp_id * kTilePoints;

    uint32_t threshold_bits = 0x7f800000u;
    uint32_t threshold_idx  = 0xffffffffu;

    // Initialization:
    // - For K < 256, consume the entire first 256-point tile so the threshold tightens early.
    // - For K >= 256, fill top[] with the first K points (which is tile-aligned for the
    //   supported K values 256/512/1024), then sort once.
    if constexpr (K < kTilePoints) {
        const int global = threadIdx.x;
        if (global < data_count) {
            sh_tile[threadIdx.x] = data[global];
        }
        __syncthreads();

        if (active) {
#pragma unroll
            for (int p = lane; p < K; p += kWarpSize) {
                const float2 pt = sh_tile[p];
                const float dx = qx - pt.x;
                const float dy = qy - pt.y;
                const float dist = fmaf(dx, dx, dy * dy);
                top[p] = pack_key_bits(__float_as_uint(dist), static_cast<uint32_t>(p));
            }

            __syncwarp();
            bitonic_sort_shared<K>(top);
            __syncwarp();

            const uint64_t threshold_key = top[K - 1];
            threshold_bits = static_cast<uint32_t>(threshold_key >> 32);
            threshold_idx  = static_cast<uint32_t>(threshold_key);

            const int first_tile_valid = (data_count < kTilePoints) ? data_count : kTilePoints;
            process_tile_candidates<K>(
                sh_tile, 0, K, first_tile_valid, qx, qy,
                top, tmp, cand, threshold_bits, threshold_idx);
        }

        __syncthreads();
    } else {
        for (int base = 0; base < K; base += kTilePoints) {
            // For the supported K >= 256 cases, this initialization region is always full.
            sh_tile[threadIdx.x] = data[base + threadIdx.x];
            __syncthreads();

            if (active) {
#pragma unroll
                for (int t = 0; t < kTileIters; ++t) {
                    const int p = t * kWarpSize + lane;
                    const float2 pt = sh_tile[p];
                    const float dx = qx - pt.x;
                    const float dy = qy - pt.y;
                    const float dist = fmaf(dx, dx, dy * dy);
                    top[base + p] = pack_key_bits(__float_as_uint(dist),
                                                  static_cast<uint32_t>(base + p));
                }
            }

            __syncthreads();
        }

        if (active) {
            bitonic_sort_shared<K>(top);
            __syncwarp();

            const uint64_t threshold_key = top[K - 1];
            threshold_bits = static_cast<uint32_t>(threshold_key >> 32);
            threshold_idx  = static_cast<uint32_t>(threshold_key);
        }

        __syncthreads();
    }

    // Main streaming pass over the remaining data, tile by tile.
    for (int base = kMainBase; base < data_count; base += kTilePoints) {
        const int global = base + threadIdx.x;
        if (global < data_count) {
            sh_tile[threadIdx.x] = data[global];
        }
        __syncthreads();

        if (active) {
            int valid = data_count - base;
            if (valid > kTilePoints) {
                valid = kTilePoints;
            }

            process_tile_candidates<K>(
                sh_tile, base, 0, valid, qx, qy,
                top, tmp, cand, threshold_bits, threshold_idx);
        }

        __syncthreads();
    }

    // The final top-k is already globally sorted by ascending (distance, index).
    if (active) {
        ResultPair* out = result + static_cast<std::size_t>(qid) * static_cast<std::size_t>(K);

#pragma unroll
        for (int j = lane; j < K; j += kWarpSize) {
            const uint64_t key = top[j];
            out[j].first  = unpack_index(key);
            out[j].second = unpack_dist(key);
        }
    }
}

template <int K>
inline void launch_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    ResultPair* result) {

    const std::size_t smem = shared_mem_bytes<K>();

    // The public interface is void, so launch/configuration errors are intentionally not
    // propagated here; callers can use standard CUDA error handling around run_knn().
    static_cast<void>(cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    static_cast<void>(cudaFuncSetAttribute(
        knn_kernel<K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    const dim3 block(kBlockThreads);
    const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

    knn_kernel<K><<<grid, block, smem>>>(
        query, query_count, data, data_count, result);
}

}  // namespace

void run_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k) {

    if (query_count <= 0) {
        return;
    }

    auto* result_view = reinterpret_cast<ResultPair*>(result);

    switch (k) {
        case 32:   launch_knn<32>(query, query_count, data, data_count, result_view);   break;
        case 64:   launch_knn<64>(query, query_count, data, data_count, result_view);   break;
        case 128:  launch_knn<128>(query, query_count, data, data_count, result_view);  break;
        case 256:  launch_knn<256>(query, query_count, data, data_count, result_view);  break;
        case 512:  launch_knn<512>(query, query_count, data, data_count, result_view);  break;
        case 1024: launch_knn<1024>(query, query_count, data, data_count, result_view); break;
        default:   break;  // Inputs are specified to be valid; keep a safe fallback.
    }
}