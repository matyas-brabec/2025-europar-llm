#include <cuda_runtime.h>
#include <utility>
#include <cstdint>
#include <cstddef>

namespace
{
    // Fixed block shape:
    // - 256 threads = 8 warps.
    // - 1 warp handles 1 query, so each CTA processes 8 queries.
    // This is a good fit for A100/H100 because each staged data tile is reused by 8 queries
    // while still leaving enough shared memory for k up to 1024.
    constexpr int kWarpSize        = 32;
    constexpr int kBlockThreads    = 256;
    constexpr int kWarpsPerBlock   = kBlockThreads / kWarpSize;

    // A100's opt-in per-CTA shared-memory limit; the chosen tile sizes below are kept within
    // this smaller limit so the same binary also runs well on H100.
    constexpr std::size_t kA100MaxOptinSharedBytes = 163840;

    // Packed key layout:
    //   high 32 bits = IEEE-754 bits of the (non-negative) squared distance
    //   low  32 bits = data index
    //
    // Because all distances here are non-negative, unsigned integer comparison on the packed
    // key is equivalent to lexicographic comparison on (distance, index). The index is used
    // only as a deterministic tie-breaker to make the ordering total; the problem statement
    // allows arbitrary tie resolution.
    using KeyT = unsigned long long;

    constexpr unsigned int kFullMask = 0xffffffffu;
    constexpr KeyT kMaxKey =
        ~KeyT{0};
    constexpr KeyT kInvalidKey =
        (KeyT{0x7f800000u} << 32) | KeyT{0xffffffffu};  // (+inf distance, invalid index)

    template <int K, int TILE_POINTS>
    constexpr std::size_t shared_bytes_for()
    {
        return static_cast<std::size_t>(TILE_POINTS) * sizeof(float2) +
               static_cast<std::size_t>(kWarpsPerBlock) * static_cast<std::size_t>(2 * K) * sizeof(KeyT);
    }

    __device__ __forceinline__ KeyT pack_key(const float dist, const int idx)
    {
        return (static_cast<KeyT>(__float_as_uint(dist)) << 32) |
               static_cast<KeyT>(static_cast<std::uint32_t>(idx));
    }

    __device__ __forceinline__ int unpack_index(const KeyT key)
    {
        return static_cast<int>(static_cast<std::uint32_t>(key));
    }

    __device__ __forceinline__ float unpack_distance(const KeyT key)
    {
        return __uint_as_float(static_cast<unsigned int>(key >> 32));
    }

    // Warp-local bitonic sort of exactly 32 keys, ascending across lanes.
    // Each lane contributes one key and receives one key back.
    __device__ __forceinline__ KeyT warp_sort32_asc(KeyT key)
    {
        const int lane = threadIdx.x & (kWarpSize - 1);

        for (int k = 2; k <= kWarpSize; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j >>= 1)
            {
                const KeyT other = __shfl_xor_sync(kFullMask, key, j);

                // Standard bitonic compare-exchange logic for XOR partners.
                const bool keep_small = (((lane & j) == 0) == ((lane & k) == 0));
                if (keep_small)
                {
                    if (other < key) key = other;
                }
                else
                {
                    if (key < other) key = other;
                }
            }
        }

        return key;
    }

    // Merge-path partition for merging:
    //   A = current top-k list in shared memory, length K, sorted ascending
    //   B = current 32-lane candidate list, sorted ascending, but only its first b_len entries matter
    //
    // For the output diagonal 'diag', return how many items are taken from A.
    // Because b_len <= 32, the binary-search window is also <= 32 entries wide, so this is cheap.
    template <int K>
    __device__ __forceinline__ int merge_path_partition(
        const KeyT* __restrict__ a,
        const int diag,
        const KeyT sorted_b_lane_value,
        const int b_len)
    {
        int low  = diag - b_len;
        int high = diag;

        if (low  < 0) low  = 0;
        if (high > K) high = K;

        while (low < high)
        {
            const int mid = (low + high) >> 1;
            const int j   = diag - mid;

            const KeyT a_mid  = (mid < K) ? a[mid] : kMaxKey;
            const KeyT b_jm1  = (j > 0) ? __shfl_sync(kFullMask, sorted_b_lane_value, j - 1) : KeyT{0};

            // If A[mid] is still smaller than B[j-1], then the partition needs more A-items.
            if ((j > 0) && (mid < K) && (a_mid < b_jm1))
            {
                low = mid + 1;
            }
            else
            {
                high = mid;
            }
        }

        return low;
    }

    // Core kernel:
    // - one warp owns one query
    // - the CTA stages a large batch of data points in shared memory
    // - each warp scans that tile in groups of 32 candidates
    // - each warp keeps its private top-k in warp-private shared memory as a sorted array of keys
    // - only candidate groups containing at least one key < current cutoff are sorted/merged
    //
    // The top-k list is stored in two shared-memory buffers (ping-pong) to avoid in-place merge
    // hazards and to avoid local-memory spills for large k (up to 1024).
    template <int K, int TILE_POINTS>
    __global__ __launch_bounds__(256)
    void knn_kernel(
        const float2* __restrict__ query,
        const int query_count,
        const float2* __restrict__ data,
        const int data_count,
        std::pair<int, float>* __restrict__ result)
    {
        static_assert((K & (K - 1)) == 0, "K must be a power of two.");
        static_assert(K >= 32 && K <= 1024, "K must be in [32, 1024].");
        static_assert((K % kWarpSize) == 0, "K must be divisible by warp size.");
        static_assert((TILE_POINTS % kBlockThreads) == 0, "Tile size must be divisible by block size.");
        static_assert((TILE_POINTS % kWarpSize) == 0, "Tile size must be divisible by warp size.");

        constexpr int kPerLane       = K / kWarpSize;
        constexpr int kLoadsPerThread = TILE_POINTS / kBlockThreads;

        extern __shared__ unsigned char smem_raw[];
        float2* tile = reinterpret_cast<float2*>(smem_raw);

        // Shared layout after the tile:
        //   [warp0 cur K][warp0 tmp K][warp1 cur K][warp1 tmp K]...
        KeyT* warp_buffers = reinterpret_cast<KeyT*>(tile + TILE_POINTS);

        const int lane       = threadIdx.x & (kWarpSize - 1);
        const int warp_local = threadIdx.x >> 5;
        const int query_idx  = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp_local;
        const bool active    = (query_idx < query_count);

        KeyT* cur = warp_buffers + static_cast<std::size_t>(warp_local) * static_cast<std::size_t>(2 * K);
        KeyT* tmp = cur + K;

        // Initialize the current top-k buffer to +inf.
        #pragma unroll
        for (int pos = lane; pos < K; pos += kWarpSize)
        {
            cur[pos] = kInvalidKey;
        }
        __syncwarp();

        // Broadcast the query point once per warp.
        float qx = 0.0f;
        float qy = 0.0f;
        if ((lane == 0) && active)
        {
            const float2 q = query[query_idx];
            qx = q.x;
            qy = q.y;
        }
        qx = __shfl_sync(kFullMask, qx, 0);
        qy = __shfl_sync(kFullMask, qy, 0);

        // Current kth key. Starts at +inf and monotonically decreases.
        KeyT cutoff = kInvalidKey;

        for (int tile_base = 0; tile_base < data_count; tile_base += TILE_POINTS)
        {
            int tile_count = data_count - tile_base;
            if (tile_count > TILE_POINTS) tile_count = TILE_POINTS;

            // Cooperative CTA load of the current data tile into shared memory.
            #pragma unroll
            for (int load = 0; load < kLoadsPerThread; ++load)
            {
                const int local = load * kBlockThreads + threadIdx.x;
                if (local < tile_count)
                {
                    tile[local] = data[tile_base + local];
                }
            }
            __syncthreads();

            if (active)
            {
                // Process the staged tile 32 points at a time so that a warp naturally owns
                // one candidate per lane.
                #pragma unroll 1
                for (int round = 0; round < tile_count; round += kWarpSize)
                {
                    const int local = round + lane;

                    KeyT cand = kInvalidKey;
                    if (local < tile_count)
                    {
                        const float2 p = tile[local];
                        const float dx = qx - p.x;
                        const float dy = qy - p.y;
                        const float dist = fmaf(dx, dx, dy * dy);  // squared L2, no sqrt
                        cand = pack_key(dist, tile_base + local);
                    }

                    // Exact cutoff test:
                    // if cand >= cutoff, then cand cannot enter the new top-k because the current
                    // top-k already contains K keys <= cutoff under the packed-key ordering.
                    const unsigned int eligible_mask = __ballot_sync(kFullMask, cand < cutoff);
                    if (eligible_mask == 0u)
                    {
                        continue;
                    }

                    // Sort all 32 candidates. After sorting, the first m entries are exactly the
                    // keys that beat the current cutoff.
                    const int m = __popc(eligible_mask);
                    cand = warp_sort32_asc(cand);

                    // Each lane merges a contiguous slice of the output:
                    //   [lane * kPerLane, (lane + 1) * kPerLane)
                    // The candidate side is tiny (<= 32), so merge-path partitioning is cheap.
                    const int out_begin = lane * kPerLane;
                    int a_idx = merge_path_partition<K>(cur, out_begin, cand, m);
                    int b_idx = out_begin - a_idx;

                    KeyT last_out = kInvalidKey;

                    #pragma unroll
                    for (int i = 0; i < kPerLane; ++i)
                    {
                        const KeyT a_key = (a_idx < K) ? cur[a_idx] : kMaxKey;
                        const KeyT b_key = (b_idx < m) ? __shfl_sync(kFullMask, cand, b_idx) : kMaxKey;

                        const bool take_b = (b_idx < m) && ((a_idx == K) || (b_key < a_key));
                        const KeyT out_key = take_b ? b_key : a_key;

                        if (take_b)
                        {
                            ++b_idx;
                        }
                        else
                        {
                            ++a_idx;
                        }

                        tmp[out_begin + i] = out_key;
                        last_out = out_key;
                    }

                    // Shared-memory communication inside the warp requires an explicit warp barrier
                    // on modern GPUs with independent thread scheduling.
                    __syncwarp();

                    // Lane 31 owns the final output element K-1 of the merged list, i.e. the new cutoff.
                    cutoff = __shfl_sync(kFullMask, last_out, kWarpSize - 1);

                    // Swap ping-pong buffers.
                    KeyT* old = cur;
                    cur = tmp;
                    tmp = old;
                }
            }

            // All warps must finish consuming the staged tile before the next tile overwrites it.
            __syncthreads();
        }

        // Write the final sorted top-k back to global memory.
        // Results are sorted ascending by the packed key, i.e. by distance with an index tie-break.
        if (active)
        {
            std::pair<int, float>* out = result + static_cast<std::size_t>(query_idx) * static_cast<std::size_t>(K);

            #pragma unroll
            for (int pos = lane; pos < K; pos += kWarpSize)
            {
                const KeyT key = cur[pos];
                out[pos].first  = unpack_index(key);
                out[pos].second = unpack_distance(key);
            }
        }
    }

    template <int K, int TILE_POINTS>
    inline void launch_knn_impl(
        const float2* query,
        const int query_count,
        const float2* data,
        const int data_count,
        std::pair<int, float>* result)
    {
        constexpr std::size_t smem_bytes = shared_bytes_for<K, TILE_POINTS>();
        static_assert(smem_bytes <= kA100MaxOptinSharedBytes,
                      "Chosen (K, TILE_POINTS) configuration exceeds A100 shared-memory capacity.");

        const dim3 block(kBlockThreads);
        const dim3 grid((query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);

        // Opt in to large dynamic shared memory; required for the larger-k specializations.
        cudaFuncSetAttribute(
            knn_kernel<K, TILE_POINTS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        cudaFuncSetAttribute(
            knn_kernel<K, TILE_POINTS>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100);

        knn_kernel<K, TILE_POINTS><<<grid, block, smem_bytes>>>(
            query, query_count, data, data_count, result);
    }
}  // namespace

void run_knn(
    const float2* query,
    int query_count,
    const float2* data,
    int data_count,
    std::pair<int, float>* result,
    int k)
{
    // Asynchronous launch, matching normal CUDA library behavior.
    if (query_count <= 0)
    {
        return;
    }

    // Tile sizes are chosen for A100's 164 KiB shared-memory ceiling and therefore also fit H100.
    // Resulting A100 shared-memory footprints per CTA:
    //   k=  32 ->  36 KiB  (4 CTAs/SM)
    //   k=  64 ->  40 KiB  (4 CTAs/SM)
    //   k= 128 ->  48 KiB  (3 CTAs/SM)
    //   k= 256 ->  64 KiB  (2 CTAs/SM)
    //   k= 512 ->  80 KiB  (2 CTAs/SM)
    //   k=1024 -> 160 KiB  (1 CTA/SM, large tile to minimize barriers)
    switch (k)
    {
        case 32:   launch_knn_impl<32,   4096>(query, query_count, data, data_count, result); break;
        case 64:   launch_knn_impl<64,   4096>(query, query_count, data, data_count, result); break;
        case 128:  launch_knn_impl<128,  4096>(query, query_count, data, data_count, result); break;
        case 256:  launch_knn_impl<256,  4096>(query, query_count, data, data_count, result); break;
        case 512:  launch_knn_impl<512,  2048>(query, query_count, data, data_count, result); break;
        case 1024: launch_knn_impl<1024, 4096>(query, query_count, data, data_count, result); break;
        default:
            // Per the problem statement this path is unreachable.
            break;
    }
}