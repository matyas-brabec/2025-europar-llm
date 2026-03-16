#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#include <climits>
#include <cstddef>
#include <utility>

namespace {

// Targeted for modern data-center GPUs (A100/H100 class).
//
// Hyper-parameter choice:
// - One CTA owns one query. With the "no extra device memory" constraint, this is the
//   simplest exact strategy because the entire per-query top-k state stays on chip.
// - 256 threads/CTA is the sweet spot here:
//     * fully coalesced 256-point streaming tiles from the data set,
//     * good occupancy on A100/H100,
//     * and, because k <= 1024, the persistent top-k state is at most 4 pairs/thread.
//
// Core idea:
// - Maintain the current exact top-k in a *striped* register layout across the CTA:
//     thread t owns ranks t, t + 256, t + 512, ...
// - On an update, append one new candidate per thread and radix-sort the combined set
//   with CUB, again asking for striped output.
// - In striped output, the last "stripe" is exactly the largest 256 items, so discarding
//   it leaves the smallest k items already laid out correctly for the next iteration.
// - For k < 256, extra lanes are simply padded with +inf sentinels.
//
// To get a tighter initial kth-distance threshold, we bootstrap from as many points as the
// update machinery can absorb in one shot: (TOP_ITEMS + 1) * 256 points. This is exact,
// costs no extra global-memory traffic, and often removes an otherwise inevitable early update.

constexpr int kBlockThreads = 256;
using ResultPair = std::pair<int, float>;

template <int K>
struct KnnTraits {
    static_assert(K >= 32 && K <= 1024 && (K & (K - 1)) == 0,
                  "k must be a power of two in [32, 1024].");

    static constexpr int TOP_ITEMS    = (K + kBlockThreads - 1) / kBlockThreads;  // 1, 2, or 4
    static constexpr int UPDATE_ITEMS = TOP_ITEMS + 1;                             // 2, 3, or 5
    static constexpr int INIT_CAPACITY = UPDATE_ITEMS * kBlockThreads;             // 512, 768, 1280
    static constexpr bool NEED_MASK   = (K < kBlockThreads);

    // In striped layout, rank r lives in:
    //   thread = r % 256
    //   item   = r / 256
    static constexpr int KTH_THREAD = (K - 1) % kBlockThreads;
    static constexpr int KTH_ITEM   = (K - 1) / kBlockThreads;

    static_assert(TOP_ITEMS >= 1 && TOP_ITEMS <= 4, "Unexpected TOP_ITEMS.");
    static_assert(UPDATE_ITEMS >= 2 && UPDATE_ITEMS <= 5, "Unexpected UPDATE_ITEMS.");
    static_assert(KTH_ITEM < TOP_ITEMS, "Invalid kth element location.");

    using UpdateSort = cub::BlockRadixSort<float, kBlockThreads, UPDATE_ITEMS, int>;
    using TempStorage = typename UpdateSort::TempStorage;
};

__device__ __forceinline__ float squared_l2(const float qx, const float qy, const float2 p) {
    // Squared Euclidean distance. No sqrt is taken because the interface asks for the
    // squared L2 norm directly.
    const float dx = qx - p.x;
    const float dy = qy - p.y;
    return fmaf(dx, dx, dy * dy);
}

template <int K>
static __global__ __launch_bounds__(256)
void knn_kernel(const float2* __restrict__ query,
                const float2* __restrict__ data,
                int data_count,
                ResultPair* __restrict__ result) {
    using Traits = KnnTraits<K>;
    using UpdateSort = typename Traits::UpdateSort;

    constexpr int TOP_ITEMS     = Traits::TOP_ITEMS;
    constexpr int UPDATE_ITEMS  = Traits::UPDATE_ITEMS;
    constexpr int INIT_CAPACITY = Traits::INIT_CAPACITY;
    constexpr bool NEED_MASK    = Traits::NEED_MASK;
    constexpr int KTH_THREAD    = Traits::KTH_THREAD;
    constexpr int KTH_ITEM      = Traits::KTH_ITEM;

    __shared__ typename Traits::TempStorage sort_storage;
    __shared__ float shared_tau;

    const int tid = static_cast<int>(threadIdx.x);
    const int qid = static_cast<int>(blockIdx.x);

    // The query is tiny and reused for the full scan; keeping it in registers is ideal.
    const float2 q = query[qid];
    const float qx = q.x;
    const float qy = q.y;

    const float inf = CUDART_INF_F;
    const int invalid_index = INT_MAX;

    // Persistent exact top-k state for this thread, stored in striped layout.
    float top_keys[TOP_ITEMS];
    int   top_vals[TOP_ITEMS];

    // -------------------------------------------------------------------------
    // Bootstrap:
    // Sort the largest batch that the regular update path can consume in one shot.
    // This gives a much tighter initial threshold than starting from only the first k
    // elements and uses the same per-thread state size as every later update.
    // -------------------------------------------------------------------------
    const int init_points = (data_count < INIT_CAPACITY) ? data_count : INIT_CAPACITY;

    {
        float init_keys[UPDATE_ITEMS];
        int   init_vals[UPDATE_ITEMS];

        #pragma unroll
        for (int i = 0; i < UPDATE_ITEMS; ++i) {
            const int data_idx = tid + i * kBlockThreads;
            if (data_idx < init_points) {
                const float2 p = data[data_idx];
                init_keys[i] = squared_l2(qx, qy, p);
                init_vals[i] = data_idx;
            } else {
                init_keys[i] = inf;
                init_vals[i] = invalid_index;
            }
        }

        // Global sort of the bootstrap batch, with striped output.
        UpdateSort(sort_storage).SortBlockedToStriped(init_keys, init_vals);

        // Keep only the smallest k entries; for k < 256, only the first k lanes of the
        // first stripe are valid and the rest are masked back to +inf.
        #pragma unroll
        for (int i = 0; i < TOP_ITEMS; ++i) {
            if constexpr (NEED_MASK) {
                const int striped_rank = tid + i * kBlockThreads;
                if (striped_rank < K) {
                    top_keys[i] = init_keys[i];
                    top_vals[i] = init_vals[i];
                } else {
                    top_keys[i] = inf;
                    top_vals[i] = invalid_index;
                }
            } else {
                top_keys[i] = init_keys[i];
                top_vals[i] = init_vals[i];
            }
        }

        if (tid == KTH_THREAD) {
            shared_tau = top_keys[KTH_ITEM];
        }

        // This barrier serves both purposes:
        // 1) makes shared_tau visible to the whole CTA;
        // 2) satisfies CUB's requirement before reusing TempStorage.
        __syncthreads();
    }

    float tau = shared_tau;

    // -------------------------------------------------------------------------
    // Stream the rest of the data in 256-point tiles.
    // We only invoke the expensive block-wide sort if at least one point in the tile
    // beats the current kth distance. Using strict '< tau' is intentional:
    // - ties may be resolved arbitrarily per the problem statement,
    // - and avoiding '<=' reduces needless update sorts on boundary ties.
    // -------------------------------------------------------------------------
    for (int base = init_points; base < data_count; base += kBlockThreads) {
        const int data_idx = base + tid;

        float cand_key = inf;
        int   cand_val = invalid_index;
        bool  candidate = false;

        if (data_idx < data_count) {
            const float2 p = data[data_idx];
            const float dist = squared_l2(qx, qy, p);
            candidate = (dist < tau);
            if (candidate) {
                cand_key = dist;
                cand_val = data_idx;
            }
        }

        // Block-uniform branch: every thread gets the same result.
        const int any_update = __syncthreads_or(candidate);

        if (any_update) {
            float work_keys[UPDATE_ITEMS];
            int   work_vals[UPDATE_ITEMS];

            // Existing top-k + this thread's new candidate.
            #pragma unroll
            for (int i = 0; i < TOP_ITEMS; ++i) {
                work_keys[i] = top_keys[i];
                work_vals[i] = top_vals[i];
            }
            work_keys[TOP_ITEMS] = cand_key;
            work_vals[TOP_ITEMS] = cand_val;

            // Sort K + 256 logical items (plus +inf padding when k < 256).
            // In striped output, dropping the last stripe is exactly equivalent to
            // removing the 256 largest elements.
            UpdateSort(sort_storage).SortBlockedToStriped(work_keys, work_vals);

            #pragma unroll
            for (int i = 0; i < TOP_ITEMS; ++i) {
                if constexpr (NEED_MASK) {
                    const int striped_rank = tid + i * kBlockThreads;
                    if (striped_rank < K) {
                        top_keys[i] = work_keys[i];
                        top_vals[i] = work_vals[i];
                    } else {
                        top_keys[i] = inf;
                        top_vals[i] = invalid_index;
                    }
                } else {
                    top_keys[i] = work_keys[i];
                    top_vals[i] = work_vals[i];
                }
            }

            if (tid == KTH_THREAD) {
                shared_tau = top_keys[KTH_ITEM];
            }

            // Visible tau update + legal CUB TempStorage reuse barrier.
            __syncthreads();
            tau = shared_tau;
        }
    }

    // -------------------------------------------------------------------------
    // Write back the final exact top-k for this query.
    // The internal representation is already globally sorted in striped order:
    //   rank = tid + item * 256
    // -------------------------------------------------------------------------
    ResultPair* const out =
        result + static_cast<size_t>(qid) * static_cast<size_t>(K);

    #pragma unroll
    for (int i = 0; i < TOP_ITEMS; ++i) {
        const int rank = tid + i * kBlockThreads;

        if constexpr (NEED_MASK) {
            if (rank < K) {
                out[rank].first  = top_vals[i];
                out[rank].second = top_keys[i];
            }
        } else {
            out[rank].first  = top_vals[i];
            out[rank].second = top_keys[i];
        }
    }
}

template <int K>
inline void launch_knn_specialized(const float2* query,
                                   int query_count,
                                   const float2* data,
                                   int data_count,
                                   ResultPair* result) {
    const dim3 grid(static_cast<unsigned int>(query_count));
    const dim3 block(kBlockThreads);

    // Intentionally asynchronous: enqueue on the current default stream.
    knn_kernel<K><<<grid, block>>>(query, data, data_count, result);
}

}  // namespace

void run_knn(const float2* query,
             int query_count,
             const float2* data,
             int data_count,
             std::pair<int, float>* result,
             int k) {
    if (query_count <= 0) {
        return;
    }

    // Runtime dispatch over the only legal k values. This keeps all of the important
    // quantities (state size, loop trip counts, radix-sort arity, kth register location)
    // compile-time constants in the actual kernel.
    switch (k) {
        case 32:   launch_knn_specialized<32>(query, query_count, data, data_count, result);   return;
        case 64:   launch_knn_specialized<64>(query, query_count, data, data_count, result);   return;
        case 128:  launch_knn_specialized<128>(query, query_count, data, data_count, result);  return;
        case 256:  launch_knn_specialized<256>(query, query_count, data, data_count, result);  return;
        case 512:  launch_knn_specialized<512>(query, query_count, data, data_count, result);  return;
        case 1024: launch_knn_specialized<1024>(query, query_count, data, data_count, result); return;
        default:
            // Input validity guarantees this path is unreachable.
            return;
    }
}