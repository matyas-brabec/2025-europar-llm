#include <cassert>

#include <utility>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define BLOCK_SIZE 128
#define MAX_K 1024
#define WARP_SIZE 32
#define IPT 7

namespace {


template <size_t K, std::size_t K_PER_LANE, size_t OUTER, size_t INNER>
__device__ void bitonic_sort_step_local(int (&indices)[K_PER_LANE], float (&distances)[K_PER_LANE]) {
    // K: size of the array
    // OUTER: size of sorted sequences (output)
    // INNER: size of the stride

    static_assert(K_PER_LANE > INNER);
    static_assert(K_PER_LANE % INNER == 0);

    static_assert(OUTER > INNER);
    static_assert(OUTER % INNER == 0);

#pragma unroll
    for (int i = 0; i < K_PER_LANE / 2; ++i) {
        const auto low_bits = i & (INNER - 1);
        const auto high_bits = i & ~(INNER - 1);

        const auto left_idx = low_bits | (high_bits << 1);
        const auto right_idx = left_idx ^ INNER;

        assert(left_idx < right_idx);

        bool ascending;
        if constexpr (OUTER >= K_PER_LANE) {
            const auto block = cooperative_groups::this_thread_block();
            const auto warp = cooperative_groups::tiled_partition<WARP_SIZE>(block);
            const auto laneId = warp.thread_rank();

            ascending = (laneId & (OUTER / K_PER_LANE)) == 0;
        } else {
            ascending = (left_idx & OUTER) == 0;
        }
        const auto right_greater = distances[right_idx] > distances[left_idx];
        const auto swap = ascending ^ right_greater;
        if (swap) {
            {
                auto tmp = distances[left_idx];
                distances[left_idx] = distances[right_idx];
                distances[right_idx] = tmp;
            }
            {
                auto tmp = indices[left_idx];
                indices[left_idx] = indices[right_idx];
                indices[right_idx] = tmp;
            }
        }
    }

    if constexpr (INNER / 2 > 0) {
        bitonic_sort_step_local<K, K_PER_LANE, OUTER, INNER / 2>(indices, distances);
    }
}

template <size_t OUTER, size_t INNER, size_t K_PER_LANE>
__device__ void bitonic_sort_step_shuffle(int (&indices)[K_PER_LANE], float (&distances)[K_PER_LANE]) {
    // K: size of the array
    // OUTER: size of sorted sequences (output)
    // INNER: size of the stride

    static_assert(OUTER > INNER);
    static_assert(OUTER % INNER == 0);

    const auto block = cooperative_groups::this_thread_block();
    const auto warp = cooperative_groups::tiled_partition<WARP_SIZE>(block);

    const auto laneId = warp.thread_rank();

    const auto ascending = ((laneId & (OUTER)) == 0);

#pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        for (int stride = INNER; stride > 0; stride >>= 1) {
            const auto is_right = (laneId & stride) != 0;
            const auto other_dist = warp.shfl_xor(distances[i], stride);
            const auto other_index = warp.shfl_xor(indices[i], stride);
            const auto other_greater = other_dist > distances[i];
            const auto swap = ascending ^ other_greater ^ is_right;
            if (swap) {
                distances[i] = other_dist;
                indices[i] = other_index;
            }
        }
    }
}

template <size_t K, size_t K_PER_LANE, size_t MIN = 1, size_t MAX = K>
__device__ void bitonic_sort(int (&indices)[K_PER_LANE], float (&distances)[K_PER_LANE]) {
    // K: size of the array
    // MIN: size of sorted sequences (input)
    // MAX: size of sorted sequences (output)

    if constexpr (MIN < MAX) {
        if constexpr (MIN >= K_PER_LANE) {
            bitonic_sort_step_shuffle<MIN * 2 / K_PER_LANE, MIN / K_PER_LANE>(indices, distances);
            if constexpr (K_PER_LANE > 1) {
                bitonic_sort_step_local<K, K_PER_LANE, MIN * 2, K_PER_LANE / 2>(indices, distances);
            }
        } else {
            bitonic_sort_step_local<K, K_PER_LANE, MIN * 2, MIN>(indices, distances);
        }

        bitonic_sort<K, K_PER_LANE, MIN * 2, MAX>(indices, distances);
    }
}

template <size_t K, std::size_t K_PER_LANE, class T>
__device__ void broadcast(T (&values)[K_PER_LANE], T &value) {
    const auto block = cooperative_groups::this_thread_block();
    const auto warp = cooperative_groups::tiled_partition<WARP_SIZE>(block);

    value = warp.shfl(values[K_PER_LANE - 1], WARP_SIZE - 1);
}

template <size_t K>
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *results, int k) {
    namespace cg = cooperative_groups;

    constexpr auto warp_count = BLOCK_SIZE / WARP_SIZE;
    constexpr auto K_PER_LANE = K / WARP_SIZE;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);
    const auto tid = block.thread_rank();

    // Each warp processes a single query.
    const auto warpIdx = grid.thread_rank() / WARP_SIZE;
    const auto warpId = block.thread_rank() / WARP_SIZE;
    const auto laneId = block.thread_rank() % WARP_SIZE;

    const auto queryIdx = warpIdx;

    assert(k <= K);
    assert(K <= MAX_K);
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(K % WARP_SIZE == 0);

    assert(block.size() == BLOCK_SIZE);

    if (queryIdx >= query_count) return;

    // Read the query point
    float2 q = query[queryIdx];

    // Allocate space in registers for the top-k intermediate result.
    float distances[K_PER_LANE];
    int indices[K_PER_LANE];

    // Initialize with max values
#pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        distances[i] = std::numeric_limits<float>::max();
        indices[i] = -1;
    }

    // Allocate space in shared memory for the new batch of candidates.
    __shared__ float2 batch_data[IPT * BLOCK_SIZE];

    // Allocate shared buffer for candidates
    __shared__ float distances_buffer[K * warp_count];
    __shared__ int indices_buffer[K * warp_count];
    __shared__ int buffer_size[warp_count];

    // Initialize the shared buffer
    if (laneId == 0) {
        buffer_size[warpId] = 0;
    }

    float radius = std::numeric_limits<float>::max();

    int offset = 0;
    for (int i = tid; i < data_count + tid; i += IPT * BLOCK_SIZE, offset += IPT * BLOCK_SIZE) {
        // Read a batch of candidates
        for (int j = 0; j < IPT; ++j) {
            const float2 d = (i + j * BLOCK_SIZE < data_count) ? data[i + j * BLOCK_SIZE] : make_float2(0, 0);
            batch_data[tid + j * BLOCK_SIZE] = d;
        }

        block.sync();

        // Each warp processes the batch of candidates
        for (int j = laneId; j < IPT * BLOCK_SIZE; j += WARP_SIZE) {
            const float2 d = batch_data[j];
            const float dx = q.x - d.x;
            const float dy = q.y - d.y;
            const float sq_dist = offset + j < data_count ? dx * dx + dy * dy : std::numeric_limits<float>::max();

            // Allocate a slot in the shared buffer
            const int idx = (sq_dist < radius) ? atomicAdd(&buffer_size[warpId], 1) : -1;

            // Write the candidate to the shared buffer
            if (idx >= 0 && idx < (int)K) {
                indices_buffer[warpId * K + idx] = offset + j;
                distances_buffer[warpId * K + idx] = sq_dist;
            }

            warp.sync();
            if (buffer_size[warpId] >= K || offset + j + WARP_SIZE >= data_count) {
                // Move the shared buffer to registers
#pragma unroll
                for (int l = 0; l < K_PER_LANE; ++l) {
                    const auto idx = warpId * K + laneId * K_PER_LANE + l;
                    {
                        auto tmp = distances[l];
                        distances[l] = distances_buffer[idx];
                        distances_buffer[idx] = tmp;
                    }
                    {
                        auto tmp = indices[l];
                        indices[l] = indices_buffer[idx];
                        indices_buffer[idx] = tmp;
                    }
                }

                // Sort the buffer
                bitonic_sort<K>(indices, distances);

                if (laneId == 0) {
                    buffer_size[warpId] -= K;
                }

                // Merge the buffer with the current candidates
#pragma unroll
                for (int l = 0; l < K_PER_LANE; ++l) {
                    const auto other_id = warpId * K + K - (laneId * K_PER_LANE + l + 1);
                    const auto other_dist = distances_buffer[other_id];

                    if (other_dist < distances[l]) {
                        distances[l] = other_dist;
                        indices[l] = indices_buffer[other_id];
                    }

                    distances_buffer[other_id] = std::numeric_limits<float>::max();
                    indices_buffer[other_id] = -1;
                }

                bitonic_sort<K>(indices, distances);
                broadcast<K>(distances, radius);

                // Write the rest of the candidates to the shared buffer
                if (idx >= (int)K) {
                    indices_buffer[warpId * K + idx - K] = offset + j;
                    distances_buffer[warpId * K + idx - K] = sq_dist;
                }
            }
        }

        block.sync();
    }

    // Write the result to global memory
#pragma unroll
    for (int i = 0; i < K_PER_LANE; ++i) {
        results[queryIdx * k + laneId * K_PER_LANE + i] = {indices[i], distances[i]};
    }
}

} // namespace

void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    const auto blocks = (query_count * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (k <= 32) {
        knn_kernel<32><<<blocks, BLOCK_SIZE>>>(query, query_count, data, data_count, result, k);
    } else if (k <= 64) {
        knn_kernel<64><<<blocks, BLOCK_SIZE>>>(query, query_count, data, data_count, result, k);
    } else if (k <= 128) {
        knn_kernel<128><<<blocks, BLOCK_SIZE>>>(query, query_count, data, data_count, result, k);
    } else if (k <= 256) {
        knn_kernel<256><<<blocks, BLOCK_SIZE>>>(query, query_count, data, data_count, result, k);
    } else if (k <= 512) {
        knn_kernel<512><<<blocks, BLOCK_SIZE>>>(query, query_count, data, data_count, result, k);
    } else if (k <= 1024) {
        knn_kernel<1024><<<blocks, BLOCK_SIZE>>>(query, query_count, data, data_count, result, k);
    }
}
