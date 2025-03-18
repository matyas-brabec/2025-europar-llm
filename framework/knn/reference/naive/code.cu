#include <cassert>
#include <cfloat>

#include <utility>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define BLOCK_SIZE 256
#define MAX_K 1024

namespace {

template <size_t k>
__device__ void heap_bubble(int (&indices)[k], float (&distances)[k]) {
    // We want to ensure the heap property:
    // 1. distances[i] >= distances[2 * i + 1]
    // 2. distances[i] >= distances[2 * i + 2]

    size_t i = 0;
    while (i < k) {
        const auto left = 2 * i + 1;
        const auto right = 2 * i + 2;

        if (right >= k) {
            // Only one child; check if the heap property is violated.
            if (distances[i] < distances[2 * i + 1]) {
                std::swap(distances[i], distances[2 * i + 1]);
                std::swap(indices[i], indices[2 * i + 1]);
            }
        } else {
            // Determine which child is larger.
            if (distances[left] > distances[right]) {
                // Check if the heap property is violated.
                if (distances[i] < distances[left]) {
                    std::swap(distances[i], distances[left]);
                    std::swap(indices[i], indices[left]);
                    i = left;
                    continue;
                }
            } else {
                // Check if the heap property is violated.
                if (distances[i] < distances[right]) {
                    std::swap(distances[i], distances[right]);
                    std::swap(indices[i], indices[right]);
                    i = right;
                    continue;
                }
            }
        }
        break;
    }
}

template <size_t k>
__device__ void heap_insert(int (&indices)[k], float (&distances)[k], int idx, float dist) {
    // Insert a new candidate (dist, idx) into the local k-array, maintaining
    // ascending order by distance. The largest distance is always at list[0].

    // If the candidate is not better (i.e. is larger or equal distance)
    // than the worst (largest) distance in the list, skip it.
    if (dist >= distances[0]) {
        return;
    }

    // Place the new candidate at the worst position.
    distances[0] = dist;
    indices[0] = idx;

    // Bubble the new value up until the order is restored.
    heap_bubble<k>(indices, distances);
}

template <size_t K>
__global__ void knn_kernel(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Each thread processes a single query.

    assert(k <= K);
    assert(K <= MAX_K);

    namespace cg = cooperative_groups;

    const auto grid = cg::this_grid();
    const auto queryIdx = grid.thread_rank();

    if (queryIdx >= query_count) return;

    // Read the query point (two floats).
    float2 q = query[queryIdx];

    // Allocate space in registers for the top-k heap.
    int indices[K];
    float distances[K];

    // Initialize distances to "infinity".
#pragma unroll
    for (int i = 0; i < k; ++i) {
        indices[i] = -1;
        distances[i] = FLT_MAX;
    }

    // Each thread iterates over data points.
    for (int i = 0; i < data_count; ++i) {
        float2 d = data[i];
        float dx = q.x - d.x;
        float dy = q.y - d.y;
        float sq_dist = dx * dx + dy * dy;
        heap_insert<K>(indices, distances, i, sq_dist);
    }

    // Write the result to global memory.
    for (int i = 0; i < k; ++i) {
        result[queryIdx * k + i] = {indices[i], distances[i]};
    }
}

} // namespace

void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    const auto blocks = (query_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
