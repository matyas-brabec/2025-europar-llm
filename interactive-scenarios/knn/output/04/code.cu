#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <utility>

// Compute the squared Euclidean distance between two 2D points.
__device__ float squared_distance(float2 a, float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Device function to insert a candidate (with its index and distance) into the current k-NN arrays.
// The arrays 'knn_indices' and 'knn_dists' are maintained in ascending order of distance,
// meaning that the best (smallest) distance is at index 0 and the worst is at index K-1.
// Only if the new candidate's distance is less than the current worst candidate will it be inserted.
template <int K>
__device__ void insert_into_knn_arrays(int *knn_indices, float *knn_dists, int index, float dist) {
    // Check if the new candidate is closer than the worst candidate in the list.
    if (dist < knn_dists[K - 1]) {
        int j = K - 1;
        // Shift worse candidates one position to the right until the correct insertion spot is found.
        while (j > 0 && knn_dists[j - 1] > dist) {
            knn_dists[j] = knn_dists[j - 1];
            knn_indices[j] = knn_indices[j - 1];
            j--;
        }
        // Insert the new candidate at the found position.
        knn_dists[j] = dist;
        knn_indices[j] = index;
    }
}

// Templated CUDA kernel for the k-NN search where K is known at compile time.
// Each thread processes one query point and computes its k nearest neighbors.
template <int K>
__global__ void knn_kernel_templated(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx < query_count) {
        float2 q = query[query_idx];

        // Local arrays to hold the k nearest neighbor indices and distances.
        int knn_indices[K];
        float knn_dists[K];

        // Initialize the arrays: distances to infinity and indices to -1.
        #pragma unroll
        for (int i = 0; i < K; i++) {
            knn_dists[i] = std::numeric_limits<float>::infinity();
            knn_indices[i] = -1;
        }

        // Loop over all data points and update the k-NN arrays.
        for (int i = 0; i < data_count; i++) {
            float dist = squared_distance(q, data[i]);
            insert_into_knn_arrays<K>(knn_indices, knn_dists, i, dist);
        }

        // Write the final k nearest neighbors for this query into the global result array.
        // The final result is stored as a pair of (index, distance) for each neighbor.
        #pragma unroll
        for (int i = 0; i < K; i++) {
            result[query_idx * K + i] = std::make_pair(knn_indices[i], knn_dists[i]);
        }
    }
}

// Host function to dispatch the templated k-NN kernel based on the runtime k value.
// The runtime parameter 'k' is assumed to be a power of two between 32 and 1024 (inclusive).
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    int threads_per_block = 256;
    int blocks = (query_count + threads_per_block - 1) / threads_per_block;
    switch (k) {
        case 32:
            knn_kernel_templated<32><<<blocks, threads_per_block>>>(query, query_count, data, data_count, result);
            break;
        case 64:
            knn_kernel_templated<64><<<blocks, threads_per_block>>>(query, query_count, data, data_count, result);
            break;
        case 128:
            knn_kernel_templated<128><<<blocks, threads_per_block>>>(query, query_count, data, data_count, result);
            break;
        case 256:
            knn_kernel_templated<256><<<blocks, threads_per_block>>>(query, query_count, data, data_count, result);
            break;
        case 512:
            knn_kernel_templated<512><<<blocks, threads_per_block>>>(query, query_count, data, data_count, result);
            break;
        case 1024:
            knn_kernel_templated<1024><<<blocks, threads_per_block>>>(query, query_count, data, data_count, result);
            break;
        default:
            // Unsupported k value. No kernel is launched.
            break;
    }
    cudaDeviceSynchronize();
}
