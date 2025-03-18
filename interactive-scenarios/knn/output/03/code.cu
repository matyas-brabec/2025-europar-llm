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

// Templated function to insert a new candidate into the sorted k-NN list.
// The list is maintained in descending order (largest distance first).
template <int K>
__device__ void insert_into_knn(std::pair<int, float> (&knn)[K], int index, float dist) {
    int pos = 0;
    // Find the correct position for the new candidate.
    while (pos < K && knn[pos].second < dist) {
        pos++;
    }
    if (pos < K) {
        // Shift elements to make room for the new candidate.
        for (int j = K - 1; j > pos; j--) {
            knn[j] = knn[j - 1];
        }
        knn[pos] = std::make_pair(index, dist);
    }
}

// Templated CUDA kernel for k-NN with compile-time constant K.
// Each thread processes one query point.
template <int K>
__global__ void knn_kernel_templated(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx < query_count) {
        float2 q = query[query_idx];
        // Allocate a local array to hold the k nearest neighbors.
        std::pair<int, float> knn[K];
        // Initialize the k-NN array with worst-case (infinite) distances.
        #pragma unroll
        for (int i = 0; i < K; i++) {
            knn[i] = std::make_pair(-1, std::numeric_limits<float>::infinity());
        }
        // Loop over all data points and update the k-NN list.
        for (int i = 0; i < data_count; i++) {
            float dist = squared_distance(q, data[i]);
            insert_into_knn<K>(knn, i, dist);
        }
        // Write the k nearest neighbors for this query back to global memory.
        #pragma unroll
        for (int i = 0; i < K; i++) {
            result[query_idx * K + i] = knn[i];
        }
    }
}

// Host function to dispatch the templated k-NN kernel based on the runtime k value.
// k is assumed to be a power of two between 32 and 1024 (inclusive).
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    int threads_per_block = 256;
    int blocks = (query_count + threads_per_block - 1) / threads_per_block;
    switch(k) {
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
            // For unsupported k values, no kernel is launched.
            break;
    }
    cudaDeviceSynchronize();
}
