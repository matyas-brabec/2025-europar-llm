#include <cuda_runtime.h>
#include <cmath>
#include <utility>

__device__ float squared_distance(float2 a, float2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

__device__ void insert_into_knn(std::pair<int, float> *knn, int k, int index, float dist) {
    // Perform insertion of (index, dist) into the k-NN heap
    int i = 0;
    while (i < k && knn[i].second < dist) {
        i++;
    }

    if (i < k) {
        // Shift larger distances to make room
        for (int j = k - 1; j > i; j--) {
            knn[j] = knn[j - 1];
        }
        knn[i] = std::make_pair(index, dist);
    }
}

__global__ void knn_kernel(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is working on a valid query
    if (query_idx < query_count) {
        float2 q = query[query_idx];

        // Create a temporary array to store the k nearest neighbors
        std::pair<int, float> knn[k];

        // Initialize knn with large distances
        for (int i = 0; i < k; i++) {
            knn[i] = std::make_pair(-1, FLT_MAX);
        }

        // Compute squared distances to all data points and insert into the knn heap
        for (int i = 0; i < data_count; i++) {
            float dist = squared_distance(q, data[i]);
            insert_into_knn(knn, k, i, dist);
        }

        // Store the results back to global memory
        for (int i = 0; i < k; i++) {
            result[query_idx * k + i] = knn[i];
        }
    }
}

void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k) {
    // Define the number of threads per block
    int threads_per_block = 256;
    int blocks = (query_count + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    knn_kernel<<<blocks, threads_per_block>>>(query, query_count, data, data_count, result, k);

    // Check for kernel launch errors
    cudaDeviceSynchronize();
}
