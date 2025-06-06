#ifndef KNN_HPP
#define KNN_HPP

#include <utility>

#include <cuda_runtime.h>

void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k);
void run_knn_baseline(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k);

#endif // KNN_HPP
