Write an optimized CUDA kernel that implements the k-nearest neighbors (k-NN) algorithm for 2D points in Euclidean space. The declaration of its C++ interface is as follows:

```cpp
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k);
```

The function receives `query_count` queries and `data_count` data points, each with two floating-point features. Both the `query` and the `data` points are stored consecutively in memory in row-major order (i.e., each `float2` element encodes coordinates of a single point). The function returns the indices of the `k` nearest data points and their corresponding distances for each query in the `result` array. The `result` array is also stored consecutively in memory in row-major order, each element being a pair of an integer index and a floating-point distance. For `query[i]`, `result[i * k + j]` corresponds to its `j`-th nearest neighbor `data[result[i * k + j].first]` at a distance of `result[i * k + j].second`.

The distance between two points is calculated using the squared Euclidean distance formula (squared L2 norm of the difference of the two points). There are no further requirements on the resolution of ties or the numerical stability of the distance calculation.

Assume that all arrays `query`, `data`, and `results` are duly allocated by `cudaMalloc` and that `data_count` is always greater than or equal to `k`. Additionally, `k` is a power of two between 32 and 1024, inclusive. Other input parameters are always valid, and, in a typical use case, they will be large enough to warrant parallel processing using a GPU; the number of queries will be in the order of thousands, and the number of data points will be at least in the order of millions. The kernel should not allocate any additional device memory.

Choose the most appropriate values for any algorithm hyper-parameters, such as the number of threads per block.
