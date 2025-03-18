@include knn01.prompt.md

Compute each query's k nearest neighbors using a single thread warp (32 threads). The threads in a warp can communicate via warp shuffles or shared memory. In both cases, they should synchronize with appropriate barriers.

For each query, prepare a private copy of indices and distances of the intermediate result of its k nearest neighbors.

For each query, prepare a buffer for `k` indices and distances of candidates in shared memory.

Process the input data iteratively in batches. Each batch of input points is loaded into shared memory by the whole thread block. Each warp then uses the cached data points to compute their corresponding distances from its own query point and add them to the buffer of candidates. Skip the candidates that are not closer than the k-th nearest neighbor in the intermediate result.

Whenever the buffer is full, merge it with the intermediate result, utilizing multiple threads in a warp and appropriate communication mechanisms.

After the last batch of candidates is processed, remember to merge the buffer with the intermediate result (if it is not empty).
