@include knn01.prompt.md

Compute each query's k nearest neighbors using a single thread warp (32 threads). The threads in a warp can communicate via warp shuffles or shared memory. In both cases, they should synchronize with appropriate barriers.

For each query, prepare a private copy of indices and distances of the intermediate result of its k nearest neighbors.

Process the input data iteratively in batches. Each batch of input points is loaded into shared memory by the whole thread block. Each warp then uses the cached data points to compute their corresponding distances from its own query point.

Use the computed distances and indices of the candidates to update the intermediate result.
