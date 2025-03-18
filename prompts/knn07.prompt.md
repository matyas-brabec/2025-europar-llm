@include knn01.prompt.md

Compute each query's k nearest neighbors using a single thread warp (32 threads). The threads in a warp can communicate via warp shuffles or shared memory. In both cases, they should synchronize with appropriate barriers.

For each query, prepare a private copy of indices and distances of the intermediate result of its k nearest neighbors.

For each query, prepare a buffer for `k` indices and distances of candidates in shared memory. For each such buffer, define a shared variable that indicates the number of currently stored candidates.

Define a `max_distance` variable that stores the distance of the k-th nearest neighbor. When a newly computed distance of a data point is lower than `max_distance`, it is added to the candidate buffer; otherwise, it can be safely ignored. Update this variable whenever the k-th nearest neighbor in the intermediate result is updated.

Process the input data iteratively in batches. Each batch of input points is loaded into shared memory by the whole thread block. Each warp then uses the cached data points to compute their corresponding distances from its own query point. The data points are then filtered by `max_distance` and, if they are closer than `max_distance`, added to the buffer of candidates. Use `atomicAdd` to update the number of stored candidates and to determine an appropriate position in the buffer for each new candidate.

Whenever the buffer is full, merge it with the intermediate result in the following way:

0. (Invariant) The intermediate result is sorted in ascending order.
1. Sort the buffer using the Bitonic Sort algorithm in ascending order.
2. Merge the buffer and the intermediate result. Each element of the merged result is the minimum of the two elements at positions `i` and `k - i - 1`, one from the buffer and the other from the intermediate result. The merged result is a bitonic sequence containing the first `k` elements of the buffer and the intermediate result. The remaining elements can be safely discarded.
3. Sort the merged result in ascending order using the Bitonic Sort algorithm. This produces an updated intermediate result.

You can use the following unoptimized pseudocode for serial Bitonic Sort as a reference for the implementation of the Bitonic Sort algorithm:

```pseudocode
for (k = 2; k <= n; k *= 2) // k is doubled every iteration
    for (j = k/2; j > 0; j /= 2) // j is halved at every iteration, with truncation of fractional parts
        for (i = 0; i < n; i++)
            l = bitwiseXOR (i, j); // in C-like languages this is "i ^ j"
            if (l > i)
                if (  (bitwiseAND (i, k) == 0) AND (arr[i] > arr[l])
                    OR (bitwiseAND (i, k) != 0) AND (arr[i] < arr[l]) )
                        swap the elements arr[i] and arr[l]
```

After the last batch of candidates is processed, remember to merge the buffer with the intermediate result (if it is not empty).
