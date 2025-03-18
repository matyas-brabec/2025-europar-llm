# k-NN

Throughout the interactive scenario, we test the provided code on the H100 NVIDIA GPU using the `knn` framework without any modifications except setting `K=32` or `K=1024` to identify any issues with the code at the two extremes of the `k` range. The language model will be informed about the conducted tests and the results whenever necessary. If the next milestone requires redesign of the solution, the performance of the previous milestone is irrelevant. However, if the test recognize a data race or a deadlock, the language model will be informed about any such issues before the next milestone, as it will likely use the previous solution as a reference for the next one. Thus, continuing is allowed only if the language model provides a correct solution.

## Milestone #1: Working parallel solution

**Objective:** The language model will probably generate a solution that computes the nearest neighbors for each query independently in each thread. The most issues in the kernel design happen due to encoding the candidates in pairs of indices and distances.

The more natural (and also the more efficient) way is to store the distances and the indices in separate arrays. Indices will be often loaded less frequently than distances (as they are loaded only if the distances of two compared candidates satisfy a certain condition - depending on the specific situation). This applies regardless of the specific parallelization strategy.

The initial prompt is equivalent to the static `knn01.prompt.md`:

---

You are an experienced CUDA programmer trying to provide the most optimized code for the given task and the target hardware.
Assume that the target hardware is a modern NVIDIA GPU for data centers such as the H100 or A100, and the code is compiled using the latest CUDA toolkit and the latest compatible host compiler.
The output should contain only the requested source code. The code must not contain any placeholders or unimplemented parts. All high-level overviews and specific explanations should be included in code comments. Do not output any non-source text.

Write an optimized CUDA kernel that implements the k-nearest neighbors (k-NN) algorithm for 2D points in Euclidean space. The declaration of its C++ interface is as follows:

```cpp
void run_knn(const float2 *query, int query_count, const float2 *data, int data_count, std::pair<int, float> *result, int k);
```

The function receives `query_count` queries and `data_count` data points, each with two floating-point features. Both the `query` and the `data` points are stored consecutively in memory in row-major order (i.e., each `float2` element encodes coordinates of a single point). The function returns the indices of the `k` nearest data points and their corresponding distances for each query in the `result` array. The `result` array is also stored consecutively in memory in row-major order, each element being a pair of an integer index and a floating-point distance. For `query[i]`, `result[i * k + j]` corresponds to its `j`-th nearest neighbor `data[result[i * k + j].first]` at a distance of `result[i * k + j].second`.

The distance between two points is calculated using the squared Euclidean distance formula (squared L2 norm of the difference of the two points). There are no further requirements on the resolution of ties or the numerical stability of the distance calculation.

Assume that all arrays `query`, `data`, and `results` are duly allocated by `cudaMalloc` and that `data_count` is always greater than or equal to `k`. Additionally, `k` is a power of two between 32 and 1024, inclusive. Other input parameters are always valid, and, in a typical use case, they will be large enough to warrant parallel processing using a GPU; the number of queries will be in the order of thousands, and the number of data points will be at least in the order of millions. The kernel should not allocate any additional device memory.

Choose the most appropriate values for any algorithm hyper-parameters, such as the number of threads per block.

---

**Prepared hints:**

- Store the distances and indices in separate arrays. Do not use pairs of indices and distances except for the final result.

# Milestone #2: Each query processed by a thread warp, data points loaded in shared memory by thread blocks

**Objective:** The LLM will generate a solution that computes the nearest neighbors for each query using a single thread warp (32 threads). Each thread warp computes its own k nearest neighbors completely independently (except for the input data points loaded in shared memory). The main goal is to better utilize the computing resources of the GPU (threads and shared memory).

The follow-up prompt is based on the static `knn02.prompt.md` and the hints are based on `knn03.prompt.md`:

---

Compute each query's k nearest neighbors using a single thread warp (32 threads). The threads in a warp can communicate via warp shuffles or shared memory. In both cases, they should synchronize with appropriate barriers.

For each query, prepare a private copy of indices and distances of the intermediate result of its k nearest neighbors.

Process the input data iteratively in batches cashed in shared memory.

---

**Prepared hints:**

- Process the input data iteratively in batches. Each batch of input points is loaded into shared memory by the whole thread block. Each warp then uses the cached data points to compute their corresponding distances from its own query point.
- Use the computed distances and indices of the candidates to update the intermediate result.

## Milestone #3: The intermediate result of the k nearest neighbors is updated with multiple candidates simultaneously

**Objective:** The LLM will update the solution so that the intermediate result of the k nearest neighbors is updated with multiple candidates simultaneously, utilizing multiple threads in a warp. The easiest way to do this is to define a loop over the data points in the shared memory and update the intermediate result locally for each thread in the warp. If the threads share the same physical copy of the intermediate result, they can use the Bitonic sort algorithm to sort the intermediate result efficiently.

The follow-up prompt is based on the static `knn04.prompt.md`:

---

Update the intermediate result with multiple candidates simultaneously, utilizing multiple threads in a warp and appropriate communication mechanisms.

---

If the language model defines a shared physical copy of the intermediate result but not a shared buffer of candidates, continue with the next milestone.

**Prepared hints:**

- Each warp should iterate over a separate subset of the data points in shared memory and update its intermediate result accordingly.
- If the input data point is farther than the k-th nearest neighbor in the intermediate result, it can be ignored.

## Milestone #4: The shared buffer is introduced and the input data are prefiltered by the maximum distance

**Objective:** The LLM will generate a solution that uses a buffer shared by all threads in the warp. Data points that are closer than the maximum distance (the k-th nearest neighbor) are enqueued in the shared buffer. After filling up, the buffer is merged with the intermediate result of the k nearest neighbors.

The follow-up prompt is based on the static `knn05.prompt.md` and the hints are based on `knn06.prompt.md`:

---

For each query, prepare a buffer for `k` indices and distances of candidates in shared memory.

When processing the input data points stored in shared memory, enqueue the candidates that are closer than the maximum distance (the k-th nearest neighbor) in the shared buffer. Whenever the buffer is full, merge it with the intermediate result.

---

**Prepared hints:**

- After the last batch of candidates is processed, remember to merge the buffer with the intermediate result (if it is not empty).
- Define a `max_distance` variable that stores the distance of the k-th nearest neighbor. When a newly computed distance of a data point is lower than `max_distance`, it is added to the candidate buffer; otherwise, it can be safely ignored. Update this variable whenever the k-th nearest neighbor in the intermediate result is updated.
- For each candidate buffer, define an associated variable that stores the number of currently stored candidates. Update the variable using `atomicAdd` to get the unique order of an candidate enqueued in the buffer. Store all candidates with orders fitting within the buffer capacity. If the variable overflows the capacity of the buffer (`k`), merge the buffer with the intermediate result and reduce the number of stored candidates and orders of still unenqueued candidates by `k`.
- Separate the different stages of enqueuing and merging the candidates in the shared buffer and use the appropriate synchronization barriers for the threads in the warp.

# Milestone #5: The shared buffer is merge using the Bitonic sort algorithm

**Objective:** The LLM will generate a solution that uses the Bitonic sort algorithm to merge the shared buffer with the intermediate result of the k nearest neighbors. The pseudo code for the Bitonic sort algorithm is provided to the LLM so it can better reason about the implementation. The specific use of the Bitonic sort algorithm is also outlined in the prompt.

The follow-up prompt is based on the static `knn07.prompt.md`:

---

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

---

**Prepared hints (if everything is stored in shared memory and the solution does not use warp shuffles):**

- If `k > 32`, compute `k / 64` items by each thread in the warp. Otherwise, simply use the warp shuffle instructions.
- If `k > 32`, each thread iterates numbers `lane_id * (k / 64) + i'` for `i' = 0, 1, ..., k / 64 - 1`. For each such number, the index `i` of the computed item is defined by shifting the bits in `i'` higher or equal to `log2(j)` upwards so that `i ^ j` is equivalent to `i + j`. This ensures that `l > i`.

Otherwise, move to the next milestone.

# Milestone #6: The intermediate result is stored distributed among the threads in the warp

**Objective:** The LLM will generate a solution that uses the Bitonic sort algorithm to merge the shared buffer with the intermediate result of the k nearest neighbors as in the previous milestone. Furthermore, the intermediate result is stored distributed among the threads in the warp and the Bitonic sort algorithm uses the warp shuffle instructions.

The follow-up prompt is based on the static `knn08.prompt.md`:

---

The `k` nearest neighbors (distances and indices) of the intermediate result should be stored so that each thread keeps `k / 32` consecutive nearest neighbors.

Revise the use of the Bitonic sort algorithm as follows:

0. (Invariant) The intermediate result is sorted in ascending order.
1. Swap the content of the buffer and the intermediate result so that the buffer is in registers.
2. Sort the buffer using the Bitonic Sort algorithm in ascending order.
3. Merge the buffer and the intermediate result into the registers. Each element of the merged result is the minimum of the two elements at positions `i` and `k - i - 1`, one from the buffer and the other from the intermediate result. The merged result is a bitonic sequence containing the first `k` elements of the buffer and the intermediate result. The remaining elements can be safely discarded.
4. Sort the merged result in ascending order using the Bitonic Sort algorithm. This produces an updated intermediate result.

In the implementation of the Bitonic Sort algorithm, use the warp shuffle instructions to exchange elements between different threads. Since each thread stores consecutive elements, the exchanged elements always have the same index within the thread's registers.
If the exchanged elements are stored in registers of the same thread, they can be simply swapped in the registers without any communication with other threads.

---

**Prepared hints:**

- In each `k` step of the Bitonic sort algorithm, perform the `j < k / 32` steps and the `j >= k / 32` steps separately. The former do not exceed the boundaries of subarrays stored on each thread, perform them in the registers without any inter-thread communication. For the latter, use the warp shuffle instructions to exchange elements between different threads.
- For the steps `j >= k / 32`, iterate over the locally stored elements and (while properly synchronizing) exchange the elements between the threads using the warp shuffle instructions. Note that this swapping has to be performed by two threads simultaneously and they have to mirror the computation of each other.
- You can limit the number of nested conditions in the Bitonic sort algorithm by using the `^` operator on elementary conditions (`arr[i] > arr[l]`, `(i & k) == 0`, etc.)
