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

[output/01/code.cu](output/01/code.cu)

---

the code won't compile:

(line 39) error: identifier "FLT_MAX" is undefined

---

To fix the compilation error, you should use `std::numeric_limits<float>::infinity()` from the `<limits>` header instead of `FLT_MAX`, which is not defined in this context.

[output/02/code.cu](output/02/code.cu)

Now, the code should compile without issues. The `std::numeric_limits<float>::infinity()` provides the appropriate large value to initialize the distances for comparison.
---

the code won't compile:

(line 36) error: expression must have a constant value
          std::pair<int, float> knn[k];
                                    ^

---

[output/03/code.cu](output/03/code.cu)

---

Store the distances and indices in separate arrays. Do not use pairs of indices and distances except for the final result.

---

[output/04/code.cu](output/04/code.cu)

---

Compute each query's k nearest neighbors using a single thread warp (32 threads). The threads in a warp can communicate via warp shuffles or shared memory. In both cases, they should synchronize with appropriate barriers.

For each query, prepare a private copy of indices and distances of the intermediate result of its k nearest neighbors.

Process the input data iteratively in batches cashed in shared memory.

---

[output/05/code.cu](output/05/code.cu)

---

Since each thread stores just `k / 32` elements and processes a different part of the input data without any communication between the threads, the final result is not correct.

---

[output/06/code.cu](output/06/code.cu)

---

(line 216) error: identifier "K" is undefined
      size_t shm_per_warp = K * sizeof(int) + K * sizeof(float) +
                            ^

---

[output/07/code.cu](output/07/code.cu)

---

Update the intermediate result with multiple candidates simultaneously, utilizing multiple threads in a warp and appropriate communication mechanisms.

---

[output/08/code.cu](output/08/code.cu)

---

If the input data point is farther than the k-th nearest neighbor in the intermediate result, it can be ignored.

---

[output/09/code.cu](output/09/code.cu)

---


For each query, prepare a buffer for `k` indices and distances of candidates in shared memory.

When processing the input data points stored in shared memory, enqueue the candidates that are closer than the maximum distance (the k-th nearest neighbor) in the shared buffer. Whenever the buffer is full, merge it with the intermediate result.

---

[output/10/code.cu](output/10/code.cu)

---

cudaGetLastError(): an illegal memory access was encountered

---

[output/11/code.cu](output/11/code.cu)

---

After computing the candidate's position using `atomicAdd`, if `pos < K` fails, the candidate is mistakenly discarded. No candidates should be discarded except for those farther than the maximum distance in the intermediate result.

---

[output/12/code.cu](output/12/code.cu)

---

The kernel contains a synchronization point inside the loop that is reached only by the lanes whose candidates' positions are out of bounds. This causes the kernel to deadlock.

---

[output/13/code.cu](output/13/code.cu)

---

(line 119) error: expected a ")"
                           warpSize * 8 * (sizeof(int) + sizeof(float)));
                                                                        ^

---

[output/14/code.cu](output/14/code.cu)

---

cudaGetLastError(): an illegal memory access was encountered

---

[output/15/code.cu](output/15/code.cu)

---

(line 227) error: identifier "candidate_count" is undefined
              *candidate_count = 0;
               ^

---

It seems that the issue occurs because `candidate_count` was defined in the `merge_buffer_with_intermediate` function but not in the global context. To resolve this, we'll define `candidate_count` properly in shared memory, ensuring that it's available across the entire warp.

Here is the corrected version:

[output/16/code.cu](output/16/code.cu)

### Changes made:
- I added `candidate_count` into the shared memory space to track how many candidates have been added to the buffer.
- I fixed the memory size calculation for the shared memory to accommodate the `candidate_count` at the end.
- In the main loop, I handled the merging process correctly by checking the `candidate_count` variable for overflows and performing the merge. This ensures that the buffer doesn't overflow and we don't encounter any race conditions.

This should fix the issue. Let me know if you encounter any other problems!

---

The result contains the initial values that represent infinity.

---

[output/17/code.cu](output/17/code.cu)
The issue with getting infinity values in the result is likely due to incorrect synchronization or failure to properly initialize and update the intermediate candidate list. Specifically, the initialization of `inter_indices` and `inter_dists` (set to `-1` and `infinity` respectively) might not be properly synchronized across warps, or the merge process may not be correctly updating the list.

To resolve this, we need to ensure the following:

1. Proper initialization of the intermediate candidate list (`inter_indices` and `inter_dists`).
2. Correct updates to the intermediate candidate list in all cases when a better candidate is found.
3. Proper synchronization during each merge step.
4. Writing the final results after all batches are processed.

Here’s the corrected version of the code:

[output/17/code.cu](output/17/code.cu)

### Key updates:
1. **Initialization:** The `inter_indices` and `inter_dists` are correctly initialized to `-1` and `infinity`, respectively, for all threads at the start of each warp.
2. **Buffer and Candidate Management:** I've fixed the logic to ensure that no candidates are discarded unless they are farther than the current worst candidate in the intermediate list. The `candidate_count` is used to track how many candidates are in the buffer.
3. **Merge Logic:** The merge of the candidate buffer and the intermediate list is correctly handled after each tile and after the final set of candidates are processed.

Let me know if the issue persists, and I will help debug further!
