# kNN

This file contains an in-depth review of the "one-shot" generated solutions.

## Correctness table

(Evaluated on an Nvidia H100 GPU with NVCC 12.8 and GCC 13.2.0)

**k=1024, n=4'194'304, m=4'096, r=10**

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ❌  | ✅  | ✅  | ❌🛠️ | ❌🛠️ | ❌💥 | ❌🛠️ | ✅  | ✅  | ✅  |
|    kNN02     | ❌  | ❌  | ❌💥 | ❌  | ❌💥🛠️| ❌  | ❌💥 | ❌💥 | ❌  | ❌💥🛠️|
|    kNN03     | ❌💥 | ❌  | ❌🛠️ | ❌  | ❌💥 | ❌🛠️ | ❌💥🛠️| ❌🛠️ | ❌  | ❌🛠️ |
|    kNN04     | ❌🛠️ | ❌💥 | ❌🛠️ | ❌  | ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌🛠️ | ❌💥 | ❌💥 |
|    kNN05     | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️|
|    kNN06     | ❌💥🛠️| ❌💥 | ❌⚙️🛠️| ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥 | ❌💥 |
|    kNN07     | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌🛠️|
|    kNN08     | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥 | ❌💥 | ❌💥 |

**k=32, n=4'194'304, m=4'096, r=10**

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ❌  | ✅  | ✅  | ✅🛠️ | ✅🛠️ | ❌💥 | ❌🛠️ | ✅  | ✅  | ✅  |
|    kNN02     | ❌  | ❌  | ❌💥 | ❌  | ❌🛠️| ❌  | ❌💥 | ❌  | ❌  | ❌🛠️ |
|    kNN03     | ❌💥 | ❌  | ❌🛠️ | ❌  | ❌💥 | ❌🛠️ | ❌🛠️ | ❌🛠️ | ❌  | ❌🛠️ |
|    kNN04     | ❌🛠️ | ❌💥 | ❌🛠️ | ❌  | ❌💥🛠️| ❌  | ❌💥🛠️| ❌🛠️ | ✅   | ❌💥 |
|    kNN05     | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️|
|    kNN06     | ❌🛠️ | ❌💥 | ❌⚙️🛠️| ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥 | ❌  | ❌ |
|    kNN07     | ❌  | ❌💥🛠️| ❌  | ❌💥 | ❌💥 | ❌💥🛠️| ❌🛠️ | ❌🛠️| ❌  | ❌🛠️|
|    kNN08     | ❌💥 | ❌💥🛠️| ❌ | ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥 | ❌  | ❌💥 |

✅ – Correct solution (compiled successfully and returned the correct GoL grid).

❌ – Compiled and ran without a runtime error but returned incorrect results.

❌💥 – Compiled but crashed during execution. (Or timed out)

❌⚙️ – Did not compile.

🛠️ – Indicator denotes the source code a small edit to make it compile (this mark is added alongside one of the above). The erroneous line(s) was/were commented and prefixed with `/// @FIXED` comment.

---

## Review process

We mostly focus the memory management of the intermediate result copies (how they are stored, how they are shared, data structure used, etc.) and the processing of the input data points. Prologue and epilogue code holds less interest as its execution time is negligible for the expected input sizes.

### Common patterns that should be looked for in the solutions

- User-defined `pair` type (commonly named `Candidate`, `Pair`, or `Neighbor`)
  - Always defined as a trivial `struct` with two data members (`int` and `float`)

- One of the following representations of the intermediate result of the k-nearest neighbors search:
  - The intermediate result is represented by a binary heap (SoA)
  - The intermediate result is represented by a binary heap (AoS)
  - The intermediate result is represented by a sorted array (SoA)
  - The intermediate result is represented by a sorted array (AoS)
  - The intermediate result is represented by an unsorted array (SoA)
  - The intermediate result is represented by an unsorted array (AoS)

- One of the following strategies
  - The intermediate result is not shared, stored in registers
  - The intermediate result is shared among threads in a warp, master copy stored in registers of a single thread
  - The intermediate result is shared among threads in a warp, local copies stored in registers
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - The intermediate result is shared among threads in a block, master copy stored in registers of a single thread
  - The intermediate result is shared among threads in a block, local copies stored in registers
  - The intermediate result is shared among threads in a block, stored in registers distributed among the block threads
  - The intermediate result is shared among threads in a block, stored in shared memory

- A batch of input data points stored in shared memory

## Prompt 01

**Requested/suggested features:**

For details, see [knn01.prompt.md](../prompts/knn01.prompt.md).

- Just a general description of the solution and the expected input data size
- Direction to optimize the code and use the expected best-performing hyper-parameters (such as the number of threads per block, etc.)

**Common features:**

- User-defined `pair` type (`Candidate`, `Pair`, `Neighbor`, or `ResultPair`)

**Special features per solution:**

- 01  | ❌ | ❌
  - The intermediate result is represented by a sorted array (SoA)
  - The intermediate result is shared among threads in a block, stored in registers distributed among the block threads
  - A batch of input data points stored in shared memory

- 02  | ✅  | ✅
  - A batch of input data points stored in shared memory
  - The intermediate result is not shared, stored in registers
  - The intermediate result is represented by a binary heap (AoS)

- 03  | ✅  | ✅
  - The intermediate result is represented by a binary heap (AoS)
  - The intermediate result is not shared, stored in registers

- 04🛠️ | ❌  | ✅
  - No user-defined `pair` type
  - The intermediate result is represented by an unsorted array (SoA)
  - The intermediate result is shared among threads in a block, stored in registers distributed among the block threads
  - Uses Bitonic sort to sort the final result

- 05🛠️ | ❌  | ✅
  - The intermediate result is represented by a binary heap (AoS)
  - The intermediate result is shared among threads in a block, local copies stored in registers

- 06  | ❌💥| ❌💥
  - The intermediate result is represented by a sorted array (AoS)
  - The intermediate result is shared among threads in a block, stored in registers distributed among the block threads
  - A batch of input data points stored in shared memory
  - Uses Bitonic sort to sort the final result

- 07🛠️ | ❌  | ❌
  - The intermediate result is represented by an unsorted array (AoS)
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - Uses Bitonic sort to sort the final result

- 08  | ✅  | ✅
  - The intermediate result is not shared, stored in registers
  - A batch of input data points stored in shared memory
  - The intermediate result is represented by a binary heap (SoA)

- 09  | ✅  | ✅
  - The intermediate result is shared among threads in a block, local copies stored in registers
  - The intermediate result is represented by a binary heap (SoA)

- 10  | ✅  | ✅
  - The intermediate result is not shared, stored in registers
  - The intermediate result is represented by a binary heap (SoA)
  - A batch of input data points stored in shared memory
  - No user-defined `pair` type

**Conclusion:**

If the intermediate result is distributed among the threads in a warp, each local copy typically contains `k / 32` candidates. This means that the algorithm actually performs a `k / 32`-nearest neighbors search (as all the expected `k` nearest neighbors can be, theoretically, processed by a single thread). Four solutions (02, 03, 08, and 09) avoid this problem by associating each thread with a different query point --- then, the intermediate result has to contain `k` candidates as it is not distributed.

Most of the solutions use a binary heap to represent the intermediate result. This is the most efficient structure for keeping the `k` minimum values as each update takes `O(log(k))` time. At the same time, it also allows for efficient pre-filtering of the candidates, which requires just a single comparison per candidate as opposed to `k` comparisons for the unsorted structure. For sorted arrays (used by solutions 01 and 06), the pre-filtering also requires just a single comparison per candidate. However, the insertion of a new candidate takes `O(k)` time, which is typically worse than the `O(log(k))` time of the binary heap. Two solutions (04 and 07) use an unsorted array to represent the intermediate result.

Half of the solutions use the SoA format of the intermediate result (01, 04, 08, 09, 10). The other half use the AoS format (02, 03, 05, 06, 07).

Since, for this prompt, the number of solutions that use the binary heap representation of the intermediate result and no distribution or sharing of the intermediate result is the highest, the prompt also has the highest number of correct solutions. Since the solutions with these characteristics are generally simpler (in terms of code complexity and in terms of design choices). If the intermediate result is not shared/distributed, the algorithm generally requires no synchronization barriers or other inter-thread communication. Since the later prompts show that the language model struggles with these features, this is yet another reason for the higher number of correct solutions for this prompt.

## Prompt 02

**Requested/suggested features:**

For details, see [knn02.prompt.md](../prompts/knn02.prompt.md).

- Each query is computed by a single thread warp (32 threads)
- A private copy of the intermediate result of the k nearest neighbors search for each query
- A batch of input data points stored in shared memory

**Common features:**

- No user-defined `pair` type

**Special features per solution:**

<!-- |    kNN02     | ❌  | ❌  | ❌💥 | ❌  | ❌💥🛠️| ❌  | ❌💥 | ❌💥 | ❌  | ❌💥🛠️| -->
<!-- |    kNN02     | ❌  | ❌  | ❌💥 | ❌  | ❌🛠️| ❌  | ❌💥 | ❌  | ❌  | ❌🛠️ | -->

- 01  | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)
  - Uses warp-global pre-filtering by communicating the maximum distance for each batch of input data points (tile)
  - Uses Bitonic sort to sort the final result

- 02  | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)
  - The merging process takes k iterations of the following:
    - each thread finds a local minimum
    - the threads communicate the global minimum via warp-shuffle instructions
    - the global minimum is stored in the intermediate result at the appropriate position
    - the thread containing the global minimum erases the corresponding value from its intermediate result (overwrites it with the maximum possible value)

- 03  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by a sorted array (SoA)
  - For the merging process, the first thread inserts the candidates from the intermediate results of all threads into a sorted array
    - The values are communicated via warp-shuffle instructions

- 04  | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)
  - Uses Bitonic sort to sort the final result
    - All local values are sent to the lane 0 of the warp via warp-shuffle instructions
    - The lane 0 performs serial sorting

- 05🛠️ | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)
  - Merging process:
    - Each thread sorts its intermediate result via insertion sort
    - The lane 0 of the warp performs a serial merge of the sorted intermediate results
      - Uses the multi-way merge (chooses the minimum of the 32 heads of the sorted arrays and pops the selected head)

- 06  | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)
  - Uses Bitonic sort to sort the final result
  - Merging process:
    - Each thread sorts its intermediate result via insertion sort
    - All local values are stored to a shared memory array
    - The lane 0 of the warp sorts the shared memory array via Bitonic sort

- 07  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, master copy stored in registers of a single thread
  - The intermediate result is represented by a binary heap (SoA)
  - All candidates are, during the processing of the input data points, sent to the lane 0 of the warp via warp-shuffle instructions
  - No merging process is performed
  - The lane 0 sorts the candidates via heap sort

- 08  | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)
  - Merging process:
    - Each thread sorts its intermediate result via insertion sort
    - All local values are stored to a shared memory array
    - The lane 0 of the warp performs a serial merge of the sorted intermediate results
      - Uses the multi-way merge (chooses the minimum of the 32 heads of the sorted arrays and pops the selected head)

- 09  | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)
  - Merging process:
    - The lane 0 gathers the candidates from all threads in the warp via warp-shuffle instructions

      (other threads are inactive during the warp-shuffle instructions; so the result is undefined)

    - The collected candidates are sorted via insertion sort

- 10🛠️ | ❌💥 | ❌
  - User-defined `pair` type (`Pair`)
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by a sorted array (SoA)
  - Merging process:
    - Each thread stores its intermediate result to a shared memory array
    - The lane 0 of the warp performs a serial merge of the sorted intermediate results
      - Uses the multi-way merge (chooses the minimum of the 32 heads of the sorted arrays and pops the selected head)

**Conclusion:**

Each local intermediate result typically contains `k / 32` candidates. This means that the algorithm actually performs a `k / 32`-nearest neighbors search (as all the expected `k` nearest neighbors can be, theoretically, processed by a single thread).

Most of the solutions keep the intermediate result unsorted. This performs worse than sorted or heap-based structure as it does not allow for efficient pre-filtering of the candidates. For the other approaches, pre-filtering requires a single comparison per candidate as opposed to `k` comparisons for the unsorted structure. Only one solution uses a binary heap to represent the intermediate result and only two solutions use a sorted array.

All solutions use the SoA format of the intermediate result.

The language model struggles with tracking the code semantics. It is prone to trivial mistakes such as creating synchronization barriers that are not reached by all synchronizing threads. It makes these mistakes even when the semantically inconsistent code fits in very small contexts (e.g., a few lines).

## Prompt 03

**Requested/suggested features:**

For details, see [knn03.prompt.md](../prompts/knn03.prompt.md).

- Each query is computed by a single thread warp (32 threads)
- A private copy of the intermediate result of the k nearest neighbors search for each query
- A batch of input data points stored in shared memory by the thread block
- Each thread warp computes the distances from the input data points in the shared memory

**Common features:**

- No user-defined `pair` type

**Special features per solution:**

<!-- |    kNN03     | ❌💥 | ❌  | ❌🛠️ | ❌  | ❌💥 | ❌🛠️ | ❌💥🛠️| ❌🛠️ | ❌  | ❌🛠️ | -->
<!-- |    kNN03     | ❌💥 | ❌  | ❌🛠️ | ❌  | ❌💥 | ❌🛠️ | ❌🛠️ | ❌🛠️ | ❌  | ❌🛠️ | -->

- 01  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)

- 02  | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)

- 03🛠️ | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by sorted arrays (SoA)
  - For each candidate, the algorithm finds the lane containing the nearest neighbor with the maximum distance and replaces it
    - Each insertion broadcasts the candidate to all threads and finds the maximum distance via warp-shuffle instructions
    - The thread with the maximum distance then replaces the maximum distance with the candidate
  - Uses Bitonic sort to sort the final result

- 04  | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by an unsorted array (SoA)

- 05  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - The intermediate result is represented by a sorted array (AoS)
  - The lane 0 of the warp inserts processed candidates communicated via warp-shuffle instructions into the intermediate result
  - All remaining threads are inactive during the warp-shuffle instructions; so the result is undefined

- 06🛠️ | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by a sorted array (SoA)
  - User-defined `pair` type (`Pair`)

- 07🛠️ | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by a sorted array (SoA)
  - User-defined `pair` type (`Pair`)

- 08🛠️ | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by a sorted array (SoA)

- 09  | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by unsorted arrays (SoA)
  - For each candidate, the algorithm finds the lane containing the nearest neighbor with the maximum distance and replaces it
    - In each iteration, each thread finds a local maximum
    - The threads communicate the global maximum via warp-shuffle instructions
    - The thread with the global maximum then replaces the maximum distance with the candidate
    - **Mistake**: each thread looks at a different candidate (they are equally split into 32 groups)
      - So a candidate can be inserted during an iteration only if the associated thread currently has the maximum distance
      - This makes no sense - the language model mixed different algorithms

- 10🛠️ | ❌  | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The intermediate result is represented by a sorted array (SoA)

**Conclusion:**

Each local intermediate result typically contains `k / 32` candidates. This means that the algorithm actually performs a `k / 32`-nearest neighbors search (as all the expected `k` nearest neighbors can be, theoretically, processed by a single thread).

All solutions except one use the SoA format of the intermediate result. Solution 05 is the only one that uses the AoS format.

Roughly half of the solutions keep the intermediate result unsorted. This performs worse than sorted or heap-based structure as it does not allow for efficient pre-filtering of the candidates. For the other approaches, pre-filtering requires a single comparison per candidate as opposed to `k` comparisons for the unsorted structure. The other half of the solutions use a sorted array to represent the intermediate result.

## Prompt 04

**Requested/suggested features:**

For details, see [knn04.prompt.md](../prompts/knn04.prompt.md).

- Each query is computed by a single thread warp (32 threads)
- A private copy of the intermediate result of the k nearest neighbors search for each query
- A batch of input data points stored in shared memory by the thread block
- Each thread warp computes the distances from the input data points in the shared memory
- The intermediate result is updated by inserting multiple candidates simultaneously in parallel

**Common features:**

- The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
- The intermediate result is represented by a sorted array (SoA)

**Special features per solution:**

<!-- |    kNN04     | ❌🛠️ | ❌💥 | ❌🛠️ | ❌  | ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌🛠️ | ❌💥 | ❌💥 | -->
<!-- |    kNN04     | ❌🛠️ | ❌💥 | ❌🛠️ | ❌  | ❌💥🛠️| ❌  | ❌💥🛠️| ❌🛠️ | ✅   | ❌💥 | -->

- 01🛠️ | ❌  | ❌
  - User-defined `pair` type (`Candidate`)
  - The lane 0 collects all local intermediate results using warp-shuffle instructions
  - The lane 0 of the warp combines the local intermediate results using the insertion sort algorithm

- 02  | ❌💥 | ❌💥
  - User-defined `pair` type (`Pair`)
  - For each candidate, the algorithm finds the lane containing the nearest neighbor with the maximum distance and replaces it
    - In each iteration, each thread finds a local maximum
    - Insertion:
      - Each thread proposes a single candidate, the algorithm then iteratively broadcasts each candidate to all threads
        - The threads communicate the global maximum via warp-shuffle instructions
        - The thread with the global maximum then replaces the maximum distance with the candidate
      - Mistake: the algorithm starts by finding the global maximum distance and filters out the candidates that are not closer than the global maximum distance
        - The threads with the filtered-out candidates are inactive during the whole process
        - Some threads do not reach synchronization barriers

- 03🛠️ | ❌  | ❌

- 04  | ❌  | ❌

- 05🛠️ | ❌💥 | ❌💥
  - User-defined `pair` type (`Candidate`)

- 06  | ❌💥 | ❌
  - The intermediate result is represented by a sorted array (AoS)
  - User-defined `pair` type (`Candidate`)

- 07🛠️ | ❌💥 | ❌💥
  - Each candidate is inserted in the intermediate result using the insertion to a local temporary array
  - The sorted temporary array and the sorted intermediate result are merged using the 2-way merge algorithm

- 08🛠️ | ❌  | ❌

- 09  | ❌💥 | ✅
  - **This is just wrong**

- 10  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - Each thread builds a local temporary array of candidates sorted by insertion sort
  - The lane 0 of the warp merges the sorted temporary arrays using the insertion sort algorithm
  - The lane 0 of the warp retrieves the temporary arrays by communicating with the inactive threads via warp-shuffle instructions
  - This makes no sense

**Conclusion:**

Same as before, each local intermediate result typically contains `k / 32` candidates. This means that the algorithm actually performs a `k / 32`-nearest neighbors search (as all the expected `k` nearest neighbors can be, theoretically, processed by a single thread).

The language model struggles with tracking the code semantics. It is prone to trivial mistakes such as creating synchronization barriers that are not reached by all synchronizing threads. It makes these mistakes even when the semantically inconsistent code fits in very small contexts (e.g., a few lines).

All solutions except one use the SoA format of the intermediate result. Solution 06 is the only one that uses the AoS format.

Virtually all solutions use a sorted array to represent the intermediate result. This is typically less efficient for low numbers of candidates stored in a local copy of the intermediate result. However, it still allows for efficient pre-filtering of the candidates, which requires just a single comparison per candidate as opposed to `k` comparisons for the unsorted structure. Only one solution uses an unsorted array to represent the intermediate result. None of the solutions use a heap-based structure.

## Prompt 05

**Requested/suggested features:**

For details, see [knn05.prompt.md](../prompts/knn05.prompt.md).

- Each query is computed by a single thread warp (32 threads)
- A private copy of the intermediate result of the k nearest neighbors search for each query
- A batch of input data points stored in shared memory by the thread block
- Each thread warp computes the distances from the input data points in the shared memory
- For each query, a shared buffer of `k` candidates is defined
- The candidates are inserted in the buffer if they are closer than the current maximum distance
- The buffer is merged with the intermediate result when it is full in parallel

**Common features:**

- User-defined `pair` type (`Candidate`, `Pair`, `Neighbor`, or `ResultPair`)
- The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
- The intermediate result is represented by a sorted array (AoS)
- each shared buffer is associated with a shared variable storing the current size of the buffer
- `atomicAdd` used to update the size of the buffer and to allocate a new position in the buffer

**Special features per solution:**

<!-- |    kNN05     | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️| -->
<!-- |    kNN05     | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️| -->

- 01  | ❌💥 | ❌💥
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense

- 02  | ❌💥 | ❌💥
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense
    - Merging utilizes the Bitonic sort algorithm
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 03🛠️ | ❌💥 | ❌💥
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 04  | ❌💥 | ❌💥
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 05🛠️ | ❌💥 | ❌💥
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense
    - Merging utilizes the Bitonic sort algorithm
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 06🛠️ | ❌💥 | ❌💥
  - Two user-defined `pair` types (`Candidate` & `Pair`)
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 07  | ❌💥 | ❌💥
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense

- 08🛠️ | ❌💥 | ❌💥
  - Buffer merging:
    - Performed in parallel
    - No synchronization before checking the buffer size
    - The buffer size resets to zero after the merge (which is not correct)
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 09  | ❌💥 | ❌💥
  - Buffer merging:
    - One lane chosen to perform the merge
    - After the merge, the lane communicates with the inactive threads via warp-shuffle instructions
    - This makes no sense
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 10🛠️ | ❌💥 | ❌💥
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

**Conclusion:**

Same as before, each local intermediate result typically contains `k / 32` candidates. This means that the algorithm actually performs a `k / 32`-nearest neighbors search (as all the expected `k` nearest neighbors can be, theoretically, processed by a single thread).

The language model struggles with tracking the code semantics. It is prone to trivial mistakes such as creating synchronization barriers that are not reached by all synchronizing threads. It makes these mistakes even when the semantically inconsistent code fits in very small contexts (e.g., a few lines).

The language model also struggles with the semantics of the task at hand. It actively discards a portion of the input data (candidates) that is not supposed to be discarded.

Even though the language model uses insertion sort as its primary sorting algorithm, some solutions use the Bitonic sort algorithm despite not being requested in the prompt.

The solutions use the AoS format for the intermediate result.

All solutions use a sorted array to represent the intermediate result. This is typically less efficient for low numbers of candidates stored in a local copy of the intermediate result. However, it still allows for efficient pre-filtering of the candidates, which requires just a single comparison per candidate as opposed to `k` comparisons for the unsorted structure. None of the solutions use unsorted or heap-based structures to represent the intermediate result.

## Prompt 06

**Requested/suggested features:**

For details, see [knn06.prompt.md](../prompts/knn06.prompt.md).

- Each query is computed by a single thread warp (32 threads)
- A private copy of the intermediate result of the k nearest neighbors search for each query
- A batch of input data points stored in shared memory by the thread block
- Each thread warp computes the distances from the input data points in the shared memory
- For each query, a shared buffer of `k` candidates is defined
- The candidates are inserted in the buffer if they are closer than `max_distance`; `max_distance` is updated after each merge of the buffer with the intermediate result
- Insertion is directed by a shared variable storing the current size of the buffer
- Positions in the buffer are assigned using `atomicAdd` on the shared variable
- The buffer is merged with the intermediate result when it is full in parallel

**Common features:**

- User-defined `pair` type (`Candidate`, `Pair`, `Neighbor`, `NN`, or `KNNResult`)
- The intermediate result is represented by a sorted array (AoS)

**Special features per solution:**

<!-- |    kNN06     | ❌💥🛠️| ❌💥 | ❌⚙️🛠️| ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥 | ❌💥 | -->
<!-- |    kNN06     | ❌🛠️ | ❌💥 | ❌⚙️🛠️| ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥 | ❌  | ❌ | -->

- 01🛠️  | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - Buffer merging:
    - All private copies and the shared buffer added to a single merge buffer in shared memory
    - The lane 0 of the warp performs a serial insertion sort on the merge buffer
    - The first `k` elements of the merge buffer are copied to the private copy of the intermediate result
    - `max_distance` is updated

- 02  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 03⚙️🛠️
  - Two user-defined `pair` types (`Candidate` & `KNNResult`)
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - This makes no sense

- 04  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane loads the intermediate result and the shared buffer into registers
    - The elements are sorted using the insertion sort algorithm
    - The first `k` elements of the sorted array are copied to the intermediate result

- 05  | ❌💥 | ❌💥
  - No user-defined `pair` type
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - Buffer merging:
    - One lane chosen to perform the merge
      - selected using two independent criteria, each targeting possibly different threads <- makes no sense
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 06 | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, master copy stored in registers of a single thread
  - Buffer merging:
    - All private copies and the shared buffer added to a single merge buffer in shared memory
    - The merge buffer is sorted using a parallel Bitonic sort algorithm
  - `atomicCAS` used to update the size of the buffer and to allocate a new position in the buffer
    - retrying until the `atomicCAS` returns the expected value
  - No synchronization before checking the buffer size
    - Synchronization point not reached by all threads in the warp

- 07🛠️ | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a parallel merge communicating with the inactive threads via warp-shuffle instructions
    - No synchronization before checking the buffer size
    - Warp-wide synchronization in places that are not guaranteed to be executed by all threads in the warp

- 08  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Buffer merging utilizes the Bitonic sort algorithm

- 09  | ❌💥 | ❌
  - No user-defined `pair` type
  - The intermediate result is shared among threads in a warp, master copy stored in registers of a single thread
  - Buffer merging:
    - The private copy and the shared buffer added to a single merge buffer in shared memory
    - The lane 0 of the warp performs a serial insertion sort on the merge buffer
    - The first `k` elements of the merge buffer are copied to the private copy of the intermediate result
    - `max_distance` is updated

- 10  | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - The buffer is stored in the AoS format
  - Buffer merging utilizes the Bitonic sort algorithm
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

**Conclusion:**

Same as before, each local intermediate result typically contains `k / 32` candidates. This means that the algorithm actually performs a `k / 32`-nearest neighbors search (as all the expected `k` nearest neighbors can be, theoretically, processed by a single thread).

The language model is especially biased towards the use of serial insertion sort (even for a higher number of elements and at the cost of leaving other threads inactive). Even though the language model uses insertion sort as its primary sorting algorithm, some solutions use the Bitonic sort algorithm despite not being requested in the prompt.

The language model struggles with tracking the code semantics. It is prone to trivial mistakes such as creating synchronization barriers that are not reached by all synchronizing threads. It makes these mistakes even when the semantically inconsistent code fits in very small contexts (e.g., a few lines).

The language model also struggles with the semantics of the task at hand. It actively discards a portion of the input data (candidates) that is not supposed to be discarded.

All solutions use the AoS format for the intermediate result.

All solutions use a sorted array to represent the intermediate result. This is typically less efficient for low numbers of candidates stored in a local copy of the intermediate result. However, it still allows for efficient pre-filtering of the candidates, which requires just a single comparison per candidate as opposed to `k` comparisons for the unsorted structure. None of the solutions use unsorted or heap-based structures to represent the intermediate result.

## Prompt 07

**Requested/suggested features:**

For details, see [knn07.prompt.md](../prompts/knn07.prompt.md).

- Each query is computed by a single thread warp (32 threads)
- A private copy of the intermediate result of the k nearest neighbors search for each query
- A batch of input data points stored in shared memory by the thread block
- Each thread warp computes the distances from the input data points in the shared memory
- For each query, a shared buffer of `k` candidates is defined
- The candidates are inserted in the buffer if they are closer than `max_distance`; `max_distance` is updated after each merge of the buffer with the intermediate result
- Insertion is directed by a shared variable storing the current size of the buffer
- Positions in the buffer are assigned using `atomicAdd` on the shared variable
- The buffer is merged with the intermediate result when it is full using the Bitonic sort algorithm (which is outlined in the prompt)

**Common features:**

- User-defined `pair` type (`Candidate`, `Pair`, or `Neighbor`)
- The intermediate result is represented by a sorted array (AoS)

**Special features per solution:**

<!-- |    kNN07     | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️| ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌🛠️| -->
<!-- |    kNN07     | ❌  | ❌💥🛠️| ❌  | ❌💥 | ❌💥 | ❌💥🛠️| ❌🛠️ | ❌🛠️| ❌  | ❌🛠️| -->

- 01  | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a serial Bitonic sort algorithm
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 02🛠️ | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - `block_size` is 32

- 03  | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 04  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane performs a serial Bitonic sort algorithm
  - `block_size` is 32

- 05  | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane communicates with the inactive threads via warp-shuffle instructions
    - The lane performs a serial Bitonic sort algorithm
    - This makes no sense

- 06🛠️ | ❌💥 | ❌💥
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Buffer merging:
    - One lane chosen to perform the merge
    - The lane communicates with the inactive threads via warp-shuffle instructions
    - The lane performs parallel Bitonic sort by itself
    - This makes no sense

- 07🛠️ | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in shared memory
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 08🛠️ | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 09  | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

- 10🛠️ | ❌💥 | ❌
  - The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded

**Conclusion:**

If the intermediate result is distributed among the threads in a warp, then each thread keeps a private copy of the intermediate result. This means that the algorithm actually performs a `k / 32`-nearest neighbors search (as all the expected `k` nearest neighbors can be, theoretically, processed by a single thread).

The language model struggles with tracking the code semantics. It is prone to trivial mistakes such as creating synchronization barriers that are not reached by all synchronizing threads. It makes these mistakes even when the semantically inconsistent code fits in very small contexts (e.g., a few lines).

The language model also struggles with the semantics of the task at hand. It actively discards a portion of the input data (candidates) that is not supposed to be discarded.

All solutions use the AoS format for the intermediate result.

All solutions use a sorted array to represent the intermediate result. This is typically less efficient for low numbers of candidates stored in a local copy of the intermediate result. However, it still allows for efficient pre-filtering of the candidates, which requires just a single comparison per candidate as opposed to `k` comparisons for the unsorted structure. None of the solutions use unsorted or heap-based structures to represent the intermediate result.

## Prompt 08

**Requested/suggested features:**

For details, see [knn08.prompt.md](../prompts/knn08.prompt.md).

- Each query is computed by a single thread warp (32 threads)
- A private copy of the intermediate result of the k nearest neighbors search for each query is distributed among the threads in the warp
- A batch of input data points stored in shared memory by the thread block
- Each thread warp computes the distances from the input data points in the shared memory
- For each query, a shared buffer of `k` candidates is defined
- The candidates are inserted in the buffer if they are closer than `max_distance`; `max_distance` is updated after each merge of the buffer with the intermediate result
- Insertion is directed by a variable storing the current size of the buffer
- Positions in the buffer are assigned using ballot instructions and updating the variable that indicates the current size of the buffer
- The buffer is merged with the intermediate result when it is full using the Bitonic sort algorithm (which is outlined in the prompt)
  - The Bitonic sort is performed using warp-shuffle instructions and local swaps in registers

**Common features:**

- User-defined `pair` type (`Candidate`)
- The intermediate result is shared among threads in a warp, stored in registers distributed among the warp threads
- The intermediate result is represented by a sorted array (AoS)

**Special features per solution:**

<!-- |    kNN08     | ❌💥 | ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥 | ❌💥 | ❌💥 | -->
<!-- |    kNN08     | ❌💥 | ❌💥🛠️| ❌ | ❌💥🛠️| ❌💥🛠️| ❌💥 | ❌💥🛠️| ❌💥 | ❌  | ❌💥 | -->

- 01  | ❌💥 | ❌💥
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Warp synchronization in a place that is not guaranteed to be executed by all threads in the warp
  - Correct balloting; offsets determined by prefix sums, the buffer size updated using `__popc` on the whole ballot
  - Forgets that each threads keeps a private copy of the buffer size, only the lane 0 updates its copy
    - And only if its candidate passes the pre-filtering

- 02🛠️ | ❌💥 | ❌💥
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Each thread computes the position of the candidate in the buffer using the ballot instruction
  - After computing the position, the thread discards the assigned position and re-computes it using the `atomicAdd` instruction
  - Uses a shared variable to store the current size of the buffer

- 03  | ❌💥 | ❌
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Ignores the direction of using ballot instructions, computes the position of the candidate in the buffer using the `atomicAdd` instruction
  - Uses a shared variable to store the current size of the buffer

- 04🛠️ | ❌💥 | ❌💥
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Correct balloting; offsets determined by prefix sums, the buffer size updated using `__popc` on the whole ballot
  - Forgets that each threads keeps a private copy of the buffer size, only the lane 0 updates its copy
  - Warp synchronization in a place that is not guaranteed to be executed by all threads in the warp

- 05🛠️ | ❌💥 | ❌💥
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Ignores the direction of using ballot instructions, computes the position of the candidate in the buffer using the `atomicAdd` instruction
  - Uses a shared variable to store the current size of the buffer

- 06  | ❌💥 | ❌💥
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Correct balloting; offsets determined by prefix sums, the buffer size updated using `__popc` on the whole ballot

- 07🛠️ | ❌💥 | ❌💥
  - Warp synchronization in a place that is not guaranteed to be executed by all threads in the warp
  - Each thread determines whether a merge is needed according to the position of its candidate in the buffer
    - Only the threads whose candidates pass the pre-filtering are active
    - Only the threads whose candidates do not fit in the buffer are active
    - Then, the merge process assumes that all threads are active
    - This makes no sense
  - The private copy of the buffer size is updated according to the position of the candidate in the buffer
    - Each thread sees a different value of the buffer size

- 08  | ❌💥 | ❌💥
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Ignores the direction of using ballot instructions, computes the position of the candidate in the buffer using the `atomicAdd` instruction
  - Uses a shared variable to store the current size of the buffer

- 09  | ❌💥 | ❌
  - When requesting a position in the buffer, if the candidate does not fit, it is completely discarded
  - Ignores the direction of using ballot instructions, computes the position of the candidate in the buffer using the `atomicAdd` instruction
  - Uses a shared variable to store the current size of the buffer
  - Determines that insertion sort is more appropriate than Bitonic sort (ignores the direction of the prompt)

- 10  | ❌💥 | ❌💥
  - Warp synchronization in a place that is not guaranteed to be executed by all threads in the warp
  - Ignores the direction of using ballot instructions, computes the position of the candidate in the buffer using the `atomicAdd` instruction
  - Uses a shared variable to store the current size of the buffer
  - if the candidate does not fit in the buffer, it is written out of bounds
  - When the buffer is full, a selected lane of the warp performs a parallel Bitonic sort algorithm
    - This makes no sense

**Conclusion:**

The language model struggles with tracking the code semantics. It is prone to trivial mistakes such as creating synchronization barriers that are not reached by all synchronizing threads or mistaking private variables for shared variables (or vice versa). It makes these mistakes (e.g., a synchronization barrier in a conditional branch) even when the semantically inconsistent code fits in very small contexts (e.g., a few lines).

The language model is inconsistent with the use of warp-wide primitives, such as `__ballot`. Sometimes, it shows a good understanding of the semantics of these primitives, while at other times, it uses them in a way that makes no sense to an experienced programmer.

The language model also struggles with the semantics of the task at hand. It actively discards a portion of the input data (candidates) that is not supposed to be discarded.

Some solutions ignore the directions of the prompt. For example, the prompt directs the model to use warp ballot instructions to assign positions in the buffer, but the model chooses to use `atomicAdd` instead. Sometimes, these choices are made as the model recognizes that following the prompt would be more challenging than using a simpler approach. Other times (e.g., in the case of the `atomicAdd` instruction), the model simply ignores the prompt and uses a more familiar approach even though both approaches are applicable to the used design.

All solutions use the AoS format for the intermediate result.

All solutions use a sorted array to represent the intermediate result. This is typically less efficient for low numbers of candidates stored in a local copy of the intermediate result. However, it still allows for efficient pre-filtering of the candidates, which requires just a single comparison per candidate as opposed to `k` comparisons for the unsorted structure. None of the solutions use unsorted or heap-based structures to represent the intermediate result.
