# Histogram reviews

This file contains an in-depth review of the single-response generated solutions.


## Correctness table

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Histogram01  | ✅  |✅🛠️|✅🛠️|✅🛠️| ✅ | ✅ | ✅  | ✅  | ✅ | ✅ |
| Histogram02  |✅🛠️| ✅  | ✅ | ✅  | ✅ | ✅ | ✅  |✅🛠️| ✅ |✅🛠️|
| Histogram03  | ✅  | ✅ |✅🛠️| ✅  | ✅ | ✅ | ✅  | ✅  | ✅ | ✅ |
| Histogram04  | ✅  | ✅ | ✅  | ✅ | ✅ | ✅  | ✅ | ✅  | ✅  | ✅ |
| Histogram05  | ✅  |✅🛠️| ✅  | ✅ | ✅ | ✅  | ✅ | ✅  |✅🛠️| ✅ |
| Histogram06  | ✅  | ✅ |✅🛠️|✅🛠️| ✅ | ✅  | ❌ | ✅  | ✅  | ✅ |
| Histogram07  | ✅  | ✅ | ✅  | ✅ | ✅ | ✅  | ✅ | ✅  | ✅  | ✅ |

✅ – Correct solution (compiled successfully and returned the correct GoL grid).

❌ – Compiled and ran without a runtime error but returned incorrect results.

❌💥 – Compiled but crashed during execution.

❌⚙️ – Did not compile.

🛠️ – Indicator denotes the source code a small edit to make it compile (this mark is added alongside one of the above). The erroneous line(s) was/were commented on and prefixed with `/// @FIXED` comment.

All compilation errors that need to be fixed were caused by extra `extern "C"` in the run function declaration (which prevents linking with main C++ code).

---

## Histgram01 - no optimization hints

All solutions automatically introduced shared memory optimization without any hints to do so. Furthermore, all solutions implemented a loop over the input which makes them ready to process multiple chars per thread (the second suggested optimization); however, all of them (except number `08`) use a grid size equal to the input, so each thread processes only one char (the loop performs exactly one iteration). Furthermore, all solutions used `256` threads per block which is suboptimal from the perspective of shared memory utilization (the more values are aggregated in the local copy the better).

All solutions properly initialize the shared memory to zeroes and concurrently add the local histogram to the global one at the end. The three phases of the histogram kernel were correctly separated by block-wise barriers (`syncthreads`).

**Variations:**
- Versions `03` and `10` contain a small optimization that checks if the count is greater than zero before adding it to the global memory. This optimization reduces the number of atomic operations and improves performance (roughly `1.5x` speedup for the lorem ipsum dataset).
- Version `08` limits the number of blocks to `65536` which for 1G input leads to `256` chars per thread. This solution is almost optimal for large inputs as it employs coalesced loading (since threads iterate over the input with a stride equal to the number of threads, thus adjacent threads load adjacent input chars).

The ten generated versions exhibit minor differences in the code like whether a `for` or `while` loop is used to iterate over the input or whether all corner cases (like an empty input) are handled properly. However, these variations have no impact on the performance.


## Histogram02 and 03 - shared memory optimization hint

Histogram02 suggests using shared memory, and histogram03 explains that a local copy of the histogram should be stored there and then merged with a global copy.

All 10 versions of Histogram02 and most of Histogram03 versions use the most basic shared memory optimization as most versions from Histogram01. It is possible, that a vague hint about shared memory hindered LLM's reasoning process in some way as it did not try to apply any additional optimizations (not even the simple `if`-test for the final atomic add operation).

**Histogram03 variations:**
- Version `07` uses the same if-optimization as versions `03` and `10` from Histogram01.
- Version `10` limits the number of thread blocks to `1024`, which has the same effect as in version `08` from Histogram01.


## Histogram04 - shared memory and multiple chars per thread optimization hints

All versions properly introduce the `itemsPerThread` constant and use it to iterate over the input. There were two versions of how the iteration was designed.
Some versions used a continuous block of chars (stride 1), others used a block size as the stride for the loop (which has a better chance for coalesced memory access).
The following table overviews the differences between the versions (adding measured times for H100 GPU using the lorem ipsum dataset):

| Test Case        | 01       | 02       | 03       | 04     | 05      | 06       | 07      | 08       | 09       | 10      |
| ---------------- |:--------:|:--------:|:--------:|:------:|:-------:|:--------:|:-------:|:--------:|:--------:|:-------:|
| items per thread | 16       | 8        | 8        | 16     | 128     | 32       | 16      | 128      | 8        | 32      |
| stride           | 1        | blockDim | blockDim | 1      | 1       | blockDim | 1       | blockDim | blockDim | 1       |
| ref. time        | 13.26 ms | 2.85 ms  | 2.85 ms  | 1.5 ms | 5.89 ms | 2.22 ms  | 1.16 ms | 2.2 ms   | 2.86 ms  | 20.5 ms |

**Variations:**

- Version `01` Uses only the first thread to merge the local histogram with the global one. This is unnecessarily suboptimal (as H threads could do this concurrently where H is the histogram range).
- Version `07` has two differences: The `itemsPerThread` is declared as a constant, but in the runner function only, and it is passed to the kernel as an argument. Additionally, it places the if-condition (whether a bin is > 0) to skip unnecessary global atomic adds in the final merge (same as version `08` from Histogram01).
- Version `10` was quite different as it created a local copy of the histogram for each thread. To ensure that, both the shared memory and the per-thread copy were allocated statically for 256 bins (the maximal histogram range). Since 256 is a large number, it is not likely that the local copy will fit the registers (and it uses input values for indexing), so this structure is likely to be placed in *local memory* (i.e., global memory cached in L1). The final merge then uses two steps (first, local copies are merged to the shared memory, then the shared memory is merged to the global memory). This version also uses the if-condition optimization for the atomic adds (at both levels).


## Histogram05 - reducing shared memory collisions

All versions of Histogram05 correctly allocate multiple copies of histogram in shared memory. The generated versions can be divided into three groups based on the selected approach and number of copies:
- Versions `01`, `03`, `06`, `07`, `08`, and `10` used exactly `8` copies which were placed in shared memory so that each copy uses a continuous block (i.e., will not prevent bank conflicts). Each thread selects its copy as `threadIdx.x % 8`.
- Version `05` - uses the same approach, but it allocates 32 copies (which is the number of banks).
- Versions `02`, `04`, and `09` use as many copies as there were warps in the block (incidentally using 256 threads per block also leads to 8 copies). Each thread selects its copy as `threadIdx.x / warpSize` which is even less practical (threads of a warp need different copies in different banks).

Additionally, version `02` used a local histogram per thread (similar to Histogram04, version `10`).

The versions used different numbers of items per thread so even the versions with the same approach can have different performances.


## Histogram06

All versions allocated 32 copies of the histogram in shared memory (as requested in the prompt); however, only 4 of 10 versions chose the proper layout to avoid bank conflicts.

- Versions `01`, `03`, `04`, and `09` - used the requested solution (32 copies, each copy per bank), versions `03` and `04` also employed explicit loop unroll pragma
- Versions `02` and `10` allocated 32 copies that were stored sequentially and they used +1 item padding for each histogram copy. This implementation trick is often used to easily resolve bank conflicts if the threads in a warp use identical access pattern; however, since the data access pattern is purely data-driven in the histogram kernel, this padding helps only in rare cases where the input comprised mainly one (or very few different) character(s).
- Version `05` used 32 copies, but with a continuous layout (with bank conflicts)
- Version `06` ❌ used 32 copies, but with continuous layout (with bank conflicts) and is not computing the right results due to a mistake in indexing
- Version `07` used 32 copies, but with a continuous layout (with bank conflicts), and the copy index is selected by warpId (which is worse than the selection by line ID)
- Version `08` used 32 copies, but with a continuous layout (with bank conflicts) and it uses padding to align histogram copy size to a multiple of 32 (which is completely unnecessary in this case)

The versions used different numbers of items per thread so even the versions with the same approach can have different performances.


## Histogram07

All versions used the requested approach (32 histogram copies with a stridden layout that prevents bank conflicts). The only variation was observed in version `04` which also stored a local copy per thread (as Histogram04, version `10`). This approach hindered the performance (by an order of magnitude).

The versions used different numbers of items per thread so even the versions with the same approach can have different performances.

To evaluate the effect of the tutoring for bank conflict prevention, here is a table summarizing the number of copies and whether they used the proper layout to avoid bank conflicts (✅) or not (❌/💥, the latter indicates it uses a copy per warp which is slightly worse). The number of copies is filled in only when it differs from 32.

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Histogram05  | 8❌ | 8💥| 8❌ | 8💥 | ❌ | 8❌ | 8❌ | 8❌ | 8💥| 8❌ |
| Histogram06  | ✅  | ❌ | ✅ | ✅ | ❌ | ❌ | 💥 | ❌  | ✅  | ❌ |
| Histogram07  | ✅  | ✅ | ✅  | ✅ | ✅ | ✅  | ✅ | ✅  | ✅  | ✅ |

The table indicates that the LLM is capable of following the instructions on how to divide the copies into the banks, but it is not able to deduce this optimization on its own.
