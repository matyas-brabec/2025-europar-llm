# Game of Life

This file contains an in-depth review of the "one-shot" generated solutions.

## Correctness table

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GoL_01       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| GoL_02       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ❌   | ❌   | ✅   | ✅   |
| GoL_02_tiled | ❌   | ❌   | ❌   | ❌   | ❌   | ✅   | ❌   | ❌   | ❌   | ❌   |
| GoL_03       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ❌   |
| GoL_04       | ✅   | ❌   | ✅   | ✅   | ✅   | ✅   | ❌   | ✅   | ✅   | ✅   |
| GoL_05       | ✅   | ❌   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ❌   | ✅   |
| GoL_06       | ✅   | ❌   | ✅   | ✅   | ✅🛠️ | ✅   | ✅   | ✅   | ✅   | ✅   |

✅ – Correct solution (compiled successfully and returned the correct GoL grid).

❌ – Compiled and ran without a runtime error but returned incorrect results.

❌💥 – Compiled but crashed during execution.

❌⚙️ – Did not compile.

---

🛠️ –  Indicator denotes the source code a small edit to make it compile (this mark is added alongside one of the above). The erroneous line(s) was/were commented and prefixed with `/// @FIXED` comment.

---

*Note: Many of the solutions used `extern "C"`. Since we did not specify its usage, we opted to remove it from the code, and we do not consider that an error.*

## Prompt 01

**Common features:**

- `byte` per cell
- shared memory
- block size = `1024`
- 2D block
- simple `if` for new state
- solution with `__restrict__` - `01`, `04`, `05`, `06`, `07`, `09`, `10`
  
**Special features per solution:**

- 03 ✅
  - ❗smarter `if` for new state - `(alive_neighbors == 3) || (current && (alive_neighbors == 2))`
- 04 ✅
  - shared memory ❗dynamically allocated
- 05 ✅
  - shared memory ❗dynamically allocated
- 06 ✅
  - ❗ Uses the assumption that the grid is always a power of `2` and greater than `512`. When defining the CUDA grid, it uses `grid_dimensions / TILE_DIM`, unlike most of the solutions that use `(grid_dimensions + blockDim.x - 1) / blockDim.x` or similar.
- 07 ✅
  - shared memory ❗dynamically allocated
  - ❗nice codestyle - using `const` for most of the variables
- 08 ✅
  - ❗ Uses the assumption that the grid is always a power of `2` and greater than `512`. When defining the CUDA grid, it uses `grid_dimensions / TILE_WIDTH`, unlike most of the solutions that use `(grid_dimensions + blockDim.x - 1) / blockDim.x` or similar.
- 09 ✅
  - ❗concise solution even if shared memory has been used (in other cases the code is polluted with the `shm` initialization)
  - ❗smarter `if` for new state - `(alive_neighbors == 3) || (current && (alive_neighbors == 2))`

**Summary:**  All solutions opted for a `byte` encoding of a cell, and none used bit encoding, even though the latter is likely the best option when optimizing for maximum performance. The prompt even specifies that only the kernel computing the Game of Life is measured, while all other code in the `run_game_of_life` function is ignored in terms of execution time.

All solutions utilize shared memory. As demonstrated by other sources, shared memory provides no significant speedup in cases where threads access memory in an orderly manner. However, shared memory is generally associated with GPU optimization and is often the first technique CUDA programmers attempt. Most solutions used statically allocated shared memory, though three solutions employed dynamically allocated shared memory. In this case, dynamic allocation is unnecessary, especially given that all solutions hardcoded the block size into the source.

Regarding block size and shape, all solutions correctly use the maximum block size of 1024. Given the small amount of work per thread, this seems to be a reasonable choice for optimal performance. Notably, all solutions used a 2D block layout rather than a long "linear chain" of threads. This is ideal, as it maximizes data reuse in cache. However, we suspect this choice was a mere consequence of using shared memory, as a square configuration is the only sensible option in that context.

Furthermore, only two solutions utilized the assumption that the grid is always a power of `2` and greater than `512`. This is reflected in their use of `grid_dimensions / TILE_WIDTH` when computing CUDA grid dimensions. While this is not a significant issue—since the alternative approach is safer—it demonstrates that some instances recognized the special condition and simplified the code accordingly.

None of the solutions heavily optimized the computation of a new cell state. All but two solutions used a classical `if` statement, while the remaining two employed a slightly improved version: `(alive_neighbors == 3) || (current && (alive_neighbors == 2))`. However, none utilized a lookup table, which is a well-known optimization technique for the Game of Life. Lookup tables are particularly beneficial as they reduce branching, which is notoriously inefficient on GPUs.

On a positive note, all solutions compiled without issues, adhered to the requested interface, and produced correct results. The correctness in all cases is somewhat surprising, as using shared memory requires careful consideration of indexing and is a common pitfall for human programmers. We attribute this success to the widespread use of shared memory in this rectangular manner, which likely provided the LLM with lots of training data to learn from.

**Conclusion:** All solutions were far from optimal, performing very close to the naive baseline we provided. Additionally, we observed that they were all highly similar, differing only in small details.

## Prompt 02

**Common Features:**  

- **Block size** is `256`, with varying dimensions. The chosen block size is sensible, as testing has shown it to be a good choice. However, the shape varied significantly.  
  - **Optimal choice:** A `32×8` block - one warp accesses memory in an aligned fashion while the rectangular shape improves caching efficiency.  
    - *Used in:* `01`, `10`
  - **Less optimal:** A `16×16` block - it disrupts aligned memory access.  
    - *Used in:* `02`,`03`,`05`,`06`,`07`
  - **Worst choice:** A linear `256×1` block - no loaded words are reused within the block, making caching almost useless.  
    - *Used in:* `04`,`08`,`09`
- Each CUDA thread processes one `64`-bit word.
- Most solutions use the more efficient `if` condition (e.g., `(nb == 3) || (cell && (nb == 2))`), but none implement smarter techniques like lookup tables.
- All solutions use bit masks and count neighbors bit by bit.  
- Most solutions utilize `#pragma unroll`.
- Two main variants of handling edge cases appeared:  
  - One approach handled edge cases inside the loop.  
  - The other moved edge case handling outside the loop, which is more efficient as it eliminates two conditional checks each iteration.
  - **Unusual Variant:** Two solutions attempted a different approach (`05` and `06`): removing unused bits from the three left/right words and replacing them with bits from the respective central words. This method is quite clever, as it eliminates the need for special-case handling in the main loop. However, only one out of the two succeeded.

**Special features per solution:**

- 02 ✅
  - simple `if` for new state
- 05 ✅
  - ❗creative edge case handling (as explained above)
- 07 ❌
  - ❗tried creative edge case handling (as explained above)
  - the error is not immediately apparent—likely due to incorrect indexing
- 08 ❌  
  - ❗ Used a `C` array (i.e., `std::uint64_t nbr[3][3];`) but incorrectly applied it. All words were used in the computation of all bits, which is incorrect.  
  - Good code style—made use of `const`.
- 10 ✅
  - simple `if` for new state

**Conclusion:** The solutions performed as expected overall. Most solutions were similar to each other and did not use any clever optimizations beyond what was suggested in the hints. However, there were two exceptions that took an unusual approach. Unfortunately, one of them failed.

## Prompt 03

This prompt provided hints that turned out to be unnecessary, as all possible issues were addressed by the LLM in the previous prompt. Specifically:

1. Each CUDA thread should handle a single `std::uint64_t`.
2. To read a cell’s neighborhood, use bit masks.
3. The 0th and 63rd bits of each word require special handling.

None of the previous solutions attempted to handle one bit per thread (hint 1). Hints 2 and 3 were related to correcting the solution, not optimizing performance. Since nearly all of the solutions above were correct, these hints were not needed. As a result, the solutions are similar to those from Prompt 2 and share the common features mentioned above.

The exception is that none of the solutions tried the "unusual" approach described earlier. It seems that providing the LLM with more requirements might limit its ability to come up with creative solutions, but we cannot be certain based on only 10 samples per prompt.

Furthermore, none of the solutions used the optimal block dimension of `32x8`. Half of the solutions used the `16x16` block size (`03`, `05`, `06`, `09`, `10`), while the other half used the suboptimal linear `256x1` approach (`01`, `02`, `04`, `07`, `08`). This result is surprising.

**Special features per solution:**

- 02 ✅
  - ❗nicer codestyle - solution defined  `__device__` function `get_bit(...)`
- 08 ✅
  - simple `if` for new state
- 09 ✅
  - simple `if` for new state
- 10 ❌
  - general idea is correct - probably a bad indexing along the way
  - ❗nice codestyle - solution uses a lambda `get_bit` to make the code cleaner (it could have been a `__device__` function instead but this is way as well)

**Conclusion:** The solutions again performed as expected and were very similar to each other. Compared to those in Prompt 2, these solutions showed less variation, and the LLM demonstrated less creativity—likely due to the additional hints, which may have acted as constraints.

## Prompt 04

In the promtt 4 there eas only one additional hint and that is the usage of `popc`. Here we expect quite concreate solution and we leftvery little space for the creativity.

**Common Features:**

We were surprised to see that, even when given explicit instructions to use the `popc` function for counting bits, most solutions ignored it entirely. Only cases `06`, `08` and `09` included a call to the function in the code. Interestingly, solutions `01`, `05`, and `07❌` mentioned the use of `popc` in comments but never actually implemented it in the code.  

All other solutions followed the same techniques discussed previously in Prompt 2 and Prompt 3. As a result, we will not summarize these properties again and will instead focus on individual peculiarities in this test case.  

**Special features per solution:**

- 01 ✅
  - Unusual Variant - further discussed in Prompt 2 section.
  - `... we rely on the __popc intrinsic ...` in the comment but never used it
- 02 ❌
  - ❗ attempt to use a look up table instead of simple `if`
  - The first problem was in the lookup table logic. In principle, there are two correct ways to implement it. Either you index the table like this `(cell_is_alive * 9) + number_of_neighbors` assuming the table is defined in the standard way:
  ```c++
  const int lut[18] = {0,0,0,1,0,0,0,0,0,
                       0,0,1,1,0,0,0,0,0};
  ```
  Alternatively, there is an optimization to replace `*9` with `<< 3` (i.e., `*8`). However, in that case, the second row of the table must be shifted, like this:
  ```c++
  const int lut[18] = {0,0,0,1,0,0,0,0,0,
                         0,1,1,0,0,0,0,0,0};
  ```
  The solution apparently attempted to use the optimized version (as suggested by the "optical shift" in the table definition and the presence of `<< 3`) but failed to remove one zero from the beginning of the second row.

  When we attempted to fix this issue, another mistake became apparent—likely an indexing error.

- 05 ✅
  - `The __popc intrinsic is mentioned in the prompt to help with bit-counting.`
  - The solution acknowledges the hint but ignores it—at least it does not falsely claim at the beginning that it uses `popc`.
- 06 ✅
  - ❗ The solution uses `popc` to count neighbors; however, it does so in a suboptimal way. It still extracts bits one by one, but instead of adding them together using `+`, it combines them into a single `64`-bit word (using `or`). Then, it calls `popc` to count the bits. In this version, no instructions are saved.  
- 07 ❌
  - `The __popc intrinsic was considered for fast bit–counting but`
  - Solution considered the usage of `popc` and decided not to use it.
  - The code style is unusual—at some points, there are three nested `if` statements, making the solution hard to read. Additionally, there is an oddly nested `if` statement that could have been written as `else if`.  
  ```c++
    if (...) {
      ...
    } else {
        if (...) { ... }
    }
  ```
  - The mistake is difficult to identify—once again, it appears to be a bad indexing issue, as the overall logic seems correct. However, due to poor code clarity, it is hard to be certain.  
- 08 ✅
  - ❗ The solution uses `popc` but has the same issue as in case `06`.
- 09 ✅
  - ❗ The solution uses `popc` but has the same issue as in case `06` and `08`.

**Conclusion:** Overall, these solutions were disappointing—only three actually used `popc`, but in an unexpected and incorrect way. Given these results, it might be interesting to expand the explanation of how `popc` was intended to be used in an additional prompt.

However, we did see one solution that attempted to use a lookup table, unfortunately with a semantic error. It would also be interesting to explore hinting at that approach in additional prompts.

## Prompt 05

In this prompt, the hint provided was more of a suggestion than a precise instruction. We asked whether the LLM could devise a method to compute multiple cells at once, pointing in the direction of full-adder technique. As a result, the solutions varied significantly in both approach and performance. Compared to previous prompts, we observed three novel approaches alongside several previously discussed methods. Below is a brief overview:

- **Full-adder approach:** Multiple solutions implemented the optimal full-adder logic strategy, which was the technique we aimed for in the next prompt.
  - `01`, `05`, `07`
- **Unusual approach (see Prompt 2):** A few solutions attempted the unconventional approach described in Prompt 2. Among them were two solutions that failed to deliver correct output—both due to incorrectly combining the left/right words with the center words.
  - `02❌`, `04`, `09❌`
- **Simple bit-by-bit processing:** Some solutions ignored the hint entirely and continued using the basic bit-masking technique from previous prompts.
  - `03`, `06`
- **Using a word like a vector register:** One solution attempted to treat a `64`-bit word like a vector register, reserving `4` bits per cell and accumulating its neighborhood in a vector-like fashion. This idea is quite clever and unusual for Conway’s Game of Life. However, the implementation turned out to be inefficient, with multiple areas where optimization could have improved performance. Still, the concept was interesting.
  - `08`
- **Large lookup table:** One solution used a lookup table, but not in the classical way typically seen in Game of Life optimizations. Instead, it allocated a 512-byte table in constant memory, encoding all possible outcomes for `3×3` neighborhoods. While unconventional, the execution was again inefficient. A significant amount of work was spent composing the `3×3` neighborhood into an index for the table—only to eliminate one or two `if` statements. Performance could likely be improved with a more efficient `3×3` composition method and compression of the large `512`-byte table to minimize redundancy.
  - `10`

**Special features per solution:**

- 02 ❌
  - ❗ Solution probably intended to use the unusual approach described in Prompt 2 but failed to correctly combine the left/right words with the center words.  
- 09 ❌
  - ❗ Solution probably intended to use the unusual approach described in Prompt 2 but failed to correctly combine the left/right words with the center words.  

**Conclusion:** We observed a greater variety of solutions when the LLM was asked an open-ended question rather than given direct instructions. In addition to cases where it successfully discovered the optimal solution or failed to use the suggestion altogether, we also saw novel approaches that we had not anticipated. This is particularly interesting because it suggests that LLMs can be valuable not only for generating correct and fully functional code but also for providing innovative ideas. Even when these ideas are not executed to their full potential, they can still serve as inspiration for domain experts to refine and optimize further.

## Prompt 06

In the final prompt, we provided explicit instructions to use the full-adder technique along with a brief explanation. As a result, the variety of solutions was low, with most correctly implementing the full-adder—though there were a few exceptions (see below).

This time, the correct solutions utilized a `__device__` functions to simplify the code. In most cases, the implementations were both readable and performant.

**Special features per solution:**

- 01 ✅
  - ❗ Ideal block size: `32x8`
  - The fastest solution.
- 02 ❌
  - ❗ The main idea appears to be correct. However, the code is messy, making it difficult to identify the error.
- 03 ✅
  - ❗ No vectorization - bit by bit-masking solution. ("Unusual" approach as described in Prompt 2.)
- 04 ✅
  - ❗ No vectorization - bit by bit-masking solution. ("Unusual" approach as described in Prompt 2.)  
- 05 ✅🛠️
  - header of lambda did not compile: `[=, input] (int r, int w)`.
- 06 ✅
  - Less clear and lengthy solution compared to others.
- 07 ✅
  - ❗ No vectorization - bit by bit-masking solution. ("Unusual" approach as described in Prompt 2.)
- 08 ✅
  - ❗  Used a full-adder, but not the optimal version. Allocated four bits to count the neighborhood, though only three are necessary—any count higher than eight automatically results in a dead cell. Additionally, the new state determination was suboptimal: a `for` loop iterates over all 64 bits to determine the final state of each cell, whereas a vectorized approach would be more efficient. As a result, this solution is approximately `5×` slower than the optimal implementation.

**Conclusion:** In most cases, the LLM correctly understood the intent behind our prompt and implemented the full-adder technique as expected. Compared to previous prompts, we observed only two types of solutions: *the optimal full-adder approach* and, in three cases, simple *bit-by-bit masking*. Notably, none of the solutions attempted the more creative or unexpected approaches we had seen in earlier prompt. This suggests that providing precise instructions increases correctness but may limit the LLM’s ability to explore novel or unconventional strategies.

## Prompt 02 Tiled

We decided to take a detour from the linear addition hints and experiment with an alternative memory layout, using 8x8-bit tiles stored in a single `uint64_t` word. Unlike previous hints, this is not a standard way to solve the Game of Life, but it offers significant speedup, as demonstrated by our reference solution. This test, therefore, provided an opportunity to evaluate the LLM’s reasoning skills when faced with an unseen problem. Unfortunately, as the results show, only one solution turned out to be correct, while the others failed. However, based on the comments in the code, the LLM did understand the problem and the general steps required to solve it. We attribute these failures to the overwhelming number of edge cases that arise when working with tiles rather than rows. While all the solutions followed the correct general approach, a small mistake—such as an incorrect constant, index, or loop boundary—was enough to cause failure.

There were three main types of attempted solutions:

- **Row by row**: The approach involved breaking the tile down into eight `8`-bit rows and processing them one by one using two nested loops. Special handling was needed for the first and last row, as well as for the `0`th and `7`th bits within each row.
  - *Used in:* `01❌`, `04❌`, `06✅`, `08❌`
- **10x10 matrix:** The approach extracted bits from the appropriate words and stored them in a temporary `10x10` `byte` matrix, which included the `8x8` cells and a halo region. After building the matrix, computing the next state for all `64` cells was straightforward.
  - *Used in:* `03❌`, `05❌`, `07❌`, `10❌`
- **Bit array:** The last approach was similar to the matrix method but packed the bits into an integer array rather than using a `byte` matrix. While the computation for the next state in this case was a bit trickier, it was still manageable overall.
  - *Used in:* `02❌`, `09❌`,

**Special features per solution:**

- 06 ✅
  - Row by row based
  - A very long solution (comparatively) with many edge cases (tree-level nesting at many points)

Conclusion: Although all but one solution failed, all of them understood the problem at hand. Given the complexity of the task, this cannot be considered a complete failure. However, none of the three approaches taken were optimal. The use of tiles presents an opportunity to count neighbors in a single instruction (using `popc`) for the inner cells, but unfortunately, all solutions resorted to bit-by-bit handling instead.
