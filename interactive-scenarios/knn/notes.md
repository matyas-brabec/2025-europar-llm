# kNN Interactive Scenario Notes

## Milestone 01

**First prompt:** The code won't compile. error on line 39: identifier "FLT_MAX" is undefined

**Second prompt:** The language models makes very conservative and straight to the point change: includes `<limits>` header and changes the `FLT_MAX` constant to `std::numeric_limits<float>::max()`. The code won't compile once again. This time, the error is on line 36: "k" is not a constant expression (the line is `std::pair<int, float> knn[k];`).

**Third prompt:** The language model makes a less conservative change: it templates the kernel and specializes it for each expected `k` value. This is a common practice in high-performance code as it allows using runtime values as compile-time constants (and thus, even as array sizes). Any mention of the `k` variable throughout the kernel code is replaced with the template parameter. Furthermore, the language model adds some more comments to the code and revises the already existing ones. It also adds explicit unroll directives to all loops that use the new template parameter as a loop bound. Other than that, the code is unchanged. The code compiles and runs, but it is extremely slow as the kernel uses linear insertion for each new candidate. For `k=1024`, the kernel times out; for `k=32`, it is almost more than 7 times slower than the `naive` implementation that uses a thread-local binary heap for each query and almost 70 times slower than the reference solution for the static knn08 prompt.

The solution uses the "Array of Structures" (AoS) format instead of the "Structure of Arrays" (SoA) format. In the reference state-of-the-art implementation, the SoA format performs better than the AoS format. Therefore, we use the prepared hint that suggest this change before.

**Fourth prompt:** The language model makes the expected change. Interestingly, it also changes the insertion algorithm; it is still linear, however in the previous version, it first found the correct position for the new candidate and then pushed the higher elements to the right. Now, it stores the new candidate at the end of the array, replacing the highest element, and bubbles it up to the correct position. On `k=32`, this performs almost 10 times better than the previous version (better than the `naive` implementation).

As the language model provides the expected solution, we move on to the next milestone.

## Milestone 02

**Fifth prompt:** The language model has made a radical redesign of the solution. In the new solution, each thread stores `k / 32` elements of the intermediate result in a local array. The solution compiles and runs without errors. And the code contains the expected changes and seemingly correct logic. However, since each thread stores just `k / 32` elements and processes a different part of the input data, the final result is not correct.

We make a new prompt that tells this oversight to the language model: Since each thread stores just `k / 32` elements and processes a different part of the input data without any communication between the threads, the final result is not correct.

**Sixth prompt:** The revised solution won't compile. error on line 216: identifier "K" is undefined (the line is `size_t shm_per_warp = K * sizeof(int) + K * sizeof(float) +`)

**Seventh prompt:** The language model makes a very minor, expected change: it replaces `K` with `k` and slightly modifies some related comments. The rest of the code is unchanged. The code compiles and runs, but it is extremely inefficient. The lane 0 of each warp performs a serial merge of the result for each tile (batch) of the input data. However, for the next milestone, making this code efficient is not a priority as the next prepared hint already addresses this issue.

We move on to the next milestone.

## Milestone 03

**Eighth prompt:** The language model makes quite conservative changes with the motivation to replace the serial merge with a parallel one using the Bitonic Sort algorithm. Most of the code is unchanged. The code compiles and runs without errors. However, the performance is over 2 times worse than the `naive` implementation. This is worse than the last solution in the first milestone. The performance would benefit from pre-filtering the input data and using the distance of the last element in the intermediate result as a threshold for the candidate's inclusion to the temporary buffer that is merged with the intermediate result --- thus, we use the second prepared hint for this milestone.

**Ninth prompt:** The language model makes a minor change to the code following the suggestion. The performance slightly improves, but it is still roughly 2 times worse than the `naive` implementation. However, we can move on to the next milestone that already addresses the present performance issue.

## Milestone 04

**Tenth prompt:** The language model redesigns the relevant parts of the code and makes some minor changes to the rest of the code. The code compiles and runs with a cuda error: "an illegal memory access was encountered". We tell the language model about this error and ask it to fix it. If it cannot, we will investigate the source of the error.

**Eleventh prompt:** The language model identifies the source of the error and fixes it. It adds bounds checks to the code that ensure that elements with `i >= k` are not used after the merging as we are interested only in the first `k` elements. This change should not affect the functionality of the code. However, the kernel provides a wrong result due to an unrelated issue. When enqueueing a new candidate, if its deduced position is out of bounds, the candidate is discarded.

We make a new prompt that addresses this issue: After computing the candidate's position using `atomicAdd`, if `pos < K` fails, the candidate is mistakenly discarded. No candidates should be discarded except for those farther than the maximum distance in the intermediate result.

**Twelfth prompt:** The language model changes the relevant part of the code to loop until the candidate is inserted into the intermediate result. However, the kernel contains a synchronization point inside the loop that is reached only by the lanes whose candidates' positions are out of bounds. The kernel has to be forcefully killed as it deadlocks.

We make a new prompt that addresses this issue: The kernel contains a synchronization point inside the loop that is reached only by the lanes whose candidates' positions are out of bounds. This causes the kernel to deadlock.

**Thirteenth prompt:** The language model rerolls some of the changes associated with the introduction of the shared buffer. The new solution reintroduces local buffers that are filled with the candidates in the tile/batch of the input data. This still satisfies the requirement of introducing a shared buffer (even though it is not stored in shared memory). However, the code does not compile. There is a typo on line 119 (the line is `warpSize * 8 * (sizeof(int) + sizeof(float)));` --- the compiler expects a `)` before the `;`).

**Fourteenth prompt:** The revised solution compiles, but encounters a cuda error: "an illegal memory access was encountered". We tell the language model about this error and ask it to fix it. If it cannot, we will investigate the source of the error.

**Fifteenth prompt:** In the new solution, there is a syntax error on line 227: identifier "candidate_count" is undefined (the line is `*candidate_count = 0;`).

**Sixteenth prompt:** The generation was interrupted and required clicking the "Continue generating" button. The new solution compiles and runs without errors; however, the result contains the initial values that represent infinity.

We make a new prompt that addresses this issue: The result contains the initial values that represent infinity.

**Seventeenth prompt:** The generation was interrupted and required clicking the "Continue generating" button. The provided solution is an exact copy of the previous one.

Since the language model provided multiple incorrect solutions in a row and also repeated the same solution, continuing beyond this point is not productive. Also, the dialog takes too long for a reasonable use of the language model.

## Conclusion

The model already surpassed the static prompts in the first three milestones. We can confirm that iterative tutoring improves the quality of the generated code and allows the language model to use the simpler solutions as a base for the more complex ones. However, the language model struggles with the semantics of CUDA parallelism and the semantics of the kNN algorithm to an extent that makes it impractical to generate a working solution without more direct intervention. Tutoring itself is not enough unless it provides concrete low-level directions that are beyond the point where the language model allows for automatization of the programming process.

It is likely that future models will be more capable of generating high-performance CUDA code requiring inter-thread communication. However, the current state-of-the-art GPT-o3-mini-high model is not capable of this even with the iterative tutoring.

## Transcript

Manual copy: [transcript.md](transcript.md)
