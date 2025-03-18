# Histogram

## Milestone #1: Working parallel solution

**Objective:** The LLM will generate a solution similar to our baseline: Each CUDA thread processes one character from the input buffer and if it falls within the specified range, it increments the corresponding histogram bin in the global memory using atomic increment (or atomic add) instruction.

The initial prompt is equivalent to the static `histogram01.prompt.md`:

---

You are an experienced CUDA programmer trying to provide the most optimized code for the given task and the target hardware.
Assume that the target hardware is a modern NVIDIA GPU for data centers such as the H100 or A100, and the code is compiled using the latest CUDA toolkit and the latest compatible host compiler.
The output should contain only the requested source code. The code must not contain any placeholders or unimplemented parts. All high-level overviews and specific explanations should be included in code comments. Do not output any non-source text.

Write a CUDA kernel that will compute a histogram of a text file restricted to a particular char range. The input is plain text stored as an array of chars. The function gets parameters `from` and `to` (0 <= `from` < `to` <= 255) that specify a continuous range `from`--`to` of the histogram that should be computed. The output is an array of `to` - `from` + 1 unsigned integers where position `i` holds the count of occurrences of character with ordinal value `i` + `from`. Besides the CUDA kernel, write a regular C++ function with the following descriptor that invokes the CUDA kernel with appropriate parameters.
```c++
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
```
Assume the `input` and `histogram` arrays are duly allocated by `cudaMalloc` and the `inputSize` is the number of chars in the `input` buffer. Any host-device synchronization is handled by the caller of `run_histogram`.

---

**Prepared hints:**
- The CUDA kernel should be designed to process one character per thread.
- The kernel should use atomic operations to increment the histogram bins.


## Milestone #2: Shared memory privatization

**Objective:** The LLM will generate a solution that keeps a copy of the histogram in the shared memory (which must be initialized to zeros first). Each thread processes one character from the input buffer and if it falls within the specified range, it increments the corresponding histogram bin in the privatized copy using atomic instructions. After all threads in the block finish, the first `H` threads (where `H` denotes the number of histogram bins) will update the global histogram using atomic instructions (each thread updates a single value). The initialization, aggregation, and write-back phases must be separated by explicit barriers.

The follow-up prompt is based on the static `histogram03.prompt.md`:

---

Optimize the previously generated kernel by utilization of shared memory for histogram privatization to reduce atomic writes to the global memory.

---

**Prepared hints:**
- Place a copy of the histogram in the shared memory and initialize it to zeros.
- Use the private copy of the histogram to accumulate the values before writing them back to the global memory.
- Privatized histogram values should be merged with global histogram values using atomic operations. Each value should be updated by a single thread.


## Milestone #3: Multiple items per thread

**Objective:** The previous solution will be updated so that each thread processes multiple characters from the input buffer. The exact number is specified by a constant and its value should be reasonably selected by the LLM (reasonable value should be at least 8 or 16). This way, multiple values are aggregated into the privatized histogram before writing them back to the global memory, which reduces the number of global memory accesses (and, consequently, atomic conflicts).


The follow-up prompt is based on the static `histogram04.prompt.md`:

---

Add a constant `itemsPerThread` to the code that controls how many input chars are processed by each thread. Select the appropriate default value for `itemsPerThread`, assuming the kernel should be optimized for the latest NVIDIA GPUs and the input size is large.

---

**Additional issue:** The outlined objective does not specify which characters are processed by each thread. The most straightforward approach is to assign a block of `itemsPerThread` consecutive characters to each thread. However, the hardware is designed to optimize for scenarios where each thread loads one 32-bit or 64-bit value and threads in a warp load one compact segment of memory (so the coalesced load is achieved). If the `itemsPerThread` gets higher than 8, each load (performed by a warp) is broken into multiple memory transactions and the efficiency of the load operations decreases.

A more suitable solution would be to make each thread process 4 consecutive characters (32-bits), and then move ahead by warp size (or block size) `* 4` bytes (`itemsPerThread / 4` times). This way, the memory accesses are coalesced and the efficiency of the load operations is maximized regardless of the `itemsPerThread` value.

This issue is not addressed in our prompts or hints and we did not plan on tutoring LLM towards it. However, we will monitor whether this issue is addressed by the LLM without our intervention.


## Milestone #4: Avoiding shared memory bank conflicts

**Objective:** Storing a private copy of the histogram in the shared memory may still cause conflicts in atomic instructions at the warp level. Atomic collision within a warp inevitably leads to the serialization of the involved threads. This can be mitigated by storing multiple private copies in shared memory. If each warp lane (`threadIdx.x % 32`) has its own copy of the histogram, the atomic operations will not collide within a warp.

However, the copies must be stored in a way that each copy occupies a separate memory bank to avoid bank conflicts (which would be as severe as the atomic collisions). Consecutive 32-bit words are in consecutive banks (modulo 32), so the histogram should be stored in a stridden manner in the shared memory. Value `i` (`0 <= i < from-to+1`) of histogram copy `c` should be stored at index `c + i * 32` in the shared memory (for 32 copies).

The follow-up prompt is based on the static `histogram06.prompt.md`:

---

Update the previously generated solution to avoid shared memory bank conflicts by using multiple copies of the histogram per thread block. Create 32 copies of the histogram and place each copy in a separate bank (using appropriate stride). Make sure that each copy is accessed by threads from the same warp lanes to avoid intra-warp bank collisions.

---

**Prepared hints:**
- Make each thread access a copy based on the thread's warp lane ID `threadIdx.x % 32`.
- Store the histogram values in the shared memory so that value `i` of histogram copy `c` (0 <= `c` < 32) is located at offset `i * 32 + c`.
- Make sure the run function allocates enough shared memory for the 32 copies of the histogram.
