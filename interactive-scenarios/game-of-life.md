# Game of Life Stencil

## Milestone #1: Working Parallel Solution

**Objective**: The LLM will generate a baseline solution where each CUDA thread processes one cell from the grid. If the cell is alive, the thread will update the state of the cell in the next generation based on the surrounding neighbors. The solution should also avoid usage of shared or texture memory as these have been proven unnecessary and only add to code's complexity.

The initial prompt is equivalent to the static `game_of_life01.prompt.md`:

---

You are an experienced CUDA programmer trying to provide the most optimized code for the given task and the target hardware.
Assume that the target hardware is a modern NVIDIA GPU for data centers such as the H100 or A100, and the code is compiled using the latest CUDA toolkit and the latest compatible host compiler.
The output should contain only the requested source code. The code must not contain any placeholders or unimplemented parts. All high-level overviews and specific explanations should be included in code comments. Do not output any non-source text.

Write a CUDA implementation of Conway’s Game of Life.

Given a 2D grid of cells, each cell can be in one of two states: dead (0) or alive (1). A cell’s neighborhood comprises eight adjacent cells (horizontal, vertical, and diagonal neighbors). The next generation of the grid is computed based on the following rules:
- Any alive cell with fewer than 2 alive neighbors dies (underpopulation).
- Any alive cell with 2 or 3 alive neighbors survives.
- Any alive cell with more than 3 alive neighbors dies (overpopulation).
- Any dead cell with exactly 3 alive neighbors becomes alive (reproduction).

Assume that all cells outside the grid are dead, this is important for handling boundary cells. Additionally, the grid dimensions are always a power of 2, greater than 512, and small enough to ensure both the input and output fit within GPU memory.

Besides the CUDA kernel, implement a `run_game_of_life` function that calls the kernel. The performance of the kernel and the simulation itself is the only concern. Any necessary data transformations can be performed inside this function, as they will not be measured. Any host-device synchronization is handled by the caller of `run_game_of_life`.

The function `run_game_of_life` executes one step of Conway’s Game of Life using a CUDA kernel.

- `input`: Pointer to the input grid, where each cell is represented as `true` (alive) or `false` (dead).
- `output`: Pointer to the output grid, storing the next state.
- `grid_dimensions`: The width/height of the square grid (always a power of 2).

```cpp
void run_game_of_life(const bool* input, bool* output, int grid_dimensions);
```

Assume the input and output grids are duly allocated by `cudaMalloc`.

---

**Prepared hints:**

- The CUDA kernel should process one cell per thread, with each thread determining the next state based on its neighboring cells.
- Avoid the use of shared or texture memory, as these have proven unnecessary and only add complexity.

## Milestone #2: Bitwise Grid Representation

**Objective:** The LLM will transition the grid representation from a boolean matrix to a more efficient bitwise encoding, where each thread processes one word, representing multiple cells, and uses the `popc` function.

The follow-up prompt is based on the static `game_of_life02.prompt.md`. Additionally, the hints are based on `game_of_life03.prompt.md` and `game_of_life04.prompt.md`:

---

Optimize the previously generated kernel by changing the grid representation to a bitwise encoding, where each cell is encoded as one bit.

---

**Additional issue:** Observe that the prompt does not specify the desired encoding (row-wise or tile-based). The primary goal is to use a row-wise encoding, as it is easier to implement than the tile-based approach. However, if the LLM suggests a more advanced tile-wise encoding, we will proceed with that approach (most of the hints still apply). Additionally, we avoid specifying the word size (`32` or `64` bits) and let the LLM choose. If the LLM opts for a 32-bit word, the code will need to be patched manually (to fit to our framework). The patch is a simple matter of `reinterpret_cast` since the content of the memory is identical, regardless of the word size.

**Second issue:** We need to test both for efficiency and performance. In "one shot" tests, we made the task easier for the LLM by encoding the grid ourselves. The LLM was asked to implement only the Game of Life logic, not the conversion. Here, however, we kept the boolean interface and let the LLM implement the conversion on its own. We check the correctness of the transformation code, but it was not used during performance measurements. Only the kernel itself was considered for the performance test. The function `run_game_of_life` was patched accordingly.

**Prepared hints:**

- The grid should be bit-packed, meaning each word represents consecutive cells within the same row (one bit per cell).
- Each CUDA thread will process a single word, which eliminates the need for atomic operations.
- Use bit masks to extract each cell’s neighborhood from the word.
- Special handling is required for the first and last bits of each word. For the first bit, consider the three words to the left, while for the last bit, consider the three words to the right.
- Instead of performing individual checks for neighbors, utilize the `__popc` intrinsic function to count the number of set bits in a word, which can significantly improve performance.

## Milestone #3: Full-adder Approach

**Objective:** The final goal is to use full-adder logic to compute multiple cells at once in a vectorized manner. Initially, the LLM is asked to devise a method for processing multiple cells simultaneously without mentioning the full-adder technique or any other key details. If the response is unsatisfactory, we provide the final hint.

The follow-up prompt is based on the static `game_of_life05.prompt.md` and `game_of_life06.prompt.md`.

---

Computing one cell at a time is computationally expensive. Can you devise a method to process multiple cells simultaneously to improve performance? Note that we are using a row-based bitwise encoding of the grid.

---

**Additional issue:** If the LLM adopts a tile-based approach during the previous milestone, we need to steer it back to a row-based solution. In that case, return to the beginning of the second milestone and repeat the prompt with the first hint from the second milestone, which specifies the details about the row-based encoding. The other hints are unnecessary, as we only need to adjust the encoding.

**Prepared hints:**

- To efficiently compute the neighbor count, apply full adder logic to add the bits from adjacent neighbor words concurrently. That is, for three neighbor words, compute the sum as the XOR of the three inputs and the carry as the majority function, and then combine these results across groups of neighbors to obtain the total count for each bit position in parallel. This allows you to process all cells in the word simultaneously using simple bitwise operations.
