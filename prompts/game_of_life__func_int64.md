The function `run_game_of_life` executes one step of Conway’s Game of Life using a CUDA kernel.

- `input`: Pointer to the input grid, where each `std::uint64_t` encodes 64 cells. A bit value of `1` represents a live cell, while `0` represents a dead cell.
- `output`: Pointer to the output grid, storing the next state using the same bit-packed encoding as the `input` grid.
- `grid_dimensions`: The width and height of the square grid (always a power of 2).

```cpp
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions);
```

Assume the input and output grids are duly allocated by `cudaMalloc`.
