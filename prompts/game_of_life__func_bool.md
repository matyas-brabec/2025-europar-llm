The function `run_game_of_life` executes one step of Conway’s Game of Life using a CUDA kernel.

- `input`: Pointer to the input grid, where each cell is represented as `true` (alive) or `false` (dead).
- `output`: Pointer to the output grid, storing the next state.
- `grid_dimensions`: The width/height of the square grid (always a power of 2).

```cpp
void run_game_of_life(const bool* input, bool* output, int grid_dimensions);
```

Assume the input and output grids are duly allocated by `cudaMalloc`.
