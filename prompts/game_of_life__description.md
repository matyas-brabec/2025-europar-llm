Write a CUDA implementation of Conway’s Game of Life.

Given a 2D grid of cells, each cell can be in one of two states: dead (0) or alive (1). A cell’s neighborhood comprises eight adjacent cells (horizontal, vertical, and diagonal neighbors). The next generation of the grid is computed based on the following rules:
- Any alive cell with fewer than 2 alive neighbors dies (underpopulation).
- Any alive cell with 2 or 3 alive neighbors survives.
- Any alive cell with more than 3 alive neighbors dies (overpopulation).
- Any dead cell with exactly 3 alive neighbors becomes alive (reproduction).

Assume that all cells outside the grid are dead, this is important for handling boundary cells. Additionally, the grid dimensions are always a power of 2, greater than 512, and small enough to ensure both the input and output fit within GPU memory.

Besides the CUDA kernel, implement a `run_game_of_life` function that calls the kernel. The performance of the kernel and the simulation itself is the only concern. Any necessary data transformations can be performed inside this function, as they will not be measured. Any host-device synchronization is handled by the caller of `run_game_of_life`.
