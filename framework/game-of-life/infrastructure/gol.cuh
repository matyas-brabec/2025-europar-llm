#ifndef GOL_CUH
#define GOL_CUH

#include <cstdint>

/**
 * Runs one step of Conway's Game of Life using a CUDA kernel.
 *
 * @param input Pointer to the input grid, where each cell is represented as true (alive) or false (dead).
 * @param output Pointer to the output grid, storing the next state.
 * @param grid_dimensions The width/height of the square grid (always a power of 2).
 */
void run_game_of_life(const bool* input, bool* output, int grid_dimensions);

/**
* Runs one step of Conway's Game of Life using a CUDA kernel.
*
* @param input Pointer to the input grid, where each word contains 64 cells.
* @param output Pointer to the output grid, storing the next state.
* @param grid_dimensions The width/height of the square grid (always a power of 2).
*/
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions);

void initialize_internal_data_structures(int grid_dimensions);

#endif // GOL_CUH
