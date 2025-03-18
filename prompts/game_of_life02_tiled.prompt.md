@include ./game_of_life__description.md
@include ./game_of_life__func_int64.md

Using shared or texture memory in this case has proven to be unnecessary and only adds complexity to the code.
The input grid is bit-packed, with each `std::uint64_t` representing an 8×8 tile of cells.
