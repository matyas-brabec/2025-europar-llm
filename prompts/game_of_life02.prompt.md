@include ./game_of_life__description.md
@include ./game_of_life__func_int64.md

Using shared or texture memory in this case has proven to be unnecessary and only adds complexity to the code.
The input grid is bit-packed, meaning each `std::uint64_t` word represents 64 consecutive cells within the same row (one bit per cell).
