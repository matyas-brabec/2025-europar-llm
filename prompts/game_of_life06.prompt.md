@include ./game_of_life__description.md
@include ./game_of_life__func_int64.md

Using shared or texture memory in this case has proven to be unnecessary and only adds complexity to the code.
The input grid is bit-packed, meaning each `std::uint64_t` word represents 64 consecutive cells within the same row (one bit per cell).
Each CUDA thread should handle a single `std::uint64_t` to eliminate the need for atomic operations.

To efficiently compute the neighbor count, apply full adder logic to add the bits from adjacent neighbor words concurrently. That is, for three neighbor words, compute the sum as the XOR of the three inputs and the carry as the majority function, and then combine these results across groups of neighbors to obtain the total count for each bit position in parallel. This allows you to process 64 cells simultaneously using simple bitwise operations.

Additionally, the 0th and 63rd bits of each word require special handling. For the 0th bit, you must also consider the three words to the left, while for the 63rd bit, you need to account for the three words to the right. In all cases, you must also check the words in the rows above and below.
