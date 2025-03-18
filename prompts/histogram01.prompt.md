Write a CUDA kernel that will compute a histogram of a text file restricted to a particular char range. The input is plain text stored as an array of chars. The function gets parameters `from` and `to` (0 <= `from` < `to` <= 255) that specify a continuous range `from`--`to` of the histogram that should be computed. The output is an array of `to` - `from` + 1 unsigned integers where position `i` holds the count of occurrences of character with ordinal value `i` + `from`. Besides the CUDA kernel, write a regular C++ function with the following descriptor that invokes the CUDA kernel with appropriate parameters.
```c++
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
```
Assume the `input` and `histogram` arrays are duly allocated by `cudaMalloc` and the `inputSize` is the number of chars in the `input` buffer. Any host-device synchronization is handled by the caller of `run_histogram`.
