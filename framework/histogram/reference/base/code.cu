#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>


template<typename T = char, typename RES = unsigned int>
__global__ void hist(const char *data, unsigned int N, unsigned int *result, int fromValue, int toValue)
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    int c = (int)data[idx];
    if (c >= fromValue && c <= toValue) {
        atomicAdd(&result[c-fromValue], 1);
    }
}


void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
	constexpr unsigned int blockSize = 256;
	hist<<<(inputSize + blockSize - 1) / blockSize, blockSize>>>(input, inputSize, histogram, from, to);
}
