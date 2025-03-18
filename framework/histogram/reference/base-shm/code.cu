#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>


template<typename T = char, typename RES = unsigned int>
__global__ void hist(const char *data, unsigned int N, unsigned int *result, int fromValue, int toValue)
{
    const unsigned int histogramSize = toValue - fromValue + 1;

    // initialize shared memory (cooperatively by the thread block)
    __shared__ unsigned int resultShm[256];
    for (unsigned int i = threadIdx.x; i < histogramSize; i += blockDim.x) {
        resultShm[i] = 0;
    }

    __syncthreads();

    // aggregate histograms in shared memory
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int c = (int)data[idx] - fromValue;
        if (c >= 0 && c < histogramSize) {
            atomicAdd(&resultShm[c], 1);
        }
    }

    __syncthreads();

    // merge shared memory copy with global memory copy (cooperatively by the thread block)
    for (unsigned int i = threadIdx.x; i < histogramSize; i += blockDim.x) {
        atomicAdd(&result[i], resultShm[i]);
    }
}


void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
	constexpr unsigned int blockSize = 1024;
    unsigned int blocks = (inputSize + blockSize - 1) / blockSize;
	hist<<<blocks, blockSize>>>(input, inputSize, histogram, from, to);
}
