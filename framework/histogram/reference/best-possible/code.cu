#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

constexpr unsigned int itemsPerThread = 512/**IPT**/;
constexpr unsigned int privCopies = 32/**PRIV**/;

using char_vec_t = char[8/**VEC**/];


template<typename T = char, typename RES = unsigned int>
__global__ void hist(const char *data, unsigned int N, unsigned int *result, int fromValue, int toValue)
{
    const char_vec_t *dataVec = reinterpret_cast<const char_vec_t*>(data);
    constexpr unsigned int batchesPerThread = itemsPerThread / sizeof(char_vec_t);
    const unsigned int histogramSize = toValue - fromValue + 1;

    // initialize shared memory
    extern __shared__ unsigned int resultShm[];
    for (unsigned int i = threadIdx.x; i < histogramSize * privCopies; i += blockDim.x) {
        resultShm[i] = 0;
    }

    unsigned int *resultShmPriv = resultShm + threadIdx.x % privCopies;
    __syncthreads();

    // aggregate histograms in shared memory
	unsigned int blockOffset = (blockIdx.x * blockDim.x) * batchesPerThread;
    unsigned int blockEnd = min(blockOffset + blockDim.x * batchesPerThread, N);
    for (unsigned int i = blockOffset + threadIdx.x; i < blockEnd; i += blockDim.x) {
        #pragma unroll
        for (unsigned int b = 0; b < sizeof(char_vec_t); ++b) {
            int c = (int)dataVec[i][b] - fromValue;
            if (c >= 0 && c < histogramSize) {
                atomicAdd(&resultShmPriv[c * privCopies], 1);
            }
        }
    }

    __syncthreads();

    // merge shared memory histograms
    for (unsigned int i = threadIdx.x; i < histogramSize; i += blockDim.x) {
        unsigned int sum = 0;
        for (unsigned int j = 0; j < privCopies; ++j) {
            sum += resultShm[i * privCopies + j];
        }
        if (sum > 0) {
            atomicAdd(&result[i], sum);
        }
    }
}


void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
	constexpr unsigned int blockSize = 1024/**BS**/;
    unsigned int threads = inputSize / itemsPerThread;
    inputSize /= sizeof(char_vec_t);
    unsigned int shmSize = (to - from + 1) * privCopies * sizeof(unsigned int);
	hist<<<(threads + blockSize - 1) / blockSize, blockSize, shmSize>>>(input, inputSize, histogram, from, to);
}
