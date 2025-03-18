#include <cuda_runtime.h>

// CUDA kernel to compute the histogram of an input text (array of chars) for a restricted character range.
// This kernel uses a two-step reduction approach. Each block allocates a shared-memory histogram,
// processes a portion of the input with a strided loop and updates its private histogram with atomic adds,
// then accumulates the block’s partial histogram into the global histogram with further atomic adds.
__global__ void histogram_kernel(const char *input,
                                 unsigned int *global_hist,
                                 unsigned int inputSize,
                                 int from,
                                 int numBins)
{
    // Declare dynamic shared memory for the block-local histogram.
    extern __shared__ unsigned int shared_hist[];

    // Initialize the shared histogram bins to zero.
    // Each thread initializes multiple bins if necessary.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        shared_hist[bin] = 0;
    }
    __syncthreads();

    // Compute global index and stride.
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    // Process the input array in a strided loop.
    // Only count characters that are within the range [from, from + numBins - 1].
    for (unsigned int i = globalIndex; i < inputSize; i += stride) {
        // Interpret input as unsigned to avoid negative values.
        unsigned char ch = static_cast<unsigned char>(input[i]);
        if (ch >= from && ch < (from + numBins)) {
            // Update the shared histogram with atomic addition.
            atomicAdd(&shared_hist[ch - from], 1);
        }
    }
    __syncthreads();

    // Each thread accumulates its portion of the shared histogram into the global histogram.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        atomicAdd(&global_hist[bin], shared_hist[bin]);
    }
}

// Host function to launch the CUDA histogram kernel.
//
// Parameters:
//   input       - Device pointer to an array of chars representing the text.
//   histogram   - Device pointer to an array of unsigned ints of length (to - from + 1)
//                 where the computed histogram will be stored.
//   inputSize   - Number of characters in the input array.
//   from, to    - Define the inclusive range [from, to] of character ordinal values to be histogrammed.
//
// Assumes that the input and histogram arrays have been allocated with cudaMalloc and that
// appropriate host-device synchronization is handled by the caller.
/// @FIXED
/// extern "C" void run_histogram(const char *input,
void run_histogram(const char *input,
                              unsigned int *histogram,
                              unsigned int inputSize,
                              int from,
                              int to)
{
    // Calculate the number of bins in the histogram.
    int numBins = to - from + 1;

    // Define CUDA kernel launch parameters.
    // A block size of 256 threads is chosen as a common value for modern GPUs.
    int blockSize = 256;
    int gridSize  = (inputSize + blockSize - 1) / blockSize;

    // Launch the CUDA kernel with dynamic shared memory size allocated to hold 'numBins' unsigned ints.
    histogram_kernel<<<gridSize, blockSize, numBins * sizeof(unsigned int)>>>(input,
                                                                              histogram,
                                                                              inputSize,
                                                                              from,
                                                                              numBins);
}
