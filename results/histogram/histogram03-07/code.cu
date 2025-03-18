#include <cuda_runtime.h>

// CUDA kernel to compute a histogram over a specified character range using shared memory privatization.
// The kernel processes a text input stored in the device memory and produces a histogram for characters
// in the inclusive range [from, to]. Each block maintains its own privatized histogram in shared memory
// to reduce contention on global memory. At the end of the kernel, the shared histograms are merged with
// the global histogram via atomic operations.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, int from, int to, unsigned int *histogram)
{
    // Compute the number of bins in the histogram.
    int histSize = to - from + 1;

    // Declare shared memory for the privatized histogram.
    // The dynamic shared memory size is provided at kernel launch.
    extern __shared__ unsigned int shared_hist[];

    // Initialize shared memory histogram bins to zero in parallel.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // Compute a unique thread index across the entire grid and the stride for processing.
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    // Process the input text in a strided loop.
    // Each thread loads characters and if the character falls into the specified range [from, to],
    // its corresponding bin in the private shared histogram is incremented via atomic addition.
    for (int i = globalId; i < inputSize; i += gridStride) {
        // Cast to unsigned char to ensure correct value for characters.
        unsigned char c = static_cast<unsigned char>(input[i]);
        if (c >= from && c <= to) {
            int bin = c - from;
            atomicAdd(&shared_hist[bin], 1U);
        }
    }
    __syncthreads();

    // Write the per-block histogram to global memory.
    // Each thread is responsible for iterating over a subset of bins.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        // Only perform an atomic addition if the bin is non-zero.
        if (shared_hist[i] != 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

// Host function that invokes the CUDA kernel to compute the character histogram.
// The input parameters:
//   input: pointer to device memory containing the text data.
//   histogram: pointer to device memory where the histogram will be stored.
//   inputSize: number of characters in the input text.
//   from, to: specify the inclusive range of character ordinals for which the histogram is computed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins required for the histogram.
    int histSize = to - from + 1;

    // Initialize the output histogram in global memory to zero.
    cudaMemset(histogram, 0, histSize * sizeof(unsigned int));

    // Set up execution configuration parameters.
    int blockSize = 256; // Number of threads per block.
    int gridSize = (inputSize + blockSize - 1) / blockSize; // Number of blocks required.

    // Launch the histogram kernel.
    // The third parameter specifies the dynamic shared memory size, which is histSize * sizeof(unsigned int).
    histogram_kernel<<<gridSize, blockSize, histSize * sizeof(unsigned int)>>>(input, inputSize, from, to, histogram);
}