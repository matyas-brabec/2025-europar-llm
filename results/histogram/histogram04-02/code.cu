// This file implements a CUDA kernel to compute a histogram for characters in a given range,
// and a host function run_histogram() that launches the kernel.
// The input is a plain text array of characters, and the histogram is computed only over
// the continuous ASCII range [from, to]. The output histogram array (of size to - from + 1)
// will have at index i the count of occurrences of the character with ordinal value (i + from).
//
// The kernel uses shared memory to privatize a per-block histogram which is later reduced
// to the global histogram. Each thread processes a fixed number of input items (ITEMS_PER_THREAD)
// to increase arithmetic intensity and hide memory latency on modern NVIDIA GPUs (like A100/H100).
//
// The input and histogram arrays are assumed to be allocated on the device (e.g., via cudaMalloc).
// No host-device synchronization is performed by run_histogram(); it is the caller's responsibility.

#include <cuda_runtime.h>

// Constant that controls how many input characters are processed per thread.
// This value is tuned for modern GPUs when the input size is large.
constexpr int ITEMS_PER_THREAD = 8;

// CUDA kernel to compute a histogram over a restricted range of character ordinals.
// Parameters:
//   input     - Pointer to the input text (in device memory).
//   inputSize - Number of characters in the input.
//   from      - Lower bound of the character range (inclusive).
//   to        - Upper bound of the character range (inclusive).
//   histogram - Pointer to the global histogram (in device memory). Must have size (to - from + 1).
__global__ void histogramKernel(const char *input, unsigned int inputSize, int from, int to, unsigned int *histogram)
{
    // Compute the number of histogram bins for the given range.
    int histSize = to - from + 1;

    // Allocate shared memory for a block-local histogram.
    // The size in bytes is specified at kernel launch. The shared memory array is uninitialized.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared histogram bins to 0. Use a strided loop over the bins.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of the input.
    // Each thread processes ITEMS_PER_THREAD characters.
    // Compute the base index for the current block.
    unsigned int baseIndex = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;

    // Loop over ITEMS_PER_THREAD segments for each thread.
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        unsigned int idx = baseIndex + threadIdx.x + i * blockDim.x;
        if (idx < inputSize)
        {
            // Cast to unsigned char to correctly map values 128..255 when char is signed.
            unsigned int value = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
            // Check if the character falls within the specified range.
            if (value >= static_cast<unsigned int>(from) && value <= static_cast<unsigned int>(to))
            {
                // Update the block-local histogram using atomics in shared memory.
                atomicAdd(&s_hist[value - from], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the block-local histogram into the global histogram.
    // Each thread adds one or more histogram bins to global memory.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x)
    {
        // Use atomicAdd to avoid race conditions from multiple blocks.
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function that configures and launches the CUDA kernel for histogram computation.
// The input and histogram arrays must be allocated on the device (with cudaMalloc or similar).
//
// Parameters:
//   input     - Pointer to the device memory containing the input text.
//   histogram - Pointer to the device memory for the output histogram. Its size should be (to - from + 1).
//   inputSize - Number of characters in the input text.
//   from      - Lower bound of the character range (inclusive), 0 <= from < to <= 255.
//   to        - Upper bound of the character range (inclusive).
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Set the number of threads per block (must be a multiple of the warp size, typically 256).
    int threadsPerBlock = 256;
    // Each thread processes ITEMS_PER_THREAD items.
    // Compute the number of blocks required to cover the input size.
    int blocks = (inputSize + threadsPerBlock * ITEMS_PER_THREAD - 1) / (threadsPerBlock * ITEMS_PER_THREAD);

    // Compute the number of histogram bins.
    int histSize = to - from + 1;
    // Calculate the amount of shared memory needed per block.
    size_t sharedMemSize = histSize * sizeof(unsigned int);

    // Launch the kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);

    // Note: Host-device synchronization (e.g., cudaDeviceSynchronize) should be handled by the caller.
}