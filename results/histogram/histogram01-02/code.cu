#include <cuda_runtime.h>

// This CUDA kernel computes a histogram over a restricted character range.
// It reads an input array of chars and, for each character within the range [from, to],
// increments a corresponding bin in a shared-memory histogram. After processing the input,
// each block accumulates its partial histogram into the global histogram.
// Parameters:
//   input: pointer to the input text (device memory)
//   histogram: pointer to the global histogram array (device memory)
//   inputSize: number of characters in the input array
//   from: lower bound (inclusive) of the char range
//   to: upper bound (inclusive) of the char range
__global__ void histogram_kernel(const char *input, unsigned int *histogram,
                                 unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins in the histogram (range is inclusive).
    int histSize = to - from + 1;

    // Allocate shared memory for a per-block histogram.
    // The size of s_hist is specified at kernel launch as dynamic shared memory.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to 0.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process the input array using a grid-stride loop.
    // Each thread processes multiple elements separated by gridDim.x * blockDim.x.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < inputSize;
         idx += blockDim.x * gridDim.x)
    {
        // Convert the input char to an unsigned char to get its ordinal value.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // Only count the character if it lies within the specified range.
        if (c >= from && c <= to) {
            int bin = c - from;
            // Atomic update on shared memory histogram to avoid race conditions.
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();

    // Accumulate the per-block shared histogram into the global histogram.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        // Atomic update ensures correct accumulation from multiple blocks.
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// This host function launches the CUDA kernel to compute the histogram.
// It assumes that the input and histogram arrays have been allocated in device memory.
// Parameters:
//   input: pointer to the input text (device memory)
//   histogram: pointer to the global histogram array (device memory)
//   inputSize: number of characters in the input array
//   from: lower bound (inclusive) of the char range to histogram
//   to: upper bound (inclusive) of the char range to histogram
//
// Note: No host-device synchronization is performed in this function; the caller is
// responsible for synchronizing the device if necessary.
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram,
void run_histogram(const char *input, unsigned int *histogram,
        unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins (range [from, to] inclusive).
    int histSize = to - from + 1;

    // Select the number of threads per block.
    int threadsPerBlock = 256;

    // Compute the number of blocks required to cover the input.
    // A grid-stride loop inside the kernel will ensure complete processing.
    int numBlocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    // Launch the kernel with dynamic shared memory allocated for the per-block histogram.
    // The size of the shared memory is histSize * sizeof(unsigned int).
    histogram_kernel<<<numBlocks, threadsPerBlock, histSize * sizeof(unsigned int)>>>(
        input, histogram, inputSize, from, to
    );

    // Note: Any necessary host-device synchronization (e.g., cudaDeviceSynchronize)
    // is assumed to be handled by the caller.
}
