#include <cuda_runtime.h>

// Kernel function to compute histogram from an input text buffer over a specified range.
// Each block computes its own partial histogram in shared memory and then merges it into the global histogram.
// Parameters:
//   input: Pointer to the input text (device memory).
//   inputSize: Number of characters in the input text.
//   histogram: Pointer to the global histogram array (device memory) where each bin corresponds
//              to characters in the range [from, to].
//   from: Lower bound (inclusive) of the character range.
//   to: Upper bound (inclusive) of the character range.
__global__ void histogram_kernel(const char *input, unsigned int inputSize,
                                 unsigned int *histogram, int from, int to)
{
    // Calculate number of bins in the histogram.
    int numBins = to - from + 1;
    
    // Declare shared memory for the block's private histogram.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to 0 in parallel.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads(); // Ensure all bins are zeroed before processing.

    // Calculate the global index and stride for strided loop to cover the whole input.
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Process the input text in a strided parallel loop.
    while (global_idx < inputSize)
    {
        // Read the character; cast to unsigned char to get the proper numeric value.
        unsigned char c = static_cast<unsigned char>(input[global_idx]);

        // If the character is within the specified range, update the block-local histogram.
        if (c >= from && c <= to)
        {
            // Use atomic operation in shared memory to avoid race conditions.
            atomicAdd(&s_hist[c - from], 1);
        }
        global_idx += stride;
    }
    __syncthreads(); // Ensure all threads have finished updating the shared histogram.

    // Merge the shared histogram results into the global histogram.
    // Each thread updates a subset of the bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        // Atomic add to global memory to handle potential inter-block concurrent updates.
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function to launch the CUDA kernel for histogram computation.
// Assumes that "input" and "histogram" pointers are allocated in device memory via cudaMalloc.
// The "histogram" array is reset to zeros before launching the kernel.
// Parameters:
//   input: Pointer to the input text (device memory).
//   histogram: Pointer to the histogram array (device memory).
//   inputSize: Number of characters in the input text buffer.
//   from: Lower bound (inclusive) of the character range to include in the histogram.
//   to: Upper bound (inclusive) of the character range.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins: one bin for each character between 'from' and 'to' inclusive.
    int numBins = to - from + 1;

    // Reset the global histogram memory to 0.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Determine the block and grid sizes.
    // A typical block size; adjust as needed based on profiling and hardware.
    int blockSize = 256;
    // Calculate grid size such that all input characters are processed.
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    // Cap gridSize to a reasonable number if input is small. This can be tuned further.
    if (gridSize < 1) gridSize = 1;

    // Calculate the required shared memory size per block in bytes.
    size_t sharedMemSize = numBins * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, inputSize, histogram, from, to);

    // Note: Host-device synchronization (e.g., cudaDeviceSynchronize) is assumed to be handled by the caller.
}