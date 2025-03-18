#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel to compute a histogram of a text buffer for characters in the range [from, to].
// The kernel uses shared memory for histogram privatization to reduce contention on global memory.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, int from, int to, unsigned int *global_hist) {
    // Number of bins in the histogram is (to - from + 1)
    int numBins = to - from + 1;
    
    // Declare dynamic shared memory for the block-private histogram.
    // The size is specified at kernel launch as (numBins * sizeof(unsigned int)).
    extern __shared__ unsigned int shared_hist[];

    // Initialize shared histogram bins to zero.
    // Each thread initializes one or more bins if necessary.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // Calculate the global index and stride for processing the input buffer.
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Loop over the input array with a stride to cover all elements.
    // Cast the char to unsigned char to correctly interpret the value.
    for (unsigned int i = index; i < inputSize; i += stride) {
        unsigned char c = static_cast<unsigned char>(input[i]);
        // Check if the character is within the desired range, and update the block's histogram.
        if (c >= from && c <= to) {
            atomicAdd(&shared_hist[c - from], 1);
        }
    }
    __syncthreads();

    // Merge the block private histograms into the global histogram.
    // Each thread processes one or more bins to reduce the number of global atomic operations.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&global_hist[i], shared_hist[i]);
    }
}

// Host function that launches the CUDA kernel to compute the histogram.
// Parameters:
// - input: pointer to the device memory array containing the text data (chars)
// - histogram: pointer to the device memory array that will hold the histogram counts.
//              It must have been allocated to have at least (to - from + 1) unsigned ints.
// - inputSize: number of characters in the input buffer
// - from, to: the inclusive range of character ordinal values to consider (0 <= from < to <= 255)
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Calculate the number of bins in the histogram.
    int numBins = to - from + 1;

    // Initialize the global histogram to zero.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Choose block size and compute grid size.
    // A block size of 256 threads is typically a good balance on modern GPUs.
    int blockSize = 256;
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Launch the kernel.
    // The third kernel launch parameter allocates dynamic shared memory of size (numBins * sizeof(unsigned int)).
    histogram_kernel<<<gridSize, blockSize, numBins * sizeof(unsigned int)>>>(input, inputSize, from, to, histogram);

    // Note: Any host-device synchronization and error checking is assumed to be handled by the caller.
}