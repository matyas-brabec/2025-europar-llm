#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// The CUDA kernel computes the histogram of characters within the inclusive range [from, to].
// It uses dynamic shared memory (privatized histogram) per thread block to reduce contention on global memory.
// Each thread block first initializes a shared memory histogram array (of length histSize = to - from + 1) to 0,
// then each thread processes a subset of the input array and updates the shared histogram using atomic additions.
// Finally, the shared histogram is merged into the global histogram array via atomicAdd operations.
__global__ void histogram_kernel(const char *input, unsigned int *globalHistogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins for the specified range.
    int histSize = to - from + 1;

    // Declare dynamic shared memory array for the privatized histogram.
    extern __shared__ unsigned int s_hist[];

    // Each thread initializes part of the shared histogram to zero.
    for (int i = threadIdx.x; i < histSize; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the global thread index and the total stride.
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple input elements based on the grid stride.
    while (idx < inputSize) {
        // Convert the input character to an unsigned value.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        // Check if the character falls within the desired range [from, to].
        if (c >= from && c <= to) {
            // Compute the local histogram index.
            int histIdx = c - from;
            // Update the shared histogram with an atomic add to prevent write conflicts.
            atomicAdd(&s_hist[histIdx], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Merge the per-block shared histogram into the global histogram.
    // Only threads that correspond to a valid histogram index perform this merge.
    if (threadIdx.x < histSize) {
        atomicAdd(&globalHistogram[threadIdx.x], s_hist[threadIdx.x]);
    }
}

// Host function that launches the CUDA histogram kernel.
// Parameters:
//   input       - Device pointer to the input text array.
//   histogram   - Device pointer to the output histogram array (pre-allocated).
//                 Its length must be at least (to - from + 1) unsigned ints.
//   inputSize   - Number of characters in the input array.
//   from, to    - Define the inclusive range of characters [from, to] to be considered for the histogram.
// Note: The caller is responsible for host-device synchronization.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins in the histogram.
    int histSize = to - from + 1;

    // Set the number of threads per block.
    int blockSize = 256;

    // Compute the grid size such that all input characters are processed.
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    // Launch the kernel with dynamic shared memory allocation for the privatized histogram.
    // Each block allocates "histSize * sizeof(unsigned int)" bytes in shared memory.
    histogram_kernel<<<gridSize, blockSize, histSize * sizeof(unsigned int)>>>(input, histogram, inputSize, from, to);
}