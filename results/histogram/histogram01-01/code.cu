#include <cuda_runtime.h>

// CUDA kernel to compute the histogram of characters in the range [from, to].
// Each block computes a private histogram (using shared memory) for its portion
// of the input and then atomically accumulates it into the global histogram.
__global__ void histogram_kernel(const char* input, unsigned int inputSize, unsigned int* histogram, int from, int to)
{
    // Compute the number of bins in the requested histogram.
    int range = to - from + 1;

    // Allocate a shared memory histogram array with fixed size 256.
    // Since to <= 255 and from >= 0, range is at most 256.
    __shared__ unsigned int s_hist[256];

    // Initialize the shared histogram bins to zero.
    // Use a strided loop in case blockDim.x is less than 'range'.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread processes multiple elements in the input array.
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    for (int i = global_id; i < inputSize; i += total_threads) {
        // Cast to unsigned char to avoid negative values.
        unsigned char c = static_cast<unsigned char>(input[i]);
        // Check if the character falls within the desired range.
        if (c >= from && c <= to) {
            // Compute the index in the histogram.
            int bin = c - from;
            // Use an atomic operation to update the shared histogram.
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();

    // Merge the shared histogram into the global histogram.
    // Each thread takes care of a subset of bins.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function to launch the histogram kernel.
// The input text and histogram arrays are assumed to be allocated in GPU memory via cudaMalloc.
// The histogram array will be initialized to zero before kernel execution.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins needed.
    int range = to - from + 1;

    // Clear the histogram array on the device.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Choose an appropriate block size. 256 threads per block is typical on modern GPUs.
    int blockSize = 256;
    // Calculate grid size to cover the entire input.
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    if (gridSize < 1)
        gridSize = 1;

    // Launch the kernel.
    // The kernel uses a statically-allocated shared memory array of size 256.
    histogram_kernel<<<gridSize, blockSize>>>(input, inputSize, histogram, from, to);
}