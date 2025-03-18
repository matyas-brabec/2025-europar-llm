#include <cuda_runtime.h>

// Optimized CUDA kernel for computing a histogram of a character array, restricted to
// a specified continuous range [from, to]. The kernel uses shared memory for histogram
// privatization per block. Each thread processes a fixed number of input characters,
// controlled by the constant "itemsPerThread".
//
// Parameters:
//   input       - device pointer to the input text array (allocated with cudaMalloc)
//   histogram   - device pointer to the output histogram array. The array has (to-from+1)
//                 unsigned integers. Each position i accumulates the count for character (i + from).
//   inputSize   - number of characters in the input array
//   from, to    - define the continuous character range of interest (0 <= from < to <= 255)
//   itemsPerThread - number of input characters processed by each thread.
__global__ void histogramKernel(const char *input, unsigned int *histogram,
                                unsigned int inputSize, int from, int to, int itemsPerThread)
{
    // Compute the number of bins needed for the histogram.
    int range = to - from + 1;

    // Allocate dynamic shared memory for the private histogram of this block.
    // The shared memory size is determined at kernel launch to be (range * sizeof(unsigned int)).
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to 0.
    // Use a loop stride by blockDim.x to allow any block size.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        s_hist[bin] = 0;
    }
    __syncthreads();

    // Compute the unique global thread id.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Determine the starting index for this thread's work.
    int baseIndex = globalThreadId * itemsPerThread;

    // Each thread processes 'itemsPerThread' consecutive characters from the input.
    for (int offset = 0; offset < itemsPerThread; offset++) {
        int idx = baseIndex + offset;
        if (idx < inputSize) {
            unsigned char c = static_cast<unsigned char>(input[idx]);
            // Check if the character falls within the specified range.
            if (c >= from && c <= to) {
                int bin = c - from;
                // Atomically increment the bin in the shared (private to block) histogram.
                atomicAdd(&s_hist[bin], 1);
            }
        }
    }
    __syncthreads();

    // Now, each thread cooperatively writes the results from the shared histogram into
    // the global histogram array using atomic updates.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        // Only update the global histogram if the bin count is non-zero.
        unsigned int count = s_hist[bin];
        if (count > 0) {
            atomicAdd(&histogram[bin], count);
        }
    }
}

// Host function that configures and launches the CUDA histogram kernel.
// Assumptions:
//   - 'input' and 'histogram' are device pointers allocated via cudaMalloc.
//   - 'inputSize' is the number of elements (chars) in the 'input' buffer.
//   - Synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Define the number of input characters each thread processes.
    // This value is chosen as a default for modern NVIDIA GPUs and large inputs.
    constexpr int itemsPerThread = 16;
    
    // Choose a typical number of threads per block for high occupancy.
    constexpr int threadsPerBlock = 256;
    
    // Calculate the total number of threads needed so that each thread processes 'itemsPerThread' items.
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;
    
    // Calculate the number of thread blocks needed.
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks < 1) {
        blocks = 1;
    }
    
    // Compute the number of histogram bins for the given character range.
    int range = to - from + 1;
    // Allocate dynamic shared memory of size (range * sizeof(unsigned int)) per block.
    int sharedMemSize = range * sizeof(unsigned int);
    
    // Launch the histogram kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to, itemsPerThread);
}