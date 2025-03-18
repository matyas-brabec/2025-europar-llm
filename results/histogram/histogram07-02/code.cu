#include <cuda_runtime.h>

// Constant to control the number of input characters each thread processes.
// A value of 8 is chosen as a good balance for modern NVIDIA GPUs when processing large inputs.
constexpr int itemsPerThread = 8;

// CUDA kernel that computes a histogram for characters within the range [from, to].
// The kernel uses shared memory privatization with 32 copies to avoid shared memory bank conflicts.
// Each copy is laid out so that bin i of copy c is stored at offset (i * 32 + c) within shared memory.
__global__ void histogram_kernel(const char *input, unsigned int inputSize,
                                 unsigned int *global_hist, int from, int to) {
    // Compute the number of bins in the histogram.
    int range = to - from + 1;

    // Declare shared memory for privatized histograms.
    // There are 32 copies of the histogram, each of which has 'range' bins.
    // The total shared memory size is (range * 32) unsigned ints.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram arrays to zero in parallel.
    // Use a stride of blockDim.x to cover all shared memory locations.
    for (int i = threadIdx.x; i < range * 32; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread processes items from a private histogram copy based on its thread index modulo 32.
    int copyIdx = threadIdx.x % 32;

    // Compute the global starting index for the data processed by this thread.
    // Each thread processes 'itemsPerThread' elements with a stride of blockDim.x.
    int globalStart = blockIdx.x * blockDim.x * itemsPerThread + threadIdx.x;

    // Process a contiguous chunk of input characters.
    for (int i = 0; i < itemsPerThread; i++) {
        int pos = globalStart + i * blockDim.x;
        if (pos < inputSize) {
            // Read a character from the input buffer.
            unsigned char ch = static_cast<unsigned char>(input[pos]);
            // Check if the character is within the histogram range.
            if (ch >= from && ch <= to) {
                // Determine the histogram bin corresponding to the character.
                int bin = ch - from;
                // Update the shared memory histogram for the thread's private copy using an atomic addition.
                atomicAdd(&s_hist[bin * 32 + copyIdx], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 private histogram copies for each bin and update the global histogram.
    // Threads cooperate by processing bins in a strided manner.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum over the 32 copies for the current bin.
        for (int c = 0; c < 32; c++) {
            sum += s_hist[bin * 32 + c];
        }
        // Atomically add the block's contribution to the global histogram.
        atomicAdd(&global_hist[bin], sum);
    }
}

// Host function that launches the histogram kernel. The arrays 'input' and 'histogram'
// are expected to be allocated on the device (cudaMalloc), and 'inputSize' is the number 
// of characters in the input array. The parameters 'from' and 'to' define the character 
// range for which the histogram is computed. Any necessary host-device synchronization is 
// assumed to be handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Choose a block size; 256 threads per block is a common choice.
    int threadsPerBlock = 256;
    // Compute the number of blocks needed so that each thread processes 'itemsPerThread' input characters.
    int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);

    // The histogram has 'range' bins where range = to - from + 1.
    int range = to - from + 1;
    // Allocate shared memory for 32 copies of the histogram.
    size_t sharedMemSize = range * 32 * sizeof(unsigned int);

    // Launch the CUDA kernel with the computed grid dimensions and shared memory size.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, histogram, from, to);
}