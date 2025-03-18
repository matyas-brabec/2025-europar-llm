#include <cuda_runtime.h>

// Default constant controlling the number of input items processed per thread.
// For modern GPUs and large inputs, 64 tends to work well.
constexpr int itemsPerThread = 64;

// CUDA kernel to compute histogram for characters in the range [from, to].
// The kernel uses shared memory privatization with 32 copies to avoid bank conflicts:
//    - Shared memory is allocated as an array of 32 copies of the histogram,
//      with layout such that for a given bin j, the 32 copies are stored at
//      indices (0 + 32*j), (1 + 32*j), ..., (31 + 32*j).
//    - Each thread uses its warp lane (threadIdx.x % 32) to select the copy to update;
//      threads in the same warp have unique lane ids, so they update distinct addresses
//      thereby avoiding intra-warp bank conflicts.
// After processing, the kernel reduces each block's shared histogram and adds the result
// atomically into the global output histogram.
__global__ void histogram_kernel(const char *input, unsigned int *global_hist,
                                 unsigned int inputSize, int from, int range)
{
    // Allocate shared memory histogram.
    // There are 32 copies for 'range' bins.
    extern __shared__ unsigned int s_hist[]; // size: 32 * range elements

    // Compute thread id
    int tid = threadIdx.x;
    // Each thread determines its bank index based on its warp lane.
    int bank = tid % 32; 

    // Total number of shared memory elements is 32 * range.
    int totalSharedElements = 32 * range;
    // Initialize the shared memory histogram.
    for (int i = tid; i < totalSharedElements; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the starting index for this thread.
    // Each thread block processes (blockDim.x * itemsPerThread) input elements.
    // Each thread starts at:
    //   base = blockIdx.x * blockDim.x * itemsPerThread + threadIdx.x
    int base = blockIdx.x * blockDim.x * itemsPerThread + tid;
    
    // Process up to itemsPerThread characters assigned to this thread.
    // The stride between successive items is blockDim.x.
    for (int i = 0; i < itemsPerThread; i++) {
        int idx = base + i * blockDim.x;
        if (idx < inputSize) {
            // Read the character from the input.
            unsigned char ch = input[idx];
            // Check if the character is within the desired range [from, to].
            if (ch >= from && ch <= (from + range - 1)) {
                int bin = static_cast<int>(ch) - from;
                // Each thread updates the histogram copy corresponding to its bank.
                // The index in shared memory is computed as: bank + 32 * bin.
                atomicAdd(&s_hist[bank + 32 * bin], 1);
            }
        }
    }
    __syncthreads();

    // Each thread now reduces a subset of the histogram bins by summing all 32 copies,
    // and then atomically adds the partial sum to the corresponding global histogram bin.
    for (int bin = tid; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum the contributions from all 32 copies for this bin.
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            sum += s_hist[j + 32 * bin];
        }
        if (sum > 0) {
            // Atomic update to global memory histogram.
            atomicAdd(&global_hist[bin], sum);
        }
    }
}

// Host function that sets up and invokes the histogram kernel.
// Assumption:
//    - 'input' is a device pointer to a buffer with 'inputSize' characters.
//    - 'histogram' is a device pointer allocated with enough space
//      to hold (to - from + 1) unsigned integers.
//    - The caller is responsible for any host-device synchronization.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of histogram bins.
    int range = to - from + 1;

    // Choose block size.
    int threadsPerBlock = 256;
    // Each block processes (threadsPerBlock * itemsPerThread) input items.
    int blockItemCount = threadsPerBlock * itemsPerThread;

    // Calculate grid size ensuring all input elements are processed.
    int numBlocks = (inputSize + blockItemCount - 1) / blockItemCount;

    // Calculate the required dynamic shared memory size: 32 copies * range bins * sizeof(unsigned int).
    size_t sharedMemSize = 32 * range * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, range);
}