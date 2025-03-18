#include <cuda_runtime.h>

// Constant controlling how many input chars are processed by each thread.
// A value of 32 is chosen as a good compromise for large inputs on modern GPUs.
#define ITEMS_PER_THREAD 32

// __global__ kernel to compute a histogram for characters in the range [from, to].
// It uses shared memory for privatizing histogram updates with 32 copies
// to avoid shared memory bank conflicts.
//
// Each block allocates shared memory for "32 copies" of the histogram bins.
// Each copy has "nbins" entries, where nbins = (to - from + 1).
// Each thread processes ITEMS_PER_THREAD characters and adds counts into
// its warp-lane designated shared memory copy (identified by lane = threadIdx.x & 31).
// After processing, the 32 copies are reduced and then atomically added to the global histogram.
__global__ void histogram_kernel(const char *d_input, unsigned int *d_histogram,
                                 unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins in the histogram.
    int nbins = to - from + 1;

    // Allocate shared memory for histogram copies.
    // There will be 32 copies, each of size "nbins", to avoid bank conflicts.
    // The total shared memory size required is 32 * nbins * sizeof(unsigned int).
    extern __shared__ unsigned int s_hist[];

    // Initialize all shared memory histogram copies to zero.
    int tid = threadIdx.x;
    int total_shared_bins = 32 * nbins;
    for (int i = tid; i < total_shared_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the starting global index for this thread.
    // Each block processes blockDim.x * ITEMS_PER_THREAD characters.
    int blockOffset = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;
    int globalIndex = blockOffset + threadIdx.x;

    // Each thread uses its warp-lane index (0-31) to select its private histogram copy.
    int lane = threadIdx.x & 31;

    // Process ITEMS_PER_THREAD input characters.
    // Each successive character is spaced blockDim.x apart.
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int index = globalIndex + i * blockDim.x;
        if (index < inputSize) {
            // Read a character from global memory.
            unsigned char c = d_input[index];
            // If the character is within the specified range [from, to], update histogram.
            if (c >= (unsigned char)from && c <= (unsigned char)to) {
                int bin = c - from;
                // Update the histogram copy corresponding to this thread's lane.
                // Use atomicAdd because threads (from different warps sharing the same lane)
                // may update the same location concurrently.
                atomicAdd(&s_hist[lane * nbins + bin], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 private histogram copies into the global histogram.
    // Each thread loops over one or more bins.
    for (int bin = threadIdx.x; bin < nbins; bin += blockDim.x) {
        unsigned int binSum = 0;
        // Sum counts from all 32 copies for this bin.
        for (int i = 0; i < 32; i++) {
            binSum += s_hist[i * nbins + bin];
        }
        // Atomically accumulate the reduced bin sum into the global histogram.
        atomicAdd(&d_histogram[bin], binSum);
    }
}

// Host function that invokes the CUDA kernel.
// The input and histogram arrays are expected to be allocated on the device using cudaMalloc.
// 'inputSize' is the number of characters in the input buffer.
// 'from' and 'to' specify the continuous range of character ordinal values to histogram.
// Any host-device synchronization is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    // Define the number of threads per block.
    const int threadsPerBlock = 256;

    // Each block processes (threadsPerBlock * ITEMS_PER_THREAD) characters.
    // Calculate the required number of blocks to cover the entire input.
    int blocks = (inputSize + (threadsPerBlock * ITEMS_PER_THREAD) - 1) / (threadsPerBlock * ITEMS_PER_THREAD);

    // Determine the number of bins in the histogram.
    int nbins = to - from + 1;

    // Compute the shared memory size needed per block (32 copies of nbins integers).
    size_t sharedMemSize = 32 * nbins * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to);
}