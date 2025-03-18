// This file implements an optimized CUDA kernel to compute a histogram for a text buffer
// restricted to a specified continuous range of character codes [from, to]. The kernel
// uses shared memory privatization with multiple copies (one per warp) to reduce global
// and shared memory atomic contention and avoid bank conflicts via padding. Each thread
// processes a fixed number of input items (itemsPerThread) to amortize the cost of
// memory loads over many histogram updates.
// The host function run_histogram launches the kernel with computed grid/block dimensions
// and shared memory size based on the histogram range and a chosen block size.
//
// The code is optimized for modern NVIDIA GPUs (e.g., A100 or H100) and compiled with the
// latest CUDA toolkit and host compiler.

#include <cuda_runtime.h>

// Constant that specifies how many input characters are processed by each thread.
// A value of 64 is chosen to balance memory throughput and per-thread work for modern GPUs.
constexpr int itemsPerThread = 64;

// CUDA kernel that computes a histogram for characters in the range [from, to].
// The input text is provided in device memory in the array "input" of length "inputSize" (chars).
// The output histogram is stored in "outputHistogram" (device memory) at index i corresponding
// to the count of character with code (i + from). Only characters in [from, to] are counted.
//
// The kernel uses shared memory with multiple (replicated) copies of the histogram per block.
// Each warp gets its own private histogram copy stored in shared memory with one extra padding
// element to avoid bank conflicts.  Once each thread processes its "itemsPerThread" consecutive
// characters and accumulates a private count in registers, the counts are atomically added into
// the per-warp shared memory copy. Finally, one thread per bin reduces the contributions from all
// warps and updates the global (output) histogram using atomicAdd.
/// @FIXED
/// extern "C" __global__
extern "C" __global__
void histogram_kernel(const char *input, unsigned int inputSize, unsigned int *outputHistogram, int from, int to)
{
    // Compute the number of bins.
    const int nBins = to - from + 1;

    // Determine warp-related values.
    const int warpSize = 32;
    const int numWarps = (blockDim.x + warpSize - 1) / warpSize;  // number of warps per block
    const int warpId = threadIdx.x / warpSize;
    const int lane   = threadIdx.x % warpSize;

    // Allocate dynamic shared memory for replicated histograms.
    // Each warp gets its own histogram copy padded to (nBins+1) elements (last element is unused).
    extern __shared__ unsigned int s_hist[];
    // Pointer to the histogram copy for the current thread's warp.
    unsigned int* localHist = s_hist + warpId * (nBins + 1);

    // Initialize the per-warp histogram to zero.
    // Use only the lanes in the warp to cover all bins.
    for (int bin = lane; bin < nBins; bin += warpSize) {
        localHist[bin] = 0;
    }
    // (The extra padding element is not used.)
    __syncthreads();

    // Calculate the global thread ID.
    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes itemsPerThread consecutive characters starting at:
    int start = globalThreadId * itemsPerThread;

    // Each thread keeps a private (register) histogram for its assigned items.
    // Maximum possible histogram range is 256 bins, so we declare an array of size 256.
    unsigned int localCount[256];
    // Only use the first nBins elements; initialize them to zero.
    for (int i = 0; i < nBins; i++) {
        localCount[i] = 0;
    }

    // Process "itemsPerThread" characters; check bounds to avoid overrunning the input.
    for (int i = 0; i < itemsPerThread; i++) {
        int index = start + i;
        if (index < inputSize) {
            // Load the character and interpret it as unsigned.
            unsigned char ch = input[index];
            // Update local histogram if the character is within the desired range.
            if (ch >= from && ch <= to) {
                localCount[ch - from]++;
            }
        }
    }

    // Merge the thread-local histogram into the shared memory copy for its warp.
    for (int bin = 0; bin < nBins; bin++) {
        if (localCount[bin] != 0) {
            // atomicAdd in shared memory is available on modern GPUs.
            atomicAdd(&localHist[bin], localCount[bin]);
        }
    }
    __syncthreads();

    // One thread (or a set of threads strided by blockDim.x) per bin reduces the
    // contributions from all warp-level histogram copies and updates the global histogram.
    for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate contributions from each warp's histogram.
        for (int w = 0; w < numWarps; w++) {
            sum += s_hist[w * (nBins + 1) + bin];
        }
        // Update the global histogram using atomicAdd, since multiple blocks may write concurrently.
        atomicAdd(&outputHistogram[bin], sum);
    }
}

// Host function that sets up and invokes the CUDA kernel to compute the histogram.
// Parameters:
//   input         - pointer to the input text (device memory), as an array of chars
//   histogram     - pointer to the output histogram (device memory); the histogram
//                   has (to - from + 1) elements where index i holds the count for character (i+from)
//   inputSize     - number of characters in the input array
//   from, to      - the range of character ordinal values to include in the histogram (0 <= from < to <= 255)
//
// This function computes grid and block dimensions based on the total number of items and launches
// the kernel with the appropriate dynamic shared memory size.
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins required.
    const int nBins = to - from + 1;

    // Zero out the destination histogram in device memory.
    // It is assumed that "histogram" is allocated using cudaMalloc and has room for nBins unsigned ints.
    cudaMemset(histogram, 0, nBins * sizeof(unsigned int));

    // Choose a block size.
    // A block size of 256 threads is a typical choice for modern GPUs.
    const int threadsPerBlock = 256;

    // Compute the total number of threads needed.
    // Each thread processes "itemsPerThread" characters.
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;

    // Calculate the number of blocks required.
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Compute the number of warps per block.
    int warpsPerBlock = (threadsPerBlock + 31) / 32;
    // Compute size (in bytes) of the dynamic shared memory to allocate per block.
    // Each warp gets (nBins + 1) unsigned integers (the extra element is for padding to avoid bank conflicts).
    size_t sharedMemSize = warpsPerBlock * (nBins + 1) * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, histogram, from, to);
}
