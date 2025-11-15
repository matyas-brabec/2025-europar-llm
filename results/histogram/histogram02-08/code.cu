#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a partial histogram of an input character buffer.
 *
 * - The input is a device pointer to a buffer of 'char' (treated as bytes).
 * - The histogram range is [from, from + numBins - 1] with numBins = to - from + 1.
 * - The output histogram is a device pointer to 'numBins' unsigned int counters.
 *
 * Optimization strategy:
 *  - Use per-block shared-memory histograms to reduce contention on global memory.
 *  - Each block:
 *      1. Initializes its shared histogram.
 *      2. Processes a strided portion of the input, accumulating counts in shared memory
 *         using atomicAdd on shared memory (fast on modern GPUs).
 *      3. Flushes its shared histogram into the global histogram with atomicAdd.
 *  - To improve memory throughput, the main loop reads the input as uchar4 (4 bytes at
 *    a time) when possible, then handles the tail bytes separately.
 *
 * Notes:
 *  - 'from' and 'to' are in [0, 255] and define the inclusive character range.
 *  - Characters with ordinal values outside [from, to] are ignored.
 *  - The kernel expects 'histogram' to be zero-initialized before launch.
 */

__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int numBins)
{
    // Dynamic shared memory: one unsigned int per histogram bin.
    extern __shared__ unsigned int sharedHist[];

    // Initialize shared histogram to zero.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads   = blockDim.x * gridDim.x;

    // Process as uchar4 (4 bytes per load) to improve global memory throughput.
    const unsigned int n4 = inputSize / 4;  // Number of full uchar4 elements.
    const uchar4 *input4  = reinterpret_cast<const uchar4*>(input);

    for (unsigned int idx4 = globalThreadId; idx4 < n4; idx4 += totalThreads) {
        uchar4 v = input4[idx4];

        // Each element of uchar4 is an unsigned char in [0, 255].
        int bin0 = static_cast<int>(v.x) - from;
        int bin1 = static_cast<int>(v.y) - from;
        int bin2 = static_cast<int>(v.z) - from;
        int bin3 = static_cast<int>(v.w) - from;

        // Range check using unsigned comparison to fold both >=0 and <numBins.
        if (static_cast<unsigned int>(bin0) < static_cast<unsigned int>(numBins)) {
            atomicAdd(&sharedHist[bin0], 1);
        }
        if (static_cast<unsigned int>(bin1) < static_cast<unsigned int>(numBins)) {
            atomicAdd(&sharedHist[bin1], 1);
        }
        if (static_cast<unsigned int>(bin2) < static_cast<unsigned int>(numBins)) {
            atomicAdd(&sharedHist[bin2], 1);
        }
        if (static_cast<unsigned int>(bin3) < static_cast<unsigned int>(numBins)) {
            atomicAdd(&sharedHist[bin3], 1);
        }
    }

    // Handle remaining tail bytes (if inputSize is not a multiple of 4).
    const unsigned int tailStart = n4 * 4;
    for (unsigned int i = tailStart + globalThreadId; i < inputSize; i += totalThreads) {
        unsigned char c = static_cast<unsigned char>(input[i]);
        int bin = static_cast<int>(c) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
            atomicAdd(&sharedHist[bin], 1);
        }
    }

    __syncthreads();

    // Flush the shared histogram to the global histogram.
    // Each thread handles a subset of bins to distribute work evenly.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = sharedHist[i];
        if (val > 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host wrapper that configures and launches the histogram kernel.
 *
 * Parameters:
 *  - input:      device pointer to the input text buffer (chars).
 *  - histogram:  device pointer to the output histogram array of size (to - from + 1).
 *  - inputSize:  number of characters in the input buffer.
 *  - from, to:   inclusive character range [from, to] for which the histogram is computed.
 *
 * Assumptions:
 *  - 'input' and 'histogram' are allocated with cudaMalloc by the caller.
 *  - The caller is responsible for any required host-device synchronization
 *    (e.g., cudaDeviceSynchronize) after this function returns.
 *  - The caller ensures that 'histogram' has enough space for (to - from + 1) bins.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic parameter validation for robustness.
    if (from < 0 || to > 255 || from > to) {
        // Invalid parameters: no work is performed.
        return;
    }

    const int numBins = to - from + 1;

    // Zero the global histogram buffer.
    // cudaMemset is asynchronous with respect to the host; the caller is
    // responsible for synchronization if needed.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // If there is no input, we are done after zeroing the histogram.
    if (inputSize == 0) {
        return;
    }

    // Choose a reasonable block size for modern GPUs.
    const int threadsPerBlock = 256;

    // Compute the number of blocks needed to cover the input.
    int blocks = static_cast<int>((inputSize + threadsPerBlock - 1) / threadsPerBlock);
    if (blocks <= 0) {
        blocks = 1;
    }

    // Clamp the block count to a safe maximum for gridDim.x.
    const int maxBlocks = 65535;
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    // Shared memory size: one unsigned int per bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input,
                                                                 histogram,
                                                                 inputSize,
                                                                 from,
                                                                 numBins);
}