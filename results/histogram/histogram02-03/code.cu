#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram for a restricted character range [from, to].
 *
 * - input:     device pointer to an array of chars (plain text).
 * - histogram: device pointer to an array of unsigned int of size (to - from + 1).
 *              Each element i counts occurrences of character with value (from + i).
 * - inputSize: number of chars in the input array.
 * - from, to:  inclusive range of characters (0 <= from <= to <= 255).
 *
 * Optimization strategy:
 * - Each thread block builds a private partial histogram in shared memory
 *   to minimize global memory atomic operations.
 * - Threads atomically update shared memory counters (fast on modern GPUs).
 * - After processing, threads cooperatively add the shared histogram to the
 *   global histogram using atomicAdd (one atomic per bin per block).
 * - A grid-stride loop is used so the kernel works efficiently with any input size.
 */
__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    // Dynamically sized shared memory: one bin per character in [from, to].
    extern __shared__ unsigned int s_hist[];

    const int numBins = to - from + 1;

    // Initialize shared histogram to zero.
    // Use a strided loop so all bins are covered even if numBins > blockDim.x.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over the input data.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int idx = globalThreadId; idx < inputSize; idx += stride) {
        unsigned char c = static_cast<unsigned char>(input[idx]);  // ensure 0..255

        // Check if character lies in the requested range.
        if (c >= static_cast<unsigned char>(from) && c <= static_cast<unsigned char>(to)) {
            int bin = static_cast<int>(c) - from;  // index into shared histogram
            atomicAdd(&s_hist[bin], 1u);
        }
    }
    __syncthreads();

    // Accumulate the per-block shared histogram into the global histogram.
    // Each thread handles a subset of bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val != 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host function to configure and launch the histogram kernel.
 *
 * Parameters:
 * - input:     device pointer to input text (char array).
 * - histogram: device pointer to histogram array (size: to - from + 1).
 * - inputSize: number of characters in the input array.
 * - from, to:  inclusive character range for which the histogram is computed.
 *
 * Notes:
 * - Memory for 'input' and 'histogram' must be allocated with cudaMalloc by the caller.
 * - This function zeros the histogram on device before launching the kernel.
 * - No device synchronization is performed here; the caller is responsible
 *   for any required synchronization or error checking after this call.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute the number of bins in the histogram for range [from, to].
    const int numBinsInt = to - from + 1;
    if (numBinsInt <= 0) {
        // Invalid range; nothing to do. Caller contract should avoid this case.
        return;
    }
    const unsigned int numBins = static_cast<unsigned int>(numBinsInt);

    // If there is no input data, just clear the histogram and return.
    if (inputSize == 0) {
        cudaMemset(histogram, 0, numBins * sizeof(unsigned int));
        return;
    }

    // Clear the histogram on device to start from a clean slate.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Kernel launch configuration.
    //  - 256 threads per block is a good balance for modern GPUs (A100/H100).
    //  - Grid size is chosen so that (gridSize * blockSize) >= inputSize.
    const int blockSize = 256;
    unsigned int gridSize = (inputSize + blockSize - 1) / blockSize;
    if (gridSize == 0) {
        gridSize = 1;
    }

    // Shared memory size: one unsigned int per bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input,
                                                             histogram,
                                                             inputSize,
                                                             from,
                                                             to);
}