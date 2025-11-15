#include <cuda_runtime.h>

// Number of input characters processed by each thread.
// A value of 8 provides a good balance between memory throughput and
// thread-level parallelism on recent NVIDIA data-center GPUs.
static constexpr int itemsPerThread = 8;

/*
 * Kernel to compute a histogram over a specified character range [from, from + nBins - 1].
 *
 * input      - device pointer to input characters
 * histogram  - device pointer to global histogram array of size nBins
 * inputSize  - number of characters in input
 * from       - lowest character value in range (inclusive)
 * nBins      - number of histogram bins (to - from + 1)
 *
 * The kernel uses shared memory privatization with 32 copies of the histogram
 * per block to reduce contention and avoid bank conflicts.
 * Layout of shared memory:
 *   For histogram bin i (0 <= i < nBins) and copy c (0 <= c < 32),
 *   the value is stored at index: i * 32 + c.
 * Each thread uses copy index (threadIdx.x % 32).
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int nBins)
{
    extern __shared__ unsigned int shist[];

    const int tid       = threadIdx.x;
    const int lane      = tid & 31;            // copy index: 0..31
    const int stridePerBin = 32;               // distance between copies for a single bin

    // Zero the shared histograms.
    // Each block has nBins * 32 counters in shared memory.
    for (int i = tid; i < nBins * stridePerBin; i += blockDim.x) {
        shist[i] = 0;
    }
    __syncthreads();

    const unsigned int globalThreadId = blockIdx.x * blockDim.x + tid;
    const unsigned int baseIndex      = globalThreadId * itemsPerThread;

    // Process up to itemsPerThread characters per thread.
    // Characters outside [from, from + nBins - 1] are ignored.
    #pragma unroll
    for (int k = 0; k < itemsPerThread; ++k) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(k);
        if (idx >= inputSize)
            break;

        unsigned char ch = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(ch) - from;
        if (bin >= 0 && bin < nBins) {
            unsigned int offset = static_cast<unsigned int>(bin) * stridePerBin + static_cast<unsigned int>(lane);
            atomicAdd(&shist[offset], 1U);
        }
    }

    __syncthreads();

    // Reduce the 32 copies in shared memory into the global histogram.
    // Each thread handles multiple bins, striding by blockDim.x.
    for (int bin = tid; bin < nBins; bin += blockDim.x) {
        unsigned int sum = 0;
        unsigned int base = static_cast<unsigned int>(bin) * stridePerBin;

        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += shist[base + static_cast<unsigned int>(c)];
        }

        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host function to launch the histogram kernel.
 *
 * input      - device pointer to input characters (cudaMalloc'd)
 * histogram  - device pointer to output histogram (cudaMalloc'd),
 *              must have space for (to - from + 1) unsigned ints
 * inputSize  - number of characters in input
 * from       - lowest character value in range (inclusive, 0 <= from <= 255)
 * to         - highest character value in range (inclusive, from < to <= 255)
 *
 * The function:
 *  - computes the number of bins
 *  - zeros the histogram on the device
 *  - launches the CUDA kernel with appropriate grid/block dimensions
 *  - uses dynamic shared memory sized for 32 privatized copies
 *
 * Synchronization (e.g., cudaDeviceSynchronize) and error checking are
 * assumed to be handled by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int nBins = to - from + 1;

    if (nBins <= 0 || inputSize == 0) {
        // Still clear the histogram to be safe, then return.
        if (nBins > 0) {
            cudaMemsetAsync(histogram, 0, static_cast<size_t>(nBins) * sizeof(unsigned int), 0);
        }
        return;
    }

    // Clear global histogram (asynchronously on default stream).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(nBins) * sizeof(unsigned int), 0);

    // Choose a reasonable block size for modern GPUs.
    constexpr int blockSize = 256;

    // Each thread processes itemsPerThread characters.
    const unsigned int charsPerBlock = static_cast<unsigned int>(blockSize) * static_cast<unsigned int>(itemsPerThread);
    const unsigned int numBlocks =
        (inputSize + charsPerBlock - 1u) / charsPerBlock;

    if (numBlocks == 0) {
        return;
    }

    // Dynamic shared memory size: nBins * 32 copies * sizeof(unsigned int)
    const size_t sharedMemSize = static_cast<size_t>(nBins) * 32u * sizeof(unsigned int);

    histogram_kernel<<<numBlocks, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        nBins
    );
}