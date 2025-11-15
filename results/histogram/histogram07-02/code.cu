#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA GPUs (e.g., A100/H100).
 *
 * Key optimization points:
 *  - Use shared memory to privatize the histogram per thread block.
 *  - Avoid shared-memory bank conflicts by using 32 interleaved copies
 *    of the histogram. Each thread updates a copy determined by its lane ID
 *    (threadIdx.x % 32). For a given histogram bin i and copy c, the value
 *    is stored at offset i * 32 + c in shared memory.
 *  - Each thread processes multiple input characters (itemsPerThread) using
 *    a grid-stride loop to achieve good global memory coalescing.
 */

static constexpr int itemsPerThread   = 8;   // Number of chars processed per thread (max; actual may be less near the end).
static constexpr int NUM_HIST_COPIES = 32;  // Number of privatized histogram copies per block (one per warp lane).

// CUDA kernel: compute histogram for character range [from, to] in 'input'.
__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ globalHist,
                                 unsigned int inputSize,
                                 int from,
                                 int to,
                                 int numBins)
{
    // Dynamically allocated shared memory. Size in bytes is specified at kernel launch:
    // numBins * NUM_HIST_COPIES * sizeof(unsigned int)
    extern __shared__ unsigned int s_hist[];

    const int tidx = threadIdx.x;
    const int copyIndex = tidx & (NUM_HIST_COPIES - 1); // threadIdx.x % 32

    const int sharedSize = numBins * NUM_HIST_COPIES;

    // Initialize all shared histogram copies to zero.
    for (int i = tidx; i < sharedSize; i += blockDim.x) {
        s_hist[i] = 0;
    }

    __syncthreads();

    // Compute per-thread global index and total number of threads.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + tidx;
    const unsigned int totalThreads   = gridDim.x * blockDim.x;

    // Grid-stride loop: each thread processes up to itemsPerThread elements.
    // This ensures good memory coalescing: in a given iteration, consecutive
    // threads in a warp access consecutive memory locations.
    unsigned int processed = 0;
    for (unsigned int idx = globalThreadId;
         idx < inputSize && processed < static_cast<unsigned int>(itemsPerThread);
         idx += totalThreads, ++processed)
    {
        unsigned char ch = static_cast<unsigned char>(input[idx]);

        // Check if character is within the target range [from, to].
        if (ch >= static_cast<unsigned char>(from) &&
            ch <= static_cast<unsigned char>(to))
        {
            int bin = static_cast<int>(ch) - from;  // 0 .. numBins-1
            int offset = bin * NUM_HIST_COPIES + copyIndex;

            // Atomic add in shared memory: collisions are limited to threads
            // from different warps that share the same copyIndex.
            atomicAdd(&s_hist[offset], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 shared-memory copies into the global histogram.
    // Each thread handles multiple bins (strided by blockDim.x).
    for (int bin = tidx; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        int base = bin * NUM_HIST_COPIES;

        #pragma unroll
        for (int c = 0; c < NUM_HIST_COPIES; ++c) {
            sum += s_hist[base + c];
        }

        // Avoid unnecessary global atomics when the bin is zero in this block.
        if (sum != 0) {
            atomicAdd(&globalHist[bin], sum);
        }
    }
}

/*
 * Host function to run the histogram kernel.
 *
 * Parameters:
 *  - input:      Pointer to device memory containing the input text (chars).
 *  - histogram:  Pointer to device memory for the output histogram. Must have
 *                space for (to - from + 1) unsigned ints.
 *  - inputSize:  Number of characters in the input buffer.
 *  - from, to:   Character range [from, to] over which to compute the histogram.
 *
 * Notes:
 *  - This function assumes 'input' and 'histogram' have been allocated with cudaMalloc.
 *  - The caller is responsible for any host-device synchronization (e.g.,
 *    cudaDeviceSynchronize) after this function returns.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Number of histogram bins for the specified range.
    const int numBins = to - from + 1;

    // Zero the global histogram buffer on the device.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Early exit if there's no input.
    if (inputSize == 0) {
        return;
    }

    // Choose a reasonable block size for modern GPUs.
    const int blockSize = 256;

    // Compute how many threads are needed so that:
    //   totalThreads * itemsPerThread >= inputSize
    // This guarantees full coverage of the input in the grid-stride loop.
    const unsigned int threadsNeeded =
        (inputSize + itemsPerThread - 1u) / static_cast<unsigned int>(itemsPerThread);

    unsigned int numBlocks =
        (threadsNeeded + static_cast<unsigned int>(blockSize) - 1u) / static_cast<unsigned int>(blockSize);

    if (numBlocks == 0) {
        numBlocks = 1;
    }

    // Dynamic shared memory size: 32 copies of the histogram, interleaved to avoid bank conflicts.
    const size_t sharedMemSize =
        static_cast<size_t>(numBins) * NUM_HIST_COPIES * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<numBlocks, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to,
        numBins
    );
}