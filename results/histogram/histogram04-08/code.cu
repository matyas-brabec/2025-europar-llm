#include <cuda_runtime.h>

// Tunable parameter: number of input characters processed by each thread.
// For modern NVIDIA data-center GPUs (A100/H100) and large inputs,
// 8 items per thread gives a good balance between memory throughput and occupancy.
static constexpr int itemsPerThread = 8;

/*
 * CUDA kernel that computes a histogram over a specific character range [from, to].
 *
 * Parameters:
 *   input      - device pointer to input characters
 *   histogram  - device pointer to global histogram array of size (to - from + 1)
 *   inputSize  - number of characters in the input buffer
 *   from, to   - inclusive range of character codes to count (0 <= from <= to <= 255)
 *
 * Implementation details:
 *   - Each block builds its own histogram in shared memory (sharedHist).
 *   - Shared histogram is initialized to zero by the threads of the block.
 *   - Each thread processes 'itemsPerThread' characters, reading them in a
 *     coalesced fashion.
 *   - For each character c, we compute bin = (unsigned char)c - from.
 *     If bin is within [0, numBins), we atomically increment sharedHist[bin].
 *   - After all threads in the block have updated sharedHist, the histogram is
 *     added to the global histogram using atomicAdd (one atomic per bin per block).
 */
__global__ void histogramKernel(const char *__restrict__ input,
                                unsigned int *__restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    const int numBins = to - from + 1;

    // Dynamically sized shared memory for per-block histogram
    extern __shared__ unsigned int sharedHist[];

    // Initialize shared histogram to zero. Multiple threads cooperate.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    const unsigned int blockChunkSize = blockDim.x * itemsPerThread;
    const unsigned int blockStart     = blockIdx.x * blockChunkSize;

    // Process itemsPerThread characters per thread.
    // Access pattern:
    //   For i in [0, itemsPerThread):
    //     idx = blockStart + i * blockDim.x + threadIdx.x
    // This ensures that, for each i, threads in a warp access consecutive
    // positions, resulting in coalesced global memory reads.
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = blockStart + i * blockDim.x + threadIdx.x;
        if (idx < inputSize) {
            // Load character and convert to unsigned to avoid sign-extension issues.
            unsigned char c = static_cast<unsigned char>(input[idx]);

            // Compute bin relative to 'from'. Use unsigned comparison to check range:
            //   0 <= bin < numBins  <=>  (unsigned)bin < (unsigned)numBins
            int bin = static_cast<int>(c) - from;
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
                atomicAdd(&sharedHist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // Accumulate per-block histogram into global histogram.
    // Each thread handles multiple bins to reduce the number of threads
    // performing atomics on the same locations.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int count = sharedHist[i];
        if (count > 0) {
            atomicAdd(&histogram[i], count);
        }
    }
}

/*
 * Host function that launches the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input characters (cudaMalloc'ed)
 *   histogram  - device pointer to histogram array (cudaMalloc'ed),
 *                size must be at least (to - from + 1) * sizeof(unsigned int)
 *   inputSize  - number of characters in the input buffer
 *   from, to   - inclusive range of character codes to count (0 <= from < to <= 255)
 *
 * Notes:
 *   - This function assumes 'input' and 'histogram' are already allocated
 *     on the device.
 *   - The global histogram is zeroed before the kernel is launched.
 *   - All operations are enqueued in the default stream; any required
 *     synchronization must be performed by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0 || from > to) {
        return;  // Nothing to do
    }

    const int numBins = to - from + 1;

    // Zero the global histogram buffer on the device.
    // Use cudaMemsetAsync in the default stream for asynchronous behavior;
    // the caller is responsible for synchronization.
    cudaMemsetAsync(histogram, 0, numBins * sizeof(unsigned int), 0);

    // Choose a reasonable block size for modern GPUs. 256 threads per block is a
    // common choice that balances occupancy, register usage, and scheduling.
    const int blockSize = 256;

    // Each block processes 'blockSize * itemsPerThread' characters.
    const unsigned int blockChunkSize = blockSize * itemsPerThread;

    // Number of blocks needed to cover the entire input.
    const unsigned int gridSize =
        (inputSize + blockChunkSize - 1) / blockChunkSize;

    // Dynamic shared memory size: one unsigned int per bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel in the default stream.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(input,
                                                            histogram,
                                                            inputSize,
                                                            from,
                                                            to);
}