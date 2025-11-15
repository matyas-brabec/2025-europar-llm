#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a character histogram for a specified ordinal range [from, to].
 *
 * - input:      pointer to device memory containing the input characters
 * - inputSize:  number of characters in the input buffer
 * - from, to:   inclusive range of character ordinal values (0..255) to histogram
 * - histogram:  pointer to device memory of length (to - from + 1) unsigned ints
 *
 * The kernel uses per-block privatized histograms in shared memory to reduce
 * contention on global memory atomics. Each block:
 *   1. Initializes its shared histogram to zero.
 *   2. Processes a strided portion of the input, updating the shared histogram
 *      with fast shared-memory atomics.
 *   3. After all threads finish, merges the shared histogram into the global
 *      histogram with a small number of global atomic operations.
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int inputSize,
                                 int from,
                                 int to,
                                 unsigned int* __restrict__ histogram)
{
    extern __shared__ unsigned int shared_hist[];

    const int range = to - from + 1;

    // Initialize the per-block shared histogram.
    // Each thread zeroes multiple bins with a strided loop.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        shared_hist[i] = 0u;
    }

    __syncthreads();

    // Compute global thread index and grid-stride.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads   = gridDim.x * blockDim.x;

    // Grid-stride loop over the input.
    for (unsigned int idx = globalThreadId; idx < inputSize; idx += totalThreads) {
        // Load character and convert to unsigned to avoid sign-extension issues.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Compute bin index relative to 'from'.
        // Using the unsigned comparison trick to combine the range check into one.
        int bin = static_cast<int>(c) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
            // Fast shared-memory atomic add.
            atomicAdd(&shared_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Merge the shared histogram into the global histogram.
    // Each thread is responsible for multiple bins in a strided loop.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        unsigned int count = shared_hist[i];
        if (count > 0u) {
            atomicAdd(&histogram[i], count);
        }
    }
}

/*
 * Host function to launch the histogram kernel.
 *
 * Parameters:
 *   - input:      device pointer to input text (cudaMalloc'ed)
 *   - histogram:  device pointer to histogram array of length (to - from + 1)
 *                 (cudaMalloc'ed by the caller)
 *   - inputSize:  number of characters in the input buffer
 *   - from, to:   inclusive character ordinal range [from, to] to be histogrammed,
 *                 with 0 <= from < to <= 255
 *
 * Behavior:
 *   - The function zeros the output histogram in device memory.
 *   - It launches a CUDA kernel that computes the histogram using shared memory.
 *   - No device synchronization is performed here; the caller is responsible
 *     for synchronization and error checking if desired.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Assumptions per problem statement:
    //   0 <= from < to <= 255
    // Therefore, range is at least 2 and at most 256.
    const int range = to - from + 1;

    // Ensure the output histogram starts at zero.
    // This is asynchronous with respect to the host and ordered with the kernel
    // launch in the default stream.
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Choose a reasonable block size for modern NVIDIA data-center GPUs.
    const unsigned int blockSize = 256;

    // Compute an initial grid size based on input size, then clamp to a
    // maximum 1D grid dimension to remain portable across architectures.
    unsigned int numBlocks = (inputSize + blockSize - 1) / blockSize;
    if (numBlocks == 0) {
        // Handle inputSize == 0 case: still launch one block to run initialization.
        numBlocks = 1;
    }
    const unsigned int maxBlocks = 65535u;
    if (numBlocks > maxBlocks) {
        numBlocks = maxBlocks;
    }

    // Shared memory size per block: one unsigned int per histogram bin.
    const size_t sharedMemSize = static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<numBlocks, blockSize, sharedMemSize>>>(input, inputSize, from, to, histogram);
}