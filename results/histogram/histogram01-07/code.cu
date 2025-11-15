#include <cuda_runtime.h>

/*
 * CUDA kernel that computes a histogram for a specified character range [from, to]
 * in an input buffer of chars.
 *
 * - input:     device pointer to input characters
 * - histogram: device pointer to output histogram (size = to - from + 1)
 * - inputSize: number of characters in input
 * - from, to:  inclusive character range [from, to] (0 <= from <= to <= 255)
 *
 * The kernel uses a per-block histogram in shared memory to reduce contention on
 * global memory. Each block:
 *   1. Initializes its local histogram to zero.
 *   2. Processes a strided subset of the input.
 *   3. Atomically accumulates its local histogram into the global histogram.
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    // Dynamic shared memory: one histogram per block, limited to [from, to].
    extern __shared__ unsigned int localHist[];

    const int numBins = to - from + 1;

    // Initialize the local histogram to zero. Each thread handles multiple bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        localHist[i] = 0;
    }

    __syncthreads();

    // Compute a global thread ID and the total number of threads.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads   = blockDim.x * gridDim.x;

    // Strided loop over the input: each thread processes positions
    // globalThreadId, globalThreadId + totalThreads, ...
    for (unsigned int idx = globalThreadId; idx < inputSize; idx += totalThreads) {
        // Convert char to unsigned char to get a value in [0, 255] even if char is signed.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        int value = static_cast<int>(c);

        // Only count characters in the requested range.
        if (value >= from && value <= to) {
            int bin = value - from;  // Map value in [from, to] to bin in [0, numBins-1].
            // Use shared-memory atomics (fast on modern GPUs like A100/H100).
            atomicAdd(&localHist[bin], 1u);
        }
    }

    __syncthreads();

    // Accumulate the per-block histogram into the global histogram in device memory.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int count = localHist[i];
        if (count != 0) {
            atomicAdd(&histogram[i], count);
        }
    }
}

/*
 * Host function that prepares and launches the histogram kernel.
 *
 * Parameters:
 *   - input:     device pointer (cudaMalloc'd) to the input char buffer
 *   - histogram: device pointer (cudaMalloc'd) to the output histogram buffer
 *   - inputSize: number of chars in the input buffer
 *   - from, to:  inclusive character range [from, to] (0 <= from < to <= 255)
 *
 * Behavior:
 *   - Clears the output histogram on the device.
 *   - Configures a reasonable launch (grid and block sizes).
 *   - Launches the histogram kernel with dynamic shared memory sized to the
 *     number of bins.
 *
 * Note:
 *   - This function does not call cudaDeviceSynchronize(). The caller is
 *     responsible for any necessary synchronization and error checking.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Clamp the range defensively, even though the problem statement guarantees it.
    if (from < 0) {
        from = 0;
    }
    if (to > 255) {
        to = 255;
    }
    if (from > to) {
        // Empty or invalid range: nothing to compute.
        return;
    }

    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);

    // Zero-out the histogram on the device to prepare for accumulation.
    // This is synchronous with respect to the host thread.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Select a block size that works well across modern NVIDIA GPUs.
    const unsigned int blockSize = 256;

    // Compute grid size. We clamp to a maximum to avoid launching an excessive
    // number of tiny blocks; the kernel internally uses a strided loop to cover
    // the full input regardless of grid size.
    unsigned int gridSize;
    if (inputSize == 0) {
        // Handle empty input gracefully: launch a minimal kernel.
        gridSize = 1;
    } else {
        gridSize = (inputSize + blockSize - 1) / blockSize;
        const unsigned int maxGridSize = 65535u;
        if (gridSize > maxGridSize) {
            gridSize = maxGridSize;
        }
    }

    // Dynamic shared memory size for the per-block histogram.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the histogram kernel. No explicit synchronization here; the caller
    // should handle synchronization and error checking as needed.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input,
                                                             histogram,
                                                             inputSize,
                                                             from,
                                                             to);
}