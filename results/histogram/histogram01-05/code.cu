#include <cuda_runtime.h>

/*
 * CUDA kernel that computes a histogram for characters in the range [from, from + numBins - 1].
 *
 * Parameters:
 *   input     - pointer to device memory with 'inputSize' bytes of text
 *   hist      - pointer to device memory with 'numBins' unsigned int bins
 *   inputSize - number of bytes in 'input'
 *   from      - starting character code (inclusive)
 *   numBins   - number of histogram bins = (to - from + 1)
 *
 * Each block builds a partial histogram in shared memory, then atomically
 * accumulates it into the global histogram. A grid-stride loop is used so
 * the kernel can handle arbitrarily large inputs with a fixed launch size.
 */
__global__ void histogram_range_kernel(const char * __restrict__ input,
                                       unsigned int * __restrict__ hist,
                                       unsigned int inputSize,
                                       int from,
                                       int numBins)
{
    // Shared-memory histogram for this block. We allocate the maximum possible
    // number of bins (256) and only use the first 'numBins' entries.
    __shared__ unsigned int s_hist[256];

    // Zero out the shared histogram cooperatively.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    const unsigned int globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads =
        gridDim.x * blockDim.x;

    // Grid-stride loop over the input.
    for (unsigned int idx = globalThreadId; idx < inputSize; idx += totalThreads) {
        unsigned char c = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(c) - from;

        // Only count characters that fall into the requested range.
        if (bin >= 0 && bin < numBins) {
            // Shared-memory atomics are fast on modern GPUs and drastically
            // reduce global-memory contention compared to direct global atomics.
            atomicAdd(&s_hist[bin], 1u);
        }
    }
    __syncthreads();

    // Accumulate the per-block histogram into the global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int count = s_hist[i];
        if (count > 0) {
            atomicAdd(&hist[i], count);
        }
    }
}

/*
 * Host-side helper that sets up and launches the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input text (allocated with cudaMalloc)
 *   histogram  - device pointer to output histogram array with
 *                (to - from + 1) unsigned int entries (allocated with cudaMalloc)
 *   inputSize  - number of characters in 'input'
 *   from, to   - character range [from, to] (0 <= from <= to <= 255)
 *
 * The function:
 *   1. Zeros the histogram buffer on the device.
 *   2. Configures a reasonable grid/block size.
 *   3. Launches the CUDA kernel.
 *
 * Synchronization and error checking are left to the caller, as requested.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Number of histogram bins to compute.
    const int numBins = to - from + 1;

    if (numBins <= 0) {
        // Nothing to do; invalid range or empty.
        return;
    }

    // Initialize the device histogram to zero. This is asynchronous
    // with respect to the host but ordered within the default stream.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // If there is no input data, we are done after zeroing the histogram.
    if (inputSize == 0) {
        return;
    }

    // Choose a block size that gives good occupancy on modern GPUs.
    const int blockSize = 256;

    // Choose grid size based on input size, but cap it to avoid extremely
    // large grids. 65535 is the legacy maximum for gridDim.x and is still
    // a reasonable upper bound even on modern hardware.
    int gridSize = static_cast<int>((inputSize + blockSize - 1u) / blockSize);
    const int maxGridSize = 65535;
    if (gridSize > maxGridSize) {
        gridSize = maxGridSize;
    }
    if (gridSize < 1) {
        gridSize = 1;
    }

    // Launch the kernel. No explicit synchronization or error checking
    // is performed here, as per the problem statement.
    histogram_range_kernel<<<gridSize, blockSize>>>(input,
                                                    histogram,
                                                    inputSize,
                                                    from,
                                                    numBins);
}