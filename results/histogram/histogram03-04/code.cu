#include <cuda_runtime.h>

/*
 * CUDA kernel that computes a partial histogram for characters in the range [from, to].
 *
 * Each block maintains a private histogram in shared memory to reduce contention on global
 * memory atomics. Threads within a block increment the shared histogram using shared-memory
 * atomics, then the shared histogram is merged into the global histogram using global atomics.
 *
 * Assumptions:
 *   - 0 <= from <= to <= 255
 *   - (to - from + 1) <= 256  (max number of bins)
 *   - 'input' and 'global_hist' point to device memory.
 */
__global__ void histogram_shared_kernel(const char * __restrict__ input,
                                        unsigned int * __restrict__ global_hist,
                                        unsigned int inputSize,
                                        int from,
                                        int to)
{
    // Maximum number of bins is 256, as chars are 8-bit.
    // We always allocate 256 bins in shared memory, but only [0, numBins) are used.
    __shared__ unsigned int shared_hist[256];

    const int range_from = from;
    const int range_to   = to;
    const int numBins    = range_to - range_from + 1;

    // Initialize the per-block shared histogram to zero.
    // Each thread zeroes a subset of bins to parallelize the initialization step.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        shared_hist[i] = 0;
    }

    __syncthreads();

    // Grid-stride loop over the input data.
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < inputSize;
         idx += stride)
    {
        unsigned char c = static_cast<unsigned char>(input[idx]); // Ensure 0..255

        // If character is in the desired range, increment corresponding bin.
        if (c >= range_from && c <= range_to) {
            // Subtract range_from to map [from, to] -> [0, numBins-1]
            atomicAdd(&shared_hist[c - range_from], 1u);
        }
    }

    __syncthreads();

    // Merge the per-block shared histogram into the global histogram.
    // Again, threads cooperate so each thread handles a subset of bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = shared_hist[i];
        if (val != 0) {
            atomicAdd(&global_hist[i], val);
        }
    }
}

/*
 * Host function that launches the histogram kernel.
 *
 * Parameters:
 *   - input:      device pointer to an array of chars (text data)
 *   - histogram:  device pointer to an array of unsigned int (output histogram)
 *   - inputSize:  number of chars in the input array
 *   - from, to:   inclusive character range [from, to] in 0..255
 *
 * Behavior:
 *   - The function zeroes the output histogram for the bins corresponding to [from, to].
 *   - It then launches the CUDA kernel that computes the histogram for this range.
 *   - No device synchronization is performed here; it's expected to be handled by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of bins in the requested range and validate basic conditions.
    if (from < 0 || to > 255 || from > to) {
        // Invalid range; nothing is done. In production code, you might want to
        // handle this error case explicitly.
        return;
    }

    const int numBins = to - from + 1;

    // Initialize (zero) only the part of the histogram that we are going to use.
    // The caller is responsible for ensuring that 'histogram' has at least 'numBins'
    // elements allocated.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose a reasonable block size. 256 threads per block works well on modern GPUs.
    const int threadsPerBlock = 256;

    // Compute number of blocks.
    // We use a simple heuristic: one thread per element up to inputSize, with a minimum
    // of one block. The kernel itself is grid-stride, so larger inputs are handled
    // correctly even with limited blocks.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks <= 0) {
        blocks = 1;
    }

    // Optionally, we could cap 'blocks' to some maximum value for portability.
    // On modern GPUs (A100/H100, sm_80+), gridDim.x can be very large, so no cap is
    // strictly necessary for correctness.

    // Launch the kernel. Shared memory size is fixed (static allocation), so we pass 0.
    histogram_shared_kernel<<<blocks, threadsPerBlock>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );

    // No cudaDeviceSynchronize() here, as the caller is responsible for synchronization.
}