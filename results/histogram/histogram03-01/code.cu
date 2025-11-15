#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram of characters within a specified range [from, from + range).
 *
 * - input: pointer to device memory holding the input text as bytes (0..255).
 * - globalHist: pointer to device memory for the histogram (size == range).
 * - n: number of bytes in the input buffer.
 * - from: lowest character value (inclusive) to count.
 * - range: number of distinct character values to count (to - from + 1).
 *
 * The kernel uses per-block histograms stored in shared memory to reduce
 * contention on global memory atomics. Each block:
 *   1. Zeroes its shared histogram.
 *   2. Processes a grid-stride subset of the input, atomically incrementing
 *      the shared histogram for characters in [from, from + range).
 *   3. After all threads finish, accumulates the shared histogram into the
 *      global histogram using atomic adds (one per bin per block).
 *
 * This design significantly reduces the number of atomic operations on
 * global memory, which is important for performance on large inputs.
 */
__global__ void histogram_kernel(const unsigned char *__restrict__ input,
                                 unsigned int       *__restrict__ globalHist,
                                 unsigned int n,
                                 int from,
                                 int range)
{
    extern __shared__ unsigned int shHist[];

    const int  localRange          = range;
    const int  localFrom           = from;
    const int  upperExclusiveInt   = localFrom + localRange;  // exclusive upper bound
    const unsigned int lowerBound  = static_cast<unsigned int>(localFrom);
    const unsigned int upperBound  = static_cast<unsigned int>(upperExclusiveInt);

    // Step 1: Initialize shared histogram to zero.
    // Each thread zeroes a subset of the bins.
    for (int i = threadIdx.x; i < localRange; i += blockDim.x) {
        shHist[i] = 0;
    }
    __syncthreads();

    // Step 2: Process input in a grid-stride loop.
    unsigned int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride  = blockDim.x * gridDim.x;

    while (idx < n) {
        unsigned int c = static_cast<unsigned int>(input[idx]);

        // Only count characters within [from, from + range).
        if (c >= lowerBound && c < upperBound) {
            // Use shared memory atomics; these are fast on modern GPUs.
            atomicAdd(&shHist[c - lowerBound], 1u);
        }

        idx += stride;
    }

    __syncthreads();

    // Step 3: Accumulate shared histogram into global histogram.
    // Each thread is responsible for a subset of bins.
    for (int i = threadIdx.x; i < localRange; i += blockDim.x) {
        unsigned int count = shHist[i];
        if (count != 0) {
            // Global histogram is laid out so that bin i corresponds
            // to character with ordinal value (from + i).
            atomicAdd(&globalHist[i], count);
        }
    }
}

/*
 * Host function to launch the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input text (array of chars).
 *   histogram  - device pointer to output histogram (array of unsigned int).
 *   inputSize  - number of chars in the input buffer.
 *   from       - lowest character value (inclusive) to count (0 <= from <= 255).
 *   to         - highest character value (inclusive) to count (from <= to <= 255).
 *
 * The output histogram must have space for (to - from + 1) entries.
 * After this function returns (and after the caller synchronizes the device),
 * histogram[i] will contain the number of occurrences of character with
 * ordinal value (from + i).
 *
 * Assumptions:
 *   - 'input' and 'histogram' point to device memory (allocated with cudaMalloc).
 *   - The caller is responsible for any required host-device synchronization.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic parameter checks; keep behavior well-defined.
    if (input == nullptr || histogram == nullptr) {
        return;
    }
    if (inputSize == 0) {
        return;
    }

    // Clamp range to valid byte values (0..255), although the caller is expected
    // to pass valid values already.
    if (from < 0)   from = 0;
    if (to > 255)   to   = 255;
    if (from > to) {
        return;
    }

    const int range = to - from + 1;

    // Initialize the global histogram to zero before accumulation.
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Choose a reasonable launch configuration for modern GPUs.
    // 256 threads per block is a common choice balancing occupancy and
    // shared-memory contention for small histograms.
    const int threadsPerBlock = 256;

    // Compute number of blocks so that each thread initially processes at least
    // one element, relying on grid-stride looping for large inputs.
    int blocks = static_cast<int>((inputSize + threadsPerBlock - 1) / threadsPerBlock);
    if (blocks <= 0) {
        blocks = 1;
    }

    // Amount of shared memory per block: one unsigned int per histogram bin.
    const size_t sharedMemSize = static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch the kernel. The input is reinterpreted as unsigned char so that
    // character values are treated as 0..255 regardless of 'char' signedness.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        reinterpret_cast<const unsigned char *>(input),
        histogram,
        inputSize,
        from,
        range
    );
}