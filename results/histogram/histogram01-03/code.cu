#include <cuda_runtime.h>

/*
 * Histogram kernel:
 *  - Computes a histogram over the character range [from, to] in the input.
 *  - Each block uses a shared-memory histogram to reduce global atomic traffic.
 *  - Final per-block histograms are accumulated into the global histogram.
 *
 * Parameters:
 *  input      - device pointer to input text (chars)
 *  globalHist - device pointer to global histogram array (size = range)
 *  size       - number of characters in input
 *  from, to   - inclusive character range [from, to], with 0 <= from <= to <= 255
 *  range      - precomputed as (to - from + 1)
 *
 * Shared memory:
 *  extern __shared__ unsigned int sHist[];  // per-block histogram of length "range"
 */
__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ globalHist,
                                unsigned int size,
                                int from,
                                int to,
                                unsigned int range)
{
    extern __shared__ unsigned int sHist[];

    const unsigned int tid       = threadIdx.x;
    const unsigned int blockSize = blockDim.x;
    const unsigned int gridSize  = blockSize * gridDim.x;

    // Initialize the shared-memory histogram to zero.
    // Threads cooperate; each thread zeros multiple bins if needed.
    for (unsigned int i = tid; i < range; i += blockSize) {
        sHist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over the input text.
    // Each thread processes multiple characters separated by gridSize.
    unsigned int idx = blockIdx.x * blockSize + tid;
    while (idx < size) {
        // Load character and convert to unsigned to avoid sign issues.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // If character falls into the [from, to] range, update shared histogram.
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {
            unsigned int bin = static_cast<unsigned int>(c - static_cast<unsigned char>(from));
            // Shared-memory atomic is significantly faster than global atomic.
            atomicAdd(&sHist[bin], 1u);
        }

        idx += gridSize;
    }

    __syncthreads();

    // Accumulate the per-block shared histogram into the global histogram.
    // Each thread updates a subset of bins to distribute work.
    for (unsigned int i = tid; i < range; i += blockSize) {
        unsigned int count = sHist[i];
        if (count > 0u) {
            atomicAdd(&globalHist[i], count);
        }
    }
}

/*
 * run_histogram:
 *  Host-side convenience function that prepares and launches the histogram kernel.
 *
 * Parameters:
 *  input      - device pointer to input text (chars), allocated with cudaMalloc
 *  histogram  - device pointer to output histogram (unsigned int array)
 *               of length (to - from + 1), allocated with cudaMalloc
 *  inputSize  - number of characters in the input buffer
 *  from, to   - inclusive character range [from, to], with 0 <= from < to <= 255
 *
 * Behavior:
 *  - Asynchronously zeros the histogram on the default stream.
 *  - Configures a 1D grid and block dimensions.
 *  - Launches the histogram kernel with dynamic shared memory sized to "range".
 *  - Does not perform host-device synchronization; caller is responsible for it.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Assume caller respects preconditions: 0 <= from < to <= 255.
    // Compute number of bins in the requested range.
    const unsigned int range = static_cast<unsigned int>(to - from + 1);

    // Asynchronously zero the output histogram (default stream).
    // This ensures that the histogram starts from a clean state.
    cudaMemsetAsync(histogram, 0, range * sizeof(unsigned int), 0);

    // If there is no input, we are done after zeroing the histogram.
    if (inputSize == 0u) {
        return;
    }

    // Choose a reasonable block size for modern GPUs.
    // 256 threads per block is a good balance between occupancy and latency hiding.
    const int blockSize = 256;

    // Compute grid size to cover the input. Using a simple 1D grid.
    unsigned int gridSize = (inputSize + blockSize - 1u) / blockSize;
    if (gridSize == 0u) {
        // This can only happen if inputSize is 0, which we already guarded against,
        // but we keep the check for robustness.
        return;
    }

    // Dynamic shared memory size: one unsigned int per bin in the range.
    const size_t sharedMemBytes = static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch the histogram kernel on the default stream.
    histogramKernel<<<gridSize, blockSize, sharedMemBytes>>>(
        input,
        histogram,
        inputSize,
        from,
        to,
        range
    );
}