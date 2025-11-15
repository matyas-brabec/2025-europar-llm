#include <cuda_runtime.h>

/*
 * CUDA histogram implementation optimized for modern NVIDIA GPUs (e.g., A100/H100).
 *
 * - The histogram is computed only for characters in the inclusive range [from, to].
 * - The input is a device pointer to a char array.
 * - The output is a device pointer to an array of unsigned int counters.
 *
 * Optimization notes:
 * - itemsPerThread controls how many input characters each thread processes.
 *   A value of 8 is a good trade-off for large inputs on recent architectures:
 *   enough work per thread to hide memory latency without significantly increasing
 *   register pressure.
 *
 * - To reduce contention and shared-memory bank conflicts, each thread block
 *   uses multiple privatized copies of the histogram in shared memory.
 *   Threads are mapped to these copies based on their threadIdx.x.
 *   This approach reduces the number of threads contending for the same
 *   shared-memory location when input data is not uniformly distributed.
 *
 * - After processing, the block-local histograms are reduced into a single
 *   histogram per block and then accumulated into the global histogram using
 *   one atomicAdd per bin per block (rather than per element).
 */

static constexpr int ITEMS_PER_THREAD   = 8;    // Number of characters processed per thread
static constexpr int THREADS_PER_BLOCK  = 256;  // Reasonable default for modern GPUs
static constexpr int HISTO_SM_COPIES    = 4;    // Number of privatized histograms per block (must be power of two)

/*
 * CUDA kernel: compute partial histogram for [from, to] range.
 *
 * Parameters:
 *   input      - device pointer to input chars
 *   histogram  - device pointer to output histogram (size = to - from + 1)
 *   inputSize  - number of characters in input buffer
 *   from, to   - inclusive character range [from, to] to be histogrammed
 *
 * Shared memory layout:
 *   sHist has HISTO_SM_COPIES copies of the histogram, each of size histSize.
 *   copy k's base starts at sHist[k * histSize].
 */
__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    extern __shared__ unsigned int sHist[];

    const int tid        = threadIdx.x;
    const int histSize   = to - from + 1;
    const int blockSize  = blockDim.x;

    // Initialize shared-memory histograms to zero.
    // Each thread zeros multiple entries if necessary.
    for (int i = tid; i < histSize * HISTO_SM_COPIES; i += blockSize) {
        sHist[i] = 0;
    }

    __syncthreads();

    // Each thread processes ITEMS_PER_THREAD consecutive characters.
    // Global thread index in terms of "threads", then multiplied by ITEMS_PER_THREAD.
    unsigned int threadGlobalIndex = blockIdx.x * blockSize + tid;
    unsigned int baseIndex         = threadGlobalIndex * ITEMS_PER_THREAD;

    // Map thread to one of the privatized histograms.
    // HISTO_SM_COPIES is a power of two; using bitwise AND is efficient.
    const int histCopy = tid & (HISTO_SM_COPIES - 1);
    unsigned int *localHistBase = sHist + histCopy * histSize;

    // Process up to ITEMS_PER_THREAD characters.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        unsigned int idx = baseIndex + i;
        if (idx >= inputSize) {
            break;
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {

            int bin = static_cast<int>(c) - from;  // bin in [0, histSize-1]

            // Shared-memory atomic is much faster than global atomic, and
            // multiple privatized copies reduce contention further.
            atomicAdd(&localHistBase[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce the privatized histograms into the global histogram.
    // Each thread handles multiple bins if necessary.
    for (int bin = tid; bin < histSize; bin += blockSize) {
        unsigned int sum = 0;

        #pragma unroll
        for (int copy = 0; copy < HISTO_SM_COPIES; ++copy) {
            sum += sHist[copy * histSize + bin];
        }

        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host function to run the histogram kernel.
 *
 * Assumptions:
 * - `input` and `histogram` are device pointers allocated with cudaMalloc.
 * - `inputSize` is the number of bytes (chars) in the input buffer.
 * - Range is specified by 0 <= from <= to <= 255.
 * - The caller is responsible for any necessary synchronization (e.g.,
 *   cudaDeviceSynchronize) after this function returns.
 *
 * Behavior:
 * - The function zeros the output histogram device buffer for the specified
 *   range [from, to].
 * - It launches the CUDA kernel with an appropriate grid size and shared
 *   memory size.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0 || from > to) {
        // Nothing to do.
        return;
    }

    const int histSize = to - from + 1;

    // Zero the output histogram on the device for the relevant range.
    // The caller is responsible for ensuring that `histogram` has at least
    // histSize elements allocated.
    cudaMemset(histogram, 0, histSize * sizeof(unsigned int));

    // Configure launch parameters.
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const unsigned int itemsPerBlock = static_cast<unsigned int>(threadsPerBlock) *
                                       static_cast<unsigned int>(ITEMS_PER_THREAD);

    // Compute number of blocks needed to cover inputSize items.
    unsigned int blocks = (inputSize + itemsPerBlock - 1u) / itemsPerBlock;

    // Guard against a zero-block launch (should not happen if inputSize > 0).
    if (blocks == 0) {
        return;
    }

    // Maximum grid dimension in x for modern GPUs is 2,147,483,647.
    const unsigned int maxGridDimX = 2147483647u;
    if (blocks > maxGridDimX) {
        blocks = maxGridDimX;
    }

    // Dynamic shared memory size: histograms per block * bins per histogram.
    const size_t sharedMemSize =
        static_cast<size_t>(histSize) *
        static_cast<size_t>(HISTO_SM_COPIES) *
        sizeof(unsigned int);

    // Launch kernel. Synchronization (if needed) is handled by the caller.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}