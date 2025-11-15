#include <cuda_runtime.h>

/*
 * CUDA histogram kernel optimized for modern NVIDIA GPUs (A100/H100).
 *
 * Features:
 *  - Range-restricted histogram (characters in [from, to]).
 *  - Uses shared memory privatization to reduce global atomic contention.
 *  - 32 warp-lane-private copies of the histogram in shared memory.
 *  - Layout and stride chosen to avoid shared-memory bank conflicts.
 *  - Each thread processes ITEMS_PER_THREAD input bytes to increase ILP.
 */

constexpr int WARP_SIZE         = 32;
constexpr int ITEMS_PER_THREAD  = 8;  // Tunable; 8 is a good default for large inputs on modern GPUs


/**
 * Histogram kernel.
 *
 * @tparam ITEMS_PER_THREAD_T  Number of input items processed per thread.
 *
 * @param input      Device pointer to input characters.
 * @param histogram  Device pointer to output histogram (size = numBins).
 * @param inputSize  Number of characters in input.
 * @param from       Lower bound of character range (inclusive).
 * @param numBins    Number of histogram bins (to - from + 1).
 * @param stride     Stride between per-lane histogram copies in shared memory.
 *                   Must be >= numBins and odd to avoid bank conflicts.
 */
template <int ITEMS_PER_THREAD_T>
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int numBins,
                                 int stride)
{
    // Shared memory layout:
    //   s_hist[warp_lane * stride + bin]
    //
    // There are WARP_SIZE copies of the histogram, each of length "stride".
    // For a given bin 'b' and lane 'l':
    //   index = l * stride + b
    // Bank index = (index) % 32 (since sizeof(unsigned int) == 4 bytes).
    // With stride odd, gcd(stride, 32) == 1, so as l goes 0..31, banks are all distinct.
    extern __shared__ unsigned int s_hist[];

    const int tid       = threadIdx.x;
    const int blockSize = blockDim.x;
    const int laneId    = tid & (WARP_SIZE - 1); // threadIdx.x % 32

    const int histCopies = WARP_SIZE;
    const int sharedSize = histCopies * stride;

    // Initialize shared histogram copies to zero.
    for (int i = tid; i < sharedSize; i += blockSize)
    {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of the input:
    //   blockChunkSize = blockDim.x * ITEMS_PER_THREAD_T
    // Each thread processes ITEMS_PER_THREAD_T items spaced by blockDim.x.
    const unsigned int blockChunkSize = blockSize * ITEMS_PER_THREAD_T;
    unsigned int baseIndex = blockIdx.x * blockChunkSize + tid;

    for (int item = 0; item < ITEMS_PER_THREAD_T; ++item)
    {
        unsigned int dataIndex = baseIndex + static_cast<unsigned int>(item) * blockSize;
        if (dataIndex >= inputSize)
            break;

        unsigned char c = static_cast<unsigned char>(input[dataIndex]);
        int val = static_cast<int>(c);

        // Compute bin index relative to 'from'.
        int bin = val - from;

        // Fast range check: equivalent to (val >= from && val < from + numBins)
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins))
        {
            // Per-lane histogram copy to avoid intra-warp bank conflicts and atomics to same address.
            unsigned int* binPtr = &s_hist[laneId * stride + bin];
            atomicAdd(binPtr, 1u);
        }
    }

    __syncthreads();

    // Reduce 32 per-lane copies into global histogram.
    // Each thread handles multiple bins (strided by blockDim.x).
    for (int bin = tid; bin < numBins; bin += blockSize)
    {
        unsigned int sum = 0;
        for (int copy = 0; copy < histCopies; ++copy)
        {
            sum += s_hist[copy * stride + bin];
        }

        if (sum != 0)
        {
            // Global histogram is already zeroed by the host; we accumulate with atomicAdd.
            atomicAdd(&histogram[bin], sum);
        }
    }
}


/**
 * Host function to run the histogram kernel.
 *
 * @param input      Device pointer to input chars (allocated via cudaMalloc).
 * @param histogram  Device pointer to output histogram (allocated via cudaMalloc),
 *                   with space for (to - from + 1) unsigned ints.
 * @param inputSize  Number of chars in the input buffer.
 * @param from       Lower bound of character range (inclusive).
 * @param to         Upper bound of character range (inclusive).
 *
 * Notes:
 *  - This function does not perform any host-device synchronization
 *    (cudaDeviceSynchronize/cudaStreamSynchronize). The caller is responsible
 *    for synchronization if needed.
 *  - Uses the default stream (0) for both cudaMemsetAsync and kernel launch.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Number of bins in the requested range.
    const int numBins = to - from + 1;

    // For safety, handle empty input: just zero the histogram and return.
    if (inputSize == 0)
    {
        cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int), 0);
        return;
    }

    // Stride for per-lane histograms in shared memory:
    //  - At least numBins.
    //  - Must be odd to ensure gcd(stride, 32) == 1 and thus avoid bank conflicts.
    int stride = numBins;
    if ((stride & 1) == 0)  // if even, make it odd
        ++stride;

    // Kernel configuration.
    const int blockSize = 256; // Reasonable default for modern GPUs with this shared mem usage.
    const unsigned int blockChunkSize = blockSize * ITEMS_PER_THREAD;

    // Number of blocks needed to cover all input characters.
    const unsigned int gridSize =
        (inputSize + blockChunkSize - 1) / blockChunkSize;

    // Shared memory size: 32 (warp lanes) * stride bins * 4 bytes.
    const size_t sharedMemSize =
        static_cast<size_t>(WARP_SIZE) * static_cast<size_t>(stride) * sizeof(unsigned int);

    // Initialize output histogram to zero on device.
    cudaMemsetAsync(histogram, 0,
                    static_cast<size_t>(numBins) * sizeof(unsigned int),
                    0);

    // Launch kernel on default stream.
    histogram_kernel<ITEMS_PER_THREAD>
        <<<gridSize, blockSize, sharedMemSize, 0>>>(input,
                                                    histogram,
                                                    inputSize,
                                                    from,
                                                    numBins,
                                                    stride);
}