#include <cuda_runtime.h>

/*
 * itemsPerThread controls how many input characters are processed per thread
 * in each "segment" of the grid-stride loop inside the kernel.
 *
 * A value of 16 is a good balance for modern NVIDIA data center GPUs (A100/H100):
 * it improves memory throughput via loop unrolling and amortizes per-thread
 * overhead without significantly harming occupancy.
 */
static constexpr int itemsPerThread = 16;

/*
 * CUDA kernel that computes a histogram over a restricted character range.
 *
 * Parameters:
 *   input      - pointer to device memory containing 'inputSize' chars
 *   globalHist - pointer to device memory with (to - from + 1) uint bins
 *   inputSize  - number of characters in 'input'
 *   from, to   - inclusive range of character codes [from, to] to be counted
 *
 * The kernel uses per-block shared memory histograms to reduce contention on
 * the global histogram. Each block:
 *   1. Initializes its shared-memory histogram to zero.
 *   2. Processes its portion of the input using a grid-stride loop, updating
 *      the shared histogram using fast shared-memory atomics.
 *   3. Atomically accumulates its partial histogram into the global histogram.
 */
__global__ void histogramKernel(const char *__restrict__ input,
                                unsigned int *__restrict__ globalHist,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    extern __shared__ unsigned int s_hist[];

    const int numBins = to - from + 1;
    const int tid = threadIdx.x;
    const int blockThreads = blockDim.x;

    const unsigned int globalThreadId = blockIdx.x * blockThreads + tid;
    const unsigned int totalThreads = gridDim.x * blockThreads;

    // Use size_t for indices/strides to avoid overflow for very large inputs.
    const size_t N = static_cast<size_t>(inputSize);
    const size_t threadBase = static_cast<size_t>(globalThreadId) * itemsPerThread;
    const size_t stride = static_cast<size_t>(totalThreads) * itemsPerThread;

    // 1. Initialize shared-memory histogram (each thread zeroes a subset of bins).
    for (int bin = tid; bin < numBins; bin += blockThreads) {
        s_hist[bin] = 0;
    }
    __syncthreads();

    // 2. Process input in segments of itemsPerThread per thread, in a grid-stride loop.
    for (size_t baseIndex = threadBase; baseIndex < N; baseIndex += stride) {
#pragma unroll
        for (int i = 0; i < itemsPerThread; ++i) {
            size_t idx = baseIndex + static_cast<size_t>(i);
            if (idx >= N) {
                break;
            }

            unsigned char c = static_cast<unsigned char>(input[idx]);
            if (c >= static_cast<unsigned char>(from) &&
                c <= static_cast<unsigned char>(to)) {
                int bin = static_cast<int>(c) - from;
                // Shared-memory atomics are fast on modern GPUs.
                atomicAdd(&s_hist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // 3. Accumulate per-block histograms into the global histogram.
    for (int bin = tid; bin < numBins; bin += blockThreads) {
        unsigned int val = s_hist[bin];
        if (val != 0) {
            atomicAdd(&globalHist[bin], val);
        }
    }
}

/*
 * Host function that sets up and launches the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input characters (cudaMalloc'd)
 *   histogram  - device pointer to output histogram (cudaMalloc'd)
 *   inputSize  - number of characters in 'input'
 *   from, to   - inclusive character code range [from, to] to be counted
 *
 * Requirements/assumptions:
 *   - 0 <= from < to <= 255
 *   - 'histogram' has room for (to - from + 1) unsigned int elements.
 *   - The caller handles any required device synchronization.
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

    const int numBins = to - from + 1;

    // Ensure output histogram is zeroed before accumulation.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose a reasonable block size for modern GPUs.
    const int threadsPerBlock = 256;

    /*
     * Determine grid size.
     *
     * We aim for enough total threads such that each thread has roughly
     * itemsPerThread characters to process on the first pass. For very
     * large inputs, a grid-stride loop in the kernel ensures full coverage,
     * so we cap the number of blocks to avoid launching an excessively
     * large grid.
     */
    unsigned int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1)
                          / (threadsPerBlock * itemsPerThread);
    if (blocks == 0) {
        blocks = 1;
    }

    // Cap the number of blocks to a reasonable value for launch overhead.
    const unsigned int maxBlocks = 1024;
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    // Shared memory size: one unsigned int per histogram bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel on the default stream.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}