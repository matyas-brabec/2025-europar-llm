#include <cuda_runtime.h>
#include <stdint.h>

// Number of input characters processed by each thread.
// A value of 8 works well on recent NVIDIA data center GPUs (A100/H100)
// for large, bandwidth-bound workloads.
static constexpr int itemsPerThread = 8;

/**
 * CUDA kernel to compute a histogram of characters within a specified range.
 *
 * input      - pointer to input text (device memory)
 * histogram  - pointer to output histogram (device memory), length = to - from + 1
 * inputSize  - number of characters in input
 * from, to   - inclusive character range [from, to] to be counted
 *
 * The kernel uses shared memory to hold a per-block private histogram
 * for the given character range, then merges it into the global histogram.
 */
__global__ void histogramKernel(const char* __restrict__ input,
                                unsigned int* __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    extern __shared__ unsigned int s_hist[];

    const unsigned int ufrom  = static_cast<unsigned int>(from);
    const unsigned int uto    = static_cast<unsigned int>(to);
    const unsigned int range  = uto - ufrom + 1u;

    // Initialize per-block shared histogram to zero.
    // Only the first 'range' elements are valid; others are unused.
    for (unsigned int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    const unsigned int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads = gridDim.x * blockDim.x;

    // Each thread processes up to itemsPerThread elements in a grid-stride pattern:
    //   idx = tid + j * totalThreads, j = 0..itemsPerThread-1
    // This ensures:
    //   - All elements in [0, inputSize) are covered exactly once (if totalThreads
    //     is chosen such that totalThreads * itemsPerThread >= inputSize).
    //   - Global memory accesses are coalesced across the warp.
    #pragma unroll
    for (unsigned int j = 0; j < static_cast<unsigned int>(itemsPerThread); ++j) {
        unsigned int idx = tid + j * totalThreads;
        if (idx >= inputSize) {
            break;  // No more work for this thread.
        }

        unsigned int c = static_cast<unsigned char>(input[idx]);
        if (c >= ufrom && c <= uto) {
            unsigned int bin = c - ufrom;
            // Atomic add in shared memory (per-block histogram).
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Merge per-block histogram into the global histogram in global memory.
    for (unsigned int i = threadIdx.x; i < range; i += blockDim.x) {
        unsigned int count = s_hist[i];
        if (count > 0) {
            atomicAdd(&histogram[i], count);
        }
    }
}

/**
 * Host wrapper to launch the histogram kernel.
 *
 * input      - device pointer to input text (cudaMalloc'ed)
 * histogram  - device pointer to output histogram (cudaMalloc'ed),
 *              size = (to - from + 1) * sizeof(unsigned int)
 * inputSize  - number of characters in input
 * from, to   - inclusive character range [from, to] to be counted
 *
 * The function assumes:
 *   - input and histogram are valid device pointers.
 *   - 0 <= from <= to <= 255.
 *   - Any required host-device synchronization is handled by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0) {
        return;
    }

    const int range = to - from + 1;
    if (range <= 0) {
        return;
    }

    // Zero out the global histogram buffer on the device.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Configure launch parameters.
    // Use 256 threads per block, a good default for modern GPUs.
    const unsigned int blockSize = 256;

    // Total number of threads required so that each thread processes up to
    // 'itemsPerThread' input characters.
    const unsigned int totalThreadsNeeded =
        (inputSize + itemsPerThread - 1) / itemsPerThread;  // ceil(inputSize / itemsPerThread)

    // Number of blocks needed.
    const unsigned int gridSize =
        (totalThreadsNeeded + blockSize - 1) / blockSize;   // ceil(totalThreadsNeeded / blockSize)

    // Dynamic shared memory size per block: one bin per character in [from, to].
    const size_t sharedMemSize = static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch kernel.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}