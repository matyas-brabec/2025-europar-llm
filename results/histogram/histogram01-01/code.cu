#include <cuda_runtime.h>

/*
 * CUDA kernel for computing a per-range character histogram.
 *
 * The kernel uses a standard two-level histogram approach:
 *   1. Each block accumulates into a shared-memory histogram for the
 *      target range [from, to].
 *   2. After processing its portion of the input, each block atomically
 *      adds its shared histogram into the global histogram.
 *
 * This reduces global-atomic contention and is efficient for ranges up
 * to 256 bins (the entire byte range).
 */
__global__ void histogram_range_kernel(const char * __restrict__ input,
                                       unsigned int * __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int shHist[];  // shared histogram for this block

    const int range = to - from + 1;          // number of bins in [from, to]
    const unsigned int tid   = threadIdx.x;
    const unsigned int blockThreads = blockDim.x;
    const unsigned int gridStride = blockThreads * gridDim.x;

    // Initialize shared histogram to zero.
    for (int i = tid; i < range; i += blockThreads) {
        shHist[i] = 0;
    }
    __syncthreads();

    // Process input in a grid-stride loop.
    unsigned int idx = blockIdx.x * blockThreads + tid;
    while (idx < inputSize) {
        unsigned char c = static_cast<unsigned char>(input[idx]);
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {
            // Map character c in [from, to] to bin index [0, range-1].
            unsigned int bin = static_cast<unsigned int>(c) - static_cast<unsigned int>(from);
            atomicAdd(&shHist[bin], 1u);
        }
        idx += gridStride;
    }
    __syncthreads();

    // Accumulate shared histogram into global histogram.
    for (int i = tid; i < range; i += blockThreads) {
        unsigned int val = shHist[i];
        if (val != 0) {
            // Global histogram is indexed from 0 to range-1, corresponding to
            // characters [from, to] respectively.
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host wrapper that sets up and launches the CUDA histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to an array of chars (text data)
 *   histogram  - device pointer to an array of unsigned int with size
 *                (to - from + 1). The function will zero this buffer
 *                before computing the histogram.
 *   inputSize  - number of chars in the input buffer
 *   from, to   - inclusive range [from, to] of character codes to count
 *                (0 <= from <= to <= 255)
 *
 * Notes:
 *   - The function assumes 'input' and 'histogram' are already allocated
 *     on the device with cudaMalloc.
 *   - Operations are enqueued in the default stream; synchronization is
 *     the responsibility of the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic parameter validation (defensive; caller is expected to pass valid values).
    if (input == nullptr || histogram == nullptr) {
        return;
    }
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const int range = to - from + 1;
    const size_t histBytes = static_cast<size_t>(range) * sizeof(unsigned int);

    // If input is empty, just zero the histogram and return.
    if (inputSize == 0) {
        cudaMemset(histogram, 0, histBytes);
        return;
    }

    // Zero the global histogram on device.
    cudaMemset(histogram, 0, histBytes);

    // Configure launch parameters.
    const int threadsPerBlock = 256;

    // Use enough blocks to cover the input, but cap to a reasonable multiple of SM count
    // to avoid excessive launch overhead on very large inputs.
    unsigned int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    int device = 0;
    cudaGetDevice(&device);
    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    // Heuristic: up to 32 blocks per SM is usually enough to saturate high-end GPUs.
    unsigned int maxBlocks = static_cast<unsigned int>(smCount) * 32u;
    if (maxBlocks == 0) {
        maxBlocks = 1; // Fallback in unlikely case of a query failure
    }
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    const size_t sharedMemSize = static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch the kernel. All operations are queued in the default stream.
    histogram_range_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}