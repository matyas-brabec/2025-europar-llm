// CUDA histogram kernel optimized with shared-memory privatization.
// Target: modern NVIDIA data-center GPUs (A100/H100).
//
// This file defines:
//   1) A CUDA kernel `histogramKernel` that computes a histogram over a
//      contiguous range of character codes [from, to] (inclusive).
//   2) A host function `run_histogram` that prepares kernel launch parameters
//      and invokes the kernel.
//
// The input is a device pointer to an array of chars (plain text).
// The output is a device pointer to an array of unsigned int of length:
//      numBins = to - from + 1
// where bin i (0 <= i < numBins) holds the count for character with value
//      (from + i).
//
// Design overview:
//   - Each thread processes `itemsPerThread` characters to amortize indexing
//     and kernel launch overhead.
//   - Each block builds a private histogram in shared memory using fast
//     shared-memory atomics.
//   - After processing its portion of the input, each block merges its
//     shared-memory histogram into the global histogram using global atomics.
//   - The global histogram is zeroed at the beginning of `run_histogram`.
//
// Notes:
//   - This implementation assumes that `input` and `histogram` point to
//     device memory allocated via `cudaMalloc`.
//   - Synchronization (e.g., cudaDeviceSynchronize) is intentionally not
//     performed inside `run_histogram`; the caller is responsible for that.
//   - The kernel is tuned for large inputs; for very small inputs, the
//     overhead of shared-memory privatization is negligible relative to
//     input size.

#include <cuda_runtime.h>

// Number of input characters processed by each thread.
// This constant can be tuned for different architectures or workloads.
// For A100/H100, 16 is a good balance between memory coalescing,
// arithmetic intensity, and register usage.
static constexpr int itemsPerThread = 16;


// CUDA kernel: compute histogram over range [from, from + numBins - 1].
//
// Parameters:
//   input      - device pointer to input chars
//   histogram  - device pointer to global histogram (size = numBins)
//   inputSize  - number of characters in `input`
//   from       - starting character code (inclusive)
//   numBins    - number of histogram bins ( = to - from + 1 )
//
// The histogram for each block is privatized in shared memory and then
// merged into the global histogram.
__global__ void histogramKernel(const char * __restrict__ input,
                                unsigned int * __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int numBins)
{
    // Shared-memory histogram, one bin per character in [from, from + numBins - 1].
    extern __shared__ unsigned int shHist[];

    // Initialize shared-memory histogram to zero.
    // Threads cooperate with a simple strided loop.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        shHist[bin] = 0;
    }

    __syncthreads();

    // Each block processes a contiguous chunk of the input.
    // Each thread processes `itemsPerThread` elements from that chunk.
    const unsigned int blockStride = blockDim.x * itemsPerThread;
    const unsigned int globalThreadBase = blockIdx.x * blockStride + threadIdx.x;

    // Unroll the per-thread loop for better ILP and fewer loop-control instructions.
    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = globalThreadBase + i * blockDim.x;
        if (idx < inputSize) {
            // Convert to unsigned char so that values are in [0, 255]
            // regardless of whether `char` is signed or unsigned.
            unsigned char c = static_cast<unsigned char>(input[idx]);

            // Compute bin index relative to `from`. We use a single bounds check:
            //   bin in [0, numBins-1]  <=>  (unsigned)bin < (unsigned)numBins
            int bin = static_cast<int>(c) - from;
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
                // Fast shared-memory atomic update.
                atomicAdd(&shHist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // Merge the shared-memory histogram into the global histogram.
    // Only one write per bin per block (plus atomic for safety across blocks).
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int val = shHist[bin];
        if (val != 0) {
            atomicAdd(&histogram[bin], val);
        }
    }
}


// Host function: prepare and launch the histogram kernel.
//
// Parameters:
//   input      - device pointer to input chars (cudaMalloc-allocated)
//   histogram  - device pointer to histogram array (cudaMalloc-allocated,
//                size = (to - from + 1) * sizeof(unsigned int))
//   inputSize  - number of characters in `input`
//   from, to   - character range to histogram, inclusive; 0 <= from < to <= 255
//
// Behavior:
//   - Clamps `from` and `to` to [0, 255] and returns immediately on invalid range
//     or zero input size.
//   - Zeros the `histogram` array on the device via cudaMemsetAsync.
//   - Launches the CUDA histogram kernel with a fixed, tuned configuration.
//   - Does NOT perform synchronization; caller must synchronize before using
//     or copying back the histogram.
//
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Clamp the range to valid byte values.
    if (from < 0)   from = 0;
    if (to   > 255) to   = 255;

    // Ensure valid range and non-empty input.
    if (from > to || inputSize == 0) {
        return;
    }

    const int numBins = to - from + 1;

    // Zero the output histogram (asynchronously on the default stream).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int), 0);

    // Kernel launch configuration.
    //  - 256 threads per block works well on A100/H100 for this style of kernel.
    //  - Each block processes threadsPerBlock * itemsPerThread input elements.
    const int threadsPerBlock = 256;
    const unsigned int itemsPerBlock = threadsPerBlock * itemsPerThread;

    // Number of blocks needed to cover all `inputSize` items.
    const unsigned int numBlocks =
        (inputSize + itemsPerBlock - 1u) / itemsPerBlock;

    if (numBlocks == 0) {
        return;
    }

    // Shared memory size per block: one unsigned int per bin.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel on the default stream (stream 0).
    histogramKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        numBins
    );

    // No synchronization here; the caller is responsible for synchronizing
    // before reading back or using `histogram`.
}