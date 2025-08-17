#include <cuda_runtime.h>

// CUDA kernel to compute histogram of characters in a specified [from, to] range.
// Each block accumulates a local histogram in shared memory to minimize expensive
// global atomics. At the end, each block atomically adds its local histogram to
// the global histogram.
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    // Compute number of bins in the requested range [from, to] (inclusive).
    const int bins = to - from + 1;

    // Dynamic shared memory allocation: one counter per bin.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared histogram to zero using all threads in the block.
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over the input data for good scalability and occupancy.
    const unsigned int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    const unsigned char from_u8 = static_cast<unsigned char>(from);
    const unsigned char to_u8   = static_cast<unsigned char>(to);
    const unsigned char* __restrict__ u8input = reinterpret_cast<const unsigned char*>(input);

    for (unsigned int i = idx; i < inputSize; i += stride) {
        // Load as unsigned to avoid negative values on platforms where char is signed.
        const unsigned char c = u8input[i];
        // If within the desired range, increment the corresponding shared-memory bin.
        if (c >= from_u8 && c <= to_u8) {
            // Shared-memory atomics are fast on modern GPUs (Ampere/Hopper).
            atomicAdd(&s_hist[static_cast<unsigned int>(c - from_u8)], 1u);
        }
    }
    __syncthreads();

    // Flush the block-local histogram to the global histogram.
    // Use multiple threads to parallelize the flush and reduce atomics by skipping zeros.
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val != 0u) {
            atomicAdd(&histogram[i], val);
        }
    }
    // No need for further synchronization; the kernel is ordered by the grid launch.
}

// Public API: Run the histogram kernel on device memory.
// - input: device pointer to input chars
// - histogram: device pointer to output bins (size: to - from + 1)
// - inputSize: number of chars in input
// - from, to: range [from, to] (0 <= from < to <= 255)
// Notes:
// - This function enqueues work on the default stream. The caller is responsible for
//   synchronization if/when needed.
// - The histogram buffer is zero-initialized here before kernel execution.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute number of bins and handle trivial cases.
    const int bins = (to - from + 1);
    if (bins <= 0) {
        return;
    }

    // Zero the output histogram (asynchronously on the default stream).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int), 0);

    // Choose launch configuration. Use occupancy API to select a good block size, taking
    // into account dynamic shared memory per block (bins * sizeof(unsigned int)).
    int minGridSize = 0;
    int blockSize   = 0;
    const size_t dynamicSmem = static_cast<size_t>(bins) * sizeof(unsigned int);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       histogram_range_kernel, dynamicSmem, 0);

    // Ensure we launch enough blocks to cover the input reasonably well.
    int gridSizeFromInput = (inputSize + blockSize - 1) / blockSize;
    int gridSize = (gridSizeFromInput > minGridSize) ? gridSizeFromInput : minGridSize;
    if (gridSize <= 0) gridSize = 1;

    // Launch the kernel with dynamic shared memory sized for the requested bin range.
    histogram_range_kernel<<<gridSize, blockSize, dynamicSmem>>>(input, histogram, inputSize, from, to);
}