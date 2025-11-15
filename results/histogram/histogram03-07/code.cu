// CUDA-accelerated histogram computation for a restricted character range.
//
// The kernel uses per-block shared-memory histograms to reduce the number
// of atomic operations to global memory. Each block builds its own partial
// histogram in shared memory using (fast) shared-memory atomics, and at the
// end of the kernel each block merges its partial histogram into the global
// histogram using at most (to-from+1) global atomic operations per block.

#include <cuda_runtime.h>
#include <cstddef>  // for size_t

// CUDA kernel that computes a histogram over characters in the inclusive
// range [from, to]. The input is processed in a grid-stride loop.
// - input: device pointer to the input character array
// - histogram: device pointer to the global histogram array
// - inputSize: number of characters in the input
// - from, to: inclusive character range [from, to] (0 <= from <= to <= 255)
__global__ void histogramKernel(const char *__restrict__ input,
                                unsigned int *__restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    // Number of histogram bins corresponding to the character range.
    const int numBins = to - from + 1;

    // Dynamic shared-memory allocation: one unsigned int per bin.
    extern __shared__ unsigned int s_hist[];

    // Initialize shared histogram to zero. Each thread zeroes multiple bins
    // using a grid-stride-like pattern over the bins local to the block.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
    {
        s_hist[bin] = 0;
    }

    __syncthreads();

    // Compute global linear thread index and grid stride for the main loop.
    const size_t globalThreadId =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
        static_cast<size_t>(threadIdx.x);

    const size_t gridStride =
        static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

    const unsigned int n = inputSize;
    const int localFrom = from;
    const int localTo   = to;

    // Grid-stride loop over the input array.
    // Each thread reads elements at indices: globalThreadId, globalThreadId+gridStride, ...
    for (size_t i = globalThreadId; i < static_cast<size_t>(n); i += gridStride)
    {
        // Load character and cast to unsigned to ensure values are in [0,255]
        unsigned char c = static_cast<unsigned char>(input[i]);

        // Only count characters within the desired range [from, to].
        if (c >= localFrom && c <= localTo)
        {
            const int bin = static_cast<int>(c) - localFrom;

            // Increment the corresponding shared-memory bin.
            // Shared-memory atomics are significantly faster than global atomics.
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Merge per-block partial histogram from shared memory into global memory.
    // Each thread handles multiple bins, and we only perform a global atomic
    // add when the local bin count is non-zero.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
    {
        unsigned int count = s_hist[bin];
        if (count > 0)
        {
            atomicAdd(&histogram[bin], count);
        }
    }
}

// Host function that configures and launches the CUDA kernel.
//
// Parameters:
// - input: device pointer to input character buffer (allocated with cudaMalloc)
// - histogram: device pointer to output histogram buffer with (to - from + 1) bins
// - inputSize: number of characters in the input buffer
// - from, to: inclusive character range [from, to] to be histogrammed
//
// This function does not perform device synchronization; callers are responsible
// for synchronization and error checking (e.g., cudaDeviceSynchronize and
// cudaGetLastError) after invoking this function.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of bins; expected to satisfy numBins > 0 for valid input.
    const int numBins = to - from + 1;

    // Handle trivial cases: empty range or empty input.
    if (numBins <= 0)
    {
        return;
    }

    if (inputSize == 0)
    {
        // No input data: ensure histogram is zeroed.
        cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));
        return;
    }

    // Zero the output histogram on the device. This is required because the
    // kernel only performs atomicAdd operations and assumes initial counts are zero.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose a block size suitable for modern NVIDIA GPUs (A100/H100).
    const int threadsPerBlock = 256;

    // Compute number of blocks needed. Ensure at least one block is launched.
    int blocks = static_cast<int>((static_cast<size_t>(inputSize) + threadsPerBlock - 1) /
                                  threadsPerBlock);
    if (blocks <= 0)
    {
        blocks = 1;
    }

    // Each block needs numBins * sizeof(unsigned int) bytes of shared memory.
    const size_t sharedMemBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the histogram kernel on the default stream.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemBytes>>>(
        input, histogram, inputSize, from, to);

    // No cudaDeviceSynchronize or error checking here; caller is responsible.
}