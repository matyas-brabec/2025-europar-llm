#include <cuda_runtime.h>

// CUDA kernel to compute a histogram for a restricted character range [from, to].
// - input: device pointer to input text buffer (chars)
// - N: number of characters in the input
// - g_hist: device pointer to output histogram (with (to - from + 1) bins)
// - from, to: inclusive range of character codes to count
//
// Implementation notes:
// - Uses per-block shared memory histogram to reduce global atomic contention.
// - Each block initializes its shared histogram, processes a grid-stride portion of the input,
//   and then atomically accumulates its partial histogram to the global histogram.
// - Characters outside [from, to] are ignored.
// - Input characters are interpreted as unsigned bytes to avoid signed-char issues.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       size_t N,
                                       unsigned int* __restrict__ g_hist,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int s_hist[];  // Shared histogram: (to - from + 1) bins

    const int numBins = to - from + 1;
    if (numBins <= 0) return; // Should not happen with valid inputs

    // Initialize shared histogram to zero
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over the input for coalesced reads and load balancing
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t idx = tid; idx < N; idx += stride) {
        unsigned int c = input[idx];          // Treat as unsigned byte [0..255]
        int bin = (int)c - from;              // Map to [0..(numBins-1)] if in range
        // Use unsigned comparison to check 0 <= bin < numBins without branches for negatives
        if ((unsigned)bin < (unsigned)numBins) {
            // Shared memory atomic add: fast on modern GPUs (A100/H100)
            atomicAdd(&s_hist[bin], 1u);
        }
    }
    __syncthreads();

    // Accumulate shared histogram into global histogram
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val) {
            atomicAdd(&g_hist[i], val);
        }
    }
}

// Host function to launch the histogram kernel.
// - input: device pointer to input text buffer
// - histogram: device pointer to output histogram buffer of size (to - from + 1) unsigned ints
// - inputSize: number of chars in input
// - from, to: inclusive range [from, to] of character codes (0 <= from <= to <= 255)
//
// Notes:
// - The function zeroes the output histogram before launching the kernel.
// - It uses a heuristic grid size (8 blocks per SM) and 256 threads per block.
// - No host-device synchronization is performed here; the caller is responsible for synchronization.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = to - from + 1;
    if (numBins <= 0) {
        // Invalid range: just zero out (if possible) and return
        if (histogram) {
            cudaMemsetAsync(histogram, 0, 0);
        }
        return;
    }

    // Zero-out the output histogram (as it may not be pre-initialized)
    cudaMemsetAsync(histogram, 0, (size_t)numBins * sizeof(unsigned int));

    // If there's no input to process, we're done
    if (inputSize == 0) {
        return;
    }

    // Choose launch configuration
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    const int blockSize = 256;                    // Reasonable default for Ampere/Hopper
    const int maxBlocks = props.multiProcessorCount * 8; // 8 blocks per SM is a common heuristic

    // Upper-bound the number of blocks by the amount of work to avoid launching too many idle blocks
    int blocksByWork = (int)((inputSize + blockSize - 1) / blockSize);
    int gridSize = blocksByWork > 0 ? blocksByWork : 1;
    if (gridSize > maxBlocks) gridSize = maxBlocks;
    if (gridSize < 1) gridSize = 1;

    // Dynamic shared memory size: one 32-bit counter per bin
    size_t smemBytes = (size_t)numBins * sizeof(unsigned int);

    // Launch the kernel
    histogram_range_kernel<<<gridSize, blockSize, smemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        static_cast<size_t>(inputSize),
        histogram,
        from,
        to
    );
}