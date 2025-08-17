#include <cuda_runtime.h>

// CUDA kernel for computing the histogram over a specified character range.
// Strategy:
//   - Each block builds a private histogram in shared memory to reduce global atomic contention.
//   - Threads iterate over the input using a grid-stride loop and atomically increment shared bins for characters in [from, to].
//   - After processing, each block merges its shared histogram into the global histogram via atomicAdd (numBins adds per block).
// Notes:
//   - Characters are treated as unsigned to avoid sign-extension issues with char.
//   - Dynamic shared memory is used with size = (to - from + 1) * sizeof(unsigned int).
//   - The output histogram must be zero-initialized before the kernel runs; run_histogram handles this.
//   - The kernel is optimized for modern GPUs (A100/H100) where shared-memory atomics are efficient.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ globalHist,
                                       unsigned int n,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int s_hist[];
    const int numBins = to - from + 1;

    // Initialize shared histogram to zero (cooperatively across threads).
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over the input.
    unsigned int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    while (idx < n) {
        unsigned int c = static_cast<unsigned int>(input[idx]);  // 0..255
        int bin = static_cast<int>(c) - from;
        // Use unsigned comparison for range check: 0 <= bin < numBins
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
            atomicAdd(&s_hist[bin], 1u);
        }
        idx += stride;
    }
    __syncthreads();

    // Merge shared histogram into global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val) {
            atomicAdd(&globalHist[i], val);
        }
    }
}

// Host function that configures and launches the histogram kernel.
// Parameters:
//   - input:      device pointer to input chars (plain text)
//   - histogram:  device pointer to output histogram (size = to - from + 1)
//   - inputSize:  number of chars in input
//   - from, to:   inclusive character range [from, to] (0 <= from <= to <= 255)
// Behavior:
//   - Zero-initializes the output histogram.
//   - Chooses a launch configuration using occupancy to limit the number of blocks,
//     thus avoiding excessive global merge atomics for very large inputs.
//   - Launches the kernel with dynamic shared memory sized to the number of bins.
// Synchronization:
//   - This function does not synchronize; the caller is responsible for any required synchronization.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate range; handle degenerate or invalid ranges safely.
    if (from < 0 || to < 0 || from > 255 || to > 255 || from > to) {
        return;
    }

    const int numBins = to - from + 1;
    const size_t histBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Zero-initialize output histogram (asynchronous on the default stream).
    cudaMemset(histogram, 0, histBytes);

    // If there's nothing to process, we're done.
    if (inputSize == 0) {
        return;
    }

    // Choose a reasonable block size. 256 threads is a good balance on A100/H100.
    const int blockSize = 256;

    // Compute dynamic shared memory requirement.
    const size_t smemBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Determine a grid size that balances throughput and minimizes global merge overhead.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, histogram_range_kernel, blockSize, smemBytes);
    if (blocksPerSM <= 0) {
        blocksPerSM = 1; // Fallback
    }

    // Upper bound on blocks: a small multiple of fully resident blocks across all SMs.
    // This keeps the number of global atomic merges manageable even for huge inputs.
    const int smCount = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    const int maxUsefulBlocks = blocksPerSM * smCount * 4; // Over-subscribe by ~4x to hide memory latency.

    // Also ensure enough blocks to cover the input with at least 1 element per thread on average.
    const unsigned int gridNeeded = (inputSize + blockSize - 1) / blockSize;

    int gridSize = static_cast<int>(gridNeeded);
    if (gridSize > maxUsefulBlocks) gridSize = maxUsefulBlocks;
    if (gridSize < 1) gridSize = 1;

    // Launch kernel. Cast input to unsigned char* for 0..255 values.
    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<gridSize, blockSize, smemBytes>>>(d_input, histogram, inputSize, from, to);
}