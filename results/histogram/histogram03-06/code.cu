#include <cuda_runtime.h>
#include <algorithm>

// This kernel computes a histogram for the inclusive character range [from, to].
// Optimization notes:
// - Each thread block maintains a private histogram in shared memory (s_hist).
//   Threads atomically update s_hist instead of the global histogram, which
//   reduces the number of global atomic operations to at most (to - from + 1)
//   per block.
// - After processing, threads cooperatively add the shared histogram to the
//   global histogram using atomicAdd only for non-zero bins.
// - Input text is processed via a grid-stride loop for scalability and to keep
//   memory accesses coalesced.
// - Characters are treated as unsigned to correctly map the full 0..255 range.
// Assumptions:
// - 0 <= from < to <= 255 (range length is at most 256).
// - 'histogram' points to an array of (to - from + 1) unsigned ints in device memory.
// - 'input' points to 'inputSize' bytes in device memory.
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from, int to)
{
    const int range = to - from + 1;
    if (range <= 0) return; // Guard against invalid range.

    extern __shared__ unsigned int s_hist[];

    // Zero-initialize the block-private histogram in shared memory.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over the input.
    const unsigned int totalThreads = blockDim.x * gridDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < inputSize; idx += totalThreads) {
        unsigned char c = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(c) - from;
        // Branchless-in-bounds check: converts to unsigned to fold both lower and upper bound into one compare.
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
            // Shared memory atomics are fast on modern NVIDIA GPUs (Ampere/Hopper).
            atomicAdd(&s_hist[bin], 1u);
        }
    }
    __syncthreads();

    // Accumulate the block's shared histogram into the global histogram.
    // Only perform a global atomicAdd for non-zero bins to reduce contention and traffic.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        unsigned int count = s_hist[i];
        if (count) {
            atomicAdd(&histogram[i], count);
        }
    }
    // No need for further synchronization; completion is ordered by the stream outside.
}

// Host function to set up and launch the histogram kernel.
//
// Parameters:
// - input: device pointer to input characters (cudaMalloc'ed)
// - histogram: device pointer to output histogram (cudaMalloc'ed) with length (to - from + 1)
// - inputSize: number of characters in 'input'
// - from, to: inclusive range [from, to] within 0..255 to compute
//
// Notes:
// - The function zeros the output histogram prior to kernel launch to ensure correctness.
// - Launch configuration uses a grid-stride kernel, so the grid size is chosen to
//   cover the device reasonably well; exact synchronization is left to the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int range = to - from + 1;
    if (range <= 0) {
        // Invalid or empty range; nothing to do.
        return;
    }

    // Zero the output histogram (only the relevant bins).
    // Using cudaMemset is sufficient because the histogram bins are 32-bit
    // and we're setting to zero. No host/device synchronization is forced here;
    // ordering is ensured within the default stream.
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    if (inputSize == 0) {
        return; // No input; histogram remains zeroed.
    }

    // Choose a launch configuration.
    //  - 256 threads per block is a good default for Ampere/Hopper.
    //  - Grid size: min(problem size in blocks, a multiple of SM count for good occupancy).
    int blockSize = 256;

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    // Upper bound the number of blocks to a multiple of SMs to maintain load.
    // Using 32x SMs provides ample parallelism without excessive oversubscription.
    int maxBlocks = std::max(1, prop.multiProcessorCount * 32);

    // Limit by problem size so we don't launch more blocks than useful.
    int blocksBySize = (inputSize + blockSize - 1) / blockSize;
    int gridSize = std::min(maxBlocks, std::max(1, blocksBySize));

    // Dynamic shared memory size for the block-private histogram (range <= 256).
    size_t sharedBytes = static_cast<size_t>(range) * sizeof(unsigned int);

    histogram_range_kernel<<<gridSize, blockSize, sharedBytes>>>(input, histogram, inputSize, from, to);
}