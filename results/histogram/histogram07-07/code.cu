#include <cuda_runtime.h>

// Tunable constants.
// itemsPerThread controls how many input characters each thread processes.
// A value of 16 provides a good balance between memory throughput and per-thread work
// on modern NVIDIA data-center GPUs (e.g., A100/H100) for large inputs.
static constexpr int itemsPerThread  = 16;
// threadsPerBlock controls how many threads in each block.
// 256 threads is a common choice that achieves high occupancy with moderate shared-memory use.
static constexpr int threadsPerBlock = 256;

// CUDA kernel to compute a histogram over a specified character range [from, to].
// The input is a plain-text buffer (chars), and the output is an array of unsigned ints
// of length (to - from + 1), where histogram[i] counts occurrences of character
// (i + from) in the input.
//
// Performance notes:
// - Uses shared-memory histogram privatization with 32 copies per block to reduce
//   shared-memory bank conflicts and contention.
// - Each thread processes `itemsPerThread` characters, accessing shared memory
//   via copy index (threadIdx.x % 32).
// - Histogram copies are laid out in shared memory such that
//   value i of copy c is at offset (i * 32 + c), ensuring each copy maps
//   to its own shared-memory bank for a given bin index.
__global__ void histogram_range_kernel(const char * __restrict__ input,
                                       unsigned int * __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    // Dynamic shared memory declaration. Size is provided at kernel launch.
    extern __shared__ unsigned int sharedHist[];

    const int numBins     = to - from + 1;
    const int numCopies   = 32;          // One logical copy per bank / warp lane.
    const int laneId      = threadIdx.x & (numCopies - 1); // threadIdx.x % 32
    const int threads     = blockDim.x;
    const unsigned int blockItems = threads * itemsPerThread;

    // Guard against invalid range; should not occur with valid inputs,
    // but keeps kernel safe if from/to are mis-specified.
    if (numBins <= 0)
        return;

    // Initialize the shared-memory histograms.
    // Layout: for bin i (0 <= i < numBins) and copy c (0 <= c < 32),
    // sharedHist[i * 32 + c] holds the count.
    for (int idx = threadIdx.x; idx < numBins * numCopies; idx += threads) {
        sharedHist[idx] = 0;
    }

    __syncthreads();

    // Compute the starting global index for this block.
    const unsigned int blockStart = blockIdx.x * blockItems;

    // Process itemsPerThread characters per thread, striding by blockDim.x.
    // This pattern ensures coalesced loads from global memory.
#pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        const unsigned int globalIndex = blockStart + i * threads + threadIdx.x;
        if (globalIndex >= inputSize)
            break;

        // Convert char to unsigned to get a value in [0, 255] regardless of signedness.
        const unsigned int c = static_cast<unsigned char>(input[globalIndex]);

        // Check if the character falls within the desired [from, to] range.
        if (c >= static_cast<unsigned int>(from) && c <= static_cast<unsigned int>(to)) {
            const int bin   = static_cast<int>(c) - from;          // 0 <= bin < numBins
            const int index = bin * numCopies + laneId;            // Strided layout.

            // Atomic add in shared memory. Multiple warps can update
            // the same copy concurrently, so an atomic is required.
            atomicAdd(&sharedHist[index], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 privatized copies into the global histogram.
    // Each thread is responsible for accumulating multiple bins.
    for (int bin = threadIdx.x; bin < numBins; bin += threads) {
        unsigned int sum = 0;
        const int base = bin * numCopies;
#pragma unroll
        for (int c = 0; c < numCopies; ++c) {
            sum += sharedHist[base + c];
        }
        if (sum != 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host-side function that prepares and launches the histogram kernel.
//
// Parameters:
// - input:      device pointer to the input text buffer (cudaMalloc'ed).
// - histogram:  device pointer to the output histogram buffer (cudaMalloc'ed).
//               Must have space for (to - from + 1) unsigned ints.
// - inputSize:  number of characters in the input buffer.
// - from, to:   inclusive character range [from, to] to compute histogram over.
//
// Notes:
// - This function does not perform device synchronization; the caller is
//   responsible for synchronization as needed.
// - The function zeroes the output histogram on the device before launching
//   the kernel.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Handle trivial or invalid cases early.
    if (inputSize == 0)
        return;
    if (from > to)
        return;

    const int numBins = to - from + 1;

    // Zero initialize the global histogram buffer on the device.
    // This is asynchronous with respect to the host.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Compute launch configuration.
    const unsigned int blockItems = threadsPerBlock * itemsPerThread;
    const unsigned int numBlocks  = (inputSize + blockItems - 1) / blockItems;

    if (numBlocks == 0)
        return;

    // Compute required shared-memory size:
    // 32 copies * numBins * sizeof(unsigned int).
    const size_t sharedMemSize = static_cast<size_t>(numBins) * 32u * sizeof(unsigned int);

    // Launch the kernel. All operations are enqueued in the default stream.
    histogram_range_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}