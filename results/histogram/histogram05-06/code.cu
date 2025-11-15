#include <cuda_runtime.h>

// CUDA histogram of a text buffer over a restricted character range [from, to].
// Optimized for modern NVIDIA GPUs (e.g., A100/H100) using per-warp privatized
// histograms in shared memory to reduce contention and bank conflicts.
//
// The device input buffer and output histogram must be allocated with cudaMalloc.
// The caller is responsible for synchronization (e.g., cudaDeviceSynchronize())
// after run_histogram returns if needed.

static constexpr int WARP_SIZE      = 32;

// Number of input characters processed per thread.
// A value of 8 is a good default on modern GPUs for large inputs: it balances
// memory throughput and arithmetic intensity without causing excessive register
// pressure.
static constexpr int itemsPerThread = 8;


// CUDA kernel: computes a partial histogram for a given text range.
//
// Parameters:
//   input     - device pointer to input characters
//   histogram - device pointer to output histogram [numBins] (for range [from, to])
//   inputSize - total number of characters in input
//   from      - lower bound of character range (inclusive, 0 <= from <= 255)
//   numBins   - number of bins in histogram (to - from + 1, 1 <= numBins <= 256)
//
// The kernel uses shared memory to hold multiple copies (one per warp) of the
// histogram to reduce bank conflicts and atomic contention. Each thread processes
// 'itemsPerThread' characters.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int numBins)
{
    // Dynamic shared memory layout:
    //   sharedHist[warp 0: paddedNumBins]
    //   sharedHist[warp 1: paddedNumBins]
    //   ...
    //   sharedHist[warp (numWarps-1): paddedNumBins]
    //
    // We add +1 padding per warp histogram to break alignment patterns and reduce
    // shared memory bank conflicts when different threads access consecutive bins.
    extern __shared__ unsigned int sharedHist[];

    const int tid             = threadIdx.x;
    const int threadsPerBlock = blockDim.x;
    const int numWarps        = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
    const int warpId          = tid / WARP_SIZE;

    const int paddedNumBins   = numBins + 1;    // +1 padding to reduce bank conflicts

    // Pointer to this warp's private histogram in shared memory
    unsigned int* warpHist = sharedHist + warpId * paddedNumBins;

    // Initialize all warp-private histograms to zero
    for (int i = tid; i < numWarps * paddedNumBins; i += threadsPerBlock) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // Each thread processes 'itemsPerThread' consecutive characters.
    const unsigned int globalThreadId = blockIdx.x * threadsPerBlock + tid;
    const unsigned int baseIndex      = globalThreadId * itemsPerThread;

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = baseIndex + static_cast<unsigned int>(i);
        if (idx >= inputSize) {
            break;  // Out of range of input buffer
        }

        unsigned char c = input[idx];
        int val         = static_cast<int>(c);        // 0..255
        int bin         = val - from;                 // bin index for [from, to]

        // Only count characters within the requested range
        if (bin >= 0 && bin < numBins) {
            // Update this warp's private histogram copy in shared memory.
            // Using per-warp histograms reduces contention and shared memory
            // bank conflicts compared to a single shared histogram.
            atomicAdd(&warpHist[bin], 1U);
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into the global histogram.
    // One thread per bin (tid < numBins) accumulates this bin across all warps.
    if (tid < numBins) {
        unsigned int sum = 0;

        // Accumulate bin 'tid' across all warp-private histograms.
        for (int w = 0; w < numWarps; ++w) {
            sum += sharedHist[w * paddedNumBins + tid];
        }

        // Add the block's total for this bin into the global histogram.
        // Multiple blocks may update the same bin, hence global atomics.
        if (sum > 0) {
            atomicAdd(&histogram[tid], sum);
        }
    }
}


// Host function: run_histogram
//
// Computes a histogram over the character range [from, to] for the given input
// buffer on the GPU.
//
// Parameters:
//   input      - device pointer to input characters (cudaMalloc'ed)
//   histogram  - device pointer to output histogram (cudaMalloc'ed), size:
//                (to - from + 1) * sizeof(unsigned int)
//   inputSize  - number of chars in the input buffer
//   from, to   - character range [from, to] (0 <= from < to <= 255)
//
// Behavior:
//   - Zeroes the output histogram on device.
//   - Launches a CUDA kernel with a configuration tuned for large inputs on
//     modern GPUs, using shared memory histogram privatization.
//   - Does not perform any device synchronization; the caller is responsible
//     for synchronizing if required.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Number of bins in the requested range (inclusive)
    const int numBins = to - from + 1;

    // Handle trivial cases early
    if (numBins <= 0 || inputSize == 0) {
        return;
    }

    // Zero the histogram buffer on the device
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Launch configuration: 256 threads per block is a good general choice for
    // histogram-like workloads; combined with itemsPerThread=8, each block
    // processes 2048 characters.
    const int threadsPerBlock     = 256;
    const unsigned int elemsPerBlock =
        static_cast<unsigned int>(threadsPerBlock * itemsPerThread);

    // Number of blocks needed to cover the input
    unsigned int numBlocks = (inputSize + elemsPerBlock - 1) / elemsPerBlock;
    if (numBlocks == 0) {
        return;
    }

    const int numWarps      = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
    const int paddedNumBins = numBins + 1;

    // Shared memory size: one padded histogram per warp
    const size_t sharedMemSize =
        static_cast<size_t>(numWarps) *
        static_cast<size_t>(paddedNumBins) *
        sizeof(unsigned int);

    // Launch the kernel. The caller is responsible for error checking and
    // synchronization (e.g., cudaGetLastError(), cudaDeviceSynchronize()).
    histogram_range_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from,
        numBins);
}