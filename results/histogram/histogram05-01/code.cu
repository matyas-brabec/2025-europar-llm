#include <cuda_runtime.h>

// Optimized CUDA histogram restricted to a given char range [from, to] inclusive.
// Key optimizations:
// - Per-block shared-memory privatized histograms to minimize global atomics
// - 32-way histogram replication (one per shared memory bank) with padding to avoid bank conflicts
// - Each thread processes a fixed number of items (itemsPerThread) for high memory throughput
// - Coalesced global memory accesses within each block
//
// Notes for tuning on modern NVIDIA GPUs (A100/H100):
// - itemsPerThread = 16 is a solid default for large inputs (balances latency hiding and register usage)
// - blockDim.x = 256 gives 8 warps per block and, with ~32 KB shared memory for full 256-bin range, good occupancy
// - Shared memory atomics are fast on these architectures

// Compile-time tuning constants
static constexpr int itemsPerThread = 16;  // Number of input chars processed by each thread (default tuned for modern GPUs)
static constexpr int kSubHists      = 32;  // One copy per shared memory bank
static constexpr int kSmemPad       = 1;   // Padding to avoid bank conflicts when numBins is a multiple of 32

// CUDA kernel: builds a histogram for bytes in [from, from + numBins - 1] inclusive.
// - input: device pointer to input chars
// - n: number of chars in input
// - from: lower bound of range (0..255)
// - numBins: number of bins = to - from + 1
// - out: device pointer to output histogram of length numBins
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int n,
                                       int from,
                                       int numBins,
                                       unsigned int* __restrict__ out)
{
    extern __shared__ unsigned int sHist[];  // Size: (numBins + kSmemPad) * kSubHists
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;           // Lane ID within the warp [0..31]
    const int blockT   = blockDim.x;
    const int copyStride = numBins + kSmemPad;

    // Zero initialize shared memory histograms
    for (int i = tid; i < copyStride * kSubHists; i += blockT) {
        sHist[i] = 0u;
    }
    __syncthreads();

    // Coalesced loading pattern: each thread processes itemsPerThread bytes
    // Base index for this block
    const unsigned int blockBase = (unsigned int)blockIdx.x * (unsigned int)blockT * (unsigned int)itemsPerThread;
    const unsigned int threadBase = blockBase + (unsigned int)tid;

    const unsigned char* __restrict__ in = reinterpret_cast<const unsigned char*>(input);
#pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = threadBase + (unsigned int)i * (unsigned int)blockT;
        if (idx < n) {
            // Single-compare in-range test using unsigned arithmetic:
            // If v - from < numBins, then v in [from, from+numBins-1]
            unsigned int v = (unsigned int)in[idx];
            unsigned int localBin = v - (unsigned int)from;
            if (localBin < (unsigned int)numBins) {
                // Update lane-specific sub-histogram to avoid intra-warp bank conflicts
                // Padding ensures the same bin of different copies hits different banks
                atomicAdd(&sHist[lane * copyStride + (int)localBin], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce kSubHists copies into a single per-bin value and update global histogram
    for (int b = tid; b < numBins; b += blockT) {
        unsigned int sum = 0u;
#pragma unroll
        for (int c = 0; c < kSubHists; ++c) {
            sum += sHist[c * copyStride + b];
        }
        if (sum != 0u) {
            atomicAdd(&out[b], sum);
        }
    }
}

// Host function to invoke the histogram kernel.
// - input: device pointer to input chars (cudaMalloc'd)
// - histogram: device pointer to output histogram of length (to - from + 1) (cudaMalloc'd)
// - inputSize: number of chars in input
// - from, to: inclusive character range [from, to], where 0 <= from < to <= 255
//
// The function zeros the output histogram, configures the kernel launch,
// and uses shared memory sized for the requested range.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute number of bins and handle trivial cases
    const int numBins = (to - from) + 1;
    if (numBins <= 0) {
        return;
    }

    // Zero the output histogram (caller handles synchronization if needed)
    cudaMemset(histogram, 0, (size_t)numBins * sizeof(unsigned int));

    if (inputSize == 0) {
        return;
    }

    // Kernel configuration
    const int blockSize = 256; // 8 warps per block; good balance for shared memory usage and occupancy
    // Ensure enough threads to cover the input, given itemsPerThread per thread
    const unsigned int workPerBlock = (unsigned int)blockSize * (unsigned int)itemsPerThread;
    const unsigned int gridSize = (inputSize + workPerBlock - 1) / workPerBlock;

    if (gridSize == 0) {
        return;
    }

    // Dynamic shared memory size: (numBins + kSmemPad) per copy, with kSubHists copies
    const size_t smemSize = (size_t)(numBins + kSmemPad) * (size_t)kSubHists * sizeof(unsigned int);

    // Launch kernel
    histogram_range_kernel<<<gridSize, blockSize, smemSize>>>(input, inputSize, from, numBins, histogram);
}