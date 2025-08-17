#include <cuda_runtime.h>

/*
  CUDA histogram kernel optimized for modern NVIDIA GPUs (A100/H100).

  Key optimizations:
  - Each block processes a contiguous chunk of the input to maximize coalesced global memory reads.
  - Per-warp privatized histograms in shared memory minimize atomic contention.
  - Block-level reduction of per-warp histograms followed by a single atomic add per bin per block.
  - itemsPerThread controls the number of input bytes processed by each thread to boost arithmetic intensity.

  Output histogram is restricted to the inclusive range [from, to], producing (to - from + 1) bins.
  The caller must ensure that:
    - input points to a device buffer of size inputSize bytes (cudaMalloc'ed)
    - histogram points to a device buffer of size (to - from + 1) * sizeof(unsigned int) (cudaMalloc'ed)
    - 0 <= from <= to <= 255
*/

static constexpr int THREADS_PER_BLOCK = 256; // Must be a multiple of 32 on modern GPUs
static_assert(THREADS_PER_BLOCK % 32 == 0, "THREADS_PER_BLOCK must be a multiple of warp size (32).");

/*
  Controls how many input bytes each thread processes.
  A higher value improves throughput by increasing per-thread work and amortizing control overhead,
  especially on memory-bound workloads and large inputs. 16 is a good default on A100/H100.
*/
static constexpr int itemsPerThread = 16;

__global__ void histogram_kernel_range_shared_warp(
    const char* __restrict__ input,
    unsigned int inputSize,
    unsigned int* __restrict__ histogram,
    int from,
    int to)
{
    // Compute number of bins and quickly exit if invalid range
    const int numBins = to - from + 1;
    if (numBins <= 0) return;

    // Shared memory layout: one privatized histogram per warp
    extern __shared__ unsigned int s_hist[];
    const int warpsPerBlock = blockDim.x / warpSize;
    const int warpId        = threadIdx.x / warpSize;

    // Initialize all per-warp histograms to zero
    for (int i = threadIdx.x; i < warpsPerBlock * numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of input for coalesced access
    const size_t blockChunk = static_cast<size_t>(blockDim.x) * static_cast<size_t>(itemsPerThread);
    const size_t baseIndex  = static_cast<size_t>(blockIdx.x) * blockChunk;

    // Process itemsPerThread bytes per thread; indices are strided by blockDim.x
    // Use a single-bounds check idiom to test if the character is within [from, to]
    // by computing bin = (unsigned char)val - from and checking bin < numBins.
    unsigned int* __restrict__ warpHist = s_hist + warpId * numBins;

    #pragma unroll
    for (int it = 0; it < itemsPerThread; ++it) {
        const size_t idx = baseIndex + static_cast<size_t>(threadIdx.x) + static_cast<size_t>(it) * static_cast<size_t>(blockDim.x);
        if (idx < static_cast<size_t>(inputSize)) {
            const unsigned int c = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
            const int bin = static_cast<int>(c) - from;
            // Branchless in-range check: valid if 0 <= bin < numBins
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
                // Shared-memory atomic add; on A100/H100 this is fast and contention is limited to the warp
                atomicAdd(&warpHist[bin], 1u);
            }
        }
    }
    __syncthreads();

    // Reduce per-warp histograms into a single per-block histogram and add to global
    // Each thread reduces a subset of bins to maximize parallelism
    for (int b = threadIdx.x; b < numBins; b += blockDim.x) {
        unsigned int sum = 0;
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * numBins + b];
        }
        if (sum) {
            atomicAdd(&histogram[b], sum);
        }
    }
    // No need for further synchronization; kernel ends here
}

/*
  Host wrapper to launch the histogram kernel.

  Parameters:
    input      - device pointer to input text buffer (size: inputSize bytes)
    histogram  - device pointer to output histogram (size: (to - from + 1) * sizeof(unsigned int))
    inputSize  - number of bytes in input
    from, to   - inclusive character range [from, to] (0 <= from <= to <= 255)

  Notes:
    - The function zeros the histogram before launching the kernel.
    - No explicit device synchronization is performed here.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate trivial cases
    if (input == nullptr || histogram == nullptr) return;
    if (inputSize == 0) {
        // Ensure histogram is zeroed for consistency
        const int numBins0 = (to - from + 1);
        if (numBins0 > 0) {
            cudaMemset(histogram, 0, static_cast<size_t>(numBins0) * sizeof(unsigned int));
        }
        return;
    }

    // Compute number of bins and zero the output histogram
    const int numBins = to - from + 1;
    if (numBins <= 0) return;
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Configure launch parameters
    const int threads = THREADS_PER_BLOCK;
    const size_t blockChunk = static_cast<size_t>(threads) * static_cast<size_t>(itemsPerThread);
    const int blocks = static_cast<int>((static_cast<size_t>(inputSize) + blockChunk - 1) / blockChunk);

    if (blocks <= 0) return;

    // Dynamic shared memory size: one histogram per warp
    const int warpsPerBlock = threads / 32;
    const size_t sharedMemBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch kernel
    histogram_kernel_range_shared_warp<<<blocks, threads, sharedMemBytes>>>(input, inputSize, histogram, from, to);
}