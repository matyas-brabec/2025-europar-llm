#include <cuda_runtime.h>

/*
  CUDA range-restricted histogram
  - Computes histogram counts for character values in [from, to] inclusive.
  - Input: device pointer 'input' with 'inputSize' bytes of plain text (char array).
  - Output: device pointer 'histogram' with (to - from + 1) bins (unsigned int per bin).
  - Implementation uses per-block histograms in shared memory to minimize global memory atomics.
  - Each block:
      1) Zeroes its shared histogram.
      2) Grid-stride loops through the input, incrementing shared histogram bins (only for chars in range).
      3) Atomically adds its shared histogram into the global histogram.
  - The host function 'run_histogram' takes device pointers, zeros the global histogram,
    configures a sensible launch, and launches the kernel. Synchronization is left to the caller.
*/

static __device__ __forceinline__ unsigned int get_lane_id()
{
    unsigned int lane;
#if (__CUDACC_VER_MAJOR__ >= 9)
    lane = threadIdx.x & 0x1f;
#else
    asm("mov.u32 %0, %laneid;" : "=r"(lane));
#endif
    return lane;
}

__global__ void histogram_range_kernel(
    const char* __restrict__ input,       // device pointer to input chars
    unsigned int inputSize,               // number of input chars
    int from,                             // inclusive lower bound of char range
    int bins,                             // number of output bins (to - from + 1)
    unsigned int* __restrict__ globalHist // device pointer to output histogram (bins elements)
)
{
    extern __shared__ unsigned int s_hist[]; // shared histogram of 'bins' elements

    // 1) Initialize shared histogram to zero.
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // 2) Grid-stride loop over input to build shared histogram.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Use an unsigned comparison trick to avoid two comparisons for the range check.
    // bin = (unsigned char)val - from; if ((unsigned)bin < (unsigned)bins) in-range.
    for (unsigned int i = tid; i < inputSize; i += stride) {
        unsigned int v = static_cast<unsigned char>(input[i]);
        int bin = static_cast<int>(v) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(bins)) {
            // Shared memory atomics are fast on modern GPUs (A100/H100).
            atomicAdd(&s_hist[bin], 1u);
        }
    }
    __syncthreads();

    // 3) Flush the shared histogram to global histogram with atomic adds.
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val != 0) {
            atomicAdd(&globalHist[i], val);
        }
    }
}

/*
  Host launcher.
  - Assumes 'input' and 'histogram' are already allocated on device via cudaMalloc.
  - Zeros the histogram and launches the kernel.
  - No synchronization is performed here; caller is responsible if required.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate and compute number of bins (clamp conservatively to avoid undefined behavior).
    if (to < from) {
        // Nothing to do if invalid; for robustness we simply return.
        return;
    }
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    int bins = to - from + 1;
    if (bins <= 0) return;

    // Zero the output histogram on device (asynchronous; ordered within default stream).
    cudaMemset(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    // Choose launch configuration.
    // - 256 threads per block is a good default for histogram workloads on A100/H100.
    const int blockSize = 256;

    // Heuristic for number of blocks: min( maxBlocks, ceil(inputSize / blockSize) )
    // This bounds the grid size to avoid oversubscription while ensuring coverage.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int smCount = prop.multiProcessorCount;
    // 32 blocks per SM is a good upper bound for sustained occupancy with lightweight kernels.
    int maxBlocks = smCount * 32;

    int blocks;
    if (inputSize == 0) {
        // No input: histogram already zeroed; nothing to launch.
        return;
    } else {
        // Ceil division to cover all input at least once if using a single pass without grid-stride.
        // We still use a grid-stride loop, so capping at maxBlocks prevents diminishing returns.
        int neededBlocks = static_cast<int>((inputSize + blockSize - 1) / blockSize);
        blocks = (neededBlocks < maxBlocks) ? neededBlocks : maxBlocks;
        if (blocks < 1) blocks = 1;
    }

    // Dynamic shared memory size: one unsigned int per bin.
    size_t sharedBytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    // Launch kernel in default stream; ordering with memset is preserved.
    histogram_range_kernel<<<blocks, blockSize, sharedBytes>>>(input, inputSize, from, bins, histogram);
}