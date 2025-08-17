#include <cuda_runtime.h>
#include <stdint.h>

/*
  Histogram of a restricted character range on the GPU.

  Key optimization:
  - Each thread block builds a privatized histogram in shared memory.
  - To reduce contention further, each warp owns its own sub-histogram in shared memory.
  - At the end, per-warp histograms are reduced within the block and added to the global histogram.
  - This yields at most (gridDim * numBins) global atomics, independent of input size.

  Assumptions:
  - input points to device memory containing 'inputSize' bytes of plain text (char array).
  - histogram points to device memory with space for (to - from + 1) unsigned ints.
  - 0 <= from <= to <= 255 (full byte range), inclusive bounds; the range length is (to - from + 1).
  - Caller handles stream synchronization; this function uses the default stream.
*/

static __global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                              unsigned int* __restrict__ global_hist,
                                              unsigned int n,
                                              int from,
                                              int to)
{
    extern __shared__ unsigned int shist[];  // Layout: [warp0 bins][warp1 bins]...[warpN bins]

    const int numBins = to - from + 1;       // number of bins in [from, to], inclusive
    const int warpSize_ = 32;
    const int nWarps = (blockDim.x + warpSize_ - 1) / warpSize_;
    const int warpId = threadIdx.x / warpSize_;
    unsigned int* warp_hist = shist + warpId * numBins;

    // Initialize the per-warp histograms in shared memory to zero.
    for (int i = threadIdx.x; i < numBins * nWarps; i += blockDim.x) {
        shist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over input to maximize memory throughput and occupancy.
    const unsigned int stride = blockDim.x * gridDim.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Precompute range width for fast range check: r = v - from; if r <= rangeWidth -> in range
    const unsigned int rangeWidth = static_cast<unsigned int>(to - from);

    // Consume input and update per-warp histogram in shared memory using atomics.
    // Shared memory atomics are fast on modern GPUs; per-warp privatization reduces contention.
    while (idx < n) {
        unsigned int v = static_cast<unsigned int>(input[idx]); // 0..255
        unsigned int r = v - static_cast<unsigned int>(from);
        if (r <= rangeWidth) {
            // r is the bin index within [0, numBins-1]
            atomicAdd(&warp_hist[r], 1u);
        }
        idx += stride;
    }

    __syncthreads();

    // Reduce per-warp histograms to a single per-block histogram and add to global.
    // Each thread accumulates several bins to distribute work.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum the same bin across all warps in this block
        for (int w = 0; w < nWarps; ++w) {
            sum += shist[w * numBins + bin];
        }
        if (sum) {
            atomicAdd(&global_hist[bin], sum);
        }
    }
}

void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Nothing to do for empty input
    if (inputSize == 0) {
        return;
    }

    // Number of bins to compute (inclusive range)
    const int numBins = to - from + 1;

    // Choose launch configuration tuned for A100/H100-class GPUs.
    // 512 threads per block gives 16 warps; with per-warp histograms, shared memory scales as 16*numBins*4 bytes.
    const int threadsPerBlock = 512;

    // Heuristic: launch a grid with enough blocks to keep the device busy, bounded by input size.
    int device = 0;
    cudaDeviceProp prop;
    int smCount = 80; // reasonable default
    if (cudaGetDevice(&device) == cudaSuccess) {
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            smCount = prop.multiProcessorCount;
        }
    }
    // Aim for multiple blocks per SM to hide latency; adjust down if input is small.
    int targetBlocks = smCount * 16;
    int blocksByElems = static_cast<int>((inputSize + threadsPerBlock - 1) / threadsPerBlock);
    int gridDim = targetBlocks;
    if (blocksByElems < gridDim) gridDim = blocksByElems;
    if (gridDim < 1) gridDim = 1;

    // Dynamic shared memory size: one sub-histogram per warp in the block.
    const int nWarps = (threadsPerBlock + 32 - 1) / 32;
    const size_t shmemBytes = static_cast<size_t>(numBins) * static_cast<size_t>(nWarps) * sizeof(unsigned int);

    // Zero the output histogram before accumulation.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Launch the kernel
    histogram_range_kernel<<<gridDim, threadsPerBlock, shmemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from,
        to
    );
}