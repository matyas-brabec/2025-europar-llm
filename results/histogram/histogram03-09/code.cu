#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized histogram kernel using shared-memory privatization.

  - Each block maintains a private histogram in shared memory for the selected
    character range [from, to]. Threads atomically update the shared histogram,
    which is very fast on modern GPUs (A100/H100).
  - After processing the input with a grid-stride loop, threads cooperatively
    merge the block-private histograms into the global histogram using a single
    pass of atomicAdd per bin. This drastically reduces global atomic pressure
    compared to per-element global atomics.

  Notes:
  - The input is treated as unsigned bytes (0..255) to avoid any sign issues.
  - The range check uses a single unsigned comparison to avoid divergent branches.
  - The kernel uses dynamically-sized shared memory equal to (to-from+1) bins.
  - Caller is expected to handle stream synchronization; this implementation
    performs no cudaDeviceSynchronize.
*/

static inline __device__ bool in_range_inclusive(int x, int lo, int hi) {
    // This helper exists mainly for readability; it is not used in the hot path.
    return x >= lo && x <= hi;
}

__global__ void histogram_shared_kernel(const unsigned char* __restrict__ input,
                                        unsigned int inputSize,
                                        unsigned int* __restrict__ histogram,
                                        int from, int numBins)
{
    extern __shared__ unsigned int sHist[]; // size = numBins

    // Zero the shared histogram (cooperatively)
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        sHist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop to cover the entire input buffer
    const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    // Process input and update shared histogram. Use unsigned comparison to avoid branches:
    // if (0 <= bin < numBins) -> (unsigned)bin < (unsigned)numBins
    for (unsigned int i = tid; i < inputSize; i += stride) {
        unsigned int v  = static_cast<unsigned int>(input[i]); // 0..255
        int bin         = static_cast<int>(v) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
            atomicAdd(&sHist[bin], 1u);
        }
    }

    __syncthreads();

    // Merge the shared histogram into the global histogram
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int val = sHist[i];
        if (val) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
  Host entry point.

  Parameters:
  - input: device pointer to input text (allocated by cudaMalloc)
  - histogram: device pointer to output histogram (allocated by cudaMalloc).
               Must have space for (to - from + 1) unsigned int elements.
  - inputSize: number of chars in input
  - from, to: selected inclusive range (0 <= from <= to <= 255)

  Behavior:
  - Asynchronously zeroes the output histogram for the requested range.
  - Launches the shared-memory histogram kernel with dynamic shared memory.
  - Does not perform synchronization; caller is responsible for it.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate parameters defensively (no-op on invalid input)
    if (!input || !histogram || from < 0 || to < from || to > 255) {
        return;
    }

    const int numBins = (to - from) + 1;
    if (numBins <= 0) {
        return;
    }

    // Clear the output histogram for the requested range (async on default stream)
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int), 0);

    // If there's nothing to process, we're done
    if (inputSize == 0) {
        return;
    }

    // Simple, robust launch configuration:
    // - 256 threads per block is a good default for memory-bound kernels.
    // - Choose grid size to cover the input at least once, but clamp to a reasonable multiple
    //   of SM count to avoid excessive tiny blocks. We don't require device sync here.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    const int blockSize = 256;
    // Aim for several resident blocks per SM to hide latency well (e.g., 8x SMs),
    // but never fewer than what covers the data once.
    int gridFromData = static_cast<int>((inputSize + blockSize - 1) / blockSize);
    int gridFromSMs  = props.multiProcessorCount * 8;
    int gridSize     = gridFromData > gridFromSMs ? gridFromSMs : gridFromData;
    if (gridSize < 1) gridSize = 1;

    // Dynamic shared memory size: one unsigned int per bin
    const size_t shmemBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch kernel
    const unsigned char* d_input_u8 = reinterpret_cast<const unsigned char*>(input);
    histogram_shared_kernel<<<gridSize, blockSize, shmemBytes, 0>>>(d_input_u8, inputSize, histogram, from, numBins);
}