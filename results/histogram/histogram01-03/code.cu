#include <cuda_runtime.h>
#include <stdint.h>

/*
  CUDA kernel to compute a histogram over a restricted, contiguous character range.

  - input:       device pointer to input text (bytes)
  - global_hist: device pointer to output histogram array of length `rangeLen` where
                 bin 0 corresponds to character value `from`, bin k to `from + k`.
  - n:           number of bytes in input
  - from:        inclusive lower bound of the character range (0..255)
  - rangeLen:    number of bins == (to - from + 1)

  Strategy:
    - Each block builds a partial histogram in shared memory (size = rangeLen).
      Shared-memory atomics are fast on modern GPUs (A100/H100).
    - Threads read input in a grid-stride loop and update the block-local histogram.
    - After processing, the block reduces its partial histogram into the global histogram
      with one atomicAdd per bin (per block), which is negligible compared to the number
      of input updates.

  Notes:
    - Input bytes are treated as unsigned (0..255), independent of host char signedness.
    - The kernel assumes 0 < rangeLen <= 256 (guaranteed by 0 <= from < to <= 255).
*/
__global__ void histogram_range_kernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ global_hist,
    unsigned int n,
    unsigned int from,
    unsigned int rangeLen)
{
    extern __shared__ unsigned int s_hist[];

    // Zero-initialize the block-local histogram.
    for (unsigned int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over the input buffer.
    const unsigned int tid = threadIdx.x;
    const unsigned int block_threads = blockDim.x;
    const unsigned int grid_threads = block_threads * gridDim.x;
    unsigned int idx = blockIdx.x * block_threads + tid;

    while (idx < n) {
        unsigned int v = static_cast<unsigned int>(input[idx]);  // 0..255
        int bin = static_cast<int>(v) - static_cast<int>(from);  // can be negative
        // Use range check to include only bytes in [from, from + rangeLen - 1].
        if (bin >= 0 && static_cast<unsigned int>(bin) < rangeLen) {
            // Shared-memory atomic to avoid inter-block contention.
            atomicAdd(&s_hist[static_cast<unsigned int>(bin)], 1u);
        }
        idx += grid_threads;
    }

    __syncthreads();

    // Reduce the block-local histogram into the global histogram.
    for (unsigned int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        unsigned int count = s_hist[i];
        if (count) {
            atomicAdd(&global_hist[i], count);
        }
    }
}

/*
  Host function that prepares and launches the histogram kernel.

  Parameters:
    - input:      device pointer (cudaMalloc'd) to the input text data (char array)
    - histogram:  device pointer (cudaMalloc'd) to the output histogram array
                  of length (to - from + 1) unsigned ints
    - inputSize:  number of bytes in the input buffer
    - from, to:   inclusive range [from, to] of character codes (0 <= from < to <= 255)

  Behavior:
    - Zeros the output histogram buffer on device.
    - If inputSize is zero, returns after zeroing (result is all zeros).
    - Launches an optimized kernel with a grid large enough to saturate the GPU
      while avoiding unnecessary blocks for small inputs.
    - No device synchronization is performed here; the caller is responsible for it.

  Assumptions:
    - The pointers are valid device pointers.
    - from and to are valid and satisfy 0 <= from < to <= 255.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    const unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);

    // Zero the output histogram on device to avoid dependence on prior contents.
    cudaMemset(histogram, 0, rangeLen * sizeof(unsigned int));

    // Nothing to count.
    if (inputSize == 0) {
        return;
    }

    // Configure launch parameters.
    // Use 256 threads per block for a good balance (8 warps, low shared-memory footprint).
    const int threadsPerBlock = 256;

    // Choose a grid size that scales with the GPU while not vastly exceeding the work size.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const int sms = prop.multiProcessorCount;

    // Heuristic: 32 blocks per SM (saturates memory-bound workload) but cap by input size.
    int maxBlocksForSaturation = sms * 32;
    int blocksByInput = static_cast<int>((inputSize + threadsPerBlock - 1) / threadsPerBlock);
    int gridDim = blocksByInput > 0 ? blocksByInput : 1;
    if (gridDim > maxBlocksForSaturation) gridDim = maxBlocksForSaturation;
    if (gridDim < 1) gridDim = 1;

    // Dynamic shared memory size: one unsigned int per bin.
    size_t sharedMemSize = static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Launch the kernel.
    histogram_range_kernel<<<gridDim, threadsPerBlock, sharedMemSize>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        rangeLen
    );
}