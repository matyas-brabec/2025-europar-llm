#include <cuda_runtime.h>
#include <algorithm>

/*
  CUDA kernel: histogram_range_kernel

  Computes a histogram over a specified inclusive character range [from, to]
  for a given input text buffer. The histogram is accumulated per-block
  in shared memory to reduce global memory contention. Each block:
    - Initializes its shared histogram to zeros
    - Processes a grid-stride range of input characters
    - Atomically increments shared-memory bins for characters in [from, to]
    - Atomically adds its shared histogram to the global histogram

  Notes for performance on A100/H100:
    - Per-block shared-memory accumulation drastically reduces the number
      of global atomics (from O(N) to O(numBlocks * bins)), which is important
      for heavily skewed distributions typical of text.
    - Shared-memory atomics on 32-bit integers are fast on recent architectures.
    - Grid-stride loop ensures full device utilization regardless of input size.
    - Dynamic shared memory is sized exactly to the requested bin count (<= 256).
*/
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int input_size,
                                       unsigned int* __restrict__ global_hist,
                                       int from,
                                       int to)
{
    // Number of bins in the requested range (inclusive)
    const int nBins = to - from + 1;

    // Dynamic shared memory for per-block histogram; size provided at launch
    extern __shared__ unsigned int s_hist[];

    // Initialize shared histogram to zero
    for (int i = threadIdx.x; i < nBins; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over input
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    while (idx < static_cast<size_t>(input_size)) {
        // Treat input bytes as unsigned to get values in [0, 255]
        const unsigned char v = input[idx];

        // Compute bin index relative to 'from'. If in range, update shared histogram.
        const int bin = static_cast<int>(v) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(nBins)) {
            // Shared-memory atomic to handle intra-block collisions
            atomicAdd(&s_hist[bin], 1u);
        }
        idx += stride;
    }

    __syncthreads();

    // Accumulate per-block shared histogram into global histogram.
    // Using global atomics across blocks; the host ensures the global histogram
    // is zero-initialized before the kernel launch.
    for (int i = threadIdx.x; i < nBins; i += blockDim.x) {
        const unsigned int c = s_hist[i];
        if (c) {
            atomicAdd(&global_hist[i], c);
        }
    }
}

/*
  Host-side launcher: run_histogram

  Parameters:
    - input:      device pointer to input text (cudaMalloc'd), size = inputSize bytes
    - histogram:  device pointer to output histogram (cudaMalloc'd),
                  size must be (to - from + 1) * sizeof(unsigned int)
    - inputSize:  number of bytes in 'input'
    - from, to:   inclusive character range [from, to], where 0 <= from < to <= 255

  Behavior:
    - Zeros the output histogram on device (asynchronously, default stream).
    - Launches a single kernel that computes the range-restricted histogram using
      per-block shared memory accumulation and global atomic reduction.

  Synchronization:
    - No synchronization is performed here; all operations are enqueued in the
      default stream. The caller is responsible for any synchronization.
*/
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Validate range and compute number of bins (inclusive)
    if (histogram == nullptr || input == nullptr) return;
    if (from > to) return; // Defensive check; spec assures from < to
    const int nBins = to - from + 1;
    if (nBins <= 0) return;

    // Zero-initialize the output histogram to ensure correctness with global atomics
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(nBins) * sizeof(unsigned int));

    // Early exit if no input to process (still zeroed)
    if (inputSize == 0) return;

    // Choose a reasonable block size for modern data center GPUs
    const int blockSize = 256;

    // Determine a grid size that provides ample parallelism without oversubscription.
    // Use up to 32 blocks per SM to keep the GPU busy while avoiding tiny blocks explosion.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    // Initial grid size estimate based on input size
    unsigned int gridSize = (inputSize + blockSize - 1) / blockSize;
    if (gridSize == 0) gridSize = 1;

    const unsigned int maxBlocks = static_cast<unsigned int>(prop.multiProcessorCount) * 32u;
    if (gridSize > maxBlocks) gridSize = maxBlocks;

    // Dynamic shared memory size in bytes
    const size_t shmemSize = static_cast<size_t>(nBins) * sizeof(unsigned int);

    // Launch the kernel. Cast input to unsigned char* to avoid signed-char pitfalls.
    histogram_range_kernel<<<gridSize, blockSize, shmemSize>>>(
        reinterpret_cast<const unsigned char*>(input),
        inputSize,
        histogram,
        from,
        to
    );
}