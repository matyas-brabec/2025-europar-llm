#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
  CUDA kernel to compute a histogram restricted to a given [from, to] character range.
  - input: device pointer to the input text buffer (char array)
  - histogram: device pointer to the output histogram (size = to - from + 1, uint32 bins)
  - inputSize: number of chars in 'input'
  - from, to: inclusive character ordinal range [from, to], with 0 <= from <= to <= 255

  Performance notes:
  - Each block uses a shared-memory privatized histogram (size <= 256 bins) to absorb
    per-character updates via fast shared-memory atomics.
  - After processing its portion of the input, each block contributes its partial histogram
    back to global memory using at most (to - from + 1) global atomics per block.
  - This drastically reduces global atomic contention relative to directly atomically
    updating global memory for each character.
*/
__global__ void histogram_char_range_kernel(const char* __restrict__ input,
                                            unsigned int* __restrict__ histogram,
                                            unsigned int inputSize,
                                            unsigned int from,
                                            unsigned int to)
{
    // Dynamic shared memory sized to number of bins in [from, to].
    extern __shared__ unsigned int s_hist[];

    const unsigned int rangeLen = to - from + 1u;

    // Initialize shared histogram to zero (cooperatively by all threads in the block).
    for (unsigned int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over input to balance work across blocks and support any input size.
    const unsigned int stride = blockDim.x * gridDim.x;
    const unsigned char* __restrict__ in = reinterpret_cast<const unsigned char*>(input);

    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < inputSize; idx += stride) {
        // Load character and convert to unsigned for safe arithmetic.
        const unsigned int c = static_cast<unsigned int>(in[idx]);

        // Branchless range check via unsigned subtraction:
        // bin = c - from; valid iff bin <= (to - from)
        const unsigned int bin = c - from;
        if (bin <= (to - from)) {
            // Use shared-memory atomic to update privatized bin.
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Accumulate the per-block shared histogram into global memory.
    // We only issue a global atomicAdd for non-zero bins to reduce global traffic.
    for (unsigned int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        const unsigned int val = s_hist[i];
        if (val != 0u) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
  Host function to launch the histogram kernel.

  Assumptions:
  - 'input' and 'histogram' are device pointers allocated with cudaMalloc.
  - 'histogram' points to an array sized to (to - from + 1) elements.
  - The caller is responsible for any required synchronization (e.g., cudaDeviceSynchronize).
  - The function zeroes the output histogram before launching the kernel to produce a fresh result.

  Strategy:
  - Clamp 'from' and 'to' into [0, 255] and validate.
  - Zero the output histogram range on device via cudaMemsetAsync (default stream).
  - Choose a reasonable launch configuration using the SM count and a grid-stride kernel.
  - Use dynamic shared memory sized to the number of bins.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Clamp parameters defensively; the problem statement guarantees 0 <= from < to <= 255.
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (from > to) {
        // Invalid range: nothing to do.
        return;
    }

    const unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);

    // Zero the output histogram to ensure a fresh result.
    cudaMemsetAsync(histogram, 0, rangeLen * sizeof(unsigned int));

    // If there's no input, we're done after zeroing.
    if (inputSize == 0u) {
        return;
    }

    // Query device to compute a launch configuration.
    int device = 0;
    (void)cudaGetDevice(&device);

    int smCount = 0;
    (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    // Tuned defaults for modern data-center GPUs (A100/H100).
    // 256 threads per block offers a good balance of occupancy and latency hiding.
    const int blockSize = 256;

    // Target roughly 8 blocks per SM; clamp by input size so we don't launch excessive blocks for tiny inputs.
    int gridSize = (smCount > 0) ? (smCount * 8) : 80; // fallback if smCount is unavailable
    const unsigned int maxBlocksByInput = (inputSize + blockSize - 1) / blockSize;
    if (gridSize > static_cast<int>(maxBlocksByInput)) gridSize = static_cast<int>(maxBlocksByInput);
    if (gridSize < 1) gridSize = 1;

    // Dynamic shared memory size equals the number of bins times sizeof(unsigned int).
    const size_t shmemSize = static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Launch kernel on the default stream. Caller is responsible for synchronization if needed.
    histogram_char_range_kernel<<<gridSize, blockSize, shmemSize>>>(
        input,
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        static_cast<unsigned int>(to)
    );

    // Optionally, capture immediate launch errors (asynchronous execution continues otherwise).
    (void)cudaGetLastError();
}