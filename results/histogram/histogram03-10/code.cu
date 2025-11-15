#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
 * CUDA kernel to compute a histogram of characters in a given range [from, to].
 * 
 * - input:      device pointer to input characters (plain text)
 * - histogram:  device pointer to histogram array of length (to - from + 1)
 *               where histogram[i] counts occurrences of character (from + i)
 * - inputSize:  number of characters in input
 * - from, to:   inclusive character range (0 <= from < to <= 255)
 *
 * Optimization strategy:
 * - Each thread block builds a private histogram in shared memory.
 * - Threads atomically update the shared-memory histogram (fast).
 * - After processing their chunk of the input, blocks atomically accumulate
 *   the shared histogram into the global histogram (fewer global atomics).
 *
 * The kernel uses a grid-stride loop so that arbitrary input sizes can be handled
 * with a fixed (or capped) grid size while maintaining good load balance.
 */
__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int s_hist[];

    const int numBins   = to - from + 1;
    const int tid       = threadIdx.x;
    const int blockSize = blockDim.x;
    const int globalId  = blockIdx.x * blockSize + tid;
    const int stride    = blockSize * gridDim.x;

    // Initialize shared-memory histogram to zero.
    for (int bin = tid; bin < numBins; bin += blockSize) {
        s_hist[bin] = 0;
    }

    __syncthreads();

    // Process input in a grid-stride loop.
    for (unsigned int idx = globalId; idx < inputSize; idx += stride) {
        unsigned char c = static_cast<unsigned char>(input[idx]);
        if (c >= from && c <= to) {
            int bin = static_cast<int>(c) - from;
            // Atomic add into shared memory (low contention per block).
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Accumulate per-block shared histogram into global histogram.
    for (int bin = tid; bin < numBins; bin += blockSize) {
        unsigned int val = s_hist[bin];
        if (val > 0) {
            atomicAdd(&histogram[bin], val);
        }
    }
}

/*
 * Host wrapper that configures and launches the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input buffer (chars) allocated with cudaMalloc
 *   histogram  - device pointer to histogram buffer (unsigned int) allocated with cudaMalloc
 *   inputSize  - number of characters in input
 *   from, to   - inclusive range of characters to count (0 <= from < to <= 255)
 *
 * Notes:
 * - This function initializes the histogram buffer to zero using cudaMemset.
 * - It does not perform any host-device synchronization; the caller is responsible
 *   for synchronization and error checking if desired.
 * - The grid size is capped based on the number of SMs to avoid oversubscription
 *   while still providing enough parallelism for large inputs.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic parameter validation (silent return on invalid range).
    if (input == nullptr || histogram == nullptr) {
        return;
    }
    if (from < 0 || to > 255 || from > to) {
        return;
    }

    const int numBins = to - from + 1;

    // Zero the histogram on device before accumulation.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Choose reasonable launch configuration for modern data-center GPUs.
    const int blockSize = 256;

    // Determine device properties to cap grid size based on SM count.
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Use up to 32 blocks per SM as a simple heuristic for good occupancy.
    int maxBlocks = prop.multiProcessorCount * 32;

    // Base grid size on input size and block size.
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    if (gridSize <= 0) {
        gridSize = 1;
    }
    if (gridSize > maxBlocks) {
        gridSize = maxBlocks;
    }

    // Dynamic shared memory size: one unsigned int per histogram bin.
    size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel on the default stream.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );

    // No synchronization here; caller is responsible for cudaDeviceSynchronize()
    // or equivalent stream synchronization and error checking.
}