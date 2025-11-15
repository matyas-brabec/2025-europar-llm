#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram of character occurrences in a given
 * continuous range [from, to] on a text buffer stored in device memory.
 *
 * The histogram is computed only for characters in that range; characters
 * outside the range are ignored.
 *
 * The kernel uses a per-block shared-memory histogram (up to 256 bins)
 * to reduce global memory contention. After processing its portion of the
 * input via a grid-stride loop, each block atomically accumulates its
 * partial histogram into the global histogram buffer.
 */
__global__ void histogram_kernel(const char *__restrict__ input,
                                 unsigned int *__restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    // Maximum possible range is 256 (characters 0..255).
    __shared__ unsigned int local_hist[256];

    int range = to - from + 1;
    if (range <= 0 || range > 256) {
        // Invalid range: nothing to do.
        return;
    }

    // Initialize per-block shared histogram to zero.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        local_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop over the input.
    unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    int range_minus_1   = range - 1;

    while (tid < inputSize) {
        // Load character and convert to unsigned to avoid sign issues.
        int c = static_cast<unsigned char>(input[tid]);
        // Shift into [0, 255] and then to [-(from), 255-from].
        c -= from;

        // Single comparison range check:
        // If 0 <= c <= range_minus_1, then (unsigned)c <= (unsigned)range_minus_1.
        if (static_cast<unsigned int>(c) <= static_cast<unsigned int>(range_minus_1)) {
            // Safe to index local_hist[c] since c is in [0, range-1].
            atomicAdd(&local_hist[c], 1u);
        }

        tid += stride;
    }

    __syncthreads();

    // Accumulate per-block histogram into global histogram.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        unsigned int count = local_hist[i];
        if (count != 0u) {
            atomicAdd(&histogram[i], count);
        }
    }
}

/*
 * Host function to set up and launch the histogram CUDA kernel.
 *
 * Parameters:
 *   - input:      device pointer to text buffer (chars).
 *   - histogram:  device pointer to histogram buffer of size (to - from + 1).
 *   - inputSize:  number of characters in the input buffer.
 *   - from, to:   character ordinal range [from, to], 0 <= from <= to <= 255.
 *
 * Assumptions:
 *   - input and histogram pointers are allocated with cudaMalloc.
 *   - Caller is responsible for any required synchronization (e.g.,
 *     cudaDeviceSynchronize) and error checking after this function returns.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (input == nullptr || histogram == nullptr) {
        // Invalid pointers; nothing can be done.
        return;
    }

    // Clamp 'from' and 'to' into the valid [0, 255] range for chars.
    if (from < 0)   from = 0;
    if (to   > 255) to   = 255;

    if (from > to) {
        // Invalid range; nothing to do.
        return;
    }

    unsigned int range = static_cast<unsigned int>(to - from + 1);

    // Initialize the histogram buffer on the device to zero.
    // This is ordered before the kernel launch on the default stream.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // No further work if the input is empty.
    if (inputSize == 0u) {
        return;
    }

    // Choose launch configuration.
    // 256 threads per block is a good balance for this type of workload.
    const int threadsPerBlock = 256;

    // Number of blocks needed to cover the input at least once.
    unsigned int numBlocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Optionally cap the number of blocks based on the device SM count
    // to avoid excessively large grids. This is beneficial when inputSize
    // is very large because the kernel uses a grid-stride loop.
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Use a reasonable multiple of the number of SMs (e.g., 8 blocks per SM).
    unsigned int maxBlocks = static_cast<unsigned int>(prop.multiProcessorCount) * 8u;
    if (maxBlocks == 0u) {
        maxBlocks = 1u;
    }

    if (numBlocks > maxBlocks) {
        numBlocks = maxBlocks;
    }
    if (numBlocks == 0u) {
        numBlocks = 1u;
    }

    // Launch the kernel on the default stream.
    histogram_kernel<<<numBlocks, threadsPerBlock>>>(input, histogram, inputSize, from, to);

    // No synchronization or error checking here; the caller is expected to
    // handle it (e.g., cudaDeviceSynchronize and cudaGetLastError).
}