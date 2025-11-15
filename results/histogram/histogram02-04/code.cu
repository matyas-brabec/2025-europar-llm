#include <cuda_runtime.h>

/*
 * Optimized character histogram on modern NVIDIA GPUs (e.g., A100/H100).
 *
 * - The input is a device pointer to plain text (array of chars).
 * - The output is a device pointer to an array of unsigned int counters.
 * - We compute a histogram only for characters in the inclusive range [from, to].
 * - The output array has (to - from + 1) elements, where index i corresponds
 *   to character code (from + i).
 *
 * Implementation details:
 * -----------------------
 * 1. Each block builds a partial histogram in shared memory to reduce the
 *    number of slow global-memory atomic operations.
 * 2. Shared-memory histogram is initialized to zero by the block's threads.
 * 3. Each thread processes input characters in a grid-stride loop and
 *    atomically accumulates counts into the shared histogram.
 * 4. After processing, each block atomically merges its shared histogram into
 *    the global histogram in device memory.
 *
 * Notes:
 * ------
 * - The input and histogram pointers are assumed to be device pointers
 *   allocated with cudaMalloc().
 * - The histogram is explicitly zeroed in run_histogram() using cudaMemset().
 * - No cudaDeviceSynchronize() is called here; synchronization is left
 *   to the caller, as requested.
 */

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel
///////////////////////////////////////////////////////////////////////////////

__global__
void histogram_kernel_range_shared(const unsigned char * __restrict__ input,
                                   unsigned int       * __restrict__ global_hist,
                                   unsigned int input_size,
                                   int from,
                                   int to)
{
    // Dynamic shared memory: one histogram per block.
    extern __shared__ unsigned int shared_hist[];

    const int num_bins = to - from + 1;

    // Step 1: Initialize the shared histogram to zero.
    // Multiple threads cooperatively zero the array.
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0u;
    }

    __syncthreads();

    // Step 2: Each thread processes a grid-stride portion of the input.
    const unsigned int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int grid_stride      = blockDim.x * gridDim.x;

    for (unsigned int idx = global_thread_id; idx < input_size; idx += grid_stride) {
        unsigned char c = input[idx];

        // Restrict to the character range [from, to].
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {

            const int bin = static_cast<int>(c) - from;
            // Atomic add on shared memory (faster than global atomics).
            atomicAdd(&shared_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Step 3: Merge the shared histogram into the global histogram.
    // Multiple threads cooperatively write back, using global atomics.
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        unsigned int count = shared_hist[i];
        if (count > 0u) {
            atomicAdd(&global_hist[i], count);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Host wrapper
///////////////////////////////////////////////////////////////////////////////

/*
 * run_histogram
 * -------------
 * Launches the CUDA kernel to compute a character histogram over a specified
 * character range [from, to].
 *
 * Parameters:
 *   input      - device pointer to input text buffer (array of chars).
 *   histogram  - device pointer to output histogram (array of unsigned int).
 *                Must be allocated for (to - from + 1) elements.
 *   inputSize  - number of characters in the input buffer.
 *   from, to   - inclusive character code range to histogram (0 <= from <= to <= 255).
 *
 * Behavior:
 *   - The histogram buffer is zeroed before the kernel launch.
 *   - No host-device synchronization (e.g., cudaDeviceSynchronize) is performed.
 *     The caller is responsible for synchronization and error checking.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Validate parameters (minimal defensive checks).
    if (input == nullptr || histogram == nullptr) {
        return; // Invalid pointers; nothing to do.
    }
    if (inputSize == 0u) {
        // Still zero out the histogram for correctness.
        if (from <= to) {
            const int num_bins = to - from + 1;
            cudaMemset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(unsigned int));
        }
        return;
    }
    if (from < 0 || to > 255 || from > to) {
        return; // Invalid range; do nothing.
    }

    const int num_bins = to - from + 1;

    // Zero out the global histogram to ensure deterministic results.
    cudaMemset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(unsigned int));

    // Kernel launch configuration.
    // Using 256 threads per block is a good balance on modern GPUs (A100/H100).
    const int threads_per_block = 256;

    // Compute the number of blocks needed; grid-stride loop handles any size.
    int blocks = static_cast<int>((inputSize + threads_per_block - 1u) / threads_per_block);
    if (blocks <= 0) {
        blocks = 1;
    }

    // Dynamic shared memory size: one histogram of num_bins counters per block.
    const size_t shared_mem_size = static_cast<size_t>(num_bins) * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel_range_shared<<<blocks, threads_per_block, shared_mem_size>>>(
        reinterpret_cast<const unsigned char *>(input),
        histogram,
        inputSize,
        from,
        to
    );

    // No cudaDeviceSynchronize() here; caller handles synchronization and errors.
}