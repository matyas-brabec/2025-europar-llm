#include <cuda_runtime.h>
#include <cstddef>

// CUDA kernel to compute a partial histogram over characters in the range [from, to].
// Each block builds a histogram in shared memory to reduce global atomic contention,
// then the per-block histograms are accumulated into the global histogram.
__global__ void histogram_kernel(const char *__restrict__ input,
                                 unsigned int *__restrict__ globalHist,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int sharedHist[];

    const int range = to - from + 1;  // Number of histogram bins (inclusive range [from, to])

    // Initialize shared-memory histogram to zero.
    // All threads in the block cooperate to zero the 'range' bins.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads = blockDim.x * gridDim.x;

    // Grid-stride loop so the kernel can handle arbitrarily large input.
    for (unsigned int idx = tid; idx < inputSize; idx += totalThreads) {
        // Cast to unsigned char to avoid sign-extension issues with 'char'.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Map character code to bin index relative to 'from'.
        // Using a single subtraction avoids two comparisons against 'from' and 'to'.
        int bin = static_cast<int>(c) - from;

        // Only count characters within [from, to].
        if (bin >= 0 && bin < range) {
            // Shared-memory atomic is much faster than a global atomic.
            atomicAdd(&sharedHist[bin], 1u);
        }
    }

    __syncthreads();

    // Accumulate per-block shared histogram into the global histogram.
    // Each block updates each bin at most once, so global atomics are minimal.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        unsigned int count = sharedHist[i];
        if (count != 0u) {
            atomicAdd(&globalHist[i], count);
        }
    }
}

// Host function that sets up and launches the histogram kernel.
//
// Parameters:
//   input      - device pointer to input text buffer (allocated with cudaMalloc)
//   histogram  - device pointer to histogram buffer of size (to - from + 1) unsigned ints
//   inputSize  - number of characters in 'input'
//   from, to   - inclusive character range [from, to] to histogram (0 <= from < to <= 255)
//
// Notes:
//   - This function zeroes the device-side histogram before launching the kernel.
//   - No host-device synchronization is performed here; the caller is responsible
//     for synchronizing and copying the histogram back to host memory if needed.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Assumes input arguments satisfy 0 <= from < to <= 255.
    const int range = to - from + 1;
    if (range <= 0) {
        // Invalid range; nothing to do.
        return;
    }

    // Initialize the histogram on the device to zero so that the kernel
    // can safely accumulate into it with atomicAdd.
    const size_t histBytes = static_cast<size_t>(range) * sizeof(unsigned int);
    cudaMemset(histogram, 0, histBytes);

    // Choose a reasonable block size for modern GPUs.
    const int threadsPerBlock = 256;

    // Compute a grid size that provides enough parallelism.
    // Use a grid-stride loop in the kernel to handle remaining data.
    unsigned int blocksPerGrid;
    if (inputSize == 0) {
        blocksPerGrid = 1;  // Launch a minimal grid; kernel will do no work.
    } else {
        blocksPerGrid = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
        const unsigned int maxBlocks = 65535u;  // Conservative cap for 1D grids.
        if (blocksPerGrid > maxBlocks) {
            blocksPerGrid = maxBlocks;
        }
    }

    // Shared memory size per block: one unsigned int per bin.
    const size_t sharedMemBytes = static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch kernel. The caller is responsible for any needed synchronization.
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(
        input, histogram, inputSize, from, to);
}