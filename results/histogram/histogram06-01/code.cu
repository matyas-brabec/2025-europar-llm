#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel to compute a histogram over a restricted character range [from, to].
// The kernel uses shared memory histogram privatization with 32 copies to avoid bank conflicts.
// Each block computes a partial histogram which is then atomically accumulated into the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int inputSize, unsigned int *histogram, int from, int to) {
    // Compute the number of histogram bins needed for the range.
    int numBins = to - from + 1;

    // Declare dynamically allocated shared memory.
    // We allocate 32 copies of the histogram bins to ensure that each copy lies in a different bank.
    // Layout: s_hist[bin * 32 + copy] for bin in 0...numBins-1 and copy in 0...31.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared memory histogram: each thread initializes a portion of the array.
    int tid = threadIdx.x;
    int totalSharedElems = numBins * 32;
    for (int i = tid; i < totalSharedElems; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Define how many input chars each thread should process.
    // This constant is tuned for modern NVIDIA GPUs and large input sizes.
    const int itemsPerThread = 128;

    // Compute the starting global index for this thread.
    // Each block processes a contiguous chunk of the input.
    int globalThreadStart = blockIdx.x * blockDim.x * itemsPerThread + threadIdx.x;
    int blockStride = blockDim.x;  // to jump to the next input for this thread within the same block

    // Determine the warp lane id to select a unique copy of the histogram within the 32 copies.
    int lane = threadIdx.x & 31; // equivalent to threadIdx.x % 32

    // Process itemsPerThread input characters per thread.
    // We use a strided access pattern to ensure coalesced global memory accesses.
    for (int i = 0; i < itemsPerThread; i++) {
        int idx = globalThreadStart + i * blockStride;
        if (idx < inputSize) {
            // Cast to unsigned char to correctly interpret the value in the range [0,255].
            unsigned char c = input[idx];
            // Only process characters that fall in our [from, to] range.
            if (c >= from && c <= to) {
                int bin = c - from;
                // Write into the shared memory histogram.
                // Each bin has 32 copies; we use the copy index equal to the warp lane.
                // AtomicAdd is used because threads from different warps may update the same copy.
                atomicAdd(&s_hist[bin * 32 + lane], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 copies in shared memory for each bin to a single value.
    // We let the first numBins threads perform the reduction.
    if (tid < numBins) {
        unsigned int sum = 0;
        for (int copy = 0; copy < 32; copy++) {
            sum += s_hist[tid * 32 + copy];  // tid here represents the bin index.
        }
        // Atomically add the block's contribution for this bin to the global histogram.
        atomicAdd(&histogram[tid], sum);
    }
    // No further synchronization needed as each block finishes its update independently.
}

// Host function that configures and launches the histogram kernel.
// The input text and output histogram arrays are assumed to be allocated in device memory (via cudaMalloc).
// Host-device (cudaDeviceSynchronize) synchronization is handled externally.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Calculate the histogram range.
    int numBins = to - from + 1;

    // Define the number of threads per block.
    // 256 is chosen as a typical value optimal for modern GPUs.
    const int threadsPerBlock = 256;
    // Must match the kernel's constant.
    const int itemsPerThread = 128;

    // Calculate how many input characters each block processes.
    int itemsPerBlock = threadsPerBlock * itemsPerThread;
    // Compute the number of blocks needed to cover the input.
    int numBlocks = (inputSize + itemsPerBlock - 1) / itemsPerBlock;

    // Calculate dynamic shared memory size: 32 copies for each histogram bin.
    size_t sharedMemSize = numBins * 32 * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogram_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, histogram, from, to);
    // Note: Synchronization and error checking is assumed to be handled externally.
}