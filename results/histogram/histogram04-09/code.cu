#include <cuda_runtime.h>

// Define the number of input characters each thread processes.
// A default value of 8 is chosen to balance memory coalescing and per‐thread work,
// especially on modern GPUs with very high occupancy.
constexpr int itemsPerThread = 8;

// CUDA kernel that computes a histogram over a restricted range [from,to].
// Each block uses shared memory to privatize histogram updates.
// Each thread processes "itemsPerThread" characters from the input array.
__global__ void histogramKernel(const char *input, unsigned int inputSize,
                                unsigned int *globalHistogram, int from, int to) {
    // Compute the number of histogram bins.
    int numBins = to - from + 1;

    // Declare shared memory histogram array.
    // The size of the shared memory is set per kernel launch.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared histogram bins to zero.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Calculate the starting global index for this block.
    // Each block processes a contiguous chunk of input data.
    int blockStart = blockIdx.x * blockDim.x * itemsPerThread;
    int tid = threadIdx.x;

    // Process itemsPerThread characters per thread.
    // The loop is structured to ensure coalesced accesses: at each iteration,
    // threads in the block access a contiguous segment of the input.
    for (int i = 0; i < itemsPerThread; i++) {
        int index = blockStart + tid + i * blockDim.x;
        if (index < inputSize) {
            // Convert to unsigned char to correctly interpret character ordinal values.
            unsigned char c = static_cast<unsigned char>(input[index]);
            if (c >= from && c <= to) {
                // Atomically increment the corresponding bin in the shared histogram.
                atomicAdd(&s_hist[c - from], 1);
            }
        }
    }
    __syncthreads();

    // Merge the block-level (shared) histogram into the global histogram.
    // Each thread cooperatively updates global bins to reduce atomic contention.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&globalHistogram[i], s_hist[i]);
    }
}

// Host function that configures and launches the CUDA histogram kernel.
// The input text is stored in device memory pointed to by 'input'.
// The output 'histogram' is also a device memory pointer and is assumed to be pre-allocated.
// 'from' and 'to' define the inclusive character range for which the histogram is computed.
// The host-device synchronization is assumed to be handled by the caller.
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to) {
    // Determine the number of histogram bins.
    int numBins = to - from + 1;

    // Choose a block size (number of threads per block).
    // 256 is a good choice for modern GPUs.
    int blockSize = 256;

    // Calculate the grid size so that all items in the input are processed.
    int gridSize = (inputSize + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);

    // Initialize the output histogram in device memory to zero.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Calculate required shared memory per block (one unsigned int per histogram bin).
    size_t sharedMemSize = numBins * sizeof(unsigned int);

    // Launch the histogram kernel.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(input, inputSize, histogram, from, to);
}