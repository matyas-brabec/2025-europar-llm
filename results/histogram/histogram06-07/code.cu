#include <cuda_runtime.h>

// Constants for kernel optimization for modern NVIDIA GPUs.
// ITEMS_PER_THREAD controls how many input characters are processed by each thread.
constexpr int ITEMS_PER_THREAD = 8;    // Optimized value for high throughput on large inputs.
constexpr int BLOCK_SIZE = 256;        // Threads per block.
constexpr int NUM_COPIES   = 32;       // Create 32 copies of the histogram in shared memory.
constexpr int MAX_HIST_BINS = 256;     // Maximum possible bins (for char values 0-255).

// CUDA kernel for computing a histogram over a restricted character range [from, to].
// Each thread processes ITEMS_PER_THREAD characters and updates a privatized histogram
// located in shared memory. The privatization is done per warp (one copy per warp)
// to avoid shared memory bank conflicts. At the end, the per-warp histograms are
// reduced and atomically accumulated to the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int inputSize,
                                 int from, int to, unsigned int *global_histogram) {
    // Determine the histogram range size.
    int histSize = to - from + 1;

    // Allocate shared memory for the privatized histograms.
    // We allocate NUM_COPIES copies (one per warp) each with MAX_HIST_BINS bins.
    // Only the first 'histSize' bins in each copy are used.
    __shared__ unsigned int s_hist[NUM_COPIES * MAX_HIST_BINS];

    // Each thread initializes a portion of the shared memory histogram.
    // Only initialize the bins that will be used for this kernel launch.
    int totalSharedBins = NUM_COPIES * histSize;
    for (int i = threadIdx.x; i < totalSharedBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute global thread index.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes ITEMS_PER_THREAD consecutive characters.
    // Compute the starting index based on thread global index.
    int base = tid * ITEMS_PER_THREAD;

    // Determine the warp index within the block.
    // Assuming warp size is 32, each warp (of 32 threads) gets its own private histogram.
    int warpId = threadIdx.x / 32;
    // Base address into shared memory for this warp's private histogram copy.
    int s_hist_base = warpId * MAX_HIST_BINS;

    // Process ITEMS_PER_THREAD characters per thread.
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = base + i;
        if (idx < inputSize) {
            // Load the character from global memory.
            unsigned char c = static_cast<unsigned char>(input[idx]);
            // Check if the character is within the restricted range.
            if (c >= from && c <= to) {
                int bin = c - from;
                // Atomically increment the count for the bin in the warp's private histogram.
                atomicAdd(&s_hist[s_hist_base + bin], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the per-warp privatized histograms from shared memory.
    // Each thread in the block sums over some of the histogram bins and atomically updates the global histogram.
    for (int bin = threadIdx.x; bin < histSize; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum up counts for this bin from all NUM_COPIES private histograms.
        for (int w = 0; w < NUM_COPIES; w++) {
            sum += s_hist[w * MAX_HIST_BINS + bin];
        }
        // Atomically update the global histogram. The histogram array is laid out such that
        // histogram[i] holds count for character (i + from).
        atomicAdd(&global_histogram[bin], sum);
    }
    // End of kernel.
}

// Host function to launch the histogram kernel.
// Assumes that 'input' and 'histogram' arrays are allocated on the device (via cudaMalloc).
// 'inputSize' is the number of characters in the input buffer.
// The histogram for characters in the range [from, to] is stored in 'histogram' such that
// histogram[i] corresponds to the count of character with ordinal value (i + from).
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to) {
    // Compute the number of bins for the restricted range.
    int histSize = to - from + 1;

    // Launch configuration:
    // Each thread processes ITEMS_PER_THREAD characters.
    // Compute number of blocks required so that (gridSize * BLOCK_SIZE * ITEMS_PER_THREAD) >= inputSize.
    int gridSize = (inputSize + (BLOCK_SIZE * ITEMS_PER_THREAD) - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);

    // Zero-out the global histogram array.
    cudaMemset(histogram, 0, histSize * sizeof(unsigned int));

    // Compute required shared memory size per block.
    // We allocate NUM_COPIES copies each with MAX_HIST_BINS integers.
    size_t sharedMemSize = NUM_COPIES * MAX_HIST_BINS * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(input, inputSize, from, to, histogram);
}