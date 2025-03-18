#include <cuda_runtime.h>

// Constants to control kernel behavior.
// ITEMS_PER_THREAD determines how many input characters each thread processes.
// HISTO_REPLICAS is the number of independent (replicated) histograms allocated in shared memory per block,
// which helps reduce shared memory bank conflicts.
constexpr int ITEMS_PER_THREAD = 16;
constexpr int HISTO_REPLICAS   = 8;

// CUDA kernel to compute a histogram for characters in the range [from, to].
// The histogram bins (of size nbins = to - from + 1) are computed only for characters that fall within the desired range.
// Each block uses shared memory to hold HISTO_REPLICAS private copies of the histogram.
// After processing, the replicas are reduced and the block's contribution is atomically added to the global histogram.
__global__ void histogram_kernel(const char *input, unsigned int *global_histogram,
                                 unsigned int inputSize, int from, int to) {
    // Calculate the number of bins to compute.
    const int nbins = to - from + 1;

    // Determine the number of threads assigned to each replicated histogram.
    // It is assumed that blockDim.x is evenly divisible by HISTO_REPLICAS.
    int threadsPerReplica = blockDim.x / HISTO_REPLICAS;
    // Each thread computes its replica index (0 <= replica < HISTO_REPLICAS).
    int replica = threadIdx.x / threadsPerReplica;

    // Declare the shared memory array for histogram privatization.
    // Its size is (HISTO_REPLICAS * nbins) unsigned integers.
    extern __shared__ unsigned int shared_hist[];

    // Initialize the shared histogram copies to zero.
    // Each thread cooperatively initializes part of the shared memory array.
    for (int i = threadIdx.x; i < HISTO_REPLICAS * nbins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // Each thread processes ITEMS_PER_THREAD consecutive input characters.
    // The starting index for this block is computed based on the block index.
    int blockStart = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int index = blockStart + threadIdx.x + i * blockDim.x;
        if (index < inputSize) {
            // Cast the input character to unsigned to properly interpret its ordinal value.
            unsigned char ch = static_cast<unsigned char>(input[index]);
            // Check if the character falls within the [from, to] range.
            if (ch >= from && ch <= to) {
                int bin = static_cast<int>(ch) - from;
                // Update the corresponding bin in the private replicated histogram using atomic operation.
                atomicAdd(&shared_hist[replica * nbins + bin], 1);
            }
        }
    }
    __syncthreads();

    // Each thread aggregates the results from all replicas for a subset of the bins.
    // Then, it atomically accumulates this block's contribution into the global histogram.
    for (int bin = threadIdx.x; bin < nbins; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int r = 0; r < HISTO_REPLICAS; r++) {
            sum += shared_hist[r * nbins + bin];
        }
        atomicAdd(&global_histogram[bin], sum);
    }
}

// Host function to invoke the histogram kernel.
// 'input' and 'histogram' are assumed to be allocated on the device (via cudaMalloc).
// 'inputSize' is the total number of characters in the input buffer.
// 'from' and 'to' specify the inclusive range of character ordinal values for which the histogram should be computed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Compute the number of histogram bins.
    int nbins = to - from + 1;

    // Set up kernel launch parameters.
    // Choosing a block size that is a multiple of HISTO_REPLICAS.
    const int threadsPerBlock = 256;
    // Each block processes (threadsPerBlock * ITEMS_PER_THREAD) characters.
    int elementsPerBlock = threadsPerBlock * ITEMS_PER_THREAD;
    // Compute the number of blocks required to cover the entire input.
    int numBlocks = (inputSize + elementsPerBlock - 1) / elementsPerBlock;

    // Compute the required shared memory size per block.
    // Each block allocates HISTO_REPLICAS * nbins unsigned integers.
    size_t sharedMemSize = HISTO_REPLICAS * nbins * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to);
}