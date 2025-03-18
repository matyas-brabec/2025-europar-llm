#include <cuda_runtime.h>

// Define constant controlling how many input characters are processed by each thread.
// For modern NVIDIA GPUs and large input sizes, 128 is a good default.
#define ITEMS_PER_THREAD 128

// CUDA kernel that computes a partial histogram over a specified range [from, to] of characters.
// The kernel employs shared memory privatization with 32 copies of the histogram to avoid bank conflicts.
// Each shared memory copy is padded with an extra element so that its stride (numBins + 1) is not a multiple of 32.
__global__ void histogramKernel(const char *input, unsigned int *globalHistogram, unsigned int inputSize, int from, int to)
{
    // Compute the number of histogram bins.
    const int numBins = to - from + 1;
    // Use a padded stride to place each copy in a distinct bank.
    const int stride = numBins + 1;

    // Declare shared memory for partial histograms.
    // There will be 32 copies (one for each possible warp lane in a warp) with 'stride' elements each.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared memory histogram.
    int tid = threadIdx.x;
    int totalSharedElements = 32 * stride;
    for (int i = tid; i < totalSharedElements; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of the input.
    // Each thread processes ITEMS_PER_THREAD characters, strided by blockDim.x.
    int blockBase = blockIdx.x * blockDim.x * ITEMS_PER_THREAD;
    int index = blockBase + threadIdx.x;

    // Assign each thread in a warp a unique copy index based on its lane.
    // This ensures that within a warp each thread updates a different shared memory copy.
    int lane = threadIdx.x & 31;  // lane index in [0, 31]

    // Process ITEMS_PER_THREAD characters per thread.
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int dataIdx = index + i * blockDim.x;
        if (dataIdx < inputSize)
        {
            // Read the character and treat it as unsigned.
            unsigned char ch = static_cast<unsigned char>(input[dataIdx]);
            // Check if the character falls within the specified range.
            if (ch >= from && ch <= to)
            {
                int bin = ch - from;
                // Update the shared histogram.
                // Since threads from different warps may update the same copy concurrently,
                // use an atomicAdd in shared memory.
                atomicAdd(&s_hist[lane * stride + bin], 1);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 private copies into the global histogram.
    // Each thread cooperatively reduces a subset of bins.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
    {
        unsigned int sum = 0;
        for (int copy = 0; copy < 32; copy++)
        {
            sum += s_hist[copy * stride + bin];
        }
        // Use atomicAdd to combine the partial result from this block into global memory.
        atomicAdd(&globalHistogram[bin], sum);
    }
}

// Host function that launches the histogramKernel.
// Assumes that 'input' and 'histogram' are device pointers allocated with cudaMalloc,
// and that any host-device synchronization is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Configure the kernel launch parameters.
    const int blockSize = 256;  // number of threads per block
    // Each thread processes ITEMS_PER_THREAD characters.
    int gridSize = (inputSize + blockSize * ITEMS_PER_THREAD - 1) / (blockSize * ITEMS_PER_THREAD);

    // Calculate the number of bins and determine the shared memory size.
    int numBins = to - from + 1;
    int stride = numBins + 1;  // padded stride to avoid bank conflicts
    size_t sharedMemSize = 32 * stride * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}