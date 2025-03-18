#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// This CUDA kernel computes a histogram for a given continuous range of characters [from, to] in an input text.
// It uses shared memory privatization with multiple (32) copies of the histogram per block to avoid shared memory bank conflicts.
// Each thread processes a constant number (itemsPerThread) of input characters in a grid‐stride loop, accumulates counts in its own
// designated shared memory copy, and then the per‐block histograms are reduced (summed) and atomically added to the global histogram.
 
// The constant itemsPerThread controls how many input chars are processed by each thread.
// For modern NVIDIA GPUs and large inputs, 16 is a reasonable default.
 
// Kernel parameters:
//    input      - device pointer to input text array (chars)
//    inputSize  - number of characters in the input array
//    globalHist - device pointer to global histogram array (size = (to - from + 1))
//    from, to   - specify the inclusive range of char ordinals to consider
//
__global__ void histogramKernel(const char *input, unsigned int inputSize, unsigned int *globalHist, int from, int to)
{
    // Determine the number of histogram bins.
    int histoBins = to - from + 1;
    
    // Compute the stride (width) of each private copy in shared memory.
    // To avoid shared memory bank conflicts, we require that the stride be co-prime to the 32 banks.
    // Since the banks are 4-byte words, making the stride odd is sufficient.
    int copyStride = histoBins;
    if ((copyStride & 1) == 0) {
        copyStride++;  // Ensure the stride is odd.
    }
    
    // Use 32 copies throughout the block for bank conflict avoidance.
    const int numCopies = 32;
    
    // Declare shared memory. The caller must allocate at least (numCopies * copyStride) * sizeof(unsigned int) bytes.
    extern __shared__ unsigned int s_hist[];
    
    // Initialize the shared memory histogram copies to 0.
    int sharedSize = numCopies * copyStride;
    for (int i = threadIdx.x; i < sharedSize; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();
    
    // Each thread selects one copy based on its thread index modulo numCopies.
    // This ensures that within a warp (of 32 threads) each thread uses a different copy,
    // thereby avoiding intra-warp bank conflicts.
    int copyIndex = threadIdx.x % numCopies;
    
    // Process a fixed number of items per thread. This constant can be tuned for optimal performance.
    const int itemsPerThread = 16;
    
    // Compute the starting global index for this thread.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int start = globalThreadId * itemsPerThread;
    
    // Compute the stride in the input array for the grid-stride loop.
    int totalStride = blockDim.x * gridDim.x * itemsPerThread;
    
    // Process the input text in a grid-stride loop.
    // Each thread processes itemsPerThread consecutive characters per iteration.
    for (int i = start; i < inputSize; i += totalStride)
    {
        #pragma unroll
        for (int j = 0; j < itemsPerThread && (i + j) < inputSize; j++)
        {
            char c = input[i + j];
            // Only update the histogram if the character lies within the specified range.
            if (c >= from && c <= to)
            {
                int bin = (int)c - from;
                // Update the thread's designated private histogram copy in shared memory.
                atomicAdd(&s_hist[copyIndex * copyStride + bin], 1);
            }
        }
    }
    __syncthreads();
    
    // After all threads have processed the input, reduce the private copies into the global histogram.
    // Each thread handles multiple bins in a strided fashion.
    for (int bin = threadIdx.x; bin < histoBins; bin += blockDim.x)
    {
        unsigned int sum = 0;
        // Sum the counts for this bin from all private copies.
        for (int c = 0; c < numCopies; c++)
        {
            sum += s_hist[c * copyStride + bin];
        }
        // Atomically add the block's contribution to the global histogram.
        atomicAdd(&globalHist[bin], sum);
    }
    // No further synchronization is necessary.
}
 
// This host function invokes the CUDA kernel to compute the histogram on the device.
// The input and histogram arrays are assumed to be allocated on the device (using cudaMalloc).
//
// Parameters:
//    input      - device pointer to the text file stored as an array of chars
//    histogram  - device pointer to the output histogram array (size = (to - from + 1))
//    inputSize  - number of characters in the input array
//    from, to   - specify the inclusive range (from to to) of char ordinals to consider
//
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Ensure that the range parameters are valid: 0 <= from < to <= 255.
    // (This check is not done here explicitly because caller is assumed to supply valid parameters.)
    
    // Use the same constant itemsPerThread as in the kernel.
    const int itemsPerThread = 16;
    
    // Define the block size (number of threads per block). This can be tuned based on the target GPU.
    int blockSize = 256;
    
    // Calculate the number of blocks needed. Each thread processes itemsPerThread characters.
    int numBlocks = (inputSize + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
    if (numBlocks < 1)
        numBlocks = 1;  // Ensure at least one block is launched.
    
    // Determine the number of histogram bins.
    int histoBins = to - from + 1;
    
    // Compute the shared memory copy stride.
    // The stride must be odd to guarantee that the 32 copies map to distinct banks.
    int copyStride = histoBins;
    if ((copyStride & 1) == 0)
    {
        copyStride++;  // Make stride odd.
    }
    const int numCopies = 32;
    
    // Compute the amount of shared memory required per block (in bytes).
    size_t sharedMemSize = numCopies * copyStride * sizeof(unsigned int);
    
    // Launch the CUDA kernel with the computed grid and block dimensions and shared memory size.
    histogramKernel<<<numBlocks, blockSize, sharedMemSize>>>(input, inputSize, histogram, from, to);
}