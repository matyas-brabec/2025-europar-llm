#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernel to compute a character histogram over a specified [from, to] range.
//
// - d_input: pointer to input text on device (array of 'char')
// - d_hist : pointer to output histogram on device (array of 'unsigned int')
//            length = numBins = to - from + 1
// - inputSize: number of characters in d_input
// - from, to : inclusive range of character codes to count (0 <= from <= to <= 255)
// - numBins  : precomputed as (to - from + 1)
//
// The kernel uses per-block privatized histograms in shared memory to reduce contention on global
// memory atomics:
//
//  * Each block:
//      - Initializes its shared-memory histogram to zero.
//      - Processes a grid-stride subset of the input characters.
//      - For characters c where from <= c <= to, increments bin (c - from) in shared memory using
//        fast shared-memory atomicAdd.
//      - After processing, each block merges its shared histogram into the global histogram using
//        atomicAdd on global memory for each bin.
//
// This significantly reduces the number of global atomic operations compared to incrementing the
// global histogram directly for each character.
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void histogramKernel(const char * __restrict__ d_input,
                                unsigned int * __restrict__ d_hist,
                                unsigned int inputSize,
                                int from,
                                int to,
                                int numBins)
{
    // Dynamic shared memory for per-block histogram (numBins entries of unsigned int).
    extern __shared__ unsigned int s_hist[];

    // Initialize shared histogram to zero.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        s_hist[i] = 0;
    }

    __syncthreads();

    // Compute global thread index and grid-stride.
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    // Traverse input in a grid-stride loop.
    for (unsigned int i = globalThreadId; i < inputSize; i += stride)
    {
        // Cast to unsigned char to avoid sign issues and get range [0, 255].
        unsigned char c = static_cast<unsigned char>(d_input[i]);

        // If character is within [from, to], increment the appropriate bin.
        if (c >= static_cast<unsigned char>(from) && c <= static_cast<unsigned char>(to))
        {
            int bin = static_cast<int>(c) - from; // bin index in [0, numBins - 1]
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // Merge per-block shared histogram into global histogram.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    {
        unsigned int val = s_hist[i];
        if (val != 0)
        {
            atomicAdd(&d_hist[i], val);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host function to launch the histogram kernel.
//
// Parameters:
//   - input      : device pointer to input text (cudaMalloc'd array of chars)
//   - histogram  : device pointer to output histogram (cudaMalloc'd array of unsigned int)
//                  Must have space for (to - from + 1) entries.
//   - inputSize  : number of characters in the input buffer
//   - from, to   : inclusive range of character codes to count (0 <= from < to <= 255)
//
// Assumptions:
//   - 'input' and 'histogram' are already allocated on the device by cudaMalloc.
//   - Host-device synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
//   - Error checking on CUDA runtime calls is omitted for brevity; production code should check
//     return values of CUDA API calls.
//
// Behavior:
//   - Clears the output histogram to zero.
//   - If inputSize == 0 or the range is empty, returns immediately.
//   - Chooses a reasonable launch configuration and invokes the CUDA kernel.
////////////////////////////////////////////////////////////////////////////////////////////////////
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Compute number of histogram bins; expect 0 <= from <= to <= 255.
    const int numBins = to - from + 1;
    if (numBins <= 0)
    {
        // Invalid or empty range; nothing to do.
        return;
    }

    // Clear the histogram on device.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // If there is no input data, we are done after clearing the histogram.
    if (inputSize == 0)
    {
        return;
    }

    // Choose a block size that is a good fit for modern GPUs.
    const int blockSize = 256;

    // Compute grid size, capping to a reasonable maximum for gridDim.x.
    // Use 64-bit arithmetic to avoid overflow for very large inputSize.
    unsigned long long inputSize64 = static_cast<unsigned long long>(inputSize);
    unsigned long long blocksNeeded = (inputSize64 + blockSize - 1ULL) / blockSize;

    int gridSize;
    const int maxGridSizeX = 65535; // Safe lower bound for all CUDA-capable GPUs.
    if (blocksNeeded > static_cast<unsigned long long>(maxGridSizeX))
        gridSize = maxGridSizeX;
    else
        gridSize = static_cast<int>(blocksNeeded);

    // Dynamic shared memory size: one per-block histogram.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the kernel. Synchronization (if desired) must be done by the caller.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to,
        numBins
    );
}