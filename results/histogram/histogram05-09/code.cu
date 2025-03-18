#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define constant controlling how many input characters are processed by each thread.
// The chosen default (8) is a reasonable balance of compute and memory bandwidth for modern GPUs.
#define ITEMS_PER_THREAD 8

// -----------------------------------------------------------------------------
// CUDA kernel to compute a histogram for a specified character range [from, to].
// Input:
//   input           - pointer to the device array of characters.
//   global_histogram- pointer to the device array holding the histogram counts.
//                     Its size must be (to - from + 1).
//   inputSize       - total number of characters in the input array.
//   from, to        - define the inclusive ordinal range to accumulate.
//
// The kernel uses shared memory to privatize histograms per warp (to reduce bank
// conflicts) and later reduces them to update the global histogram.
//
// Each thread processes ITEMS_PER_THREAD characters. The shared memory is
// allocated in a 2D layout: one row per warp copy, with an extra element as padding
// to avoid bank conflicts. The row length is (histoSize + 1) where histoSize = (to-from+1).
// -----------------------------------------------------------------------------
/// @FIXED
/// extern "C" __global__ void histogram_kernel(const char *input, unsigned int *global_histogram,
__global__ void histogram_kernel(const char *input, unsigned int *global_histogram,
                                              unsigned int inputSize, int from, int to) {
    // Determine the histogram range size.
    int histoSize = to - from + 1;

    // Each block uses multiple copies of the histogram to limit shared memory bank conflicts.
    // We assign one copy per warp.
    const int warpSize = 32;
    int warpId = threadIdx.x / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    // Declare dynamically allocated shared memory.
    // We allocate numWarps copies, each of (histoSize + 1) integers (extra element as padding).
    extern __shared__ unsigned int smem[];

    // Initialize the shared memory histogram copies to zero in parallel.
    int totalSharedElements = numWarps * (histoSize + 1);
    for (int i = threadIdx.x; i < totalSharedElements; i += blockDim.x) {
        smem[i] = 0;
    }
    __syncthreads();

    // Calculate the starting index in the input for this thread.
    // Each thread processes ITEMS_PER_THREAD characters.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int base = globalThreadId * ITEMS_PER_THREAD;

    // Loop over the ITEMS_PER_THREAD characters assigned to this thread.
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = base + i;
        if (idx < inputSize) {
            // Read the character.
            unsigned char ch = static_cast<unsigned char>(input[idx]);
            // Check if the character falls in the [from, to] range.
            if (ch >= from && ch <= to) {
                int bin = ch - from;
                // Compute the index into the shared memory for this warp's histogram copy.
                // Each copy has a stride of (histoSize + 1) to avoid bank conflicts.
                int smemIndex = warpId * (histoSize + 1) + bin;
                // Update the shared histogram with an atomic add.
                atomicAdd(&smem[smemIndex], 1);
            }
        }
    }
    __syncthreads();

    // Each thread now reduces a subset of the bins from all copies to update the global histogram.
    for (int bin = threadIdx.x; bin < histoSize; bin += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate counts from each warp copy.
        for (int w = 0; w < numWarps; w++) {
            sum += smem[w * (histoSize + 1) + bin];
        }
        // Atomically accumulate the sum into the global histogram.
        if(sum > 0)
            atomicAdd(&global_histogram[bin], sum);
    }
    // No further synchronization is needed.
}

// -----------------------------------------------------------------------------
// Host function to compute and launch the histogram kernel.
// The input and histogram pointers must be allocated on the device (e.g., using cudaMalloc).
// Host-device synchronization is assumed to be managed by the caller.
// Parameters:
//   input      - device pointer to the input text buffer (array of chars).
//   histogram  - device pointer to the output histogram array; size must be (to - from + 1).
//   inputSize  - number of characters in the input buffer.
//   from, to   - specify the inclusive range [from, to] for which the histogram is computed.
// -----------------------------------------------------------------------------
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram,
void run_histogram(const char *input, unsigned int *histogram,
                              unsigned int inputSize, int from, int to) {
    // Set the number of threads per block.
    int blockSize = 256;
    // Each thread processes ITEMS_PER_THREAD characters.
    int itemsPerBlock = blockSize * ITEMS_PER_THREAD;
    // Determine the number of blocks needed to cover the entire input.
    int gridSize = (inputSize + itemsPerBlock - 1) / itemsPerBlock;

    // Compute the histogram range size.
    int histoSize = to - from + 1;
    // Calculate the number of warp copies that will be used in each block.
    const int warpSize = 32;
    int numWarps = (blockSize + warpSize - 1) / warpSize;
    // Compute the required dynamic shared memory size per block.
    // Each warp copy requires (histoSize + 1) unsigned ints.
    size_t sharedMemSize = numWarps * (histoSize + 1) * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}
