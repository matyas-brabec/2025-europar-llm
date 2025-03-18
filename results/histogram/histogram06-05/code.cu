#include <cuda_runtime.h>
#include <cuda.h>

// Constant controlling how many input chars are processed by each thread.
// Chosen as 16 to provide sufficient work per thread on modern GPUs.
constexpr int itemsPerThread = 16;

// CUDA kernel to compute a histogram of a text file restricted to characters in the range [from, to].
// Each thread processes 'itemsPerThread' characters from the input array.
// The kernel makes use of shared memory with 32 copies of the histogram (one per bank) to minimize bank conflicts:
//   - The shared memory array has dimensions: 32 (copies) x (# bins) where # bins = to - from + 1.
//   - Each thread uses its lane index (threadIdx.x % 32) to select which histogram copy to update.
// After processing, thread 0 in each block reduces the 32 copies into a block-local histogram,
// and then atomically accumulates the result into the global histogram.
__global__ void histogram_kernel(const char* input, unsigned int inputSize, int from, int to, unsigned int* global_hist) {
    // Compute number of histogram bins.
    int histoSize = to - from + 1;
    
    // Declare shared memory histogram array.
    // We allocate 32 copies to avoid bank conflicts.
    // Total shared memory elements = 32 * histoSize.
    extern __shared__ unsigned int s_hist[];

    // Initialize the shared memory histogram to 0.
    int tid = threadIdx.x;
    int totalShmElements = 32 * histoSize;
    for (int i = tid; i < totalShmElements; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each block processes blockDim.x * itemsPerThread characters.
    // Each thread processes itemsPerThread characters spaced by blockDim.x.
    int globalStartIdx = blockIdx.x * blockDim.x * itemsPerThread;
    // Each thread uses its lane id to select a dedicated bank copy.
    int myBank = tid & 31;  // equivalent to (tid % 32)

    // Process items assigned to this thread.
    for (int i = 0; i < itemsPerThread; ++i) {
        int pos = globalStartIdx + tid + i * blockDim.x;
        if (pos < inputSize) {
            // Load the character and convert to unsigned.
            unsigned char c = input[pos];
            // If the character is within the desired range, update the histogram.
            if (c >= from && c <= to) {
                int bin = c - from;
                // Update the shared memory histogram in the bank corresponding to this thread.
                // Atomic operation is used to safely update the shared copy.
                atomicAdd(&s_hist[myBank * histoSize + bin], 1);
            }
        }
    }
    __syncthreads();

    // One thread (thread 0) reduces the 32 copies of each bin into a single value
    // and then atomically adds the partial result into the global histogram.
    if (tid == 0) {
        for (int bin = 0; bin < histoSize; ++bin) {
            unsigned int sum = 0;
            for (int bank = 0; bank < 32; ++bank) {
                sum += s_hist[bank * histoSize + bin];
            }
            atomicAdd(&global_hist[bin], sum);
        }
    }
}

// Host function that invokes the CUDA kernel to compute the histogram.
// Assumes that 'input' and 'histogram' are device pointers allocated via cudaMalloc.
// The 'inputSize' is the number of characters in the 'input' array.
// Host-device synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Compute the number of bins in the histogram.
    int histoSize = to - from + 1;

    // Choose block size: 256 threads per block is usually well-suited on modern GPUs.
    int threadsPerBlock = 256;
    dim3 block(threadsPerBlock);

    // Each thread processes 'itemsPerThread' characters, so each block processes:
    int itemsPerBlock = threadsPerBlock * itemsPerThread;
    // Calculate grid size to cover the entire input.
    int gridSize = (inputSize + itemsPerBlock - 1) / itemsPerBlock;
    dim3 grid(gridSize);

    // Calculate the shared memory size per block. We need 32 copies of the histogram.
    size_t sharedMemSize = 32 * histoSize * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<grid, block, sharedMemSize>>>(input, inputSize, from, to, histogram);
}