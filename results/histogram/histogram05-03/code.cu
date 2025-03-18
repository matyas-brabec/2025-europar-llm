#include <cuda_runtime.h>

// The number of input characters each thread processes.
// Tuned for modern NVIDIA GPUs and large inputs.
static constexpr int itemsPerThread = 64;

// Number of replicated shared memory histogram copies per block.
// Multiple copies help reduce shared memory bank conflicts.
static constexpr int NUM_HISTO_COPIES = 8;

// 
// CUDA Kernel to compute a histogram over a restricted char range.
// Each block uses shared memory to accumulate a partial histogram.
// The input text is processed in chunks of "itemsPerThread" per thread.
// To reduce contention and bank conflicts, each thread updates a designated
// copy (sliced within shared memory) so that simultaneous writes hit different banks.
// Finally, the copies are reduced and the block's result is atomically accumulated
// into the global histogram.
//
__global__ void histogram_kernel(const char *input, unsigned int *global_hist,
                                 unsigned int inputSize, int from, int histSize, int itemsPerThread)
{
    // Shared memory allocation:
    // The layout is: NUM_HISTO_COPIES copies stored contiguously.
    // Each copy holds 'histSize' unsigned integers.
    extern __shared__ unsigned int s_hist[];

    // Total number of shared histogram bins in this block.
    const int totalSharedBins = NUM_HISTO_COPIES * histSize;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Initialize shared memory histogram bins to zero in parallel.
    for (int i = tid; i < totalSharedBins; i += blockSize) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the starting index for this block.
    unsigned int blockStart = blockIdx.x * blockSize * itemsPerThread;

    // Determine which copy this thread will update.
    int myCopy = threadIdx.x % NUM_HISTO_COPIES;
    // Pointer to the start of this thread's designated histogram copy in shared memory.
    unsigned int *my_hist = s_hist + myCopy * histSize;

    // Process 'itemsPerThread' characters per thread.
    // Each iteration processes one element from the block's assigned segment of the input.
    for (int i = 0; i < itemsPerThread; i++) {
        unsigned int idx = blockStart + threadIdx.x + i * blockSize;
        if (idx < inputSize) {
            char c = input[idx];
            // Check if the character is within the desired range [from, to].
            // histSize = to - from + 1.
            if ((unsigned char)c >= (unsigned char)from && (unsigned char)c <= (unsigned char)(from + histSize - 1)) {
                int bin = (int)c - from;
                // Atomically increment the bin in this thread's shared memory copy.
                atomicAdd(&my_hist[bin], 1);
            }
        }
    }
    __syncthreads();

    // Reduce all histogram copies within the block.
    // Each thread accumulates counts for bins in a strided manner.
    for (int bin = tid; bin < histSize; bin += blockSize) {
        unsigned int binSum = 0;
        // Sum the counts from all the replicated copies.
        for (int copy = 0; copy < NUM_HISTO_COPIES; copy++) {
            binSum += s_hist[copy * histSize + bin];
        }
        // Atomically update the global histogram.
        atomicAdd(&global_hist[bin], binSum);
    }
}


//
// Host function to launch the histogram kernel.
// 'input' is a device pointer to the text data.
// 'histogram' is a device pointer to an array allocated with length equal to (to - from + 1).
// 'inputSize' is the total number of characters in the input.
// 'from' and 'to' specify the inclusive range of characters for the histogram.
//
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    // Compute the size of the histogram range.
    int histSize = to - from + 1;

    // Define block size: using 256 threads per block.
    int blockSize = 256;

    // Compute grid size: each block processes (blockSize * itemsPerThread) items.
    int gridSize = (inputSize + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);

    // Calculate the required shared memory size (in bytes) per block.
    // There are NUM_HISTO_COPIES copies of the histogram, each of size 'histSize'.
    size_t sharedMemSize = NUM_HISTO_COPIES * histSize * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, histSize, itemsPerThread);
}