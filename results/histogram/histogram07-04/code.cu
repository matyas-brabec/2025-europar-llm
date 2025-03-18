#include <cuda_runtime.h>

// This CUDA kernel computes a histogram for characters in the input array that fall within a given range [from, to].
// Each thread processes a fixed number of characters (itemsPerThread) and accumulates counts in a local register array.
// These local histograms are then added to a shared memory histogram that is privatized into 32 copies 
// (to avoid bank conflicts) with a strided layout: for bin i and copy c, the location is i*32 + c.
// Finally, one reduction loop per thread block sums the 32 copies for each bin and atomically adds the result 
// to the global histogram.
 
// Constant controlling how many input characters each thread processes.
constexpr int itemsPerThread = 256;

__global__ void histogram_kernel(const char *input, unsigned int *global_hist, unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins for the histogram.
    int histSize = to - from + 1;

    // Declare externally allocated shared memory.
    // Shared memory will store 32 copies of the histogram, each of size "histSize".
    // The layout is such that for a given bin i (0 <= i < histSize) and copy c (0 <= c < 32),
    // the counter is stored at index: i * 32 + c.
    extern __shared__ unsigned int s_hist[];
    
    // Determine which shared memory copy this thread will update.
    int myCopy = threadIdx.x % 32;

    // Initialize the shared memory histogram bins to 0.
    int totalSharedBins = histSize * 32;
    for (int i = threadIdx.x; i < totalSharedBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Allocate a local histogram in registers.
    // The maximum possible size is 256 because the full range of unsigned char is 0..255.
    unsigned int local_hist[256];
    for (int i = 0; i < histSize; i++) {
        local_hist[i] = 0;
    }
    
    // Calculate the starting global position for this thread.
    // Each thread processes 'itemsPerThread' consecutive characters.
    unsigned int globalStart = (blockIdx.x * blockDim.x + threadIdx.x) * itemsPerThread;

    // Process itemsPerThread characters per thread.
    for (int i = 0; i < itemsPerThread; i++) {
        unsigned int pos = globalStart + i;
        if (pos < inputSize) {
            // Read the character as unsigned to avoid sign issues.
            unsigned char c = input[pos];
            // Only count characters in the specified range [from, to].
            if (c >= from && c <= to) {
                int bin = c - from;
                local_hist[bin]++;
            }
        }
    }
    
    // Flush the per-thread local histogram into the appropriate shared memory copy.
    // The use of atomicAdd ensures correct accumulation when multiple threads update the same bin.
    for (int bin = 0; bin < histSize; bin++) {
        if (local_hist[bin] > 0) {
            atomicAdd(&s_hist[bin * 32 + myCopy], local_hist[bin]);
        }
    }
    __syncthreads();
    
    // Each thread block reduces the 32 copies in shared memory for each bin and adds the result atomically to global memory.
    // Threads cooperate by processing different bins in a strided fashion.
    for (int bin = threadIdx.x; bin < histSize; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int c = 0; c < 32; c++) {
            sum += s_hist[bin * 32 + c];
        }
        atomicAdd(&global_hist[bin], sum);
    }
}
 
// Host function to launch the histogram CUDA kernel.
// This function takes device pointers for input and histogram arrays (allocated via cudaMalloc),
// the number of input characters, and the histogram range parameters 'from' and 'to'.
// It computes the appropriate launch configuration and allocates sufficient shared memory for 32 copies of the histogram.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of bins.
    int histSize = to - from + 1;
    
    // Choose a block size that works well on modern NVIDIA GPUs.
    int blockSize = 256;
    
    // Each thread processes 'itemsPerThread' characters. Total number of threads required:
    int totalThreads = (inputSize + itemsPerThread - 1) / itemsPerThread;
    
    // Compute the grid size needed.
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    if (gridSize < 1) {
        gridSize = 1;
    }
    
    // Calculate the amount of shared memory needed per block:
    // 32 copies of the histogram, each with 'histSize' unsigned integers.
    size_t sharedMemSize = histSize * 32 * sizeof(unsigned int);
    
    // Launch the histogram kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
    
    // Note: Any error checking and host-device synchronization is assumed to be handled by the caller.
}