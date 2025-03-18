#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel to compute a histogram for a specified range of characters.
// The input is a device array of char, and the histogram range is specified by 'from' and 'to'.
// Each thread block uses shared memory to build a private histogram and then atomically adds its results
// into the global histogram to reduce the number of global memory atomic updates.
__global__ void histogramKernel(const char *input,
                                unsigned int *histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    // Compute the number of bins in the histogram.
    int range = to - from + 1;

    // Declare shared memory for the block-local histogram. The size is determined by the kernel launch.
    extern __shared__ unsigned int s_hist[];

    // Each thread in the block initializes part of the shared histogram to 0.
    // A for-loop covers all bins if blockDim.x is less than 'range'.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }

    // Ensure the shared histogram is fully initialized.
    __syncthreads();

    // Calculate the global thread index and the stride over the input array.
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes multiple characters from the input using a stride loop.
    for (unsigned int i = globalIdx; i < inputSize; i += stride) {
        // Read the character and cast to unsigned char for proper comparison.
        unsigned char c = input[i];
        // Check if the character lies within the desired range.
        if (c >= from && c <= to) {
            // Calculate the local histogram bin and increment it atomically in shared memory.
            atomicAdd(&s_hist[c - from], 1);
        }
    }

    // Ensure all threads have finished updating the shared histogram.
    __syncthreads();

    // Each thread writes a portion of the shared histogram into the global histogram.
    // Atomic add is used since multiple blocks may update the same global bins concurrently.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        // Only perform an atomic add if there was an update, avoiding unnecessary atomic calls.
        if (s_hist[i] != 0)
            atomicAdd(&histogram[i], s_hist[i]);
    }
}

// Host function that launches the CUDA kernel to compute the histogram on the device.
// The function assumes 'input' and 'histogram' are device pointers allocated with cudaMalloc.
// No explicit host-device synchronization (e.g., cudaDeviceSynchronize) is done here.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Choose optimal block size. For modern NVIDIA GPUs, 256 is a common choice.
    int blockSize = 256;
    
    // Determine grid size based on the input size.
    // We limit the number of blocks to a reasonable maximum (e.g., 1024) to balance load.
    int gridSize = (inputSize + blockSize - 1) / blockSize;
    if (gridSize > 1024)
        gridSize = 1024;

    // Compute the number of bins in the histogram and allocate dynamic shared memory size.
    int range_size = to - from + 1;
    size_t sharedMemSize = range_size * sizeof(unsigned int);

    // Launch the histogram kernel on the GPU.
    histogramKernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);

    // Note: Any necessary error checking and device synchronization is assumed to be handled externally.
}