#include <cuda_runtime.h>
#include <cuda.h>

// CUDA kernel to compute histogram for characters in a specified range [from, to].
// The input text is provided as a device array of chars (input) with inputSize elements.
// The output global_hist array (of size (to - from + 1)) accumulates counts for each character.
// This kernel uses shared memory to build a per-block histogram before atomic accumulation to global memory.
__global__ void histogram_kernel(const char *input, unsigned int *global_hist, unsigned int inputSize, int from, int to)
{
    // Compute the number of bins in the histogram. For inclusive range, number of bins = to - from + 1.
    int nbins = to - from + 1;

    // Declare shared memory histogram. Its size must be supplied during kernel launch.
    extern __shared__ unsigned int shared_hist[];

    // Initialize the shared histogram bins to zero. Each thread initializes multiple bins if necessary.
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // Process the input array using a grid-stride loop.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (idx < inputSize) {
        unsigned char c = input[idx];
        // Check if the character falls within the specified [from, to] range.
        if (c >= from && c <= to) {
            // Compute the corresponding bin index and update shared memory histogram using atomicAdd.
            atomicAdd(&shared_hist[c - from], 1U);
        }
        idx += stride;
    }
    __syncthreads();

    // After processing, each thread writes parts of the per-block (shared) histogram to global memory.
    // Global memory updates are performed with atomicAdd to safely accumulate across all blocks.
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        atomicAdd(&global_hist[i], shared_hist[i]);
    }
}

// Host function that sets up and invokes the CUDA kernel for histogram computation.
// 'input' and 'histogram' are assumed to be allocated on the device via cudaMalloc.
// The parameter 'inputSize' is the number of characters in the input array.
// The range [from, to] specifies the inclusive interval of character codes to be histogrammed.
// Note: Host-device synchronization, error checking, and stream management are assumed to be handled by the caller.
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Choose a number of threads per block tuned for modern GPUs like the A100/H100.
    int threadsPerBlock = 256;
    // Calculate the number of blocks required to cover all input elements.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate the size of the dynamic shared memory required,
    // which is one unsigned int per histogram bin.
    size_t sharedMemSize = (to - from + 1) * sizeof(unsigned int);

    // Launch the kernel with the computed grid configuration and dynamic shared memory allocation.
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to);
    // Any necessary host-device synchronization (e.g., cudaDeviceSynchronize) is assumed to be handled externally.
}
