#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// CUDA kernel: compute histogram for characters in [from, to] range.
// Each block builds a private histogram in shared memory, then atomically
// accumulates it into the global histogram.
//
// Parameters:
//   input      - device pointer to input characters
//   histogram  - device pointer to output histogram (size = to - from + 1)
//   inputSize  - number of characters in input
//   from, to   - inclusive character range [from, to], 0 <= from <= to <= 255
// -----------------------------------------------------------------------------
__global__ void histogram_kernel_shared(const char* __restrict__ input,
                                        unsigned int* __restrict__ histogram,
                                        unsigned int inputSize,
                                        int from,
                                        int to)
{
    extern __shared__ unsigned int s_hist[];  // dynamic shared memory
    const int tid       = threadIdx.x;
    const int histRange = to - from + 1;

    // Initialize shared histogram to zero.
    for (int i = tid; i < histRange; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over input data.
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int idx = blockIdx.x * blockDim.x + tid;
         idx < inputSize;
         idx += stride)
    {
        // Load character and convert to unsigned to avoid sign issues.
        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Update shared histogram if character is within [from, to].
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to))
        {
            // Shared memory atomics are fast on modern GPUs (A100/H100).
            atomicAdd(&s_hist[c - from], 1u);
        }
    }

    __syncthreads();

    // Accumulate per-block histogram into global histogram.
    // Each thread processes a subset of bins to reduce contention.
    for (int i = tid; i < histRange; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val > 0) {
            atomicAdd(&histogram[i], val);
        }
    }
}

// -----------------------------------------------------------------------------
// Host function: run_histogram
//
// Computes a histogram for the given input buffer on the GPU using the above
// kernel. The histogram counts occurrences of characters whose ordinal value
// lies in the inclusive range [from, to].
//
// Parameters:
//   input      - device pointer to input characters (cudaMalloc'ed)
//   histogram  - device pointer to output histogram (cudaMalloc'ed),
//                must have space for (to - from + 1) unsigned ints
//   inputSize  - number of characters in 'input'
//   from, to   - inclusive character range [from, to], 0 <= from <= to <= 255
//
// Notes:
//   - This function does not perform any synchronization (no cudaDeviceSynchronize).
//     The caller is responsible for synchronizing if needed.
//   - The histogram buffer is zeroed before launching the kernel.
//   - Launch configuration is chosen to provide good occupancy on modern
//     data center GPUs (e.g., A100/H100) without overprovisioning blocks.
// -----------------------------------------------------------------------------
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Assume valid parameters as per problem statement:
    // 0 <= from <= to <= 255
    const int histRange = to - from + 1;

    // Zero the histogram on device (asynchronously).
    cudaMemset(histogram, 0, histRange * sizeof(unsigned int));

    // If there is no input data, we are done after zeroing.
    if (inputSize == 0) {
        return;
    }

    // Choose a reasonable block size.
    const int threadsPerBlock = 256;

    // Determine a sensible number of blocks.
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    // Start with enough blocks to cover the input once with one element per thread.
    int blocks = (static_cast<int>(inputSize) + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks < 1) {
        blocks = 1;
    }

    // Limit the number of blocks to avoid excessive oversubscription.
    // Using 4x number of SMs is generally a good heuristic.
    const int maxBlocks = props.multiProcessorCount * 4;
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    // Shared memory size: one unsigned int per histogram bin.
    const size_t sharedMemBytes = static_cast<size_t>(histRange) * sizeof(unsigned int);

    // Launch the kernel asynchronously.
    histogram_kernel_shared<<<blocks, threadsPerBlock, sharedMemBytes>>>(
        input, histogram, inputSize, from, to);
}