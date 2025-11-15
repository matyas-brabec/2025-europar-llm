#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram over a given character range [from, to].
 * - input: pointer to device memory containing 'inputSize' chars of text.
 * - histogram: pointer to device memory with 'histLen' bins (unsigned int).
 * - inputSize: number of chars in 'input'.
 * - from: lower bound of character range (inclusive), 0 <= from <= 255.
 * - histLen: number of histogram bins, histLen = to - from + 1, 1 <= histLen <= 256.
 *
 * The kernel uses per-block shared-memory histograms to reduce global memory
 * contention. Each block:
 *   1. Initializes its private histogram in shared memory to zero.
 *   2. Processes a grid-stride subset of the input, updating the shared histogram
 *      with fast shared-memory atomics.
 *   3. After processing, merges its shared histogram into the global histogram
 *      using atomicAdds in global memory (one per bin per block at most).
 */
__global__ void histogramKernelSharedRange(const unsigned char *__restrict__ input,
                                           unsigned int *__restrict__ histogram,
                                           unsigned int inputSize,
                                           int from,
                                           int histLen)
{
    extern __shared__ unsigned int s_hist[];

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int gridSize = blockSize * gridDim.x;
    const int upper = from + histLen;  // exclusive upper bound for character range

    // 1. Initialize the shared-memory histogram to zero.
    for (int i = tid; i < histLen; i += blockSize) {
        s_hist[i] = 0u;
    }

    __syncthreads();

    // 2. Process input data in a grid-stride loop.
    for (unsigned int idx = blockIdx.x * blockSize + tid; idx < inputSize; idx += gridSize) {
        unsigned char c = input[idx];
        // Only count characters in [from, from + histLen).
        if (c >= from && c < upper) {
            int bin = static_cast<int>(c) - from;
            // Shared-memory atomic add: fast on modern GPUs (Ampere/Hopper).
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    __syncthreads();

    // 3. Merge per-block shared histogram into global histogram.
    for (int i = tid; i < histLen; i += blockSize) {
        unsigned int val = s_hist[i];
        if (val != 0u) {
            atomicAdd(&histogram[i], val);
        }
    }
}

/*
 * Host function to run the histogram kernel.
 *
 * Parameters:
 *   - input: device pointer to input text buffer (cudaMalloc'ed).
 *   - histogram: device pointer to histogram buffer (cudaMalloc'ed).
 *   - inputSize: number of characters in 'input'.
 *   - from, to: character range [from, to] to be histogrammed.
 *
 * Behavior:
 *   - Validates the range (expects 0 <= from <= to <= 255).
 *   - Computes the number of histogram bins.
 *   - Zeroes the histogram buffer on the device.
 *   - Chooses a grid/block configuration suitable for large modern GPUs.
 *   - Launches the CUDA kernel with dynamic shared memory sized to 'histLen'.
 *
 * Synchronization:
 *   - The function does not call cudaDeviceSynchronize(). Ordering with respect
 *     to other operations is governed by the default stream semantics. The caller
 *     is responsible for any host-side synchronization.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic parameter validation (no error reporting, just return on invalid input).
    if (input == nullptr || histogram == nullptr || inputSize == 0u) {
        return;
    }
    if (from < 0 || to < from || to > 255) {
        return;
    }

    const int histLen = to - from + 1;

    // Zero the histogram on the device.
    cudaMemset(histogram, 0, static_cast<size_t>(histLen) * sizeof(unsigned int));

    // Choose block size. 256 threads is a good default for histogram-like workloads.
    const int blockSize = 256;

    // Determine a reasonable grid size.
    // Start from one block per 'blockSize' input elements, then cap by a multiple
    // of the number of SMs to avoid excessive oversubscription.
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    const int smCount = props.multiProcessorCount;
    // Use up to 32 blocks per SM; modern GPUs (A100/H100) can support this.
    const int maxBlocks = smCount * 32;

    int blocks = static_cast<int>((inputSize + blockSize - 1u) / blockSize);
    if (blocks < 1) {
        blocks = 1;
    } else if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    // Dynamic shared memory size: one unsigned int per histogram bin.
    const size_t sharedMemSize = static_cast<size_t>(histLen) * sizeof(unsigned int);

    // Launch the kernel. Input is cast to unsigned char* to work with 0..255 ordinals.
    histogramKernelSharedRange<<<blocks, blockSize, sharedMemSize>>>(
        reinterpret_cast<const unsigned char *>(input),
        histogram,
        inputSize,
        from,
        histLen
    );
}