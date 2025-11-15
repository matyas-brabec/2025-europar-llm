#include <cuda_runtime.h>

/*
 * CUDA kernel that computes a histogram for a specified contiguous range of
 * character codes [from, to] in the input text.
 *
 * - input: device pointer to an array of chars (plain text)
 * - globalHist: device pointer to an array of unsigned int of size (to - from + 1)
 * - inputSize: number of characters in the input buffer
 * - from, to: inclusive character range [from, to] with 0 <= from <= to <= 255
 *
 * The kernel uses per-block shared memory histograms to reduce contention on
 * global memory. Each block:
 *   1. Initializes its shared-memory histogram to zero.
 *   2. Processes a grid-stride slice of the input, updating the shared histogram
 *      using atomic operations in shared memory.
 *   3. After all threads in the block finish, the shared histogram is merged
 *      into the global histogram using atomic adds.
 */
__global__ void histogram_kernel(const char *__restrict__ input,
                                 unsigned int *__restrict__ globalHist,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int sharedHist[];

    const int  rangeLen = to - from + 1;
    const unsigned int u_from = static_cast<unsigned int>(from);
    const unsigned int u_to   = static_cast<unsigned int>(to);

    // Initialize per-block shared histogram to zero.
    for (int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        sharedHist[i] = 0u;
    }

    __syncthreads();

    // Grid-stride loop over the input characters.
    unsigned int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    while (idx < inputSize) {
        // Load character and convert to unsigned [0,255] to avoid sign issues.
        unsigned int c = static_cast<unsigned char>(input[idx]);

        // If character falls in the [from, to] range, update shared histogram.
        if (c >= u_from && c <= u_to) {
            unsigned int bin = c - u_from;
            atomicAdd(&sharedHist[bin], 1u);
        }

        idx += stride;
    }

    __syncthreads();

    // Merge per-block shared histogram into the global histogram.
    for (int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        unsigned int count = sharedHist[i];
        if (count > 0u) {
            atomicAdd(&globalHist[i], count);
        }
    }
}

/*
 * Host function that launches the CUDA histogram kernel.
 *
 * Parameters:
 *   - input:     device pointer (cudaMalloc'd) to an array of chars
 *   - histogram: device pointer (cudaMalloc'd) to an array of unsigned int
 *                of size (to - from + 1)
 *   - inputSize: number of chars in the input buffer
 *   - from, to:  inclusive character range [from, to], 0 <= from <= to <= 255
 *
 * Notes:
 *   - This function assumes input and histogram are device pointers.
 *   - It zeroes the histogram output buffer before launching the kernel.
 *   - Any host-device synchronization (e.g., cudaDeviceSynchronize()) is to be
 *     handled by the caller.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic validation: nothing to do if input is empty or range is invalid.
    if (inputSize == 0u || from > to) {
        return;
    }

    const int rangeLen = to - from + 1;

    // Choose a reasonable block size for modern NVIDIA GPUs.
    const int blockSize = 256;

    // Compute grid size to cover all input elements.
    int gridSize = static_cast<int>((inputSize + blockSize - 1u) / blockSize);
    if (gridSize <= 0) {
        gridSize = 1;
    }

    // Zero the output histogram on the device.
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Shared memory size: one unsigned int per histogram bin per block.
    const size_t sharedMemBytes = static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Launch the kernel on the default stream.
    histogram_kernel<<<gridSize, blockSize, sharedMemBytes>>>(input,
                                                              histogram,
                                                              inputSize,
                                                              from,
                                                              to);

    // Optional: Check for launch errors (does not synchronize the device).
    cudaGetLastError();
}