#include <cuda_runtime.h>

/*
 * CUDA kernel to compute a histogram of characters in a specified [from, to] range.
 *
 * - input:      device pointer to input text (chars)
 * - histogram:  device pointer to output histogram (length = to - from + 1)
 * - inputSize:  number of characters in input
 * - from, to:   inclusive character range [from, to]; 0 <= from < to <= 255
 *
 * Implementation details:
 * - Each block builds a private histogram in shared memory to reduce global
 *   atomic contention.
 * - Shared histogram size is (to - from + 1) bins of unsigned int.
 * - Input is processed in a grid-stride loop.
 * - Main body is vectorized using uchar4 loads for better memory throughput.
 * - Remaining (inputSize % 4) tail characters are processed by a single thread.
 * - Finally, per-block histograms are merged into the global histogram:
 *      * If gridDim.x == 1, we directly write shared histogram to global memory.
 *      * Otherwise, we use atomicAdd on global memory to accumulate blocks.
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int s_hist[];

    const int range = to - from + 1;

    // Initialize per-block shared histogram
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    const unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride    = blockDim.x * gridDim.x;

    // Number of 4-byte groups we can process with vectorized loads
    const unsigned int numVec4   = inputSize / 4;
    const uchar4* input4         = reinterpret_cast<const uchar4*>(input);

    // Process main body of the input using uchar4 loads
    for (unsigned int idx4 = globalIdx; idx4 < numVec4; idx4 += stride) {
        uchar4 v = input4[idx4];

        int bin;

        bin = static_cast<int>(v.x) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
            atomicAdd(&s_hist[bin], 1u);
        }

        bin = static_cast<int>(v.y) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
            atomicAdd(&s_hist[bin], 1u);
        }

        bin = static_cast<int>(v.z) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
            atomicAdd(&s_hist[bin], 1u);
        }

        bin = static_cast<int>(v.w) - from;
        if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
            atomicAdd(&s_hist[bin], 1u);
        }
    }

    // Handle remaining (inputSize % 4) tail characters with a single thread.
    // This introduces negligible overhead (at most 3 characters per call).
    const unsigned int tailStart = numVec4 * 4;
    if (globalIdx == 0) {
        for (unsigned int i = tailStart; i < inputSize; ++i) {
            unsigned char c = static_cast<unsigned char>(input[i]);
            int bin = static_cast<int>(c) - from;
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(range)) {
                atomicAdd(&s_hist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // Flush per-block shared histogram into global histogram.
    // For a single block, we can directly store to global memory (no atomics needed).
    if (gridDim.x == 1) {
        for (int i = threadIdx.x; i < range; i += blockDim.x) {
            histogram[i] = s_hist[i];
        }
    } else {
        for (int i = threadIdx.x; i < range; i += blockDim.x) {
            unsigned int val = s_hist[i];
            if (val != 0u) {
                atomicAdd(&histogram[i], val);
            }
        }
    }
}

/*
 * Host-side function to launch the histogram kernel.
 *
 * Parameters:
 * - input:      device pointer to input buffer (chars), allocated by cudaMalloc
 * - histogram:  device pointer to histogram buffer (unsigned int),
 *               length must be (to - from + 1), allocated by cudaMalloc
 * - inputSize:  number of characters in the input buffer
 * - from, to:   inclusive character range [from, to]; 0 <= from < to <= 255
 *
 * Notes:
 * - This function does not perform any synchronization; it relies on the caller
 *   to synchronize if needed.
 * - The histogram buffer is zero-initialized before kernel launch.
 * - Grid size is chosen based on input size and device properties for good
 *   occupancy, while remaining fully asynchronous.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int range = to - from + 1;

    // Zero out the output histogram on device (asynchronously)
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // If there is no input data, nothing more to do
    if (inputSize == 0) {
        return;
    }

    // Choose a reasonable block size; 256 threads is a good general choice
    // for modern NVIDIA GPUs (e.g., A100, H100).
    const int threadsPerBlock = 256;

    // Determine an appropriate grid size.
    // We attempt to cap the grid size to a multiple of SM count for good occupancy.
    int blocks = 0;

    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        err = cudaGetDeviceProperties(&prop, device);
    }

    if (err == cudaSuccess) {
        // Minimum grid size to cover all elements with one pass
        unsigned int minGrid = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
        if (minGrid == 0) {
            minGrid = 1;
        }

        // Aim for up to 32 blocks per SM for high occupancy
        unsigned int maxGrid = static_cast<unsigned int>(prop.multiProcessorCount) * 32u;
        if (maxGrid == 0) {
            maxGrid = 1;
        }

        blocks = static_cast<int>(minGrid < maxGrid ? minGrid : maxGrid);
    } else {
        // Fallback if device properties could not be queried
        unsigned int grid = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
        if (grid == 0) {
            grid = 1;
        }
        // Conservative upper bound compatible with older architectures
        const unsigned int maxGrid = 65535u;
        if (grid > maxGrid) {
            grid = maxGrid;
        }
        blocks = static_cast<int>(grid);
    }

    // Shared memory size: one unsigned int per histogram bin
    const size_t sharedMemSize = static_cast<size_t>(range) * sizeof(unsigned int);

    // Launch kernel (asynchronous)
    histogram_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}