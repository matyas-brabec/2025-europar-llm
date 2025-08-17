#include <cuda_runtime.h>
#include <stdint.h>

// CUDA kernel: per-warp privatized shared-memory histogram for a restricted character range.
// - Each block allocates (warpCount x range) counters in shared memory.
// - Threads update their warp's private histogram using shared-memory atomics.
// - After processing, per-warp histograms are reduced within the block and added to the global histogram.
// - Global histogram is updated via atomicAdd (low contention due to per-block reduction).
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from, int to)
{
    extern __shared__ unsigned int s_hist[];
    const int WARP_SIZE = 32;

    // Compute range length (inclusive)
    const int range = to - from + 1;

    // Warp topology
    const int warpId    = threadIdx.x / WARP_SIZE;
    const int laneId    = threadIdx.x % WARP_SIZE;
    const int warpCount = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Total shared-memory counters
    const int sharedBins = range * warpCount;

    // Zero initialize the shared memory histogram
    for (int i = threadIdx.x; i < sharedBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over input characters
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < (size_t)inputSize; idx += stride) {
        // Cast to unsigned to avoid sign issues with 'char'
        unsigned int c = static_cast<unsigned char>(input[idx]);
        if (c >= static_cast<unsigned int>(from) && c <= static_cast<unsigned int>(to)) {
            const int bin = static_cast<int>(c) - from;
            // Update the warp-private bin in shared memory
            atomicAdd(&s_hist[warpId * range + bin], 1u);
        }
    }
    __syncthreads();

    // Reduce per-warp histograms into a single per-block histogram and update global memory
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate across all warps for this bin
        for (int w = 0; w < warpCount; ++w) {
            sum += s_hist[w * range + bin];
        }
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
    // No further synchronization required; global atomics ensure correctness across blocks.
}

// Host function to run the histogram kernel.
// - input and histogram are device pointers allocated with cudaMalloc.
// - inputSize is the number of chars in input.
// - from and to define the inclusive [from, to] range to count (0 <= from < to <= 255).
// - The histogram buffer must be to - from + 1 elements long.
// - This function zeros the output histogram and launches the kernel; synchronization is managed by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate parameters (defensive; problem statement guarantees these)
    if (!input || !histogram) return;
    if (from < 0 || to > 255 || from > to) return;

    const int range = to - from + 1;

    // Zero the output histogram (so the kernel can safely atomicAdd its per-block partials)
    // Using the default stream; caller manages synchronization if needed.
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    if (inputSize == 0) return;

    // Choose launch configuration tuned for modern data center GPUs (A100/H100).
    // Use 256 threads per block (8 warps) to allow up to 8 blocks/SM (2048 threads max/SM).
    const int blockSize = 256;

    // Determine a reasonable grid size based on SM count (saturate the GPU).
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const int smCount = prop.multiProcessorCount;

    // Aim for 8 resident blocks per SM with 256-thread blocks for full occupancy.
    int gridSize = smCount * 8;

    // Dynamic shared memory size: one privatized histogram per warp
    const int warpCount = (blockSize + 31) / 32;
    const size_t sharedMemBytes = static_cast<size_t>(range) * warpCount * sizeof(unsigned int);

    histogram_range_kernel<<<gridSize, blockSize, sharedMemBytes>>>(input, histogram, inputSize, from, to);
}