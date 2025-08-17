#include <cuda_runtime.h>
#include <stdint.h>

// Optimized CUDA histogram kernel with shared-memory privatization.
// - Computes histogram for the character ordinal range [from, to].
// - Uses 32 copies of the histogram in shared memory, one per warp lane,
//   laid out with a stride of 32 so that bin i of copy c is at offset i*32 + c.
//   This arrangement ensures that when a warp updates the same bin index i,
//   each lane hits a distinct shared memory bank, avoiding bank conflicts.
// - Each thread processes itemsPerThread input characters per grid-stride iteration.
// - After local accumulation, blocks reduce their 32 copies into a single sum per bin
//   and atomically add to the global histogram.
static constexpr int itemsPerThread = 8; // Good default for A100/H100-class GPUs and large inputs.

__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ globalHist,
                                       unsigned int inputSize,
                                       int from, int to)
{
    // Validate range on device side (defensive). If invalid, do nothing.
    if (from < 0 || to > 255 || from > to) return;

    const unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);
    const unsigned int lane = threadIdx.x & 31; // lane id within warp

    // Dynamic shared memory layout:
    // 32 copies of the histogram, strided so that:
    // sHist[i*32 + c] holds bin i for copy c (0 <= c < 32).
    extern __shared__ unsigned int sHist[];

    // Zero shared histograms cooperatively
    const unsigned int sBins = rangeLen * 32u;
    for (unsigned int idx = threadIdx.x; idx < sBins; idx += blockDim.x) {
        sHist[idx] = 0u;
    }
    __syncthreads();

    // Grid-stride loop setup: each thread processes itemsPerThread items per iteration.
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockDim.x * itemsPerThread;

    // Process the input in chunks assigned to each thread
    for (size_t base = t * itemsPerThread; base < inputSize; base += gridStride) {
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            size_t idx = base + static_cast<size_t>(k);
            if (idx >= inputSize) break;
            // Read the character and map to unsigned to avoid negative values for chars >= 128.
            unsigned char uc = static_cast<unsigned char>(input[idx]);

            // Filter to the requested [from, to] range, inclusive.
            int v = static_cast<int>(uc);
            if (v >= from && v <= to) {
                unsigned int localBin = static_cast<unsigned int>(v - from);
                // Bin index for this thread's private copy (copy index = lane)
                unsigned int sIdx = localBin * 32u + lane;
                // Shared-memory atomic add (fast on modern GPUs). Collisions are minimized by copy privatization.
                atomicAdd(&sHist[sIdx], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce the 32 privatized copies into a single sum per bin and atomically add to global memory.
    // Use all threads in the block to cooperatively accumulate different bins.
    for (unsigned int bin = threadIdx.x; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0u;
        // Accumulate across the 32 lane-private copies.
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += sHist[bin * 32u + static_cast<unsigned int>(c)];
        }
        if (sum) {
            atomicAdd(&globalHist[bin], sum);
        }
    }
}

// Host function to launch the histogram kernel.
// - input: device pointer to input char buffer.
// - histogram: device pointer to output histogram buffer of length (to - from + 1) unsigned ints.
// - inputSize: number of chars in the input buffer.
// - from, to: inclusive character ordinal range [from, to], with 0 <= from < to <= 255.
// The function zeros the output histogram, configures the kernel launch using an occupancy-based
// grid size, and launches the kernel. Synchronization is left to the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (from > to) return;

    const unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);

    // Zero the output histogram on device (async on default stream).
    cudaMemsetAsync(histogram, 0, rangeLen * sizeof(unsigned int));

    // Choose a block size tuned for modern NVIDIA GPUs.
    // 256 threads is a good balance between occupancy and per-thread resource usage.
    const int blockSize = 256;

    // Dynamic shared memory size: 32 privatized copies of the histogram, strided by 32.
    const size_t sharedMemSize = static_cast<size_t>(rangeLen) * 32u * sizeof(unsigned int);

    // Compute grid size based on occupancy with the specified dynamic shared memory.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM,
                                                  histogram_range_kernel,
                                                  blockSize,
                                                  static_cast<size_t>(sharedMemSize));
    // Fallback in case occupancy query returns 0 for some corner case.
    if (maxBlocksPerSM <= 0) {
        maxBlocksPerSM = 1;
    }

    int gridSize = maxBlocksPerSM * prop.multiProcessorCount;
    // Ensure at least 1 block.
    if (gridSize <= 0) gridSize = 1;

    // Launch the kernel (async on default stream). The caller handles synchronization.
    histogram_range_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}