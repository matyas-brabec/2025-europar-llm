#include <cuda_runtime.h>
#include <stdint.h>

// Number of input characters processed per thread per outer grid-stride iteration.
// A value of 16 provides good ILP and memory throughput on modern NVIDIA data-center GPUs (A100/H100)
// for large inputs, while keeping register pressure moderate.
static constexpr int itemsPerThread = 16;

// CUDA kernel that computes a histogram for characters in the inclusive range [from, to].
// - input: device pointer to input chars.
// - N: number of chars in input.
// - from, to: inclusive range [from, to], with 0 <= from <= to <= 255.
// - output: device pointer to histogram of size (to - from + 1) unsigned ints.
//
// Optimization details:
// - Shared memory privatization: 32 copies (one per warp lane) of the histogram per block.
// - To avoid shared-memory bank conflicts, each histogram copy is laid out with a stride of 32:
//   For bin i (0 <= i < rangeLen) and copy c (0 <= c < 32), the element is at index i*32 + c.
// - Each thread uses copy number (threadIdx.x % 32) for its updates.
// - Shared-memory atomics are used to handle cross-warp updates to the same copy safely.
// - After processing, per-block histograms are reduced across the 32 copies and atomically added
//   to the global output histogram.
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int N,
                                       int from,
                                       int to,
                                       unsigned int* __restrict__ output)
{
    extern __shared__ unsigned int shist[]; // Size: (rangeLen * 32) uints

    const int lane = threadIdx.x & 31;               // warp lane index 0..31
    const int rangeLen = to - from + 1;              // number of histogram bins
    const unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gstride = blockDim.x * gridDim.x;

    // Initialize shared histogram copies to zero.
    // We have rangeLen bins, each with 32 copies, stored with stride 32.
    for (int i = threadIdx.x; i < rangeLen * 32; i += blockDim.x) {
        shist[i] = 0u;
    }
    __syncthreads();

    // Process input with grid-stride loops, each iteration handling itemsPerThread elements per thread.
    // Access pattern:
    //   For the k-th item per thread, we read index j = start + k * gstride.
    //   This maintains coalescing for the first k and spreads accesses across the dataset for ILP.
    for (unsigned int start = gtid; start < N; start += gstride * itemsPerThread) {
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            unsigned int j = start + static_cast<unsigned int>(k) * gstride;
            if (j >= N) break;

            // Cast to unsigned char to convert from potentially signed char
            // and obtain ordinal value in 0..255.
            unsigned int v = static_cast<unsigned char>(input[j]);

            // Compute local bin index; range check using unsigned comparison.
            int diff = static_cast<int>(v) - from;
            if (static_cast<unsigned int>(diff) <= static_cast<unsigned int>(to - from)) {
                // Each thread updates its lane-specific copy to avoid intra-warp conflicts.
                // Shared-memory atomics guard against inter-warp updates to the same copy.
                atomicAdd(&shist[diff * 32 + lane], 1u);
            }
        }
    }
    __syncthreads();

    // Reduce 32 copies per bin into a single value and atomically add to global histogram.
    for (int bin = threadIdx.x; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += shist[bin * 32 + c];
        }
        // Multiple blocks contribute to the same global bins concurrently; use atomicAdd.
        atomicAdd(&output[bin], sum);
    }
}

// Helper for integer ceiling division.
static inline unsigned int ceil_div_u32(unsigned int a, unsigned int b) {
    return (a + b - 1u) / b;
}

// Host function that launches the histogram kernel.
// Assumptions:
// - 'input' and 'histogram' point to device memory allocated with cudaMalloc.
// - 'inputSize' is the number of chars in 'input'.
// - The caller handles any synchronization needed after this function returns.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (!input || !histogram || inputSize == 0u || from > to) {
        return;
    }

    const int rangeLen = to - from + 1;

    // Zero the output histogram on device (asynchronously).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Choose launch configuration.
    // - 256 threads per block is a balanced choice for A100/H100 across many workloads.
    // - Grid size is chosen to cover the input and to provide sufficient blocks per SM
    //   for latency hiding on large inputs.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const int blockSize = 256;
    const unsigned int threadsNeeded = ceil_div_u32(inputSize, static_cast<unsigned int>(itemsPerThread));
    unsigned int gridFromWork = ceil_div_u32(threadsNeeded, static_cast<unsigned int>(blockSize));

    // Cap blocks to avoid excessive launch sizes while keeping the GPU busy.
    // Using up to 8 blocks per SM typically offers good occupancy without oversubscription.
    const unsigned int maxBlocks = static_cast<unsigned int>(prop.multiProcessorCount) * 8u;
    if (gridFromWork == 0u) gridFromWork = 1u;
    const unsigned int gridSize = (gridFromWork > maxBlocks) ? maxBlocks : gridFromWork;

    // Dynamic shared memory size for 32 copies of the histogram.
    const size_t shmBytes = static_cast<size_t>(rangeLen) * 32u * sizeof(unsigned int);

    // Launch kernel.
    histogram_range_kernel<<<gridSize, blockSize, shmBytes>>>(input, inputSize, from, to, histogram);
}