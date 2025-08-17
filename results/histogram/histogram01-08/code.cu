#include <cuda_runtime.h>
#include <stdint.h>

// CUDA kernel to compute a histogram restricted to a character range [from, to].
// - input: device pointer to input text buffer (array of chars)
// - histogram: device pointer to output histogram (size = to - from + 1)
// - n: number of chars in input
// - from, to: inclusive range of characters to count (0 <= from < to <= 255)
//
// Implementation details:
// - Each block uses warp-private histograms in shared memory to minimize contention.
// - Threads process the input via a grid-stride loop.
// - Each thread increments its warp's private histogram (in shared memory) using fast atomics.
// - At the end, warp histograms are reduced across the block and accumulated into the global histogram.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int n,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int s_mem[]; // Layout: [warpsPerBlock][range]
    const int tid = threadIdx.x;
    const int warpSize_ = 32;
    const int warpId = tid / warpSize_;
    const int warpsPerBlock = blockDim.x / warpSize_;
    const int range = to - from + 1;

    // Guard against degenerate configurations
    if (range <= 0 || warpsPerBlock <= 0) return;

    unsigned int* warpHist = s_mem + warpId * range;

    // Zero all shared histograms (all warps) cooperatively
    for (int i = tid; i < warpsPerBlock * range; i += blockDim.x) {
        s_mem[i] = 0;
    }
    __syncthreads();

    // Process input using a grid-stride loop
    for (unsigned int idx = blockIdx.x * blockDim.x + tid; idx < n; idx += gridDim.x * blockDim.x) {
        unsigned char c = input[idx]; // avoid sign issues
        if (c >= (unsigned char)from && c <= (unsigned char)to) {
            // Atomic add to this warp's private histogram bin
            atomicAdd(&warpHist[(int)c - from], 1u);
        }
    }
    __syncthreads();

    // Reduce warp histograms to a block histogram and accumulate to global
    for (int bin = tid; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum this bin across all warps
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_mem[w * range + bin];
        }
        // Accumulate to global histogram (atomic to handle multi-block updates)
        atomicAdd(&histogram[bin], sum);
    }
    // No need for further synchronization
}

// Host-side launcher. The input and histogram pointers are device pointers allocated with cudaMalloc.
// The histogram array must be sized to (to - from + 1) elements.
// This function zeros the output histogram, configures a performant launch, and dispatches the kernel.
// Any synchronization (e.g., cudaDeviceSynchronize or stream sync) is expected to be handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Basic parameter handling
    if (inputSize == 0) {
        // Still zero the histogram for correctness
        int range = (to - from + 1);
        if (range > 0) {
            // Zero asynchronously on the default stream to keep this function non-blocking
            cudaMemsetAsync(histogram, 0, sizeof(unsigned int) * static_cast<size_t>(range), 0);
        }
        return;
    }

    const int range = (to - from + 1);
    if (range <= 0) {
        // Nothing to do
        return;
    }

    // Zero the output histogram (required, as the kernel accumulates)
    cudaMemsetAsync(histogram, 0, sizeof(unsigned int) * static_cast<size_t>(range), 0);

    // Kernel launch configuration
    // - Use a block size that is a multiple of 32 (warp size) for warp-private histograms.
    // - Compute dynamic shared memory size as (warpsPerBlock * range * sizeof(unsigned int)).
    // - Choose grid size based on occupancy to reasonably saturate the device.

    const int BLOCK_SIZE = 256; // 8 warps per block; good balance for A100/H100
    const int warpsPerBlock = BLOCK_SIZE / 32;
    const size_t sharedMemBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(range) * sizeof(unsigned int);

    // Determine a good grid size via occupancy
    int device = 0;
    cudaGetDevice(&device);
    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int maxActiveBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM,
                                                  histogram_range_kernel,
                                                  BLOCK_SIZE,
                                                  sharedMemBytes);
    if (maxActiveBlocksPerSM <= 0) {
        // Fallback in unlikely case occupancy query fails
        maxActiveBlocksPerSM = 16;
    }

    // Use a multiplier to oversubscribe a bit for better latency hiding
    const int oversub = 4;
    int grid = smCount * maxActiveBlocksPerSM * oversub;
    if (grid <= 0) grid = 1;

    // Alternatively cap grid to avoid excessively tiny workloads per block
    // but ensure we still cover a reasonable number of blocks for large inputs.
    // This cap isn't strictly necessary; scheduler handles large grids well.
    // Here we ensure at least enough threads to touch each input element once.
    long long minGridBySize = (static_cast<long long>(inputSize) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (minGridBySize > grid) grid = static_cast<int>(minGridBySize);

    // Launch kernel
    const unsigned char* d_input_uc = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<grid, BLOCK_SIZE, sharedMemBytes>>>(d_input_uc, histogram, inputSize, from, to);
}