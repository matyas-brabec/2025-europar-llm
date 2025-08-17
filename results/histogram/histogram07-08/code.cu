#include <cuda_runtime.h>
#include <stdint.h>

// itemsPerThread controls how many input chars each thread processes per iteration.
// For modern data-center GPUs (A100/H100) and large inputs, 8 provides a good balance
// between instruction-level parallelism and occupancy. Tune if needed.
constexpr int itemsPerThread = 8;

// Reasonable default block size for histogramming on modern GPUs.
constexpr int threadsPerBlock = 256;

/*
Kernel details:
- Computes histogram for characters in [from, to] inclusive.
- Uses shared-memory histogram privatization to reduce global-memory atomics.
- Avoids shared-memory bank conflicts by keeping 32 copies of the histogram per block,
  one for each lane (threadIdx.x % 32), and by laying out these copies with a stride of 32.
  Specifically, for histogram bin i and copy c (0 <= c < 32), the element is at shared
  memory offset i*32 + c. This ensures that threads within a warp increment different banks.

Phases:
1) Zero shared-memory histograms (histLen * 32 entries).
2) Each thread processes itemsPerThread input elements per iteration of a grid-stride loop.
   For each character v in the specified range:
     - Compute local bin = v - from.
     - Atomic add to shared mem at smem[bin*32 + lane].
   Shared-memory atomicAdd is fast on modern GPUs, and the bank-conflict-free layout reduces
   contention and bank conflicts.
3) Reduce the 32 per-lane histograms to a single histogram per block and accumulate to
   global memory using atomicAdd (one atomic per bin per block).

Assumptions:
- 0 <= from < to <= 255 (caller guarantees valid inputs).
- input points to device memory of size inputSize bytes.
- histogram points to device memory of size (to - from + 1) * sizeof(unsigned int).
*/
__global__ void histogramKernel(const char* __restrict__ input,
                                unsigned int* __restrict__ histogram,
                                unsigned int inputSize,
                                int from,
                                int to)
{
    extern __shared__ unsigned int smem[]; // size: (to - from + 1) * 32
    const unsigned int histLen = static_cast<unsigned int>(to - from + 1);

    // Zero shared-memory histogram copies.
    for (unsigned int i = threadIdx.x; i < histLen * 32u; i += blockDim.x) {
        smem[i] = 0;
    }
    __syncthreads();

    const unsigned int lane = static_cast<unsigned int>(threadIdx.x) & 31u;
    const unsigned int range = static_cast<unsigned int>(to - from);

    // Grid-stride loop where each thread processes itemsPerThread items per iteration.
    size_t threadBase = static_cast<size_t>(blockIdx.x) * blockDim.x * itemsPerThread + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x * itemsPerThread;

    while (threadBase < inputSize) {
        #pragma unroll
        for (int j = 0; j < itemsPerThread; ++j) {
            size_t idx = threadBase + static_cast<size_t>(j) * blockDim.x;
            if (idx < inputSize) {
                // Convert to unsigned to avoid sign-extension issues for chars with high bit set.
                unsigned int v = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
                // Branchless range check: valid if v - from <= range.
                unsigned int bin = v - static_cast<unsigned int>(from);
                if (bin <= range) {
                    // Increment per-lane copy to avoid bank conflicts; atomic in shared memory handles collisions across warps.
                    atomicAdd(&smem[bin * 32u + lane], 1u);
                }
            }
        }
        threadBase += stride;
    }

    __syncthreads();

    // Reduce 32 copies into global histogram: parallelize across bins.
    for (unsigned int bin = threadIdx.x; bin < histLen; bin += blockDim.x) {
        unsigned int sum = 0;
        unsigned int base = bin * 32u;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += smem[base + static_cast<unsigned int>(c)];
        }
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host function that prepares and launches the kernel.
// - input: device pointer to text buffer of size inputSize bytes.
// - histogram: device pointer to (to - from + 1) unsigned ints (will be zeroed here).
// - inputSize: number of bytes in input.
// - from, to: inclusive character range [from, to].
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const unsigned int histLen = static_cast<unsigned int>(to - from + 1);

    // Initialize output histogram to zero on device.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(histLen) * sizeof(unsigned int));

    // Kernel launch configuration.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const int smCount = prop.multiProcessorCount;
    const unsigned int elementsPerBlock = static_cast<unsigned int>(threadsPerBlock * itemsPerThread);
    unsigned int blocksForInput = elementsPerBlock ? ((inputSize + elementsPerBlock - 1) / elementsPerBlock) : 1u;

    // Use a modest oversubscription factor to saturate the GPU for large inputs.
    unsigned int maxBlocks = static_cast<unsigned int>(smCount) * 8u;
    if (blocksForInput == 0) blocksForInput = 1;
    unsigned int grid = (blocksForInput < maxBlocks) ? blocksForInput : maxBlocks;

    // Dynamic shared memory size: 32 copies of the histogram, each of length histLen.
    size_t smemSize = static_cast<size_t>(histLen) * 32u * sizeof(unsigned int);

    histogramKernel<<<grid, threadsPerBlock, smemSize>>>(input, histogram, inputSize, from, to);
}