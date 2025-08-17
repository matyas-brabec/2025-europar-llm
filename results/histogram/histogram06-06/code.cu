#include <cuda_runtime.h>

// Tunable constant: how many input bytes each thread processes per loop iteration.
// 16 is a good default for A100/H100 to balance ILP and occupancy for large inputs.
static constexpr int itemsPerThread = 16;
static constexpr int WARP_SIZE = 32;

// CUDA kernel that computes a range-restricted histogram using shared-memory privatization.
// - 32 copies of the histogram are created in shared memory.
// - Each copy is mapped to a distinct shared-memory bank by using a stride of 32 elements.
// - Threads update the copy corresponding to their lane ID (0..31), ensuring that within a warp
//   all threads access distinct banks (no intra-warp bank conflicts).
// - After processing, the 32 copies are reduced and atomically accumulated to the global histogram.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int shmem[]; // size: 32 * stride (computed at launch)
    const int numBins = to - from + 1;

    // Pad stride to 32 so that address = lane + bin*stride maps to shared-memory bank "lane".
    const int stride = ((numBins + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int blockThreads = blockDim.x;
    const unsigned int globalThreadId = blockIdx.x * blockThreads + threadIdx.x;
    const unsigned int totalThreads = gridDim.x * blockThreads;

    // Zero initialize the shared histogram copies: 32 copies, each with 'stride' elements.
    for (int i = threadIdx.x; i < WARP_SIZE * stride; i += blockThreads) {
        shmem[i] = 0;
    }
    __syncthreads();

    // Process the input in chunks. Each thread handles itemsPerThread elements per iteration.
    // Grid-stride over chunks of size totalThreads * itemsPerThread to cover large inputs.
    for (unsigned int base = globalThreadId * itemsPerThread;
         base < inputSize;
         base += totalThreads * itemsPerThread)
    {
        // Unroll the per-thread work for ILP.
        #pragma unroll
        for (int j = 0; j < itemsPerThread; ++j) {
            unsigned int idx = base + static_cast<unsigned int>(j);
            if (idx < inputSize) {
                unsigned int v = static_cast<unsigned int>(input[idx]); // 0..255
                if (v >= static_cast<unsigned int>(from) && v <= static_cast<unsigned int>(to)) {
                    int bin = static_cast<int>(v) - from;          // 0..numBins-1
                    int pos = bin * stride + lane;                 // lane-specific copy
                    // Shared-memory atomic add is fast on modern GPUs.
                    atomicAdd(&shmem[pos], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce the 32 lane-private copies into a single histogram and update global memory.
    // The addressing pattern bin*stride + lane ensures that, at a given lane loop step,
    // all threads in a warp read from distinct banks (no bank conflicts).
    for (int bin = threadIdx.x; bin < numBins; bin += blockThreads) {
        unsigned int sum = 0;
        // Accumulate across the 32 lane copies.
        #pragma unroll
        for (int l = 0; l < WARP_SIZE; ++l) {
            sum += shmem[bin * stride + l];
        }
        // Accumulate per-block result into the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
}

// Host function that launches the histogram kernel.
// - input: device pointer to input chars (cudaMalloc'ed).
// - histogram: device pointer to output bins (cudaMalloc'ed).
// - inputSize: number of chars in input.
// - from, to: inclusive range [from, to] to compute (0 <= from < to <= 255).
// This function zeroes the output histogram, configures the launch, and invokes the kernel.
// No device synchronization is performed; the caller is responsible for synchronization.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (input == nullptr || histogram == nullptr || inputSize == 0 || from > to) {
        return;
    }

    const int numBins = to - from + 1;
    // Stride must match device computation to maintain bank mapping.
    const int stride = ((numBins + (WARP_SIZE - 1)) / WARP_SIZE) * WARP_SIZE;
    const size_t shmemBytes = static_cast<size_t>(WARP_SIZE) * static_cast<size_t>(stride) * sizeof(unsigned int);

    // Zero the output histogram (caller provides allocation; we initialize to 0 here).
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose a launch configuration suitable for A100/H100.
    // 512 threads per block is a good balance for occupancy and register usage.
    int blockSize = 512;

    // Determine grid size: cover the data and provide enough blocks for high occupancy.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Number of blocks needed to cover the input if each thread processes itemsPerThread items once.
    unsigned long long chunks = (static_cast<unsigned long long>(inputSize) + (blockSize * itemsPerThread - 1ULL)) /
                                static_cast<unsigned long long>(blockSize * itemsPerThread);
    if (chunks == 0ULL) chunks = 1ULL;

    // Clamp to a multiple of SM count to keep GPU busy; use 16x SMs for large inputs.
    int maxActiveBlocks = prop.multiProcessorCount * 16;
    int gridSize = static_cast<int>(chunks);
    if (gridSize > maxActiveBlocks) gridSize = maxActiveBlocks;
    if (gridSize < 1) gridSize = 1;

    // Launch the kernel with dynamic shared memory sized for 32 copies of the histogram.
    histogram_range_kernel<<<gridSize, blockSize, shmemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from,
        to
    );
}