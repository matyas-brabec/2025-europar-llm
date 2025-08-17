#include <cuda_runtime.h>
#include <stdint.h>

// Tunable compile-time constant: how many input chars each thread processes per grid-stride iteration.
// On recent NVIDIA data center GPUs (A100/H100), 16 provides good ILP and memory coalescing for large inputs.
static constexpr int itemsPerThread = 16;

/*
Kernel design notes (see inline comments below for details):
- Uses 32 privatized copies of the histogram in shared memory to avoid bank conflicts.
- Copy index is chosen as laneId = threadIdx.x % 32.
- Shared-memory histogram copies are stored with a stride of 32 so that for bin i and copy c, the location is i*32 + c.
- Updates to shared memory use atomicAdd to be correct across warps (warps share the same copy index).
- After processing input, each block reduces the 32 copies into one set of bin counts and atomically adds them to the global histogram.
*/
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int N,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int smem[]; // Size: (range * 32) elements
    const unsigned int range = static_cast<unsigned int>(to - from + 1);
    const unsigned int lane   = static_cast<unsigned int>(threadIdx.x) & 31u;

    // Zero shared-memory histogram copies
    for (unsigned int idx = threadIdx.x; idx < range * 32u; idx += blockDim.x) {
        smem[idx] = 0u;
    }
    __syncthreads();

    // Grid-stride loop where each thread processes itemsPerThread items per stride
    const unsigned int gtid   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gsize  = gridDim.x * blockDim.x;
    const unsigned int chunk  = itemsPerThread;
    const unsigned int stride = gsize * chunk;

    // Process input in contiguous per-thread chunks for coalesced reads
    for (unsigned int start = gtid * chunk; start < N; start += stride) {
        #pragma unroll
        for (int k = 0; k < itemsPerThread; ++k) {
            unsigned int idx = start + static_cast<unsigned int>(k);
            if (idx >= N) break;

            // Use unsigned char to avoid sign-extension issues with 'char'
            unsigned char ch = static_cast<unsigned char>(input[idx]);
            // Compare in int domain; to/from are ints. ch is promoted to int automatically.
            if (ch >= from && ch <= to) {
                unsigned int bin = static_cast<unsigned int>(ch - from); // 0..range-1
                // Location for (bin, lane) using stride-32 layout: i*32 + c
                unsigned int off = bin * 32u + lane;
                // Atomic in shared memory ensures correctness across warps sharing the same 'lane' copy
                atomicAdd(&smem[off], 1u);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 copies for each bin and update the global histogram
    for (unsigned int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        unsigned int base = bin * 32u;
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            sum += smem[base + static_cast<unsigned int>(c)];
        }
        // Accumulate per-block results to global histogram
        atomicAdd(&histogram[bin], sum);
    }
}

/*
Host-side runner.
- Assumes 'input' and 'histogram' are device pointers allocated via cudaMalloc.
- 'inputSize' is the number of chars in 'input'.
- The histogram covers the inclusive range [from, to] with output size (to - from + 1).
- The output 'histogram' buffer will be zeroed here before kernel launch.
- Caller is responsible for synchronization if needed (e.g., cudaDeviceSynchronize).
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (from > to) return;

    const unsigned int range = static_cast<unsigned int>(to - from + 1);

    // Zero the output histogram (required since we use atomicAdd to accumulate)
    cudaMemset(histogram, 0, range * sizeof(unsigned int));

    // Choose CUDA launch parameters
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device);

    // Reasonable defaults for modern GPUs
    const int blockSize = 256;
    // Primary block count choice based on data size and per-block work
    const unsigned long long workPerBlock = static_cast<unsigned long long>(blockSize) * static_cast<unsigned long long>(itemsPerThread);
    unsigned long long neededBlocks = (inputSize + workPerBlock - 1ull) / workPerBlock;
    if (neededBlocks == 0ull) neededBlocks = 1ull;

    // Oversubscribe SMs to hide latency but avoid launching an excessive number of blocks
    const unsigned long long maxActiveBlocks = static_cast<unsigned long long>(props.multiProcessorCount) * 32ull;
    unsigned int gridSize = static_cast<unsigned int>(neededBlocks > maxActiveBlocks ? maxActiveBlocks : neededBlocks);

    // Shared memory size: 32 copies of the histogram, each of size 'range' bins
    const size_t sharedMemBytes = static_cast<size_t>(range) * 32u * sizeof(unsigned int);

    // Launch kernel
    histogram_range_kernel<<<gridSize, blockSize, sharedMemBytes>>>(input, histogram, inputSize, from, to);
}