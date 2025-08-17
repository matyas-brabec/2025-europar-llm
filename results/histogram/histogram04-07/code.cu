#include <cuda_runtime.h>
#include <stdint.h>

/*
  CUDA histogram for a restricted character range with shared-memory privatization.

  Key design points:
  - The global histogram covers only characters in [from, to] inclusive.
  - Each block privatizes the histogram in shared memory, and each warp gets its own
    copy to reduce contention on shared-memory atomics.
  - After processing, per-warp histograms are reduced within the block and committed
    to the global histogram using atomics (one atomic add per bin per block).
  - A grid-stride loop with ILP (itemsPerThread) is used to efficiently walk large inputs.
  - The default itemsPerThread and block size are tuned for modern NVIDIA data center GPUs
    (A100/H100). Adjust if needed for other devices or workloads.

  Notes:
  - The input buffer contains raw bytes; 'char' may be signed on some platforms, so we cast
    to unsigned char before comparisons.
  - The output histogram buffer must have length (to - from + 1) and is zeroed before kernel launch.
  - No host-device synchronization is performed here; caller is responsible for that if needed.
*/

// Controls how many input chars each thread processes per grid-stride iteration.
// On A100/H100, 16 provides good ILP and memory-latency hiding for large inputs.
static constexpr int itemsPerThread = 16;

// Threads per block. Must be a multiple of warpSize (32).
// 512 threads per block (16 warps) is a good balance for modern GPUs.
static constexpr int threadsPerBlock = 512;
static_assert(threadsPerBlock % 32 == 0, "threadsPerBlock must be a multiple of 32");

__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    // Number of bins for the requested character range [from, to].
    const int bins = to - from + 1;

    // Shared memory layout: per-warp privatized histograms.
    // Total shared mem size (in 32-bit counters) = warpsPerBlock * bins.
    extern __shared__ unsigned int shmem[];
    const int warpsPerBlock = blockDim.x / warpSize;
    const int warpId        = threadIdx.x / warpSize;

    // Zero initialize the shared histograms.
    for (int i = threadIdx.x; i < warpsPerBlock * bins; i += blockDim.x) {
        shmem[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop with ILP: each thread processes 'itemsPerThread' elements
    // per outer loop iteration in a coalesced manner.
    const unsigned int gtid         = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int totalThreads = blockDim.x * gridDim.x;
    unsigned int* __restrict__ warpHist = shmem + warpId * bins;

    for (unsigned int base = gtid; base < inputSize; base += totalThreads * itemsPerThread) {
        #pragma unroll
        for (int i = 0; i < itemsPerThread; ++i) {
            unsigned int idx = base + i * totalThreads;
            if (idx < inputSize) {
                // Byte value as unsigned to avoid sign-extension issues with 'char'.
                unsigned char c = static_cast<unsigned char>(input[idx]);

                // Restrict to range [from, to].
                if (c >= static_cast<unsigned char>(from) && c <= static_cast<unsigned char>(to)) {
                    const int bin = static_cast<int>(c) - from;
                    // Shared-memory atomic add: fast on modern architectures.
                    atomicAdd(&warpHist[bin], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce per-warp histograms into a single per-block histogram and commit to global.
    for (int b = threadIdx.x; b < bins; b += blockDim.x) {
        unsigned int sum = 0;
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += shmem[w * bins + b];
        }
        // One atomic add per bin per block to the global histogram.
        atomicAdd(&histogram[b], sum);
    }
}

/*
  Host entry point.

  Parameters:
  - input:      device pointer to the input text (cudaMalloc-ed), length inputSize bytes.
  - histogram:  device pointer to output histogram (cudaMalloc-ed), length (to - from + 1) counters.
  - inputSize:  number of bytes in input.
  - from, to:   inclusive character range [from, to], 0 <= from < to <= 255.

  Behavior:
  - Zeroes the output histogram on device.
  - Launches the histogram kernel with a grid size chosen to balance throughput and minimize
    global atomic update overhead by using grid-stride loops.
  - No device synchronize is performed here; caller may synchronize if required.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int bins = to - from + 1;

    // Zero the output histogram (as required for correct counts on each call).
    // cudaMemset is asynchronous with respect to the host; ordering w.r.t. the default stream is preserved.
    cudaMemset(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    // Early exit if no input to process or empty range; memset above already handled the output.
    if (inputSize == 0 || bins <= 0) {
        return;
    }

    // Choose grid size.
    // Limit the number of blocks to a multiple of SM count to reduce the number of global atomic updates.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Blocks needed to cover the input if each thread processes itemsPerThread items.
    const unsigned long long denom = static_cast<unsigned long long>(threadsPerBlock) * itemsPerThread;
    int blocksNeeded = static_cast<int>((static_cast<unsigned long long>(inputSize) + denom - 1ull) / denom);

    // Aim for a modest oversubscription (e.g., 8 blocks per SM) to keep the GPU busy while
    // minimizing the number of global atomic updates during the final accumulation.
    const int blocksPerSM = 8;
    const int maxBlocks   = prop.multiProcessorCount * blocksPerSM;

    int grid = blocksNeeded > 0 ? blocksNeeded : 1;
    if (grid > maxBlocks) grid = maxBlocks;

    // Dynamic shared memory size: one privatized histogram per warp.
    const size_t shmemSize = static_cast<size_t>(bins) * (threadsPerBlock / 32) * sizeof(unsigned int);

    // Launch the kernel. No explicit synchronization here; caller may synchronize if desired.
    histogram_range_kernel<<<grid, threadsPerBlock, shmemSize>>>(input, histogram, inputSize, from, to);
}