#include <cuda_runtime.h>
#include <stdint.h>

// Optimized per-block shared-memory privatized histogram kernel.
// - Computes histogram counts for a contiguous range of byte values [from, to] (inclusive).
// - Uses shared memory to accumulate per-block counts, minimizing global atomics.
// - Vectorized input loading (uchar4) improves memory throughput.
// - Counts from all blocks are combined into the global histogram via a single atomicAdd per bin per block.
__global__ void histogramRangeKernel(const char* __restrict__ input,
                                     unsigned int inputSize,
                                     int from,
                                     int to,
                                     unsigned int* __restrict__ globalHist)
{
    extern __shared__ unsigned int shist[]; // Dynamic shared memory: bins = (to - from + 1)
    const unsigned int bins = static_cast<unsigned int>(to - from + 1);
    const unsigned int ufrom = static_cast<unsigned int>(from);

    // Zero out shared-memory histogram. Each thread clears a strided subset of bins.
    for (unsigned int i = threadIdx.x; i < bins; i += blockDim.x) {
        shist[i] = 0;
    }
    __syncthreads();

    // Grid-stride traversal over input (vectorized as uchar4 = 4 bytes per load).
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    // Process main body in 4-byte chunks
    const unsigned int n4 = inputSize / 4;
    const uchar4* __restrict__ data4 = reinterpret_cast<const uchar4*>(input);

    for (unsigned int idx4 = tid; idx4 < n4; idx4 += stride) {
        uchar4 v = data4[idx4];

        // For each byte in the vector, compute relative bin index and update shared histogram.
        // Using unsigned arithmetic avoids branches on negative differences.
        unsigned int r0 = static_cast<unsigned int>(v.x) - ufrom;
        if (r0 < bins) atomicAdd(&shist[r0], 1u);

        unsigned int r1 = static_cast<unsigned int>(v.y) - ufrom;
        if (r1 < bins) atomicAdd(&shist[r1], 1u);

        unsigned int r2 = static_cast<unsigned int>(v.z) - ufrom;
        if (r2 < bins) atomicAdd(&shist[r2], 1u);

        unsigned int r3 = static_cast<unsigned int>(v.w) - ufrom;
        if (r3 < bins) atomicAdd(&shist[r3], 1u);
    }

    // Handle remaining tail bytes (if inputSize % 4 != 0)
    const unsigned int tailStart = n4 * 4;
    for (unsigned int i = tailStart + tid; i < inputSize; i += stride) {
        unsigned int r = static_cast<unsigned int>(static_cast<unsigned char>(input[i])) - ufrom;
        if (r < bins) atomicAdd(&shist[r], 1u);
    }

    __syncthreads();

    // Merge per-block shared histograms into global histogram.
    // Each thread writes a strided subset of bins; skip zero counts to reduce global atomics.
    for (unsigned int i = threadIdx.x; i < bins; i += blockDim.x) {
        unsigned int c = shist[i];
        if (c) atomicAdd(&globalHist[i], c);
    }
}

// Host-side launcher.
// - input: device pointer to input chars (cudaMalloc'ed), with inputSize bytes.
// - histogram: device pointer to output histogram array of size (to - from + 1) (cudaMalloc'ed).
// - from/to: inclusive character range [from, to], with 0 <= from <= to <= 255.
// Notes:
// - The function initializes the output histogram to zeros.
// - It launches the kernel with a configuration tuned for modern NVIDIA data center GPUs.
// - No device-wide synchronization is performed here; the caller handles synchronization.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate range and handle trivial cases.
    if (input == nullptr || histogram == nullptr) return;
    if (inputSize == 0) {
        if (from <= to) {
            const size_t bins = static_cast<size_t>(to - from + 1);
            // Async memset on the default stream; caller is responsible for synchronization.
            cudaMemset(histogram, 0, bins * sizeof(unsigned int));
        }
        return;
    }
    if (from > to) return;

    const int bins = to - from + 1;

    // Initialize output histogram to zero before accumulation.
    cudaMemset(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    // Choose a launch configuration suitable for large memory-bound workloads.
    // Use up to 8 blocks per SM to provide enough parallelism without oversubscription.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const int threadsPerBlock = 256; // Good balance for shared memory usage and occupancy
    int maxBlocks = prop.multiProcessorCount * 8;

    // Heuristic: cap blocks to the amount of work to avoid excessive tiny-block launches on small inputs.
    // Each thread processes multiple elements via grid-stride and vectorization, so this is conservative.
    int blocksByWork = static_cast<int>((static_cast<unsigned long long>(inputSize) + threadsPerBlock - 1ULL) / threadsPerBlock);
    if (blocksByWork < 1) blocksByWork = 1;

    int blocks = maxBlocks < blocksByWork ? maxBlocks : blocksByWork;
    if (blocks < 1) blocks = 1;

    // Dynamic shared memory size equals number of bins.
    const size_t shmemSize = static_cast<size_t>(bins) * sizeof(unsigned int);

    histogramRangeKernel<<<blocks, threadsPerBlock, shmemSize>>>(input, inputSize, from, to, histogram);
}