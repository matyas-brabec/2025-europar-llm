#include <cuda_runtime.h>
#include <stdint.h>

// Tunable parameter: number of input chars processed per thread.
// On modern data-center GPUs (A100/H100) with large inputs, 16-32 is a good range.
// Using 16 balances memory pressure and atomic contention well.
static constexpr int itemsPerThread = 16;

// Warp size constant
static constexpr int WARP_SIZE = 32;

// CUDA kernel to compute histogram for a restricted character range [from, to] (inclusive).
// - input: device pointer to input char buffer (as bytes)
// - global_hist: device pointer to output histogram (length = numBins = to - from + 1)
// - inputSize: number of chars in input
// - from, to: inclusive character range (0 <= from <= to <= 255)
//
// Optimization details:
// - Uses shared memory privatization with 32 copies of the histogram per block, interleaved across banks.
// - Layout: s_hist[bin * 32 + lane], so that lane 'l' of a warp maps to bank 'l' for any bin.
// - Threads update s_hist via shared-memory atomics to ensure correctness across multiple warps.
// - After processing, threads cooperatively reduce the 32 copies per bin and atomically accumulate to global memory.
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ global_hist,
                                       unsigned int inputSize,
                                       int from, int to)
{
    // Compute number of bins
    const int numBins = to - from + 1;

    // Dynamic shared memory: 32 copies (one per lane) interleaved across banks for each bin.
    extern __shared__ unsigned int s_hist[];

    // Zero initialize the shared memory histogram (all copies)
    const int smemSize = numBins * WARP_SIZE; // number of counters in shared memory
    for (int i = threadIdx.x; i < smemSize; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int threads = gridDim.x * blockDim.x;
    const int lane = threadIdx.x & (WARP_SIZE - 1); // thread's lane within warp

    // Grid-stride loop over tiles of size itemsPerThread per thread
    // Each iteration processes itemsPerThread consecutive chars per thread
    for (unsigned int base = tid * itemsPerThread; base < inputSize; base += threads * itemsPerThread) {
        // Process up to itemsPerThread items, guarding the tail
        #pragma unroll
        for (int j = 0; j < itemsPerThread; ++j) {
            unsigned int idx = base + static_cast<unsigned int>(j);
            if (idx >= inputSize) break;

            unsigned char v = input[idx];
            // Check if within requested range [from, to]
            // Casting 'v' to int to avoid unsigned wrap behavior
            int iv = static_cast<int>(v);
            if (iv >= from && iv <= to) {
                int bin = iv - from; // 0 .. numBins-1
                // Update lane-specific copy to avoid intra-warp bank conflicts.
                // Use shared-memory atomicAdd to be correct across multiple warps in the block.
                atomicAdd(&s_hist[bin * WARP_SIZE + lane], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce 32 lane-copies per bin to a single value per bin, then accumulate to global histogram
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0u;
        #pragma unroll
        for (int l = 0; l < WARP_SIZE; ++l) {
            sum += s_hist[bin * WARP_SIZE + l];
        }
        if (sum) {
            atomicAdd(&global_hist[bin], sum);
        }
    }
    // No need for __syncthreads here since the kernel is ending
}

// Host function to run the histogram kernel.
// Parameters:
// - input: device pointer to input chars (cudaMalloc'd), size = inputSize
// - histogram: device pointer to output histogram (cudaMalloc'd), size = (to - from + 1) unsigned ints
// - inputSize: number of chars in input
// - from, to: inclusive character range [from, to], with 0 <= from <= to <= 255
//
// Notes:
// - This function zeroes the output histogram before launching the kernel.
// - No explicit synchronization is performed here; the caller is responsible for it if needed.
// - Launch configuration aims for good occupancy and work distribution on modern GPUs.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (input == nullptr || histogram == nullptr) return;
    if (inputSize == 0) {
        // Zero the histogram and return
        int numBins = (to - from + 1);
        if (numBins > 0) {
            cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));
        }
        return;
    }

    // Compute number of bins, validate range
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (to < from) return;
    const int numBins = to - from + 1;

    // Zero the output histogram
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose launch configuration
    // Block size: 256 threads balances occupancy and shared-memory atomics contention.
    const int blockSize = 256;

    // Grid size: enough blocks so that each thread processes ~itemsPerThread elements initially.
    // Ensures at least 1 block is launched.
    unsigned int blocks = (inputSize + (blockSize * itemsPerThread) - 1) / (blockSize * itemsPerThread);
    if (blocks == 0) blocks = 1;

    // Optionally cap the number of blocks to a reasonable multiple of SMs (e.g., 4x SMs) to avoid oversubscription.
    // Querying device properties to set a reasonable cap.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        unsigned int maxBlocks = static_cast<unsigned int>(prop.multiProcessorCount) * 8u; // 8x SMs is a good default
        if (blocks > maxBlocks) blocks = maxBlocks;
    }

    // Dynamic shared memory size: 32 copies per bin, each an unsigned int
    size_t sharedMemSize = static_cast<size_t>(numBins) * WARP_SIZE * sizeof(unsigned int);

    // Launch the kernel
    const unsigned char* d_input_u8 = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<blocks, blockSize, sharedMemSize>>>(d_input_u8, histogram, inputSize, from, to);

    // No synchronization here; caller is responsible for it.
}