#include <cuda_runtime.h>

// --------------------------------------------------------------------------------------
// CUDA histogram implementation optimized for modern NVIDIA GPUs (A100/H100).
//
//  - Input:  device pointer to chars (`input`), size `inputSize`.
//  - Output: device pointer to unsigned ints (`histogram`), covering bins [from, to].
//  - Only characters with ordinal value in [from, to] are counted; others are ignored.
//  - The output histogram has (to - from + 1) bins, where bin i corresponds to
//    character value (from + i).
//
// Optimizations:
//  - ITEMS_PER_THREAD controls how many characters each thread processes. This allows
//    fewer threads to cover large inputs while maintaining high memory throughput.
//  - Shared memory privatization: each thread block maintains multiple private copies
//    of the histogram in shared memory (one per warp). This reduces contention and
//    shared-memory bank conflicts compared to a single shared histogram per block.
//  - Final histogram is obtained by reducing all per-warp shared histograms into
//    the global histogram via atomicAdd.
// --------------------------------------------------------------------------------------

// Number of threads per block (chosen for good occupancy and latency hiding).
static constexpr int BLOCK_SIZE        = 256;

// Number of input characters processed by each thread.
// A value of 16 is a good default for large inputs on modern GPUs.
static constexpr int ITEMS_PER_THREAD  = 16;

// One private histogram per warp to reduce shared memory contention.
static constexpr int WARP_SIZE         = 32;
static constexpr int WARPS_PER_BLOCK   = BLOCK_SIZE / WARP_SIZE;
static constexpr int NUM_SUBHISTS      = WARPS_PER_BLOCK;

// --------------------------------------------------------------------------------------
// CUDA kernel: compute histogram over a restricted character range [from, from+range-1].
//
// Parameters:
//   input      - device pointer to input characters
//   histogram  - device pointer to global histogram (range bins, zeroed before call)
//   inputSize  - number of characters in input
//   from       - starting character (0..255)
//   range      - number of bins (1..256), i.e. to-from+1
//
// Shared memory layout:
//   extern __shared__ unsigned int s_hist[];
//   Size = range * NUM_SUBHISTS elements.
//
//   Sub-histogram k (for warp k) uses the slice:
//     s_hist[k * range + 0 .. k * range + (range - 1)]
// --------------------------------------------------------------------------------------
__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int range)
{
    extern __shared__ unsigned int s_hist[];

    // Initialize all shared-memory histogram bins to 0.
    // All threads collaborate on initialization.
    const int sharedSize = range * NUM_SUBHISTS;
    for (int idx = threadIdx.x; idx < sharedSize; idx += blockDim.x) {
        s_hist[idx] = 0;
    }
    __syncthreads();

    // Each block processes a contiguous chunk of the input.
    // Each thread processes ITEMS_PER_THREAD characters, in a strided fashion
    // that ensures coalesced global memory loads.
    const unsigned int blockChunkSize = blockDim.x * ITEMS_PER_THREAD;
    const unsigned int firstIndex =
        blockIdx.x * blockChunkSize + threadIdx.x;

    // Warp-based privatization: each warp writes into its own sub-histogram.
    const int warpId        = threadIdx.x / WARP_SIZE;  // 0 .. WARPS_PER_BLOCK-1
    const int subHistOffset = warpId * range;

    // Process ITEMS_PER_THREAD input characters per thread.
    // For each character, if it lies in [from, from+range-1], increment the
    // appropriate bin in the warp's private shared-memory histogram.
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        unsigned int idx = firstIndex + i * blockDim.x;
        if (idx >= inputSize) {
            // All subsequent indices for this thread will be out of range too.
            break;
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);
        int bin = static_cast<int>(c) - from;

        if (bin >= 0 && bin < range) {
            // Shared-memory atomicAdd is fast on modern GPUs.
            atomicAdd(&s_hist[subHistOffset + bin], 1u);
        }
    }

    __syncthreads();

    // Reduce all per-warp sub-histograms into the global histogram.
    // Each thread accumulates several bins in a strided fashion.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate bin 'bin' from each warp's sub-histogram.
        for (int sub = 0; sub < NUM_SUBHISTS; ++sub) {
            sum += s_hist[sub * range + bin];
        }
        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// --------------------------------------------------------------------------------------
// Host function: run_histogram
//
// Launches the CUDA kernel to compute the histogram of the device-resident input.
//
// Parameters:
//   input      - device pointer to input buffer (chars)
//   histogram  - device pointer to histogram buffer (unsigned int), must have
//                at least (to - from + 1) elements
//   inputSize  - number of chars in input buffer
//   from       - starting character code (0 <= from <= 255)
//   to         - ending character code   (from <= to <= 255)
//
// Notes:
//   - The function assumes `input` and `histogram` are already allocated with
//     cudaMalloc on the device.
//   - This function zeroes the histogram buffer before launching the kernel.
//   - Host-device synchronization (e.g. cudaDeviceSynchronize) is left to the caller.
// --------------------------------------------------------------------------------------
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    if (inputSize == 0 || from > to) {
        // Nothing to do.
        return;
    }

    const int range = to - from + 1;  // Number of bins to compute.

    // Zero the output histogram on the device.
    cudaMemset(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int));

    // Compute grid configuration.
    const unsigned int blockSize       = BLOCK_SIZE;
    const unsigned int blockChunkSize  = blockSize * ITEMS_PER_THREAD;
    const unsigned int gridSize        =
        (inputSize + blockChunkSize - 1) / blockChunkSize;

    // Shared memory size: one sub-histogram per warp.
    const size_t sharedMemSize =
        static_cast<size_t>(range) * NUM_SUBHISTS * sizeof(unsigned int);

    // Launch the kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        range
    );
}