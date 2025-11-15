#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// CUDA histogram kernel optimized for modern NVIDIA data-center GPUs (A100/H100)
//
// - Computes a histogram over a specified ASCII/byte range [from, to].
// - Uses shared memory histogram privatization with 32 copies per block
//   to minimize shared-memory bank conflicts.
// - Each warp lane owns one copy of the histogram; bins are laid out so that
//   accesses from all 32 lanes to the same logical bin hit different banks.
// - Uses shared-memory atomics for intra-block accumulation and global-memory
//   atomics for inter-block accumulation.
// - itemsPerThread controls how many input characters each thread processes
//   (contiguously in memory, for coalesced loads).
// ----------------------------------------------------------------------------

static constexpr int WARP_SIZE          = 32;
static constexpr int THREADS_PER_BLOCK  = 256;  // Must be a multiple of WARP_SIZE
static constexpr int ITEMS_PER_THREAD   = 8;    // Good default for A100/H100-class GPUs

static_assert(THREADS_PER_BLOCK % WARP_SIZE == 0, "THREADS_PER_BLOCK must be a multiple of WARP_SIZE");
static_assert(ITEMS_PER_THREAD > 0, "ITEMS_PER_THREAD must be positive");

// Kernel computing a histogram over the range [from, to] (inclusive).
// - input: pointer to device memory holding inputSize bytes.
// - histogram: pointer to device memory with (to - from + 1) unsigned int bins.
//              The kernel atomically adds its per-block histograms into this array.
// - from, to: inclusive range of byte values (0..255) to histogram.
__global__
void histogram_range_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            int from,
                            int to)
{
    // Number of bins we are responsible for.
    const int numBins = to - from + 1;

    // Shared memory layout:
    // We allocate numBins * 32 counters.
    // Conceptually: sharedHist[bin][lane], laid out as:
    //   sharedHist[bin * WARP_SIZE + lane]
    //
    // For any bin, the 32 copies (one per warp lane) are stored in different
    // shared-memory banks, because consecutive 32-bit words map to consecutive
    // banks. Thus, when all 32 threads in a warp update the same logical bin,
    // they access different banks and avoid bank conflicts.
    extern __shared__ unsigned int sharedHist[];

    // Initialize shared histogram copies to zero.
    // Each thread zeros multiple elements in a strided fashion.
    for (int i = threadIdx.x; i < numBins * WARP_SIZE; i += blockDim.x) {
        sharedHist[i] = 0;
    }

    __syncthreads();

    const int laneId = threadIdx.x & (WARP_SIZE - 1);

    // Each block processes a contiguous chunk of input of size:
    //   blockRange = blockDim.x * ITEMS_PER_THREAD
    //
    // Within that chunk, each thread processes ITEMS_PER_THREAD characters,
    // with indices:
    //   idx = blockBase + threadIdx.x + i * blockDim.x,  i = 0..ITEMS_PER_THREAD-1
    //
    // For each i, threads in the block access a contiguous, coalesced region
    // of input memory.
    const unsigned int blockRange = blockDim.x * ITEMS_PER_THREAD;
    const unsigned int blockBase  = blockIdx.x * blockRange;

    // Process up to ITEMS_PER_THREAD characters per thread.
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const unsigned int idx = blockBase + threadIdx.x + i * blockDim.x;
        if (idx >= inputSize) {
            break;  // Out of bounds; no more work for this thread.
        }

        const unsigned char value = input[idx];
        const int bin = static_cast<int>(value) - from;

        // Only count characters within [from, to].
        if (bin >= 0 && bin < numBins) {
            // Each warp lane updates its own copy of each bin.
            // Offset is bin * WARP_SIZE + laneId.
            const unsigned int offset = static_cast<unsigned int>(bin) * WARP_SIZE + static_cast<unsigned int>(laneId);
            atomicAdd(&sharedHist[offset], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 per-lane copies of each bin into the global histogram.
    //
    // For each bin:
    //   sum = sum over lane = 0..31 of sharedHist[bin * WARP_SIZE + lane]
    // Then atomically add 'sum' to the corresponding global bin.
    //
    // Threads cooperate to reduce bins in a strided fashion.
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        const int base = bin * WARP_SIZE;

        #pragma unroll
        for (int lane = 0; lane < WARP_SIZE; ++lane) {
            sum += sharedHist[base + lane];
        }

        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
//
// void run_histogram(const char *input, unsigned int *histogram,
//                    unsigned int inputSize, int from, int to)
//
// - input: device pointer obtained with cudaMalloc (array of chars).
// - histogram: device pointer obtained with cudaMalloc; must hold
//              (to - from + 1) unsigned ints.
// - inputSize: number of characters in 'input'.
// - from, to: inclusive range [from, to], 0 <= from <= to <= 255.
//
// This function:
//   1. Zeros the device histogram array.
//   2. Launches the CUDA kernel with a configuration chosen for A100/H100.
//   3. Does not synchronize; the caller is responsible for synchronization.
// ---------------------------------------------------------------------------
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Assume caller respects 0 <= from <= to <= 255.
    const int numBins = to - from + 1;

    // Zero the output histogram on device (asynchronously, default stream).
    const size_t histBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);
    cudaMemset(histogram, 0, histBytes);

    // If there is no input data, we're done after zeroing the histogram.
    if (inputSize == 0) {
        return;
    }

    // Launch configuration:
    // - THREADS_PER_BLOCK = 256 threads, i.e., 8 warps per block.
    // - Each thread processes ITEMS_PER_THREAD items.
    //
    // Number of blocks chosen so that total coverage is at least inputSize.
    const unsigned int blockRange = THREADS_PER_BLOCK * ITEMS_PER_THREAD;
    unsigned int numBlocks = (inputSize + blockRange - 1u) / blockRange;
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    // Shared memory size: numBins * 32 copies * sizeof(unsigned int).
    const size_t sharedMemBytes = static_cast<size_t>(numBins) * WARP_SIZE * sizeof(unsigned int);

    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);

    // Launch kernel (asynchronous; caller handles synchronization and error checking if desired).
    histogram_range_kernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemBytes>>>(
        d_input,
        histogram,
        inputSize,
        from,
        to
    );
}