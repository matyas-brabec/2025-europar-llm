#include <cuda_runtime.h>

// itemsPerThread is the requested tuning knob. For large streaming inputs on modern
// data-center GPUs (A100/H100 class), 16 is a strong default: it gives enough ILP to
// amortize loop overhead while keeping register pressure modest for this atomic-heavy kernel.
constexpr int itemsPerThread = 16;

// Exactly 32 privatized copies are required so that copy c maps to shared-memory bank c.
constexpr int histogramCopies = 32;

// We choose the smallest block size that still leaves at least this many resident warps
// per SM. Because the histogram has only 32 shared copies, fewer warps/block directly
// means less intra-block contention on hot bins.
constexpr int minUsefulResidentWarps = 32;

static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

template <int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ void histogram_range_kernel(const unsigned char *__restrict__ input,
                                       unsigned int *__restrict__ histogram,
                                       unsigned int inputSize,
                                       unsigned int rangeStart,
                                       unsigned int numBins) {
    static_assert(BLOCK_SIZE % histogramCopies == 0,
                  "BLOCK_SIZE must be a multiple of 32.");

    // Shared-memory layout:
    //   sHistogram[bin * 32 + copy]
    // where copy = threadIdx.x % 32.
    //
    // Since shared memory has 32 banks and each element is 4 bytes, address
    // (bin * 32 + copy) always maps to bank 'copy'. Therefore, for the histogram update
    // path, every lane in a warp hits a distinct bank regardless of the bin value.
    extern __shared__ unsigned int sHistogram[];

    const unsigned int lane = threadIdx.x & 31u;  // == threadIdx.x % 32
    const unsigned int totalPrivatizedBins = numBins * histogramCopies;

    // Zero the block-private histogram.
    for (unsigned int i = threadIdx.x; i < totalPrivatizedBins; i += BLOCK_SIZE) {
        sHistogram[i] = 0u;
    }
    __syncthreads();

    // Process a block-wide tile in a coalesced pattern:
    //   idx = tileBase + threadIdx.x + item * blockDim.x
    // Each inner-loop iteration therefore reads a contiguous byte span across the block.
    const size_t n = static_cast<size_t>(inputSize);
    const size_t tileSize = static_cast<size_t>(BLOCK_SIZE) * itemsPerThread;
    const size_t gridStride = static_cast<size_t>(gridDim.x) * tileSize;

    for (size_t tileBase = static_cast<size_t>(blockIdx.x) * tileSize;
         tileBase < n;
         tileBase += gridStride) {
        const size_t threadBase = tileBase + threadIdx.x;

#pragma unroll
        for (int item = 0; item < itemsPerThread; ++item) {
            const size_t idx = threadBase + static_cast<size_t>(item) * BLOCK_SIZE;
            if (idx < n) {
                // Interpret char as an unsigned byte in [0, 255], independent of host-side
                // signedness of 'char'. The range test uses unsigned arithmetic so it becomes
                // one compare after subtraction.
                const unsigned int byteVal = static_cast<unsigned int>(input[idx]);
                const unsigned int bin = byteVal - rangeStart;
                if (bin < numBins) {
                    atomicAdd(&sHistogram[bin * histogramCopies + lane], 1u);
                }
            }
        }
    }
    __syncthreads();

    // Merge the 32 bank-striped copies into the final global histogram.
    // This reduces global atomic traffic from "one atomic per input byte" to
    // "one atomic per nonzero bin per block".
    for (unsigned int bin = threadIdx.x; bin < numBins; bin += BLOCK_SIZE) {
        unsigned int sum = 0u;
        const unsigned int base = bin * histogramCopies;

#pragma unroll
        for (int copy = 0; copy < histogramCopies; ++copy) {
            sum += sHistogram[base + copy];
        }

        if (sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

template <int BLOCK_SIZE>
static inline int get_active_blocks_per_sm(size_t sharedMemBytes) {
    static_assert(BLOCK_SIZE % histogramCopies == 0,
                  "BLOCK_SIZE must be a multiple of 32.");

    // This kernel is dominated by shared-memory atomics and performs streaming input reads,
    // so preferring shared memory over L1 is usually beneficial on Ampere/Hopper.
    cudaFuncSetAttribute(histogram_range_kernel<BLOCK_SIZE>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);

    int activeBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSm,
                                                  histogram_range_kernel<BLOCK_SIZE>,
                                                  BLOCK_SIZE,
                                                  sharedMemBytes);

    return (activeBlocksPerSm > 0) ? activeBlocksPerSm : 1;
}

template <int BLOCK_SIZE>
static inline void launch_histogram_kernel(const unsigned char *input,
                                           unsigned int *histogram,
                                           unsigned int inputSize,
                                           unsigned int rangeStart,
                                           unsigned int numBins,
                                           size_t sharedMemBytes,
                                           int smCount,
                                           int activeBlocksPerSm,
                                           cudaStream_t stream) {
    const size_t tileSize = static_cast<size_t>(BLOCK_SIZE) * itemsPerThread;
    const size_t blocksForInput64 =
        (static_cast<size_t>(inputSize) + tileSize - 1) / tileSize;

    // Launch one resident wave of blocks and let the kernel grid-stride over the full input.
    // That keeps the GPU busy while minimizing block finalization and global merge atomics.
    unsigned int launchBlocks =
        static_cast<unsigned int>(smCount) * static_cast<unsigned int>(activeBlocksPerSm);
    if (launchBlocks == 0u) {
        launchBlocks = 1u;
    }

    const unsigned int blocksForInput = static_cast<unsigned int>(blocksForInput64);
    if (launchBlocks > blocksForInput) {
        launchBlocks = blocksForInput;
    }

    histogram_range_kernel<BLOCK_SIZE>
        <<<static_cast<int>(launchBlocks), BLOCK_SIZE, sharedMemBytes, stream>>>(
            input, histogram, inputSize, rangeStart, numBins);
}

// Host launcher. Per the problem statement, 'input' and 'histogram' are device pointers
// allocated by cudaMalloc. The function intentionally does not synchronize; it uses the
// default stream because the required signature does not provide a stream parameter.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    cudaStream_t stream = 0;

    const unsigned int rangeStart = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // The output must contain only the result of this invocation.
    cudaMemsetAsync(histogram, 0, histogramBytes, stream);

    if (inputSize == 0u) {
        return;
    }

    // Dynamic shared memory holds 32 copies of the requested range:
    //   [bin0_copy0..31, bin1_copy0..31, ...]
    // Worst-case valid input is 256 bins -> 256 * 32 * 4 B = 32 KiB.
    const size_t sharedMemBytes =
        static_cast<size_t>(numBins) * histogramCopies * sizeof(unsigned int);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    // The histogram has only 32 shared-memory copies, so contention on hot bins scales with
    // the number of warps per block. We therefore choose the smallest block size that still
    // leaves at least minUsefulResidentWarps resident warps per SM on the current device.
    const int active64 = get_active_blocks_per_sm<64>(sharedMemBytes);
    const int active128 = get_active_blocks_per_sm<128>(sharedMemBytes);
    const int active256 = get_active_blocks_per_sm<256>(sharedMemBytes);

    const int residentWarps64 = active64 * (64 / 32);
    const int residentWarps128 = active128 * (128 / 32);
    const int residentWarps256 = active256 * (256 / 32);

    const unsigned char *inputBytes =
        reinterpret_cast<const unsigned char *>(input);

    if (residentWarps64 >= minUsefulResidentWarps) {
        launch_histogram_kernel<64>(inputBytes,
                                    histogram,
                                    inputSize,
                                    rangeStart,
                                    numBins,
                                    sharedMemBytes,
                                    smCount,
                                    active64,
                                    stream);
    } else if (residentWarps128 >= minUsefulResidentWarps) {
        launch_histogram_kernel<128>(inputBytes,
                                     histogram,
                                     inputSize,
                                     rangeStart,
                                     numBins,
                                     sharedMemBytes,
                                     smCount,
                                     active128,
                                     stream);
    } else {
        // For larger ranges, 256 threads keeps occupancy healthy when shared memory begins
        // to limit active blocks.
        (void)residentWarps256;
        launch_histogram_kernel<256>(inputBytes,
                                     histogram,
                                     inputSize,
                                     rangeStart,
                                     numBins,
                                     sharedMemBytes,
                                     smCount,
                                     active256,
                                     stream);
    }
}