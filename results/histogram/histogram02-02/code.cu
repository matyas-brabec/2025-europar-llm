#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// Histogram for a contiguous byte sub-range [from, to].
//
// Design choices for modern NVIDIA data-center GPUs:
// - One private shared-memory histogram per warp reduces shared-memory atomic contention versus a
//   single block-wide histogram.
// - The block size is 256 threads, which matches the maximum possible number of output bins.
//   That lets one thread naturally correspond to one possible output bin during initialization
//   and final reduction.
// - The input is processed through:
//     1) a short scalar prologue to reach 16-byte alignment,
//     2) a vectorized main loop using uint4 loads (16 bytes/thread/iteration),
//     3) a short scalar epilogue.
// - The final merge performs one global update per nonzero bin per block.

constexpr int kWarpSize      = 32;
constexpr int kBlockSize     = 256;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr int kMaxBins       = 256;
constexpr unsigned int kVectorBytes = sizeof(uint4);

static_assert(kBlockSize % kWarpSize == 0, "Block size must be a multiple of the warp size.");
static_assert(kBlockSize == kMaxBins, "This implementation relies on one thread per possible byte bin.");
static_assert(kVectorBytes == 16u, "uint4 is expected to be 16 bytes.");

__device__ __forceinline__ void shared_hist_atomic_add(unsigned int* address, unsigned int value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    // The shared histogram is private to the current CTA, so block-scope atomics are sufficient
    // and can be slightly cheaper than full device-scope atomics on modern architectures.
    atomicAdd_block(address, value);
#else
    atomicAdd(address, value);
#endif
}

__device__ __forceinline__ void accumulate_byte_to_hist(const unsigned int byteValue,
                                                        unsigned int* const warpHist,
                                                        const unsigned int from,
                                                        const unsigned int rangeLen) {
    // Unsigned subtraction + bounds check compactly implements:
    //   from <= byteValue <= to
    // because values smaller than `from` wrap to a large unsigned integer and fail the check.
    const unsigned int bin = byteValue - from;
    if (bin < rangeLen) {
        shared_hist_atomic_add(warpHist + bin, 1u);
    }
}

__device__ __forceinline__ void accumulate_word_to_hist(const unsigned int word,
                                                        unsigned int* const warpHist,
                                                        const unsigned int from,
                                                        const unsigned int rangeLen) {
    accumulate_byte_to_hist( word        & 0xFFu, warpHist, from, rangeLen);
    accumulate_byte_to_hist((word >>  8) & 0xFFu, warpHist, from, rangeLen);
    accumulate_byte_to_hist((word >> 16) & 0xFFu, warpHist, from, rangeLen);
    accumulate_byte_to_hist((word >> 24) & 0xFFu, warpHist, from, rangeLen);
}

template <int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void histogram_range_kernel(const unsigned char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            const unsigned int inputSize,
                            const unsigned int from,
                            const unsigned int rangeLen) {
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / kWarpSize;

    // One private 256-bin histogram per warp.
    // With BLOCK_SIZE == 256 this is 8 warps * 256 bins * 4 bytes = 8 KiB of shared memory.
    __shared__ unsigned int s_hist[WARPS_PER_BLOCK][kMaxBins];

    const unsigned int localBin = threadIdx.x;

    // Only bins [0, rangeLen) can ever be touched, so only those bins need initialization.
    if (localBin < rangeLen) {
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            s_hist[w][localBin] = 0u;
        }
    }
    __syncthreads();

    const unsigned int globalThread = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned int totalThreads = gridDim.x * BLOCK_SIZE;
    const unsigned int warpId       = threadIdx.x >> 5;
    unsigned int* const warpHist    = s_hist[warpId];

    // Even though cudaMalloc returns highly aligned pointers, this scalar prologue makes the vectorized
    // path safe if the caller passes an offset into that allocation.
    const unsigned int misalignment =
        static_cast<unsigned int>(reinterpret_cast<uintptr_t>(input) & (kVectorBytes - 1u));
    const unsigned int scalarPrefix =
        (kVectorBytes - misalignment) & (kVectorBytes - 1u);
    const unsigned int prefixCount =
        (scalarPrefix < inputSize) ? scalarPrefix : inputSize;

    // Scalar prologue: at most 15 bytes.
    for (unsigned int idx = globalThread; idx < prefixCount; idx += totalThreads) {
        accumulate_byte_to_hist(static_cast<unsigned int>(input[idx]), warpHist, from, rangeLen);
    }

    const unsigned char* const alignedInput = input + prefixCount;
    const unsigned int remaining = inputSize - prefixCount;
    const unsigned int vecCount  = remaining / kVectorBytes;
    const uint4* const input4    = reinterpret_cast<const uint4*>(alignedInput);

    // Main vectorized pass: 16 bytes per thread per iteration.
    for (unsigned int vecIdx = globalThread; vecIdx < vecCount; vecIdx += totalThreads) {
        const uint4 v = input4[vecIdx];
        accumulate_word_to_hist(v.x, warpHist, from, rangeLen);
        accumulate_word_to_hist(v.y, warpHist, from, rangeLen);
        accumulate_word_to_hist(v.z, warpHist, from, rangeLen);
        accumulate_word_to_hist(v.w, warpHist, from, rangeLen);
    }

    const unsigned int tailStart = prefixCount + vecCount * kVectorBytes;
    const unsigned int tailCount = inputSize - tailStart;

    // Scalar epilogue: at most 15 bytes.
    for (unsigned int idx = globalThread; idx < tailCount; idx += totalThreads) {
        accumulate_byte_to_hist(static_cast<unsigned int>(input[tailStart + idx]), warpHist, from, rangeLen);
    }

    __syncthreads();

    // Reduce the warp-private histograms and merge them into the final global histogram.
    if (localBin < rangeLen) {
        unsigned int sum = 0u;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += s_hist[w][localBin];
        }

        if (sum != 0u) {
            // run_histogram() zeros the destination histogram before launch, so a single-block launch
            // can store directly instead of using a global atomic.
            if (gridDim.x == 1) {
                histogram[localBin] = sum;
            } else {
                atomicAdd(histogram + localBin, sum);
            }
        }
    }
}

}  // namespace

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    const unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);

    // The result is accumulated with atomics, so the destination must start from zero.
    // Both operations are enqueued on stream 0; synchronization is intentionally left to the caller.
    (void)cudaMemsetAsync(
        histogram,
        0,
        static_cast<size_t>(rangeLen) * sizeof(unsigned int),
        0);

    if (inputSize == 0u) {
        return;
    }

    constexpr unsigned int kBytesPerBlock =
        static_cast<unsigned int>(kBlockSize) * kVectorBytes;

    unsigned int gridSize = 1u;

    // For very small inputs, a single block is enough and avoids host-side occupancy queries.
    if (inputSize > kBytesPerBlock) {
        int device = 0;
        (void)cudaGetDevice(&device);

        int smCount = 0;
        (void)cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

        int activeBlocksPerSM = 0;
        (void)cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocksPerSM,
            histogram_range_kernel<kBlockSize>,
            kBlockSize,
            0);

        unsigned int maxResidentBlocks = 128u;
        if (smCount > 0 && activeBlocksPerSM > 0) {
            maxResidentBlocks = static_cast<unsigned int>(smCount * activeBlocksPerSM);
        } else if (smCount > 0) {
            maxResidentBlocks = static_cast<unsigned int>(smCount * 4);
        }

        // Size-based heuristic: enough blocks so that, before grid-striding, each thread gets about
        // one vector load on average. Cap to resident blocks so the final global merge does not create
        // excessive atomic traffic.
        const unsigned int blocksFromSize = 1u + (inputSize - 1u) / kBytesPerBlock;
        gridSize = (blocksFromSize < maxResidentBlocks) ? blocksFromSize : maxResidentBlocks;
        if (gridSize == 0u) {
            gridSize = 1u;
        }
    }

    // Reinterpret as unsigned bytes so byte values 128..255 are counted correctly even if `char` is signed.
    histogram_range_kernel<kBlockSize><<<gridSize, kBlockSize, 0, 0>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        rangeLen);
}