#include <cuda_runtime.h>

// Histogram over a restricted byte range [from, to], where both input and output live in device memory.
//
// Performance-oriented design for modern NVIDIA GPUs (A100/H100 class):
//   1. Shared-memory privatization is done at warp granularity: each warp gets its own sub-histogram.
//      This sharply reduces contention compared to a single block-private histogram, especially for
//      skewed text distributions (e.g. spaces, newlines, vowels).
//   2. One padding element is inserted every 32 logical bins in shared memory to reduce bank conflicts.
//   3. Each thread processes multiple bytes (itemsPerThread) to amortize loop/index overhead.
//   4. The default itemsPerThread = 16 is chosen because it maps naturally to one 128-bit load
//      (uint4) per thread on the fast path, which is a strong default for large inputs on recent GPUs.
//
// The host wrapper intentionally does not synchronize; the caller requested fully asynchronous behavior.

namespace histogram_detail {

constexpr int warpWidth       = 32;
constexpr int threadsPerBlock = 256;

// Tuned default:
// - 16 bytes/thread enables one aligned 128-bit load per thread on the vectorized path.
// - This is a good balance between memory efficiency, ILP, and register pressure on A100/H100.
constexpr int itemsPerThread  = 16;

static_assert((threadsPerBlock % warpWidth) == 0,
              "threadsPerBlock must be a multiple of the warp size.");
static_assert(threadsPerBlock >= 256,
              "threadsPerBlock must be at least 256 so the final reduction can cover all 256 byte values.");
static_assert(itemsPerThread > 0,
              "itemsPerThread must be positive.");

// Total shared-memory counters needed for a padded histogram with 'bins' logical bins.
// Padding rule: add one extra element after every 32 logical bins.
__host__ __device__ inline unsigned int padded_histogram_bins(const unsigned int bins) {
    return bins + ((bins - 1u) / warpWidth);
}

// Map a logical bin index to its padded shared-memory index.
__device__ __forceinline__ unsigned int padded_bin_index(const unsigned int bin) {
    return bin + (bin / warpWidth);
}

// Increment the warp-private shared histogram if the byte falls in the requested [from, to] range.
// The range check is written as (byte - from) < bins, which is a standard branch-efficient form.
__device__ __forceinline__ void update_warp_histogram(
    const unsigned int byteValue,
    const unsigned int from_u,
    const unsigned int bins,
    unsigned int* const warpHistogram)
{
    const unsigned int localBin = byteValue - from_u;
    if (localBin < bins) {
        atomicAdd(&warpHistogram[padded_bin_index(localBin)], 1u);
    }
}

// Unpack four bytes from one 32-bit word and update the warp-private histogram.
__device__ __forceinline__ void process_packed_word(
    const unsigned int packedBytes,
    const unsigned int from_u,
    const unsigned int bins,
    unsigned int* const warpHistogram)
{
    update_warp_histogram( packedBytes        & 0xFFu, from_u, bins, warpHistogram);
    update_warp_histogram((packedBytes >>  8) & 0xFFu, from_u, bins, warpHistogram);
    update_warp_histogram((packedBytes >> 16) & 0xFFu, from_u, bins, warpHistogram);
    update_warp_histogram((packedBytes >> 24) & 0xFFu, from_u, bins, warpHistogram);
}

// Scalar fallback: used when the input pointer is not 16-byte aligned or when ITEMS_PER_THREAD is not
// a multiple of 16. This preserves correctness for any positive itemsPerThread value.
template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void process_tile_scalar(
    const unsigned char* const __restrict__ input,
    const unsigned int base,
    const unsigned int inputSize,
    const unsigned int from_u,
    const unsigned int bins,
    unsigned int* const warpHistogram)
{
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const unsigned int idx = base + static_cast<unsigned int>(i);
        if (idx < inputSize) {
            update_warp_histogram(input[idx], from_u, bins, warpHistogram);
        }
    }
}

// Vectorized fast path: assumes the input pointer is 16-byte aligned and ITEMS_PER_THREAD is a multiple
// of 16. Each thread consumes 16 bytes at a time via one uint4 load.
template <int ITEMS_PER_THREAD>
__device__ __forceinline__ void process_tile_vectorized(
    const unsigned char* const __restrict__ input,
    const unsigned int base,
    const unsigned int inputSize,
    const unsigned int from_u,
    const unsigned int bins,
    unsigned int* const warpHistogram)
{
    const uint4* const inputVec = reinterpret_cast<const uint4*>(input);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i += 16) {
        const unsigned int idx = base + static_cast<unsigned int>(i);

        // 64-bit comparison avoids corner-case wraparound when inputSize is near UINT_MAX.
        if (static_cast<unsigned long long>(idx) + 16ull <= static_cast<unsigned long long>(inputSize)) {
            const uint4 vec = inputVec[idx >> 4];
            process_packed_word(vec.x, from_u, bins, warpHistogram);
            process_packed_word(vec.y, from_u, bins, warpHistogram);
            process_packed_word(vec.z, from_u, bins, warpHistogram);
            process_packed_word(vec.w, from_u, bins, warpHistogram);
        } else {
            // Only the very end of the input reaches this path.
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const unsigned int scalarIdx = idx + static_cast<unsigned int>(j);
                if (scalarIdx < inputSize) {
                    update_warp_histogram(input[scalarIdx], from_u, bins, warpHistogram);
                }
            }
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void histogram_kernel(
    const unsigned char* const __restrict__ input,
    unsigned int* const __restrict__ histogram,
    const unsigned int inputSize,
    const unsigned int from_u,
    const unsigned int bins)
{
    constexpr int warpsPerBlock = BLOCK_THREADS / warpWidth;

    // Shared memory layout:
    //   [ warp0 padded histogram ][ warp1 padded histogram ] ... [ warpN padded histogram ]
    extern __shared__ unsigned int sharedHistogram[];

    const unsigned int tid        = threadIdx.x;
    const unsigned int warpId     = tid / warpWidth;
    const unsigned int paddedBins = padded_histogram_bins(bins);
    const unsigned int sharedSpan = static_cast<unsigned int>(warpsPerBlock) * paddedBins;

    unsigned int* const warpHistogram = sharedHistogram + warpId * paddedBins;

    // Zero all warp-private histograms.
    for (unsigned int i = tid; i < sharedSpan; i += BLOCK_THREADS) {
        sharedHistogram[i] = 0u;
    }
    __syncthreads();

    // Grid-stride over tiles. Each thread handles ITEMS_PER_THREAD consecutive bytes per iteration.
    const unsigned int globalThread = blockIdx.x * BLOCK_THREADS + tid;
    const unsigned int gridStride   = static_cast<unsigned int>(gridDim.x) * BLOCK_THREADS * ITEMS_PER_THREAD;

    // cudaMalloc returns pointers with enough alignment for 16-byte vector loads, but this runtime check
    // also keeps the kernel correct if the caller passes an offset pointer.
    const bool inputIsAligned16 = ((reinterpret_cast<unsigned long long>(input) & 0xFull) == 0ull);

    if (((ITEMS_PER_THREAD % 16) == 0) && inputIsAligned16) {
        for (unsigned int base = globalThread * ITEMS_PER_THREAD; base < inputSize; base += gridStride) {
            process_tile_vectorized<ITEMS_PER_THREAD>(input, base, inputSize, from_u, bins, warpHistogram);
        }
    } else {
        for (unsigned int base = globalThread * ITEMS_PER_THREAD; base < inputSize; base += gridStride) {
            process_tile_scalar<ITEMS_PER_THREAD>(input, base, inputSize, from_u, bins, warpHistogram);
        }
    }

    __syncthreads();

    // Reduce the warp-private shared histograms into the final global histogram.
    // Because BLOCK_THREADS is 256 and bins <= 256, the full-range case maps one thread per output bin.
    if (tid < bins) {
        const unsigned int paddedBin = padded_bin_index(tid);
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += sharedHistogram[w * paddedBins + paddedBin];
        }

        if (sum != 0u) {
            atomicAdd(&histogram[tid], sum);
        }
    }
}

} // namespace histogram_detail

void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // The output histogram stores only the requested inclusive range [from, to].
    const unsigned int bins   = static_cast<unsigned int>(to - from + 1);
    const unsigned int from_u = static_cast<unsigned int>(from);

    // The kernel accumulates with atomic adds, so the destination must start from zero.
    // This is asynchronous and ordered before the kernel launch in the same stream.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedMemBytes =
        static_cast<size_t>(histogram_detail::threadsPerBlock / histogram_detail::warpWidth) *
        static_cast<size_t>(histogram_detail::padded_histogram_bins(bins)) *
        sizeof(unsigned int);

    // Launch enough blocks to fill the machine, but do not oversubscribe unnecessarily.
    // The kernel is grid-stride, so a resident-sized grid is sufficient even for very large inputs.
    int device = 0;
    cudaGetDevice(&device);

    int multiprocessorCount = 0;
    cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device);

    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_detail::histogram_kernel<histogram_detail::threadsPerBlock, histogram_detail::itemsPerThread>,
        histogram_detail::threadsPerBlock,
        sharedMemBytes);

    const unsigned long long tileItems =
        static_cast<unsigned long long>(histogram_detail::threadsPerBlock) *
        static_cast<unsigned long long>(histogram_detail::itemsPerThread);

    const unsigned int blocksForInput = static_cast<unsigned int>(
        (static_cast<unsigned long long>(inputSize) + tileItems - 1ull) / tileItems);

    const unsigned int maxResidentBlocks =
        static_cast<unsigned int>(activeBlocksPerSM * multiprocessorCount);

    const unsigned int gridSize =
        (blocksForInput < maxResidentBlocks) ? blocksForInput : maxResidentBlocks;

    if (gridSize == 0u) {
        return;
    }

    histogram_detail::histogram_kernel<histogram_detail::threadsPerBlock, histogram_detail::itemsPerThread>
        <<<gridSize, histogram_detail::threadsPerBlock, sharedMemBytes>>>(
            reinterpret_cast<const unsigned char*>(input),
            histogram,
            inputSize,
            from_u,
            bins);

    // Intentionally no synchronization here; the caller requested asynchronous behavior.
}