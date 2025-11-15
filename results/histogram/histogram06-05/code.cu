#include <cuda_runtime.h>
#include <stdint.h>

/*
 * CUDA histogram of characters in a given ordinal range [from, to].
 *
 * - Input:  device pointer to chars (plain text).
 * - Output: device pointer to unsigned int array of size (to - from + 1),
 *           where histogram[i] counts occurrences of character (from + i).
 *
 * The implementation uses:
 * - Per-block shared-memory histograms with privatization.
 * - 32 copies of the histogram in shared memory, one per warp lane.
 * - Striding of copies so that, for a given bin, the 32 copies map to 32
 *   different shared-memory banks (no intra-warp bank conflicts).
 * - Each thread processes a fixed number of input characters (itemsPerThread).
 */

/// Number of input characters processed per thread.
/// 16 is a good default on modern NVIDIA data-center GPUs (A100/H100) for
/// large inputs: enough work per thread to amortize overhead, without
/// excessively striding global memory accesses.
static constexpr int itemsPerThread = 16;

/*
 * Compute the padded number of bins for shared memory layout.
 *
 * For each logical bin, we have 32 copies (one per warp lane) in shared
 * memory, laid out with a stride that is:
 *
 *   paddedBins = k * 32 + 1
 *
 * for some integer k. This implies:
 *
 *   (lane * paddedBins + bin) % 32 == (lane + bin) % 32
 *
 * so, for a fixed bin, the 32 lane-specific addresses land in 32 distinct
 * shared-memory banks (no intra-warp bank conflicts when all threads in a
 * warp update the same bin).
 */
__device__ __forceinline__ int computePaddedBins(int numBins) {
    // numBins <= 256 (since chars in [0,255]), so this remains small.
    return ((numBins + 31) / 32) * 32 + 1;
}

/*
 * CUDA kernel: compute histogram for range [from, to] over `inputSize` chars.
 *
 * Parameters:
 *   input      - device pointer to input characters
 *   globalHist - device pointer to histogram array of size (to - from + 1)
 *   inputSize  - number of characters in input
 *   from, to   - inclusive character ordinal range [0, 255]
 *
 * Shared memory layout:
 *   sHist has size: 32 * paddedBins
 *   sHist[lane * paddedBins + bin] is the bin-th element of lane's histogram.
 */
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ globalHist,
                                 unsigned int inputSize,
                                 int from,
                                 int to) {
    extern __shared__ unsigned int sHist[];

    const int numBins    = to - from + 1;
    const int paddedBins = computePaddedBins(numBins);
    const int lane       = threadIdx.x & 31;         // Warp lane ID [0,31]
    const int blockThreads = blockDim.x;

    const int totalSharedBins = 32 * paddedBins;     // 32 lane copies

    // Initialize shared-memory histograms to zero.
    for (int i = threadIdx.x; i < totalSharedBins; i += blockThreads) {
        sHist[i] = 0;
    }
    __syncthreads();

    // Pointer to this thread's lane-specific histogram copy.
    unsigned int* myHist = sHist + lane * paddedBins;

    // Each block processes a contiguous chunk of the input:
    //   blockChunkSize = blockDim.x * itemsPerThread characters.
    const unsigned int blockChunkSize = static_cast<unsigned int>(blockThreads) *
                                        static_cast<unsigned int>(itemsPerThread);
    const unsigned int blockOffset = blockIdx.x * blockChunkSize;

    // Process up to itemsPerThread characters per thread.
#pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        unsigned int idx = blockOffset + i * blockThreads + threadIdx.x;
        if (idx >= inputSize) {
            break;
        }

        unsigned char c = static_cast<unsigned char>(input[idx]);

        // Only count characters within [from, to].
        if (c >= static_cast<unsigned char>(from) &&
            c <= static_cast<unsigned char>(to)) {
            int bin = static_cast<int>(c) - from;  // 0 <= bin < numBins

            // Shared-memory atomic add (fast on modern GPUs).
            atomicAdd(&myHist[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce the 32 lane-specific histograms into the global histogram.
    // Assumes blockDim.x >= numBins (true with block size 256 and numBins<=256).
    if (threadIdx.x < numBins) {
        unsigned int sum = 0;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            sum += sHist[l * paddedBins + threadIdx.x];
        }

        if (sum != 0) {
            // Multiple blocks contribute to the same global histogram,
            // so we must use a global atomic add.
            atomicAdd(&globalHist[threadIdx.x], sum);
        }
    }
}

/*
 * Host-side helper: launch the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input characters (cudaMalloc'd)
 *   histogram  - device pointer to output histogram array of size (to-from+1)
 *   inputSize  - number of characters in input
 *   from, to   - inclusive character ordinal range, 0 <= from < to <= 255
 *
 * The caller is responsible for host-device synchronization (e.g. cudaDeviceSynchronize).
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // Basic validation. The problem statement guarantees valid ranges, but
    // this protects against accidental misuse.
    if (inputSize == 0 || from > to) {
        return;
    }

    const int numBins = to - from + 1;

    // Initialize device histogram to zeros.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // Choose a reasonable block size for A100/H100.
    const int blockSize = 256;  // 8 warps per block.

    // Each block processes blockSize * itemsPerThread characters.
    const unsigned int workPerBlock = static_cast<unsigned int>(blockSize) *
                                      static_cast<unsigned int>(itemsPerThread);

    // Number of blocks needed to cover the input.
    const unsigned int gridSize =
        (inputSize + workPerBlock - 1) / workPerBlock;

    // Compute padded bin count and required shared memory size.
    const int paddedBins = ((numBins + 31) / 32) * 32 + 1;
    const size_t sharedMemSize =
        static_cast<size_t>(32) *
        static_cast<size_t>(paddedBins) *
        sizeof(unsigned int);

    // Launch kernel.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        to
    );
}