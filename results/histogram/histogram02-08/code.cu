#include <cuda_runtime.h>
#include <stdint.h>

// Device helper: update histogram for 4 bytes packed in a 32-bit word.
// Uses warp-private shared memory histogram to reduce contention.
static __device__ __forceinline__ void update_word_4bytes(uint32_t w, int from, int numBins, unsigned int* warpHist) {
    // Extract 4 bytes, check against range, and update shared histogram with atomicAdd
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        unsigned int c = (w >> (8 * k)) & 0xFFu;
        unsigned int bin = c - static_cast<unsigned int>(from);
        if (bin < static_cast<unsigned int>(numBins)) {
            atomicAdd(&warpHist[bin], 1u);
        }
    }
}

// Kernel: compute histogram over a restricted character range [from, from+numBins-1].
// - input: device pointer to char data (plain text)
// - histogram: device pointer to output bins (numBins entries), must be zeroed before accumulation
// - inputSize: number of chars in input
// - from: starting character code (inclusive)
// - numBins: number of bins (to - from + 1)
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int numBins) {
    extern __shared__ unsigned int s_hist[]; // Layout: warpsPerBlock private histograms, each numBins wide.

    const int tid = threadIdx.x;
    const int warpsPerBlock = blockDim.x >> 5; // assume blockDim.x is multiple of 32
    const int warpId = tid >> 5;

    // Initialize shared memory
    for (int i = tid; i < numBins * warpsPerBlock; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Pointer to this warp's private histogram
    unsigned int* warpHist = s_hist + warpId * numBins;

    // Global thread indexing and grid-stride setup
    const size_t globalThreadId = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t totalThreads   = static_cast<size_t>(gridDim.x) * blockDim.x;

    // Use vectorized 16B loads when input is 16-byte aligned
    const bool aligned16 = ((reinterpret_cast<uintptr_t>(input) & 15u) == 0u);

    if (aligned16) {
        // Process 16-byte chunks via uint4 loads
        const size_t numVec = static_cast<size_t>(inputSize) / 16u;
        const uint4* input4 = reinterpret_cast<const uint4*>(input);

        for (size_t i = globalThreadId; i < numVec; i += totalThreads) {
            uint4 v = input4[i];
            update_word_4bytes(v.x, from, numBins, warpHist);
            update_word_4bytes(v.y, from, numBins, warpHist);
            update_word_4bytes(v.z, from, numBins, warpHist);
            update_word_4bytes(v.w, from, numBins, warpHist);
        }

        // Handle the tail bytes
        const size_t tailStart = numVec * 16u;
        for (size_t i = tailStart + globalThreadId; i < static_cast<size_t>(inputSize); i += totalThreads) {
            unsigned int c = static_cast<unsigned char>(input[i]);
            unsigned int bin = c - static_cast<unsigned int>(from);
            if (bin < static_cast<unsigned int>(numBins)) {
                atomicAdd(&warpHist[bin], 1u);
            }
        }
    } else {
        // Fallback for unaligned pointers: process byte-by-byte
        for (size_t i = globalThreadId; i < static_cast<size_t>(inputSize); i += totalThreads) {
            unsigned int c = static_cast<unsigned char>(input[i]);
            unsigned int bin = c - static_cast<unsigned int>(from);
            if (bin < static_cast<unsigned int>(numBins)) {
                atomicAdd(&warpHist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce warp-private histograms into global histogram
    for (int bin = tid; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        #pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * numBins + bin];
        }
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host function: launches the histogram kernel.
// Assumptions:
// - 'input' and 'histogram' are device pointers allocated with cudaMalloc.
// - 'histogram' has room for (to - from + 1) bins.
// - Caller handles synchronization (this function enqueues work on the default stream).
extern "C"
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    const int numBins = to - from + 1;
    if (!input || !histogram || inputSize == 0 || numBins <= 0) {
        return;
    }

    // Choose a block size that's a multiple of 32 for clean warp partitioning.
    // 256 threads per block is a good balance for shared-memory histograms.
    constexpr int blockSize = 256;
    const int warpsPerBlock = blockSize / 32;

    // Grid size heuristic:
    // Aim for roughly one block per 4096 bytes of input (assuming 16 bytes per thread iteration),
    // but clamp to 65535 to remain compatible across architectures. Grid-stride loop will cover all data.
    unsigned long long bytesPerBlockIter = static_cast<unsigned long long>(blockSize) * 16ull;
    unsigned long long gridEstimate = (static_cast<unsigned long long>(inputSize) + bytesPerBlockIter - 1ull) / bytesPerBlockIter;
    if (gridEstimate == 0ull) gridEstimate = 1ull;
    if (gridEstimate > 65535ull) gridEstimate = 65535ull;

    dim3 gridDim(static_cast<unsigned int>(gridEstimate));
    dim3 blockDim(blockSize);

    // Shared memory size: warp-private histograms
    size_t sharedMemBytes = static_cast<size_t>(numBins) * static_cast<size_t>(warpsPerBlock) * sizeof(unsigned int);

    // Ensure the output histogram starts at zero before accumulation
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Launch kernel (asynchronous on default stream)
    histogram_range_kernel<<<gridDim, blockDim, sharedMemBytes>>>(input, histogram, inputSize, from, numBins);
}