#include <cuda_runtime.h>
#include <stdint.h>

// This implementation computes a histogram for a specified character range [from, to] over an input
// text buffer. It is optimized for modern NVIDIA GPUs (A100/H100) using shared-memory privatization.
// Key optimization points:
// - Each thread processes multiple items (itemsPerThread) to amortize indexing overhead and increase ILP.
// - Coalesced global memory access via a grid-stride loop with loop unrolling by itemsPerThread.
// - Per-warp privatized histograms in shared memory to reduce contention and global atomics.
// - Multiple replicated histograms per warp (SUBHISTS_PER_WARP) to reduce shared-memory atomic contention
//   and avoid bank conflicts. A small padding (+1 per histogram) breaks 32-bank alignment.
// - Final reduction in shared memory and atomicAdd to global memory once per bin per block.

// Tunable constants chosen for high-end GPUs and large inputs.
static constexpr int THREADS_PER_BLOCK   = 256;  // 8 warps per block; good balance of occupancy and shared memory usage
static constexpr int ITEMS_PER_THREAD    = 16;   // Each thread processes this many items per outer iteration
static constexpr int SUBHISTS_PER_WARP   = 4;    // Number of replicated sub-histograms per warp (power-of-two recommended)

static_assert((SUBHISTS_PER_WARP & (SUBHISTS_PER_WARP - 1)) == 0, "SUBHISTS_PER_WARP must be a power of two");
static_assert((THREADS_PER_BLOCK % 32) == 0, "THREADS_PER_BLOCK must be a multiple of warp size (32)");

// CUDA kernel: compute per-block partial histograms and accumulate to global output.
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    const int warpSizeC = 32;
    const int rangeLen = to - from + 1;      // number of bins to compute
    const int warpsPerBlock = blockDim.x / warpSizeC;
    const int subHistCount = warpsPerBlock * SUBHISTS_PER_WARP;

    // We add 1 element of padding to each sub-histogram to break 32-bank alignment,
    // which helps reduce bank conflicts when different copies are accessed concurrently.
    const int stride = rangeLen + 1;

    extern __shared__ unsigned int s_hist[]; // layout: [subHistCount][stride]

    // Zero-initialize the shared histograms.
    for (int i = threadIdx.x; i < subHistCount * stride; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int gridStride = blockDim.x * gridDim.x;

    const int warpId  = threadIdx.x / warpSizeC;
    const int laneId  = threadIdx.x & (warpSizeC - 1);
    // Distribute lanes across SUBHISTS_PER_WARP copies to reduce contention.
    const int subCopy = laneId & (SUBHISTS_PER_WARP - 1);

    const int subHistIndex = warpId * SUBHISTS_PER_WARP + subCopy;
    unsigned int* myHist = s_hist + subHistIndex * stride;

    // Coalesced, grid-stride loop unrolled by ITEMS_PER_THREAD.
    // For fixed k, threads in a warp access consecutive input positions -> coalesced.
    for (unsigned int base = globalThreadId; base < inputSize; base += gridStride * ITEMS_PER_THREAD) {
        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            unsigned int pos = base + k * gridStride;
            if (pos < inputSize) {
                unsigned char c = static_cast<unsigned char>(input[pos]);
                if (c >= static_cast<unsigned char>(from) && c <= static_cast<unsigned char>(to)) {
                    int bin = static_cast<int>(c) - from; // 0 .. rangeLen-1
                    // Shared-memory atomic to our warp's replicated histogram copy.
                    atomicAdd(&myHist[bin], 1u);
                }
            }
        }
    }

    __syncthreads();

    // Reduce all sub-histograms in the block into global histogram.
    // Each thread sums a subset of bins across all copies, then atomically adds to global.
    for (int bin = threadIdx.x; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int sh = 0; sh < subHistCount; ++sh) {
            sum += s_hist[sh * stride + bin];
        }
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host function that launches the CUDA kernel.
// - input: device pointer to input chars (cudaMalloc'd), size inputSize
// - histogram: device pointer to output bins (cudaMalloc'd), size (to-from+1)
// - inputSize: number of chars in input
// - from, to: inclusive range [from,to] to compute histogram over (0 <= from < to <= 255)
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Defensive checks (no synchronization involved).
    if (!input || !histogram || inputSize == 0 || from > to) {
        return;
    }

    const int rangeLen = to - from + 1;

    // Ensure output histogram is zero-initialized. Caller manages synchronization, so async is fine.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Determine a suitable grid size. Use a multiple of SM count to saturate the device.
    int device = 0, smCount = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    // Heuristic: 32 blocks per SM (suits large inputs, keeps high latency hiding).
    int maxActiveBlocks = smCount * 32;

    // Also consider total work. Each block covers THREADS_PER_BLOCK * ITEMS_PER_THREAD items per "iteration".
    // Use ceiling division to estimate blocks needed; clamp to a reasonable maximum (1D grid limit).
    unsigned long long itemsPerBlockIter = static_cast<unsigned long long>(THREADS_PER_BLOCK) * ITEMS_PER_THREAD;
    unsigned int neededBlocks = (inputSize + itemsPerBlockIter - 1ULL) / itemsPerBlockIter;

    unsigned int gridBlocks = neededBlocks;
    if (gridBlocks < static_cast<unsigned int>(smCount)) {
        gridBlocks = smCount; // at least one block per SM
    }
    if (gridBlocks > static_cast<unsigned int>(maxActiveBlocks)) {
        gridBlocks = static_cast<unsigned int>(maxActiveBlocks);
    }

    // Dynamic shared memory size: subHistCount * (rangeLen + 1) counters.
    const int warpsPerBlock = THREADS_PER_BLOCK / 32;
    const int subHistCount = warpsPerBlock * SUBHISTS_PER_WARP;
    size_t sharedMemSize = static_cast<size_t>(subHistCount) * static_cast<size_t>(rangeLen + 1) * sizeof(unsigned int);

    // Launch kernel on the default stream (caller handles synchronization).
    histogram_range_kernel<<<gridBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(
        input, histogram, inputSize, from, to
    );
}