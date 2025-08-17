#include <cuda_runtime.h>
#include <stdint.h>

/*
  GPU-optimized histogram for a character range [from..to] using per-block shared-memory privatization.

  Key optimizations:
  - Each block builds multiple private copies of per-warp histograms in shared memory to reduce contention and minimize bank conflicts.
  - Final block-local histograms are reduced in shared memory and accumulated to the global histogram using atomic adds.
  - Each thread processes itemsPerThread characters (grid-tiled), improving memory throughput and reducing launch overhead.
  - Memory accesses are coalesced by having threads in a block process strided positions: idx = base + i * blockDim.x.

  Notes:
  - Input bytes are treated as unsigned (0..255) regardless of char signedness.
  - Output histogram has (to - from + 1) bins. The caller must have allocated this with cudaMalloc.
  - run_histogram() zeros the output histogram prior to kernel execution.
  - itemsPerThread is tunable; chosen default is optimized for modern NVIDIA data center GPUs (A100/H100) for large inputs.
*/

// Controls how many input chars each thread processes.
// Increase to reduce grid size and improve throughput; decrease if register pressure/occupancy becomes limiting.
static constexpr int itemsPerThread = 8;

// Threads per block. Must be a multiple of warpSize (32).
static constexpr int BLOCK_SIZE = 256;

// Number of replicated histograms per warp in shared memory to reduce bank conflicts.
// Must be >=1. Using a power-of-two (e.g., 2, 4, 8) is recommended for fast modulo via bitmask.
// 4 provides a good trade-off between contention reduction and shared memory usage.
static constexpr int SUBHIST_COPIES = 4;

static_assert((BLOCK_SIZE % 32) == 0, "BLOCK_SIZE must be a multiple of 32 (warp size).");
static_assert(SUBHIST_COPIES >= 1, "SUBHIST_COPIES must be >= 1.");

__global__ void histogramKernelRangeSM(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    // Compute number of bins for the requested range.
    const int numBins = to - from + 1;

    // Dynamic shared memory layout:
    // [ warpsPerBlock * SUBHIST_COPIES * numBins entries of unsigned int ]
    extern __shared__ unsigned int s_mem[];
    unsigned int* s_hists = s_mem;

    const int warpSize_ = 32;
    const int warpsPerBlock = blockDim.x / warpSize_;
    const int lane = threadIdx.x & (warpSize_ - 1);
    const int warp = threadIdx.x >> 5;

    // Each thread accumulates to its assigned replicated histogram to reduce bank conflicts:
    // - replication across SUBHIST_COPIES maps lanes to separate copies: copy = lane % SUBHIST_COPIES.
    //   This spreads concurrent atomic updates across multiple shared arrays.
    const int copy = (SUBHIST_COPIES == 1) ? 0 : (lane % SUBHIST_COPIES);
    const size_t perHistStride = static_cast<size_t>(numBins);
    const size_t myHistOffset = (static_cast<size_t>(warp) * SUBHIST_COPIES + static_cast<size_t>(copy)) * perHistStride;
    unsigned int* myHist = s_hists + myHistOffset;

    // Zero shared memory histograms.
    const size_t totalSharedEntries = static_cast<size_t>(warpsPerBlock) * SUBHIST_COPIES * static_cast<size_t>(numBins);
    for (size_t i = threadIdx.x; i < totalSharedEntries; i += blockDim.x) {
        s_hists[i] = 0u;
    }
    __syncthreads();

    // Process input: each thread handles itemsPerThread items in a grid-tiled manner for coalesced loads.
    const size_t tidInBlock = static_cast<size_t>(threadIdx.x);
    const size_t blockSpan = static_cast<size_t>(blockDim.x) * static_cast<size_t>(itemsPerThread);
    const size_t baseIndex = static_cast<size_t>(blockIdx.x) * blockSpan + tidInBlock;
    const size_t inputSizeSz = static_cast<size_t>(inputSize);
    const unsigned char* __restrict__ in = reinterpret_cast<const unsigned char*>(input);

    // Precompute bounds as unsigned for efficient comparisons.
    const unsigned int ufrom = static_cast<unsigned int>(from);
    const unsigned int uto   = static_cast<unsigned int>(to);

    #pragma unroll
    for (int i = 0; i < itemsPerThread; ++i) {
        const size_t idx = baseIndex + static_cast<size_t>(i) * static_cast<size_t>(blockDim.x);
        if (idx >= inputSizeSz) break;

        const unsigned int v = static_cast<unsigned int>(in[idx]);
        if (v >= ufrom && v <= uto) {
            const unsigned int bin = v - ufrom;
            // Shared-memory atomic adds are natively supported and fast on A100/H100.
            atomicAdd(&myHist[bin], 1u);
        }
    }

    __syncthreads();

    // Reduce all replicated per-warp histograms into the global histogram.
    // Each thread reduces a strided subset of bins to maximize parallelism.
    for (int b = threadIdx.x; b < numBins; b += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate across all warps and all replicated copies.
        for (int w = 0; w < warpsPerBlock; ++w) {
            const size_t warpBase = (static_cast<size_t>(w) * SUBHIST_COPIES) * perHistStride + static_cast<size_t>(b);
            #pragma unroll
            for (int c = 0; c < SUBHIST_COPIES; ++c) {
                sum += s_hists[warpBase + static_cast<size_t>(c) * perHistStride];
            }
        }
        if (sum) {
            atomicAdd(&histogram[b], sum);
        }
    }
}

/*
  Host launcher:

  - input: device pointer to chars of size inputSize (cudaMalloc'd).
  - histogram: device pointer to unsigned int array of size (to - from + 1) (cudaMalloc'd).
  - inputSize: number of chars in input.
  - from, to: inclusive range [from..to], with 0 <= from < to <= 255.

  The function zeroes the output histogram, computes an appropriate grid size based on itemsPerThread,
  and launches the kernel with dynamic shared memory sized to hold the replicated per-warp histograms.
  Synchronization is not performed here; the caller can synchronize if needed.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int numBins = to - from + 1;

    // Zero the output histogram (we compute fresh counts, not incremental updates).
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose block size and grid size to cover the input with itemsPerThread per thread.
    const int blockSize = BLOCK_SIZE;
    const unsigned int elemsPerBlock = static_cast<unsigned int>(blockSize * itemsPerThread);
    const unsigned int gridSize = (inputSize + elemsPerBlock - 1) / elemsPerBlock;

    // Dynamic shared memory size: per-warp replicated histograms.
    const int warpsPerBlock = blockSize / 32;
    const size_t smemBytes = static_cast<size_t>(warpsPerBlock) *
                             static_cast<size_t>(SUBHIST_COPIES) *
                             static_cast<size_t>(numBins) *
                             sizeof(unsigned int);

    // Launch the kernel. Caller is responsible for stream/synchronization management.
    if (gridSize > 0) {
        histogramKernelRangeSM<<<gridSize, blockSize, smemBytes>>>(input, histogram, inputSize, from, to);
    }
}