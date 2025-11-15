#include <cuda_runtime.h>

// Number of input items processed by each thread.
// For modern data-center GPUs and large inputs, 8 provides a good balance
// between arithmetic intensity and occupancy for this histogram kernel.
static const int itemsPerThread = 8;

// Number of privatized histogram copies per block, one per warp lane.
// This matches the number of shared memory banks on recent NVIDIA GPUs
// and is used together with striding to avoid bank conflicts.
static const int histogramCopies = 32;

// CUDA kernel that computes a histogram over a specified character range.
// - input: device pointer to input characters
// - histogram: device pointer to global histogram (length = to - from + 1)
// - inputSize: number of characters in input
// - from, to: inclusive character range [from, to] (0 <= from < to <= 255)
__global__ void histogram_kernel(const char *input,
                                 unsigned int *histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    // Dynamic shared memory layout:
    // We store 32 copies of the histogram, each of length (to - from + 1),
    // with striding to avoid bank conflicts.
    //
    // For bin index i (0 <= i < rangeLen) and copy index c (0 <= c < 32),
    // the element is placed at:
    //     sharedHist[i * histogramCopies + c]
    //
    // Here, each copy is effectively interleaved across banks and each
    // warp lane accesses a different copy (copy index = threadIdx.x % 32),
    // minimizing shared memory bank conflicts during atomic updates.
    extern __shared__ unsigned int sharedHist[];

    const int rangeLen = to - from + 1;
    const unsigned int tid = threadIdx.x;
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + tid;
    const unsigned int totalThreads = gridDim.x * blockDim.x;

    // Each thread uses a specific histogram copy based on its lane index
    // (threadIdx.x % 32). This aligns copies with warp lanes.
    const int copyId = tid & (histogramCopies - 1); // threadIdx.x % 32

    // 1. Initialize shared histogram copies to zero.
    // Each thread zeroes a subset of the shared memory array.
    for (int i = tid; i < rangeLen * histogramCopies; i += blockDim.x) {
        sharedHist[i] = 0;
    }

    __syncthreads();

    // 2. Process input characters.
    // Each thread processes up to itemsPerThread characters with a grid-stride
    // pattern. This ensures full coverage of the input even with large sizes.
    for (int k = 0; k < itemsPerThread; ++k) {
        unsigned int idx = globalThreadId + static_cast<unsigned int>(k) * totalThreads;
        if (idx >= inputSize) {
            break;
        }

        unsigned char ch = static_cast<unsigned char>(input[idx]);
        unsigned int v = static_cast<unsigned int>(ch);

        if (v >= static_cast<unsigned int>(from) && v <= static_cast<unsigned int>(to)) {
            int bin = static_cast<int>(v) - from; // 0 <= bin < rangeLen
            unsigned int offset = static_cast<unsigned int>(bin) * histogramCopies
                                  + static_cast<unsigned int>(copyId);
            // Atomic add in shared memory, privatized per copy to reduce contention.
            atomicAdd(&sharedHist[offset], 1u);
        }
    }

    __syncthreads();

    // 3. Reduce per-copy histograms into the global histogram.
    //
    // Each thread is responsible for reducing several bins (strided by blockDim.x).
    // For each bin, we sum the 32 privatized copies, then atomically add the
    // block's contribution to the global histogram.
    for (int bin = tid; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0;
        int base = bin * histogramCopies;

        // Sum across all copies for this bin.
        #pragma unroll
        for (int c = 0; c < histogramCopies; ++c) {
            sum += sharedHist[base + c];
        }

        if (sum > 0) {
            // Global histogram is shared across blocks; use atomicAdd.
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// Host function that configures and launches the histogram kernel.
// - input: device pointer to input characters (allocated by cudaMalloc)
// - histogram: device pointer to output histogram (cudaMalloc'ed, but not assumed zeroed)
// - inputSize: number of characters in input
// - from, to: inclusive character range [from, to] (0 <= from < to <= 255)
//
// The caller is responsible for any required host-device synchronization.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Handle trivial cases early.
    if (inputSize == 0 || from > to) {
        return;
    }

    const int rangeLen = to - from + 1;

    // Initialize the global histogram buffer to zero.
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Choose block size suitable for modern NVIDIA GPUs.
    // 256 threads per block usually offers good occupancy and flexibility.
    const int blockSize = 256;

    // Each thread processes up to itemsPerThread items.
    const unsigned long long itemsPerBlock =
        static_cast<unsigned long long>(blockSize) * static_cast<unsigned long long>(itemsPerThread);

    // Compute grid size such that total capacity >= inputSize.
    unsigned int gridSize =
        static_cast<unsigned int>((static_cast<unsigned long long>(inputSize) + itemsPerBlock - 1ULL) / itemsPerBlock);

    if (gridSize == 0) {
        gridSize = 1;
    }

    // Dynamic shared memory size: number of bins * number of copies * sizeof(unsigned int).
    const size_t sharedMemSize =
        static_cast<size_t>(rangeLen) * static_cast<size_t>(histogramCopies) * sizeof(unsigned int);

    // Launch the kernel. Synchronization (if desired) is left to the caller.
    histogram_kernel<<<gridSize, blockSize, sharedMemSize>>>(input, histogram, inputSize, from, to);
}