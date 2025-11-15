#include <cuda_runtime.h>

/// Number of input characters processed by each thread.
/// Chosen for good balance between memory throughput and occupancy
/// on recent NVIDIA data center GPUs (e.g., A100, H100).
static constexpr int ITEMS_PER_THREAD   = 16;
static constexpr int THREADS_PER_BLOCK  = 256;
static constexpr int WARP_SIZE          = 32;

/// CUDA kernel that computes a histogram over a selected character range.
/// - input:      device pointer to input characters
/// - histogram:  device pointer to output histogram (size = to - from + 1)
/// - inputSize:  number of characters in input
/// - from, to:   inclusive character range [from, to] to count
///
/// Optimization details:
/// - Uses shared memory to hold 32 privatized copies of the histogram per block
///   to reduce global memory atomics.
/// - Each thread uses a copy identified by (threadIdx.x % 32), i.e., its warp lane.
/// - Copies are stored with stride 32: bin i, copy c -> sharedHist[i * 32 + c].
///   This layout ensures that for a given bin, threads in a warp access different
///   banks, minimizing shared memory bank conflicts.
/// - Input is processed in grid-stride loops, ITEMS_PER_THREAD items per thread
///   per iteration, for scalability to large inputs.
__global__ void histogram_kernel(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    // Dynamic shared memory: 32 copies of the histogram.
    // Size (in ints) = (to - from + 1) * 32
    extern __shared__ unsigned int sharedHist[];

    const int numBins = to - from + 1;
    const int tid     = threadIdx.x;
    const int lane    = tid & (WARP_SIZE - 1);  // threadIdx.x % 32

    // Initialize all histogram copies in shared memory to zero.
    // We have numBins * WARP_SIZE entries: for bin i, copies 0..31.
    for (int i = tid; i < numBins * WARP_SIZE; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();

    // Global linear thread index.
    const size_t globalThreadId =
        static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

    // Each thread processes ITEMS_PER_THREAD characters per loop iteration.
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x * ITEMS_PER_THREAD;

    // Starting index in the input buffer for this thread.
    size_t index = globalThreadId * ITEMS_PER_THREAD;

    // Grid-stride loop over the input text.
    while (index < static_cast<size_t>(inputSize)) {

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            size_t idx = index + i;
            if (idx >= static_cast<size_t>(inputSize)) {
                break;
            }

            // Cast to unsigned char to avoid sign-extension issues.
            unsigned char ch =
                static_cast<unsigned char>(input[idx]);

            // Only count characters within [from, to].
            if (ch >= static_cast<unsigned char>(from) &&
                ch <= static_cast<unsigned char>(to)) {

                int bin = static_cast<int>(ch) - from;
                unsigned int offset =
                    static_cast<unsigned int>(bin * WARP_SIZE + lane);

                // Shared-memory atomic update to this thread's copy of the bin.
                atomicAdd(&sharedHist[offset], 1u);
            }
        }

        index += stride;
    }

    __syncthreads();

    // Reduce the 32 copies of each bin into a single value and
    // accumulate into the global histogram.
    for (int bin = tid; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        int base = bin * WARP_SIZE;

        #pragma unroll
        for (int c = 0; c < WARP_SIZE; ++c) {
            sum += sharedHist[base + c];
        }

        if (sum != 0) {
            // Global atomic add to accumulate partial histograms from all blocks.
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/// Host function that launches the histogram kernel.
///
/// Parameters:
/// - input:      device pointer to input characters (cudaMalloc'd)
/// - histogram:  device pointer to output histogram (cudaMalloc'd)
///               size must be (to - from + 1) * sizeof(unsigned int)
/// - inputSize:  number of characters in input
/// - from, to:   inclusive character range [from, to] (0 <= from < to <= 255)
///
/// Notes:
/// - This function assumes input and histogram are device pointers.
/// - It zero-initializes the histogram on device before launching the kernel.
/// - Host/device synchronization (e.g., cudaDeviceSynchronize()) is the
///   responsibility of the caller.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic validation; the caller guarantees from/to are within [0,255],
    // but we still guard against empty input or null pointers.
    if (!input || !histogram || inputSize == 0 || from > to) {
        return;
    }

    const int numBins = to - from + 1;

    // Clear the output histogram on device.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    // Choose a grid size. Since the kernel uses a grid-stride loop, we only
    // need "enough" blocks for good occupancy, not necessarily one thread per
    // ITEMS_PER_THREAD characters.
    int blocks = static_cast<int>(
        (static_cast<size_t>(inputSize) +
         (THREADS_PER_BLOCK * ITEMS_PER_THREAD - 1)) /
        (THREADS_PER_BLOCK * ITEMS_PER_THREAD));

    if (blocks <= 0) {
        blocks = 1;
    }

    // Limit block count to a large but safe upper bound.
    const int maxBlocks = 65535;
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    // Shared memory size: 32 copies of the histogram, each with numBins bins.
    // Layout: bin i, copy c -> sharedHist[i * 32 + c]
    size_t sharedMemSize =
        static_cast<size_t>(numBins) * WARP_SIZE * sizeof(unsigned int);

    histogram_kernel<<<blocks, THREADS_PER_BLOCK, sharedMemSize>>>(
        input, histogram, inputSize, from, to);
}