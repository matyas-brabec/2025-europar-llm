#include <cuda_runtime.h>

// ============================================================================
// CUDA histogram kernel with shared-memory privatization.
//
// This kernel computes a histogram over a contiguous range of character codes
// [from, to] in an input buffer of bytes. It is optimized for modern NVIDIA
// GPUs (e.g., A100/H100) and large input sizes.
//
// Key design points:
//  - Each thread processes ITEMS_PER_THREAD input bytes for more arithmetic
//    intensity and better latency hiding.
//  - Histogram bins live in shared memory and are replicated per warp-lane
//    to reduce contention and avoid shared memory bank conflicts.
//  - Each lane i (0..31) writes to bin (b * WARP_SIZE + i), so all threads
//    in a warp accessing the same bin touch different banks with no conflicts.
//  - After processing, all lane-local copies are reduced and accumulated into
//    the global histogram using atomicAdd.
// ============================================================================

static constexpr int THREADS_PER_BLOCK = 256;  // Reasonable default for A100/H100
static constexpr int ITEMS_PER_THREAD  = 8;    // Tuned for high throughput on large inputs
static constexpr int WARP_SIZE_CONST   = 32;   // Hardware warp size on current NVIDIA GPUs

// Kernel expects "input" as bytes (unsigned char) to avoid sign issues with char.
__global__ void histogram_kernel(const unsigned char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int s_hist[];  // Layout: numBins groups, each of size WARP_SIZE_CONST

    const int numBins = to - from + 1;
    const int tid     = threadIdx.x;
    const int lane    = tid & (WARP_SIZE_CONST - 1);  // lane index within warp: 0..31

    const int sharedHistSize = numBins * WARP_SIZE_CONST;  // total entries in shared histogram

    // ------------------------------------------------------------------------
    // Initialize shared histogram replicas to zero in parallel.
    // All threads in the block participate in zeroing.
    // ------------------------------------------------------------------------
    for (int i = tid; i < sharedHistSize; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Base pointer for this lane's private copy of the histogram.
    // Bins for this lane are at indices: bin * WARP_SIZE_CONST + lane
    unsigned int* s_hist_lane = s_hist + lane;

    const int globalThreadId = blockIdx.x * blockDim.x + tid;
    const int baseIndex      = globalThreadId * ITEMS_PER_THREAD;

    // ------------------------------------------------------------------------
    // Process ITEMS_PER_THREAD input characters per thread.
    // For each character:
    //  - Compute its bin index if within [from, to].
    //  - Atomically increment the lane-private shared-memory bin.
    //
    // Using lane-private bins reduces intra-warp contention and, with the
    // specific addressing (bin * WARP_SIZE_CONST + lane), avoids bank
    // conflicts for threads in the same warp accessing the same bin.
    // ------------------------------------------------------------------------
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = baseIndex + i;
        if (idx >= static_cast<int>(inputSize)) {
            break;
        }

        unsigned char c = input[idx];
        int bin = static_cast<int>(c) - from;

        if (bin >= 0 && bin < numBins) {
            // Each lane writes to its own replica of the bin.
            // Index pattern: bin * WARP_SIZE_CONST + lane
            atomicAdd(s_hist_lane + bin * WARP_SIZE_CONST, 1u);
        }
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // Reduce lane-private shared histograms into a single per-block histogram
    // and accumulate into the global histogram.
    //
    // Each thread handles multiple bins, summing across all WARP_SIZE_CONST
    // lane replicas and then performing a single atomicAdd to global memory.
    // ------------------------------------------------------------------------
    for (int bin = tid; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        int offset = bin * WARP_SIZE_CONST;

#pragma unroll
        for (int laneIter = 0; laneIter < WARP_SIZE_CONST; ++laneIter) {
            sum += s_hist[offset + laneIter];
        }

        if (sum > 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}


// ============================================================================
// Host-side wrapper to launch the histogram kernel.
//
// Parameters:
//   input      - device pointer (cudaMalloc'd) to the input text buffer.
//   histogram  - device pointer (cudaMalloc'd) to output histogram buffer.
//                Must hold at least (to - from + 1) unsigned ints.
//   inputSize  - number of characters (bytes) in the input buffer.
//   from, to   - inclusive character code range [from, to] (0 <= from < to <= 255)
//
// Notes:
//   - This function does not perform host-device synchronization; the caller
//     is responsible for synchronizing and checking for errors if needed.
//   - The histogram buffer is zeroed before kernel launch to ensure that
//     the result contains only counts from this invocation.
// ============================================================================

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Basic parameter checks; assume valid range as per problem statement.
    if (inputSize == 0 || from > to) {
        return;
    }

    const int numBins = to - from + 1;

    // Initialize the output histogram to zero.
    // This is asynchronous with respect to the host by default.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    const int threadsPerBlock = THREADS_PER_BLOCK;

    // Each thread handles ITEMS_PER_THREAD elements.
    // Total threads needed: ceil(inputSize / ITEMS_PER_THREAD)
    unsigned long long totalThreads =
        (static_cast<unsigned long long>(inputSize) + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;

    // Grid size: ceil(totalThreads / threadsPerBlock)
    unsigned long long blocks =
        (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    if (blocks == 0) {
        blocks = 1;
    }

    // dim3 uses unsigned int; clamp if extremely large (unlikely for practical cases).
    unsigned int gridX = (blocks > static_cast<unsigned long long>(UINT_MAX))
                         ? UINT_MAX
                         : static_cast<unsigned int>(blocks);

    dim3 blockDim(static_cast<unsigned int>(threadsPerBlock));
    dim3 gridDim(gridX);

    // Shared memory size: numBins * WARP_SIZE_CONST entries, each 4 bytes (unsigned int).
    size_t sharedMemSize = static_cast<size_t>(numBins) *
                           static_cast<size_t>(WARP_SIZE_CONST) *
                           sizeof(unsigned int);

    // Launch kernel; input is cast to unsigned char* to avoid sign issues.
    histogram_kernel<<<gridDim, blockDim, sharedMemSize>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        from,
        to
    );
}