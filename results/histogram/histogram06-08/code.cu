#include <cuda_runtime.h>
#include <stdint.h>

// Tunable constant: number of input chars processed by each thread in the grid-stride loop.
// For modern data center GPUs (A100/H100) and large inputs, 16 balances memory throughput
// with occupancy and minimizes loop/branch overhead.
static constexpr int itemsPerThread = 16;

// CUDA kernel that computes a histogram for characters within [from, to] (inclusive).
// The histogram is privatized in shared memory using 32 copies (one per bank/warp lane).
// Each copy is laid out with a stride of 32 so that, for a given bin, warp lanes update
// distinct banks: address = bin * 32 + laneId -> bank = laneId.
// This avoids intra-warp bank conflicts. Inter-warp updates to the same (bin, laneId) use
// fast shared-memory atomics for correctness.
// After counting, the 32 copies are reduced per-bin and atomically accumulated to the
// global histogram.
__global__ void histogramKernel_range32banks(
    const char* __restrict__ input,
    unsigned int inputSize,
    unsigned int* __restrict__ globalHist,  // length = (to - from + 1)
    int from,
    int to)
{
    const int rangeLen = to - from + 1;
    if (rangeLen <= 0) return;  // Guard (shouldn't happen per problem constraints)

    // Allocate 32 copies of the histogram in shared memory with a 32-element stride:
    // Indexing: s_hist[bin * 32 + laneId] -> maps to shared bank 'laneId'.
    extern __shared__ unsigned int s_hist[];
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane  = threadIdx.x & 31; // warp lane id
    const int tpb   = blockDim.x;
    const int gsize = gridDim.x * blockDim.x;

    // Initialize shared memory histogram (all copies to zero).
    // Each thread zeros a slice to reduce init time.
    for (int i = threadIdx.x; i < rangeLen * 32; i += tpb) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Grid-stride loop where each thread processes 'itemsPerThread' items per outer iteration.
    // For coalesced memory access, at inner iteration j, threads in a warp access consecutive
    // positions: idx = base + j * gsize, so lanes 0..31 map to a contiguous 32-byte segment.
    for (unsigned int base = tid; base < inputSize; base += gsize * itemsPerThread) {
        #pragma unroll
        for (int j = 0; j < itemsPerThread; ++j) {
            unsigned int idx = base + j * gsize;
            if (idx >= inputSize) break;

            // Treat chars as unsigned to map into [0, 255].
            unsigned char c = static_cast<unsigned char>(input[idx]);
            int bin = static_cast<int>(c) - from;

            // Fast in-range test: unsigned comparison handles both bounds.
            if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(rangeLen)) {
                // Update the per-lane copy of the bin using shared-memory atomic add.
                // Layout ensures that within a warp, each lane hits a distinct bank.
                atomicAdd(&s_hist[bin * 32 + lane], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce the 32 lane-privatized copies per bin to a single count and add to global.
    for (int bin = threadIdx.x; bin < rangeLen; bin += tpb) {
        unsigned int sum = 0;
        int base = bin * 32;
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            sum += s_hist[base + l];
        }
        // Accumulate to the global histogram. Multiple blocks contribute, so use atomicAdd.
        atomicAdd(&globalHist[bin], sum);
    }
}

// Host function to launch the histogram kernel.
// - 'input' and 'histogram' are device pointers allocated via cudaMalloc.
// - 'histogram' must have size (to - from + 1) unsigned ints.
// - The function zeros the output histogram, configures launch parameters, and launches the kernel.
// - Synchronization is left to the caller as requested.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const int rangeLen = to - from + 1;
    if (rangeLen <= 0) {
        return; // Nothing to do
    }

    // Clear the output histogram (as we accumulate with atomics in the kernel).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int), 0);

    // Choose a balanced launch configuration for A100/H100:
    // - 256 threads per block (8 warps) gives good occupancy without excessive shared atomic contention.
    // - Blocks = min(ceil(N / (TPB * itemsPerThread)), SMs * 8) to saturate GPU while keeping
    //   the number of global atomicAdd reductions reasonable (one per bin per block).
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    const int threadsPerBlock = 256;
    const unsigned int blockWork = threadsPerBlock * itemsPerThread;
    unsigned int blocksFromInput = (inputSize + blockWork - 1) / blockWork;
    unsigned int maxBlocks = static_cast<unsigned int>(prop.multiProcessorCount) * 8u;
    if (maxBlocks == 0) maxBlocks = 1;
    unsigned int numBlocks = blocksFromInput > 0 ? (blocksFromInput < maxBlocks ? blocksFromInput : maxBlocks) : 1;

    // Dynamic shared memory size: 32 copies per bin.
    size_t sharedBytes = static_cast<size_t>(rangeLen) * 32u * sizeof(unsigned int);

    histogramKernel_range32banks<<<numBlocks, threadsPerBlock, sharedBytes>>>(input, inputSize, histogram, from, to);
}