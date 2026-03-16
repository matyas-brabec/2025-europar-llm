#include <cuda_runtime.h>

namespace {

// One histogram replica per warp lane / shared-memory bank.
constexpr int kWarpLanes = 32;
constexpr int kHistogramReplicas = 32;

// 256 threads = 8 warps per block. This is a good fit for A100/H100-class GPUs while
// keeping the worst-case shared-memory footprint (256 bins * 32 replicas * 4 B = 32 KiB)
// low enough to allow multiple resident blocks per SM.
constexpr int kBlockSize = 256;
constexpr int kWarpsPerBlock = kBlockSize / kWarpLanes;

// Large-input default for modern NVIDIA GPUs: enough work per thread to amortize shared
// histogram setup/finalization, while still keeping the loop compact and fully unrollable.
constexpr int itemsPerThread = 16;
constexpr unsigned int kBlockWork = kBlockSize * itemsPerThread;

// This kernel has non-trivial per-block overhead (shared histogram zeroing + final reduction).
// Around 32 active warps/SM is typically enough on recent data-center GPUs, so we cap the
// launch at 4 blocks/SM rather than blindly launching every possible resident block.
constexpr int kTargetResidentBlocksPerSM = 4;

static_assert(kWarpLanes == 32, "This kernel relies on NVIDIA's 32-lane warp size.");
static_assert(kHistogramReplicas == kWarpLanes, "Use one replica per warp lane.");
static_assert((kBlockSize % kWarpLanes) == 0, "Block size must be a multiple of warp size.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int value) {
    #pragma unroll
    for (int offset = kWarpLanes / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/*
  Shared-memory layout:
      s_hist[bin * 32 + replica]

  For 32-bit words, the shared-memory bank is (word_index % 32). With the above layout,
  replica r of every bin always maps to bank r. Threads use replica == lane, so within a warp
  every lane touches a distinct bank regardless of which bin it increments:
      bank(bin * 32 + lane) == lane

  This is the requested 32-copy histogram privatization scheme:
    - 32 total copies of the histogram per thread block
    - each copy is associated with one warp lane / one bank
    - updates are bank-conflict-free within a warp
    - reduction is also bank-conflict-free by having a warp read all 32 replicas of one bin
*/
__global__ __launch_bounds__(kBlockSize)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int rangeSize) {
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid    = threadIdx.x;
    const unsigned int lane   = tid & (kWarpLanes - 1u);
    const unsigned int warpId = tid >> 5;

    const unsigned int replicatedBins = rangeSize * kHistogramReplicas;

    // Zero the block-private histogram. Dynamic shared memory keeps the footprint proportional
    // to the requested [from, to] range instead of always reserving the full 256-bin table.
    for (unsigned int i = tid; i < replicatedBins; i += kBlockSize) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Process the input in a blocked, fully coalesced pattern:
    // on each inner-loop iteration, threads in the block load a contiguous kBlockSize-byte span.
    // 64-bit index arithmetic is used so the code remains correct up to the full 4 GiB inputSize limit.
    const size_t n          = static_cast<size_t>(inputSize);
    const size_t blockWork  = static_cast<size_t>(kBlockWork);
    const size_t base0      = static_cast<size_t>(blockIdx.x) * blockWork + static_cast<size_t>(tid);
    const size_t gridStride = static_cast<size_t>(gridDim.x) * blockWork;

    for (size_t base = base0; base < n; base += gridStride) {
        #pragma unroll
        for (int i = 0; i < itemsPerThread; ++i) {
            const size_t idx = base + static_cast<size_t>(i) * kBlockSize;
            if (idx < n) {
                // Cast through unsigned char so byte values 128..255 are handled correctly even
                // when plain char is signed.
                const unsigned int c = static_cast<unsigned char>(input[idx]);

                // Single unsigned range test:
                //   bin = c - from
                //   bin < rangeSize  <=>  from <= c <= to
                const unsigned int bin = c - from;
                if (bin < rangeSize) {
                    atomicAdd(&s_hist[bin * kHistogramReplicas + lane], 1u);
                }
            }
        }
    }
    __syncthreads();

    // Reduce the 32 lane-private replicas. One warp reduces one bin at a time so lane r reads
    // replica r, which again maps one lane to one bank with no intra-warp bank conflicts.
    for (unsigned int bin = warpId; bin < rangeSize; bin += kWarpsPerBlock) {
        unsigned int sum = s_hist[bin * kHistogramReplicas + lane];
        sum = warp_reduce_sum(sum);

        if (lane == 0u && sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

} // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // The interface does not accept a stream, so work is enqueued on the default stream.
    // The caller is responsible for any synchronization/error handling policy.
    cudaStream_t stream = 0;

    const unsigned int rangeSize = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes  = static_cast<size_t>(rangeSize) * sizeof(unsigned int);

    // The kernel accumulates block results into the output via global atomics, so the output
    // histogram must be cleared before launch.
    cudaMemsetAsync(histogram, 0, histogramBytes, stream);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedBytes =
        static_cast<size_t>(rangeSize) * static_cast<size_t>(kHistogramReplicas) * sizeof(unsigned int);

    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int maxActiveBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM,
        histogram_range_kernel,
        kBlockSize,
        sharedBytes);

    if (maxActiveBlocksPerSM < 1) {
        maxActiveBlocksPerSM = 1;
    }

    int blocksPerSM = maxActiveBlocksPerSM;
    if (blocksPerSM > kTargetResidentBlocksPerSM) {
        blocksPerSM = kTargetResidentBlocksPerSM;
    }

    const unsigned int requiredBlocks = 1u + (inputSize - 1u) / kBlockWork;

    unsigned int launchBlocks = static_cast<unsigned int>(smCount * blocksPerSM);
    if (launchBlocks == 0u) {
        launchBlocks = 1u;
    }
    if (launchBlocks > requiredBlocks) {
        launchBlocks = requiredBlocks;
    }

    histogram_range_kernel<<<launchBlocks, kBlockSize, sharedBytes, stream>>>(
        input,
        histogram,
        inputSize,
        static_cast<unsigned int>(from),
        rangeSize);
}