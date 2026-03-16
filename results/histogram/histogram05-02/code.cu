#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Range-restricted histogram for device-resident text.
//
// Design notes:
//   * The input is processed by a persistent grid with a grid-stride loop so that
//     each block handles a large slice of a large file. That strongly amortizes
//     per-block shared-memory initialization and flush costs.
//   * Each block accumulates into a block-private histogram in shared memory.
//   * To avoid shared-memory bank conflicts, the shared histogram is replicated
//     32 times per block, matching the 32 shared-memory banks on modern NVIDIA
//     GPUs. The stride of each replica is forced to be odd so that
//       bank(copy * stride + bin)
//     becomes a permutation of the 32 banks as `copy` varies.
//   * After the input slice is processed, the 32 replicas are reduced and each
//     block performs only one global atomic add per output bin.
//
// The kernel is intentionally specialized for a fixed 256-thread block size.
// That matches the maximum 256 histogram bins and lets the flush phase map one
// thread to one bin in the worst case.

constexpr int blockSize = 256;

// One aligned 128-bit load per thread is a strong default on A100/H100 for a
// byte-stream histogram. It keeps the load side fully coalesced, keeps register
// pressure modest, and is large enough to amortize loop/control overhead.
constexpr int itemsPerThread = 16;

// 32 replicas = 32 shared-memory banks on Ampere/Hopper.
constexpr int histogramCopies = 32;

static_assert(blockSize == 256,
              "This implementation is tuned for exactly 256 threads per block.");
static_assert(itemsPerThread > 0 && (itemsPerThread % 16 == 0),
              "itemsPerThread must be a positive multiple of 16 for aligned uint4 loads.");

__host__ __device__ constexpr unsigned int padded_hist_stride(unsigned int rangeSize) {
    // Forcing the per-replica stride to be odd guarantees gcd(stride, 32) == 1,
    // so the same logical bin in different replicas maps to different banks.
    return rangeSize | 1u;
}

__device__ __forceinline__ void accumulate_char(unsigned int value,
                                                unsigned int from,
                                                unsigned int rangeSize,
                                                unsigned int* localHist) {
    // Unsigned subtraction plus a single bounds check covers both
    //   value < from
    // and
    //   value > to
    // without two comparisons.
    const unsigned int bin = value - from;
    if (bin < rangeSize) {
        atomicAdd(localHist + bin, 1u);
    }
}

__device__ __forceinline__ void accumulate_packed4(uint32_t packed,
                                                   unsigned int from,
                                                   unsigned int rangeSize,
                                                   unsigned int* localHist) {
    accumulate_char((packed      ) & 0xFFu, from, rangeSize, localHist);
    accumulate_char((packed >>  8) & 0xFFu, from, rangeSize, localHist);
    accumulate_char((packed >> 16) & 0xFFu, from, rangeSize, localHist);
    accumulate_char((packed >> 24) & 0xFFu, from, rangeSize, localHist);
}

__global__ __launch_bounds__(blockSize)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from,
                            unsigned int rangeSize) {
    extern __shared__ unsigned int sharedHist[];

    const unsigned int stride = padded_hist_stride(rangeSize);
    const unsigned int tid    = threadIdx.x;
    const unsigned int lane   = tid & 31u;
    const unsigned int warp   = tid >> 5;

    // Every lane in a warp uses a different replica, and the warp-dependent
    // rotation prevents identical lane IDs from all warps from always hitting
    // the same replica.
    const unsigned int copy = (lane + warp) & (histogramCopies - 1u);
    unsigned int* const localHist = sharedHist + copy * stride;

    const unsigned int totalSharedBins =
        static_cast<unsigned int>(histogramCopies) * stride;

    for (unsigned int i = tid; i < totalSharedBins; i += blockSize) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    const uint64_t n = static_cast<uint64_t>(inputSize);
    uint64_t idx =
        (static_cast<uint64_t>(blockIdx.x) * blockSize + tid) * itemsPerThread;
    const uint64_t gridStride =
        static_cast<uint64_t>(gridDim.x) * blockSize * itemsPerThread;

    if (n >= static_cast<uint64_t>(itemsPerThread)) {
        const uint64_t fullLimit = n - static_cast<uint64_t>(itemsPerThread);

        // The API contract says `input` is a cudaMalloc allocation. cudaMalloc
        // provides sufficient alignment for 16-byte vector loads, and the per-
        // thread starting offset is also a multiple of 16 because itemsPerThread
        // is constrained to a multiple of 16.
        const uint4* const input4 = reinterpret_cast<const uint4*>(input);

        for (; idx <= fullLimit; idx += gridStride) {
#pragma unroll
            for (int chunk = 0; chunk < itemsPerThread; chunk += 16) {
                const uint4 v = input4[(idx + static_cast<uint64_t>(chunk)) >> 4];
                accumulate_packed4(v.x, from, rangeSize, localHist);
                accumulate_packed4(v.y, from, rangeSize, localHist);
                accumulate_packed4(v.z, from, rangeSize, localHist);
                accumulate_packed4(v.w, from, rangeSize, localHist);
            }
        }
    }

    // After the full-chunk loop, each thread can have at most one partial tail.
    if (idx < n) {
        uint64_t end = idx + static_cast<uint64_t>(itemsPerThread);
        if (end > n) {
            end = n;
        }

        for (uint64_t i = idx; i < end; ++i) {
            // Cast through unsigned char so the histogram is always based on the
            // byte ordinal 0..255 regardless of host/compiler `char` signedness.
            accumulate_char(static_cast<unsigned char>(input[i]),
                            from,
                            rangeSize,
                            localHist);
        }
    }
    __syncthreads();

    // blockSize == 256 and rangeSize <= 256, so one thread can flush one bin.
    if (tid < rangeSize) {
        unsigned int sum = 0u;
        unsigned int offset = tid;

#pragma unroll
        for (int c = 0; c < histogramCopies; ++c) {
            sum += sharedHist[offset];
            offset += stride;
        }

        if (sum != 0u) {
            atomicAdd(histogram + tid, sum);
        }
    }
}

}  // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // Per the API contract, `input` and `histogram` are device pointers returned
    // by cudaMalloc, and `from`/`to` form a valid inclusive byte range.
    const unsigned int fromU     = static_cast<unsigned int>(from);
    const unsigned int rangeSize = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes  = static_cast<size_t>(rangeSize) * sizeof(unsigned int);

    // The kernel accumulates into the final output with atomicAdd, so start from
    // a clean output buffer. Stream 0 is used because the requested API does not
    // carry a stream parameter. The caller owns synchronization/completion.
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    const size_t sharedMemBytes =
        static_cast<size_t>(histogramCopies) *
        static_cast<size_t>(padded_hist_stride(rangeSize)) *
        sizeof(unsigned int);

    // Persistent-grid launch:
    // launch roughly one occupancy-limited wave of blocks and let every block
    // walk the file with a grid-stride loop. For large inputs this is better
    // than launching one tiny block per ~4 KiB chunk because it minimizes the
    // shared-memory init/flush overhead per byte processed.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 1;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount < 1) {
        smCount = 1;
    }

    int activeBlocksPerSM = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM,
        histogram_range_kernel,
        blockSize,
        sharedMemBytes);
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    const uint64_t workPerBlock =
        static_cast<uint64_t>(blockSize) * static_cast<uint64_t>(itemsPerThread);

    uint64_t requiredBlocks =
        (static_cast<uint64_t>(inputSize) + workPerBlock - 1u) / workPerBlock;
    if (requiredBlocks == 0u) {
        requiredBlocks = 1u;
    }

    uint64_t gridSize64 =
        static_cast<uint64_t>(smCount) * static_cast<uint64_t>(activeBlocksPerSM);
    if (gridSize64 > requiredBlocks) {
        gridSize64 = requiredBlocks;
    }
    if (gridSize64 == 0u) {
        gridSize64 = 1u;
    }

    histogram_range_kernel<<<static_cast<int>(gridSize64), blockSize, sharedMemBytes>>>(
        input,
        histogram,
        inputSize,
        fromU,
        rangeSize);
}