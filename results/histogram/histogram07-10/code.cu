#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Restricted-range text histogram for modern NVIDIA data-center GPUs:
//  1) Each block builds a block-private histogram in shared memory.
//  2) The shared histogram is replicated 32 times and bank-striped so that
//     bin i of copy c is stored at offset i * 32 + c.
//  3) Each thread always updates copy (threadIdx.x % 32), eliminating intra-warp
//     shared-memory bank conflicts even when many threads hit the same bin.
//  4) Blocks process the input in a persistent grid-stride loop and merge their
//     block-local histograms into the global output with one atomic add per bin.
//
// Controls how many input characters each thread processes per grid-stride
// iteration. For large inputs on A100/H100-class GPUs, 16 is a good default:
// it maps the hot path to one aligned 16-byte uint4 load per thread while
// keeping shared-memory atomic pressure and register use in a good balance.
constexpr int itemsPerThread = 16;

namespace {

// 256 threads = 8 warps, which is a good compromise between occupancy and
// per-copy contention, and it also matches the maximum possible histogram
// length (256 bins) so the epilogue can assign one thread to one bin.
constexpr int kBlockThreads = 256;

// One copy per warp lane / shared-memory bank.
constexpr unsigned int kHistogramCopies = 32u;
constexpr unsigned int kLaneMask        = kHistogramCopies - 1u;

// Vector width for the input hot path: one uint4 = 16 bytes.
constexpr int kBytesPerVector  = 16;
constexpr int kVectorsPerThread = itemsPerThread / kBytesPerVector;

// Bytes processed by one block per grid-stride iteration.
constexpr size_t kBlockWork = static_cast<size_t>(kBlockThreads) * itemsPerThread;

static_assert(sizeof(unsigned int) == 4,
              "The bank-striped shared-memory layout assumes 4-byte counters.");
static_assert(sizeof(uint4) == kBytesPerVector, "Unexpected uint4 size.");
static_assert(kHistogramCopies == 32u,
              "The bank-striped layout is defined for exactly 32 copies.");
static_assert(kBlockThreads % 32 == 0,
              "Block size should be a whole number of warps.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(itemsPerThread % kBytesPerVector == 0,
              "itemsPerThread must be a multiple of 16 for the uint4 input path.");

__device__ __forceinline__
void accumulate_byte(const unsigned int value,
                     unsigned int* const threadCopy,
                     const unsigned int rangeBegin,
                     const unsigned int rangeLen)
{
    // Unsigned subtraction turns the inclusive range check
    // [rangeBegin, rangeBegin + rangeLen - 1] into one comparison.
    // Values below rangeBegin underflow to a large unsigned number and fail.
    const unsigned int bin = value - rangeBegin;
    if (bin < rangeLen) {
        // threadCopy already points at this thread's copy column (copy = lane),
        // so stepping by 32 moves between bins while keeping the same copy:
        // address = (bin * 32) + lane.
        atomicAdd(&threadCopy[bin * kHistogramCopies], 1u);
    }
}

__device__ __forceinline__
void accumulate_packed_u32(const uint32_t packed,
                           unsigned int* const threadCopy,
                           const unsigned int rangeBegin,
                           const unsigned int rangeLen)
{
    // CUDA GPUs are little-endian, so the least-significant byte is the
    // earliest character in memory for the 32-bit word loaded through uint4.
    accumulate_byte( packed        & 0xFFu, threadCopy, rangeBegin, rangeLen);
    accumulate_byte((packed >>  8) & 0xFFu, threadCopy, rangeBegin, rangeLen);
    accumulate_byte((packed >> 16) & 0xFFu, threadCopy, rangeBegin, rangeLen);
    accumulate_byte((packed >> 24) & 0xFFu, threadCopy, rangeBegin, rangeLen);
}

__device__ __forceinline__
void accumulate_uint4(const uint4 packed16,
                      unsigned int* const threadCopy,
                      const unsigned int rangeBegin,
                      const unsigned int rangeLen)
{
    accumulate_packed_u32(packed16.x, threadCopy, rangeBegin, rangeLen);
    accumulate_packed_u32(packed16.y, threadCopy, rangeBegin, rangeLen);
    accumulate_packed_u32(packed16.z, threadCopy, rangeBegin, rangeLen);
    accumulate_packed_u32(packed16.w, threadCopy, rangeBegin, rangeLen);
}

__global__ __launch_bounds__(kBlockThreads)
void histogram_kernel(const char* __restrict__ input,
                      unsigned int* __restrict__ histogram,
                      const unsigned int inputSize,
                      const unsigned int rangeBegin,
                      const unsigned int rangeLen)
{
    // Dynamic shared storage for 32 bank-striped copies of only the active range.
    // Layout: sharedHist[bin * 32 + copy].
    extern __shared__ unsigned int sharedHist[];

    const unsigned int tid  = static_cast<unsigned int>(threadIdx.x);
    const size_t tid64      = static_cast<size_t>(tid);
    const unsigned int lane = tid & kLaneMask;  // Equivalent to threadIdx.x % 32.

    // Each thread always targets its lane-selected copy. Within a warp, if all
    // threads hit the same bin, they still touch 32 distinct banks because the
    // copy index differs by lane. Threads from different warps but the same lane
    // still share a copy, so shared-memory atomics remain necessary.
    unsigned int* const threadCopy = sharedHist + lane;

    const unsigned int sharedBins = rangeLen * kHistogramCopies;

    // Cooperative shared-memory initialization for the privatized histogram.
    for (unsigned int i = tid; i < sharedBins; i += kBlockThreads) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    // Interpret input as bytes in [0, 255] regardless of whether host-side
    // char is signed or unsigned.
    const unsigned char* const inputBytes = reinterpret_cast<const unsigned char*>(input);

    // The problem statement says the buffers are cudaMalloc-allocated; that
    // alignment is sufficient for the aligned uint4 load path below.
    const uint4* const inputVec = reinterpret_cast<const uint4*>(input);

    const size_t inputSizeSz = static_cast<size_t>(inputSize);
    const size_t gridStride  = static_cast<size_t>(gridDim.x) * kBlockWork;

    // The hot path updates shared memory only; global memory is touched just
    // once per bin in the epilogue.
    size_t chunkBase = static_cast<size_t>(blockIdx.x) * kBlockWork;

    // Full chunks: no bounds checks in the vectorized hot path.
    for (; chunkBase + kBlockWork <= inputSizeSz; chunkBase += gridStride) {
        const size_t vectorBase = chunkBase / kBytesPerVector;

        #pragma unroll
        for (int vec = 0; vec < kVectorsPerThread; ++vec) {
            const size_t vectorIndex =
                vectorBase + tid64 + static_cast<size_t>(vec) * kBlockThreads;
            const uint4 packed = inputVec[vectorIndex];
            accumulate_uint4(packed, threadCopy, rangeBegin, rangeLen);
        }
    }

    // Optional final partial chunk for this block.
    if (chunkBase < inputSizeSz) {
        const size_t vectorBase = chunkBase / kBytesPerVector;

        #pragma unroll
        for (int vec = 0; vec < kVectorsPerThread; ++vec) {
            const size_t vectorIndex =
                vectorBase + tid64 + static_cast<size_t>(vec) * kBlockThreads;
            const size_t byteIndex = vectorIndex * kBytesPerVector;

            if (byteIndex + kBytesPerVector <= inputSizeSz) {
                const uint4 packed = inputVec[vectorIndex];
                accumulate_uint4(packed, threadCopy, rangeBegin, rangeLen);
            } else {
                // The very end of the input may not fill a whole uint4.
                #pragma unroll
                for (int byte = 0; byte < kBytesPerVector; ++byte) {
                    const size_t idx = byteIndex + static_cast<size_t>(byte);
                    if (idx < inputSizeSz) {
                        accumulate_byte(static_cast<unsigned int>(inputBytes[idx]),
                                        threadCopy,
                                        rangeBegin,
                                        rangeLen);
                    }
                }
            }
        }
    }

    __syncthreads();

    // Final merge: because the legal range has at most 256 bins and the block
    // size is 256, one thread can reduce one bin across the 32 shared copies.
    if (tid < rangeLen) {
        const unsigned int base = tid * kHistogramCopies;
        unsigned int sum = 0u;

        #pragma unroll
        for (unsigned int copy = 0; copy < kHistogramCopies; ++copy) {
            sum += sharedHist[base + copy];
        }

        // One global atomic add per bin per block.
        if (sum != 0u) {
            atomicAdd(&histogram[tid], sum);
        }
    }
}

}  // namespace

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Histogram length is the inclusive character range [from, to].
    const unsigned int rangeBegin = static_cast<unsigned int>(from);
    const unsigned int rangeLen   = static_cast<unsigned int>(to - from + 1);

    // The kernel merges block-local histograms into global memory with atomic
    // adds, so the output must be cleared first. This is enqueued on stream 0;
    // the interface has no stream parameter and the caller requested that
    // synchronization be handled externally.
    const size_t histogramBytes = static_cast<size_t>(rangeLen) * sizeof(unsigned int);
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    // Dynamic shared memory for 32 copies of only the active histogram range.
    // Worst case: 256 bins * 32 copies * 4 bytes = 32 KiB, so no opt-in for
    // >48 KiB dynamic shared memory is needed.
    const size_t sharedBytes = histogramBytes * kHistogramCopies;

    // Persistent-CTA style launch: cap the grid at the number of concurrently
    // resident blocks. That keeps the GPU full while minimizing per-block
    // shared-histogram init/finalize overhead and reducing the number of global
    // atomic merges in the epilogue.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM,
                                                  histogram_kernel,
                                                  kBlockThreads,
                                                  sharedBytes);

    const int maxResidentBlocks =
        (smCount > 0 && blocksPerSM > 0) ? (smCount * blocksPerSM) : 1;

    const uint64_t blockWork = static_cast<uint64_t>(kBlockWork);
    const uint64_t blocksNeeded =
        (static_cast<uint64_t>(inputSize) + blockWork - 1ULL) / blockWork;

    const uint64_t launchBlocks64 =
        (blocksNeeded < static_cast<uint64_t>(maxResidentBlocks))
            ? blocksNeeded
            : static_cast<uint64_t>(maxResidentBlocks);

    int gridBlocks = static_cast<int>(launchBlocks64);
    if (gridBlocks < 1) {
        gridBlocks = 1;
    }

    // Launch on stream 0; caller owns synchronization and any post-launch
    // error checking.
    histogram_kernel<<<gridBlocks, kBlockThreads, sharedBytes>>>(
        input,
        histogram,
        inputSize,
        rangeBegin,
        rangeLen);
}