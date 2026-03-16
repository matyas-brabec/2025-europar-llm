#include <cuda_runtime.h>

namespace {

// Range-restricted byte histogram.
// Output bin i counts the occurrences of byte value (from + i).
//
// Tuning choices aimed at large inputs on modern NVIDIA data-center GPUs:
//   - blockThreads = 256: good balance between occupancy and intra-block contention.
//   - itemsPerThread = 16: one aligned 16-byte vector load (`uint4`) per full iteration.
//   - sharedHistogramCopies = 32: one shared histogram replica per warp lane.
//     With an odd replica stride, the same logical bin in consecutive replicas maps to
//     different shared-memory banks, eliminating intra-warp bank conflicts for hot bins.
//   - Grid-stride processing: launch only an occupancy-sized wave of blocks and let each
//     block iterate over the input, which keeps reduction traffic to global memory low.
constexpr int blockThreads = 256;
constexpr int itemsPerThread = 16;
constexpr int sharedHistogramCopies = 32;
constexpr int minBlocksPerSM = 4;
constexpr int vectorLoadBytes = static_cast<int>(sizeof(uint4));

static_assert(blockThreads > 0, "blockThreads must be positive.");
static_assert((blockThreads & 31) == 0, "blockThreads must be a multiple of 32.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert((itemsPerThread % vectorLoadBytes) == 0,
              "itemsPerThread must be a multiple of 16 to preserve aligned uint4 loads.");
static_assert((sharedHistogramCopies & (sharedHistogramCopies - 1)) == 0,
              "sharedHistogramCopies must be a power of two.");
static_assert(sharedHistogramCopies == 32,
              "This kernel is tuned for modern NVIDIA GPUs with 32-lane warps and 32 shared-memory banks.");

__host__ __device__ __forceinline__
unsigned int paddedHistStride(const unsigned int numBins) {
    // Smallest odd stride >= numBins.
    //
    // Shared-memory bank selection is address modulo 32. If the stride between
    // consecutive histogram replicas is odd, it is coprime with 32, so for any
    // logical bin `b` the addresses
    //   b, b + stride, b + 2*stride, ..., b + 31*stride
    // hit all 32 banks exactly once. This is the key bank-conflict avoidance trick.
    return numBins | 1u;
}

__device__ __forceinline__
void addByteToSharedReplica(const unsigned int value,
                            unsigned int* const replica,
                            const unsigned int from,
                            const unsigned int numBins) {
    // Inclusive range check written as subtract + unsigned compare.
    const unsigned int bin = value - from;
    if (bin < numBins) {
        atomicAdd(replica + bin, 1u);
    }
}

__device__ __forceinline__
void addPackedWordToSharedReplica(const unsigned int packed,
                                  unsigned int* const replica,
                                  const unsigned int from,
                                  const unsigned int numBins) {
    addByteToSharedReplica((packed      ) & 0xFFu, replica, from, numBins);
    addByteToSharedReplica((packed >>  8) & 0xFFu, replica, from, numBins);
    addByteToSharedReplica((packed >> 16) & 0xFFu, replica, from, numBins);
    addByteToSharedReplica((packed >> 24) & 0xFFu, replica, from, numBins);
}

__launch_bounds__(blockThreads, minBlocksPerSM)
__global__
void histogramKernel(const char* __restrict__ input,
                     unsigned int* __restrict__ histogram,
                     unsigned int inputSize,
                     int from,
                     int to) {
    extern __shared__ unsigned int sharedHist[];

    const unsigned int blockThreadsU = static_cast<unsigned int>(blockThreads);
    const unsigned int itemsPerThreadU = static_cast<unsigned int>(itemsPerThread);
    const unsigned int fromU = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const unsigned int stride = paddedHistStride(numBins);
    const unsigned int totalSharedWords =
        static_cast<unsigned int>(sharedHistogramCopies) * stride;

    // Replica selection by lane id. Because there are 32 replicas, the low 5 bits of
    // threadIdx.x choose the copy. Threads in the same warp therefore update different
    // replicas for the same logical bin.
    const unsigned int copyId =
        threadIdx.x & static_cast<unsigned int>(sharedHistogramCopies - 1);
    unsigned int* const replica = sharedHist + copyId * stride;

    // Cooperative initialization of all histogram replicas.
    for (unsigned int i = threadIdx.x; i < totalSharedWords; i += blockThreadsU) {
        sharedHist[i] = 0u;
    }
    __syncthreads();

    // Interpret input as unsigned bytes so ordinal values 128..255 are handled correctly
    // even if plain `char` is signed.
    const unsigned char* const bytes =
        reinterpret_cast<const unsigned char*>(input);

    // Grid-stride loop over large inputs. The host intentionally launches only enough
    // blocks to saturate the GPU; this loop lets those blocks cover the full buffer.
    const unsigned int threadBase =
        (blockIdx.x * blockThreadsU + threadIdx.x) * itemsPerThreadU;
    const unsigned int gridStride =
        gridDim.x * blockThreadsU * itemsPerThreadU;

    for (unsigned int index = threadBase; index < inputSize; index += gridStride) {
        const unsigned int remaining = inputSize - index;

        if (remaining >= itemsPerThreadU) {
            // Full chunk: aligned vector loads. cudaMalloc provides sufficient base
            // alignment, and the per-thread offset is always a multiple of 16 because
            // itemsPerThread is constrained to multiples of sizeof(uint4).
            const uint4* const vec =
                reinterpret_cast<const uint4*>(bytes + index);

            #pragma unroll
            for (int i = 0; i < itemsPerThread / vectorLoadBytes; ++i) {
                const uint4 v = vec[i];
                addPackedWordToSharedReplica(v.x, replica, fromU, numBins);
                addPackedWordToSharedReplica(v.y, replica, fromU, numBins);
                addPackedWordToSharedReplica(v.z, replica, fromU, numBins);
                addPackedWordToSharedReplica(v.w, replica, fromU, numBins);
            }
        } else {
            // Tail handling for the last partial chunk assigned to this thread.
            #pragma unroll
            for (int i = 0; i < itemsPerThread; ++i) {
                if (static_cast<unsigned int>(i) < remaining) {
                    addByteToSharedReplica(bytes[index + static_cast<unsigned int>(i)],
                                           replica,
                                           fromU,
                                           numBins);
                }
            }
        }
    }
    __syncthreads();

    // Reduce the 32 replicas into the final global histogram. Only one global atomic
    // per bin and block remains; all per-byte updates stayed local to shared memory.
    for (unsigned int bin = threadIdx.x; bin < numBins; bin += blockThreadsU) {
        unsigned int sum = 0u;
        const unsigned int* ptr = sharedHist + bin;

        #pragma unroll
        for (int copy = 0; copy < sharedHistogramCopies; ++copy) {
            sum += *ptr;
            ptr += stride;
        }

        if (sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

} // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // The problem statement guarantees a valid inclusive range:
    //   0 <= from < to <= 255
    // Output bin 0 corresponds to byte value `from`.
    const unsigned int numBins = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes =
        static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Enqueue the output clear on the default stream. The caller explicitly owns any
    // synchronization policy, so this function remains asynchronous with respect to host.
    cudaMemsetAsync(histogram, 0, histogramBytes, 0);

    if (inputSize == 0u) {
        return;
    }

    const unsigned int stride = paddedHistStride(numBins);
    const size_t sharedMemBytes =
        static_cast<size_t>(sharedHistogramCopies) *
        static_cast<size_t>(stride) *
        sizeof(unsigned int);

    // Launch just enough blocks to fill the machine for the current shared-memory
    // footprint. Launching many more blocks would only increase the final global
    // reduction traffic because the kernel already uses a grid-stride loop.
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
        histogramKernel,
        blockThreads,
        sharedMemBytes);
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    int gridSize = smCount * activeBlocksPerSM;

    const size_t blockWork =
        static_cast<size_t>(blockThreads) * static_cast<size_t>(itemsPerThread);
    const size_t blocksNeeded =
        (static_cast<size_t>(inputSize) + blockWork - 1u) / blockWork;

    if (static_cast<size_t>(gridSize) > blocksNeeded) {
        gridSize = static_cast<int>(blocksNeeded);
    }
    if (gridSize < 1) {
        gridSize = 1;
    }

    histogramKernel<<<gridSize, blockThreads, sharedMemBytes, 0>>>(
        input,
        histogram,
        inputSize,
        from,
        to);

    // No synchronization and no explicit error handling here by design:
    // the fixed signature returns void, and the caller owns sync/error policy.
}