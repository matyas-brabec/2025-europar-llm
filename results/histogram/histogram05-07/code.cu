#include <cuda_runtime.h>
#include <cstdint>

namespace histogram_detail {

// Tuned defaults for large inputs on modern data-center GPUs (A100/H100).
// - 256 threads/block gives 8 warps/block, enough parallelism while keeping inter-warp
//   contention on the shared histogram replicas moderate.
// - 16 items/thread maps naturally to one 128-bit (uint4) load per thread on the fast path,
//   amortizing shared-memory init/final-reduction overhead without bloating register usage.
constexpr int itemsPerThread   = 16;
constexpr int threadsPerBlock  = 256;

// One replica per warp lane / shared-memory bank.
// Together with a bank-skewed stride, this spreads updates across banks and sharply reduces
// shared-memory conflicts for hot bins.
constexpr int histogramReplicas = 32;

// Size of one block-sized chunk of input processed per grid-stride iteration.
constexpr unsigned int blockWork =
    static_cast<unsigned int>(threadsPerBlock * itemsPerThread);

// Vectorized fast path uses one uint4 load per 16 input bytes.
constexpr int vectorLoadBytes = static_cast<int>(sizeof(uint4));
constexpr bool vectorizedLoadPathAvailable =
    (itemsPerThread % vectorLoadBytes) == 0;

static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert(threadsPerBlock >= 256,
              "threadsPerBlock must be at least 256 so the epilogue can cover all possible bins.");
static_assert((threadsPerBlock & 31) == 0,
              "threadsPerBlock must be a multiple of 32.");
static_assert(histogramReplicas == 32,
              "This implementation assumes one shared-memory replica per warp lane.");
static_assert((histogramReplicas & (histogramReplicas - 1)) == 0,
              "histogramReplicas must be a power of two.");

// Warp-aggregated update into the lane-private shared-memory replica.
// All lanes participate. Lanes that target the same histogram bin are collapsed with
// __match_any_sync(), and only the first lane in each equivalence class issues the atomicAdd
// with the population count of the class. This is especially effective on text inputs, where
// repeated characters are common.
//
// On Volta+ (and therefore on A100/H100), __match_any_sync() is available and very fast.
// A conservative fallback is kept for older architectures so the code remains correct there.
__device__ __forceinline__
void accumulate_byte_warp_aggregated(unsigned int byteValue,
                                     bool present,
                                     unsigned int* replica,
                                     unsigned int lane,
                                     unsigned int from,
                                     unsigned int numBins)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    constexpr unsigned int invalidKey = 0xFFFFFFFFu;

    const unsigned int bin = byteValue - from;
    const unsigned int key = (present && bin < numBins) ? bin : invalidKey;

    const unsigned int peers  = __match_any_sync(__activemask(), key);
    const unsigned int leader = static_cast<unsigned int>(__ffs(static_cast<int>(peers)) - 1);

    if (key != invalidKey && lane == leader) {
        atomicAdd(replica + bin, static_cast<unsigned int>(__popc(peers)));
    }
#else
    if (present) {
        const unsigned int bin = byteValue - from;
        if (bin < numBins) {
            atomicAdd(replica + bin, 1u);
        }
    }
#endif
}

// Process four bytes packed in one 32-bit word.
__device__ __forceinline__
void accumulate_packed_word(unsigned int word,
                            unsigned int* replica,
                            unsigned int lane,
                            unsigned int from,
                            unsigned int numBins)
{
    accumulate_byte_warp_aggregated((word >>  0) & 0xFFu, true, replica, lane, from, numBins);
    accumulate_byte_warp_aggregated((word >>  8) & 0xFFu, true, replica, lane, from, numBins);
    accumulate_byte_warp_aggregated((word >> 16) & 0xFFu, true, replica, lane, from, numBins);
    accumulate_byte_warp_aggregated((word >> 24) & 0xFFu, true, replica, lane, from, numBins);
}

// Common prologue: zero all shared-memory replicas.
template <int BLOCK_SIZE>
__device__ __forceinline__
void clear_shared_hist(unsigned int* s_hist, unsigned int totalWords)
{
    for (unsigned int i = threadIdx.x; i < totalWords; i += BLOCK_SIZE) {
        s_hist[i] = 0u;
    }
    __syncthreads();
}

// Common epilogue: reduce all per-block replicas into the global histogram.
// Because the output has at most 256 bins, one thread per output bin is enough.
template <int REPLICAS>
__device__ __forceinline__
void flush_shared_hist(const unsigned int* s_hist,
                       unsigned int* __restrict__ histogram,
                       unsigned int numBins,
                       unsigned int smemStride)
{
    __syncthreads();

    const unsigned int bin = threadIdx.x;
    if (bin < numBins) {
        const unsigned int* replicaBin = s_hist + bin;

        unsigned int sum = 0u;
#pragma unroll
        for (int r = 0; r < REPLICAS; ++r) {
            sum += replicaBin[static_cast<unsigned int>(r) * smemStride];
        }

        if (sum != 0u) {
            atomicAdd(histogram + bin, sum);
        }
    }
}

// Fast path: requires ITEMS_PER_THREAD to be a multiple of 16 and the input pointer
// to be 16-byte aligned. With the default itemsPerThread == 16, each thread issues
// exactly one aligned uint4 load per block-chunk iteration.
template <int BLOCK_SIZE, int ITEMS_PER_THREAD, int REPLICAS>
__global__ __launch_bounds__(BLOCK_SIZE)
void histogram_kernel_vectorized(const char* __restrict__ input,
                                 unsigned int* __restrict__ histogram,
                                 unsigned int inputSize,
                                 unsigned int from,
                                 unsigned int numBins,
                                 unsigned int smemStride)
{
    static_assert((BLOCK_SIZE & 31) == 0,
                  "BLOCK_SIZE must be a multiple of 32.");
    static_assert(BLOCK_SIZE >= 256,
                  "BLOCK_SIZE must be at least 256.");
    static_assert(REPLICAS == 32,
                  "This kernel assumes one shared-memory replica per warp lane.");
    static_assert((ITEMS_PER_THREAD % vectorLoadBytes) == 0,
                  "Vectorized kernel requires ITEMS_PER_THREAD to be a multiple of 16.");

    extern __shared__ unsigned int s_hist[];

    const unsigned char* const uinput =
        reinterpret_cast<const unsigned char*>(input);

    clear_shared_hist<BLOCK_SIZE>(
        s_hist, static_cast<unsigned int>(REPLICAS) * smemStride);

    const unsigned int lane = threadIdx.x & static_cast<unsigned int>(REPLICAS - 1);
    unsigned int* const replica = s_hist + lane * smemStride;

    // Use size_t for address arithmetic so the grid-stride stepping remains safe even
    // close to the 4 GiB limit implied by the unsigned int inputSize parameter.
    const size_t inputSizeBytes   = static_cast<size_t>(inputSize);
    const size_t localBlockWork   = static_cast<size_t>(BLOCK_SIZE) * ITEMS_PER_THREAD;
    const size_t gridStride       = static_cast<size_t>(gridDim.x) * localBlockWork;
    size_t       blockBase        = static_cast<size_t>(blockIdx.x) * localBlockWork;
    const size_t threadChunkBase  = static_cast<size_t>(threadIdx.x) * ITEMS_PER_THREAD;

    // Bulk path: the whole block-sized chunk is in range, so every vector load is valid
    // and, because blockBase and threadChunkBase are both multiples of 16, aligned.
    if (inputSizeBytes >= localBlockWork) {
        const size_t bulkLimit = inputSizeBytes - localBlockWork;

        for (; blockBase <= bulkLimit; blockBase += gridStride) {
            const size_t idx = blockBase + threadChunkBase;

#pragma unroll
            for (int vec = 0; vec < (ITEMS_PER_THREAD / vectorLoadBytes); ++vec) {
                const size_t vecOffset = static_cast<size_t>(vec) * vectorLoadBytes;
                const uint4 packed =
                    *reinterpret_cast<const uint4*>(uinput + idx + vecOffset);

                accumulate_packed_word(packed.x, replica, lane, from, numBins);
                accumulate_packed_word(packed.y, replica, lane, from, numBins);
                accumulate_packed_word(packed.z, replica, lane, from, numBins);
                accumulate_packed_word(packed.w, replica, lane, from, numBins);
            }
        }
    }

    // Tail path: at most one partial block-sized chunk remains for this block.
    // Every lane still participates in every warp-aggregation step; absent bytes are
    // represented by "present == false" and simply do not contribute.
    if (blockBase < inputSizeBytes) {
        const size_t idx = blockBase + threadChunkBase;

        unsigned int remaining = 0u;
        if (idx < inputSizeBytes) {
            const size_t tail = inputSizeBytes - idx;
            remaining = static_cast<unsigned int>(
                tail < static_cast<size_t>(ITEMS_PER_THREAD)
                    ? tail
                    : static_cast<size_t>(ITEMS_PER_THREAD));
        }

#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const bool present = static_cast<unsigned int>(item) < remaining;

            unsigned int byteValue = 0u;
            if (present) {
                byteValue = static_cast<unsigned int>(
                    uinput[idx + static_cast<size_t>(item)]);
            }

            accumulate_byte_warp_aggregated(
                byteValue, present, replica, lane, from, numBins);
        }
    }

    flush_shared_hist<REPLICAS>(s_hist, histogram, numBins, smemStride);
}

// Scalar fallback: used when the device pointer is not 16-byte aligned or when
// itemsPerThread is changed to a value that is not a multiple of 16.
template <int BLOCK_SIZE, int ITEMS_PER_THREAD, int REPLICAS>
__global__ __launch_bounds__(BLOCK_SIZE)
void histogram_kernel_scalar(const char* __restrict__ input,
                             unsigned int* __restrict__ histogram,
                             unsigned int inputSize,
                             unsigned int from,
                             unsigned int numBins,
                             unsigned int smemStride)
{
    static_assert((BLOCK_SIZE & 31) == 0,
                  "BLOCK_SIZE must be a multiple of 32.");
    static_assert(BLOCK_SIZE >= 256,
                  "BLOCK_SIZE must be at least 256.");
    static_assert(REPLICAS == 32,
                  "This kernel assumes one shared-memory replica per warp lane.");

    extern __shared__ unsigned int s_hist[];

    const unsigned char* const uinput =
        reinterpret_cast<const unsigned char*>(input);

    clear_shared_hist<BLOCK_SIZE>(
        s_hist, static_cast<unsigned int>(REPLICAS) * smemStride);

    const unsigned int lane = threadIdx.x & static_cast<unsigned int>(REPLICAS - 1);
    unsigned int* const replica = s_hist + lane * smemStride;

    const size_t inputSizeBytes = static_cast<size_t>(inputSize);
    const size_t localBlockWork = static_cast<size_t>(BLOCK_SIZE) * ITEMS_PER_THREAD;
    const size_t gridStride     = static_cast<size_t>(gridDim.x) * localBlockWork;
    size_t       blockBase      = static_cast<size_t>(blockIdx.x) * localBlockWork;
    const size_t threadBase     = static_cast<size_t>(threadIdx.x);

    // Bulk path: each block-chunk is processed in a transposed layout
    // (item * BLOCK_SIZE + threadIdx.x), which gives fully coalesced scalar loads.
    if (inputSizeBytes >= localBlockWork) {
        const size_t bulkLimit = inputSizeBytes - localBlockWork;

        for (; blockBase <= bulkLimit; blockBase += gridStride) {
            const size_t base = blockBase + threadBase;

#pragma unroll
            for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
                const unsigned int byteValue = static_cast<unsigned int>(
                    uinput[base + static_cast<size_t>(item) * BLOCK_SIZE]);

                accumulate_byte_warp_aggregated(
                    byteValue, true, replica, lane, from, numBins);
            }
        }
    }

    // Tail path for the final partial chunk.
    if (blockBase < inputSizeBytes) {
        const size_t base = blockBase + threadBase;

#pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            const size_t idx = base + static_cast<size_t>(item) * BLOCK_SIZE;
            const bool present = idx < inputSizeBytes;

            unsigned int byteValue = 0u;
            if (present) {
                byteValue = static_cast<unsigned int>(uinput[idx]);
            }

            accumulate_byte_warp_aggregated(
                byteValue, present, replica, lane, from, numBins);
        }
    }

    flush_shared_hist<REPLICAS>(s_hist, histogram, numBins, smemStride);
}

// The kernel uses a grid-stride loop with a uniform per-byte cost, so one occupancy-filling
// wave of blocks is sufficient. Launching many more blocks would mostly increase the number
// of final global atomic merges.
template <typename KernelFunc>
inline unsigned int compute_launch_blocks(KernelFunc kernel,
                                          size_t sharedMemBytes,
                                          unsigned int inputSize)
{
    int device = 0;
    int smCount = 1;
    int activeBlocksPerSM = 1;

    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSM, kernel, threadsPerBlock, sharedMemBytes);

    if (smCount < 1) {
        smCount = 1;
    }
    if (activeBlocksPerSM < 1) {
        activeBlocksPerSM = 1;
    }

    const unsigned int blocksForInput =
        inputSize / blockWork + static_cast<unsigned int>((inputSize % blockWork) != 0u);

    unsigned int numBlocks = blocksForInput;
    const unsigned int maxBlocks =
        static_cast<unsigned int>(smCount) *
        static_cast<unsigned int>(activeBlocksPerSM);

    if (numBlocks > maxBlocks) {
        numBlocks = maxBlocks;
    }
    if (numBlocks == 0u) {
        numBlocks = 1u;
    }

    return numBlocks;
}

// Compile-time dispatch:
// - if itemsPerThread is a multiple of 16, keep the vectorized fast path available and
//   choose it at runtime when the input pointer is aligned;
// - otherwise fall back to the scalar kernel unconditionally.
template <bool UseVectorizedFastPath>
struct launch_dispatch;

template <>
struct launch_dispatch<true> {
    static inline void launch(const char* input,
                              unsigned int* histogram,
                              unsigned int inputSize,
                              unsigned int from,
                              unsigned int numBins,
                              unsigned int smemStride,
                              size_t sharedMemBytes,
                              cudaStream_t stream)
    {
        // cudaMalloc() returns sufficiently aligned pointers, but callers may still pass
        // an offset pointer into a larger allocation. Use the fast path only when alignment
        // is preserved.
        const std::uintptr_t inputAddr =
            reinterpret_cast<std::uintptr_t>(input);
        const std::uintptr_t alignmentMask =
            static_cast<std::uintptr_t>(alignof(uint4) - 1u);

        if ((inputAddr & alignmentMask) == 0u) {
            const unsigned int numBlocks = compute_launch_blocks(
                histogram_kernel_vectorized<threadsPerBlock, itemsPerThread, histogramReplicas>,
                sharedMemBytes,
                inputSize);

            histogram_kernel_vectorized<
                threadsPerBlock, itemsPerThread, histogramReplicas>
                <<<numBlocks, threadsPerBlock, sharedMemBytes, stream>>>(
                    input, histogram, inputSize, from, numBins, smemStride);
        } else {
            const unsigned int numBlocks = compute_launch_blocks(
                histogram_kernel_scalar<threadsPerBlock, itemsPerThread, histogramReplicas>,
                sharedMemBytes,
                inputSize);

            histogram_kernel_scalar<
                threadsPerBlock, itemsPerThread, histogramReplicas>
                <<<numBlocks, threadsPerBlock, sharedMemBytes, stream>>>(
                    input, histogram, inputSize, from, numBins, smemStride);
        }
    }
};

template <>
struct launch_dispatch<false> {
    static inline void launch(const char* input,
                              unsigned int* histogram,
                              unsigned int inputSize,
                              unsigned int from,
                              unsigned int numBins,
                              unsigned int smemStride,
                              size_t sharedMemBytes,
                              cudaStream_t stream)
    {
        const unsigned int numBlocks = compute_launch_blocks(
            histogram_kernel_scalar<threadsPerBlock, itemsPerThread, histogramReplicas>,
            sharedMemBytes,
            inputSize);

        histogram_kernel_scalar<
            threadsPerBlock, itemsPerThread, histogramReplicas>
            <<<numBlocks, threadsPerBlock, sharedMemBytes, stream>>>(
                input, histogram, inputSize, from, numBins, smemStride);
    }
};

}  // namespace histogram_detail

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    using namespace histogram_detail;

    const unsigned int fromU   = static_cast<unsigned int>(from);
    const unsigned int numBins = static_cast<unsigned int>(to - from) + 1u;

    // Round the logical histogram width up to a multiple of 32 and then add one extra slot.
    // This makes smemStride % 32 == 1, so the same logical bin in neighboring replicas is
    // shifted by one shared-memory bank each time.
    const unsigned int bankMask   = static_cast<unsigned int>(histogramReplicas - 1);
    const unsigned int smemStride = ((numBins + bankMask) & ~bankMask) + 1u;

    const size_t histogramBytes =
        static_cast<size_t>(numBins) * sizeof(unsigned int);
    const size_t sharedMemBytes =
        static_cast<size_t>(histogramReplicas) *
        static_cast<size_t>(smemStride) *
        sizeof(unsigned int);

    // The kernel merges per-block results into the global histogram with atomic adds,
    // so zero the output first. The API has no stream parameter; use the default stream
    // and leave synchronization/error observation to the caller as requested.
    cudaStream_t stream = 0;
    cudaMemsetAsync(histogram, 0, histogramBytes, stream);

    if (inputSize == 0u) {
        return;
    }

    launch_dispatch<vectorizedLoadPathAvailable>::launch(
        input,
        histogram,
        inputSize,
        fromU,
        numBins,
        smemStride,
        sharedMemBytes,
        stream);
}