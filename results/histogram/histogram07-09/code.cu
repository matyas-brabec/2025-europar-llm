#include <cuda_runtime.h>
#include <stdint.h>

// Tuned defaults for recent datacenter GPUs (A100/H100 class):
// - 16 bytes/thread maps to one aligned uint4 load per thread.
// - 256 threads/block keeps plenty of warps resident while limiting contention
//   on the 32 bank-striped shared-memory histogram copies.
constexpr int itemsPerThread   = 16;
constexpr int histogramCopies  = 32;
constexpr int threadsPerBlock  = 256;
constexpr int vectorWidthBytes = sizeof(uint4);
constexpr int vectorChunkCount = itemsPerThread / vectorWidthBytes;

static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");
static_assert((itemsPerThread % vectorWidthBytes) == 0,
              "itemsPerThread must stay a multiple of 16 so each thread can use aligned uint4 loads.");
static_assert(histogramCopies == 32,
              "The bank-conflict avoidance scheme relies on exactly 32 shared-memory histogram copies.");

__device__ __forceinline__ void update_private_histogram(
    unsigned int byte,
    unsigned int from_u,
    unsigned int range,
    unsigned int* privateCopy)
{
    // Unsigned subtraction turns the range test into a single comparison:
    // byte in [from, to]  <=>  (byte - from) < range.
    const unsigned int rel = byte - from_u;
    if (rel < range) {
        // privateCopy points to the thread's bank-striped copy:
        // bin b lives at privateCopy[b * 32], which is s_hist[b * 32 + copy].
        atomicAdd(&privateCopy[rel * histogramCopies], 1u);
    }
}

__device__ __forceinline__ void accumulate_word_bytes(
    unsigned int word,
    unsigned int from_u,
    unsigned int range,
    unsigned int* privateCopy)
{
    update_private_histogram( word        & 0xFFu, from_u, range, privateCopy);
    update_private_histogram((word >>  8) & 0xFFu, from_u, range, privateCopy);
    update_private_histogram((word >> 16) & 0xFFu, from_u, range, privateCopy);
    update_private_histogram((word >> 24) & 0xFFu, from_u, range, privateCopy);
}

__device__ __forceinline__ void accumulate_uint4_bytes(
    uint4 v,
    unsigned int from_u,
    unsigned int range,
    unsigned int* privateCopy)
{
    accumulate_word_bytes(v.x, from_u, range, privateCopy);
    accumulate_word_bytes(v.y, from_u, range, privateCopy);
    accumulate_word_bytes(v.z, from_u, range, privateCopy);
    accumulate_word_bytes(v.w, from_u, range, privateCopy);
}

__device__ __forceinline__ void load_and_accumulate_uint4(
    const char* ptr,
    unsigned int from_u,
    unsigned int range,
    unsigned int* privateCopy)
{
    // The input buffer is allocated by cudaMalloc, so it is naturally well aligned.
    // With itemsPerThread kept as a multiple of 16, every thread chunk start is also
    // 16-byte aligned, which lets the fast path use a single uint4 load.
    const uint4 v = *reinterpret_cast<const uint4*>(ptr);
    accumulate_uint4_bytes(v, from_u, range, privateCopy);
}

__global__ __launch_bounds__(threadsPerBlock)
void histogram_kernel(
    const char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    extern __shared__ unsigned int s_hist[];

    // Shared-memory layout:
    //   s_hist[bin * 32 + copy]
    // For any fixed bin, copies 0..31 map one-to-one onto the 32 shared-memory banks.
    // Each thread always updates copy = threadIdx.x % 32, eliminating intra-warp bank
    // conflicts even when many lanes hit the same logical bin.
    const unsigned int range  = static_cast<unsigned int>(to - from + 1);
    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int tid    = threadIdx.x;
    const unsigned int copy   = tid & (histogramCopies - 1);

    unsigned int* const privateCopy = s_hist + copy;

    // Initialize all 32 copies of the block-private histogram.
    const unsigned int sharedBins = range * histogramCopies;
    for (unsigned int i = tid; i < sharedBins; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    const size_t inputSize64      = static_cast<size_t>(inputSize);
    const size_t threadChunkBytes = static_cast<size_t>(itemsPerThread);
    const size_t vectorWidth      = static_cast<size_t>(vectorWidthBytes);

    const size_t globalThread =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(tid);
    const size_t start  = globalThread * threadChunkBytes;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x) * threadChunkBytes;

    // Grid-stride loop over per-thread chunks.
    for (size_t base = start; base < inputSize64; base += stride) {
        const size_t remaining = inputSize64 - base;

        if (remaining >= threadChunkBytes) {
            // Fast path: the whole per-thread chunk is available.
            #pragma unroll
            for (int chunk = 0; chunk < vectorChunkCount; ++chunk) {
                const size_t chunkOffset = static_cast<size_t>(chunk) * vectorWidth;
                load_and_accumulate_uint4(input + base + chunkOffset, from_u, range, privateCopy);
            }
        } else {
            // Tail path for the last partial chunk seen by this grid-stride thread.
            #pragma unroll
            for (int chunk = 0; chunk < vectorChunkCount; ++chunk) {
                const size_t chunkOffset = static_cast<size_t>(chunk) * vectorWidth;
                if (chunkOffset < remaining) {
                    const size_t chunkRemaining = remaining - chunkOffset;

                    if (chunkRemaining >= vectorWidth) {
                        load_and_accumulate_uint4(input + base + chunkOffset, from_u, range, privateCopy);
                    } else {
                        // Scalar tail uses unsigned-char conversion to avoid sign extension.
                        #pragma unroll
                        for (int i = 0; i < vectorWidthBytes; ++i) {
                            if (static_cast<size_t>(i) < chunkRemaining) {
                                update_private_histogram(
                                    static_cast<unsigned char>(input[base + chunkOffset + static_cast<size_t>(i)]),
                                    from_u,
                                    range,
                                    privateCopy);
                            }
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

    // Merge the 32 bank-striped copies into the final global histogram.
    // bin is relative to 'from': histogram[0] counts byte value 'from'.
    for (unsigned int bin = tid; bin < range; bin += blockDim.x) {
        const unsigned int* const binCopies = s_hist + bin * histogramCopies;

        // Four independent accumulation chains reduce dependency depth a bit.
        unsigned int sum0 = 0u;
        unsigned int sum1 = 0u;
        unsigned int sum2 = 0u;
        unsigned int sum3 = 0u;

        #pragma unroll
        for (int c = 0; c < histogramCopies; c += 4) {
            sum0 += binCopies[c + 0];
            sum1 += binCopies[c + 1];
            sum2 += binCopies[c + 2];
            sum3 += binCopies[c + 3];
        }

        const unsigned int sum = (sum0 + sum1) + (sum2 + sum3);
        if (sum != 0u) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

void run_histogram(
    const char* input,
    unsigned int* histogram,
    unsigned int inputSize,
    int from,
    int to)
{
    // Fresh result, not accumulation into a previous histogram.
    // The caller owns synchronization; this function only enqueues work on the default stream.
    const unsigned int range = static_cast<unsigned int>(to - from + 1);
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range) * sizeof(unsigned int), 0);

    if (inputSize == 0u) {
        return;
    }

    // Dynamic shared memory must hold 32 bank-striped copies of the requested sub-histogram.
    const size_t sharedMemBytes =
        static_cast<size_t>(range) * static_cast<size_t>(histogramCopies) * sizeof(unsigned int);

    // Launch only enough blocks to fill resident capacity. The kernel is grid-stride, so
    // extra blocks would mostly add overhead and increase the number of final global atomics.
    int device = 0;
    int smCount = 1;
    int blocksPerSM = 1;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        histogram_kernel,
        threadsPerBlock,
        sharedMemBytes);

    const unsigned long long workPerBlock =
        static_cast<unsigned long long>(threadsPerBlock) * static_cast<unsigned long long>(itemsPerThread);
    const unsigned long long blocksNeeded =
        (static_cast<unsigned long long>(inputSize) + workPerBlock - 1ull) / workPerBlock;
    const unsigned long long maxResidentGrid =
        static_cast<unsigned long long>(smCount) * static_cast<unsigned long long>(blocksPerSM);

    int grid = static_cast<int>(blocksNeeded < maxResidentGrid ? blocksNeeded : maxResidentGrid);
    if (grid < 1) {
        grid = 1;
    }

    histogram_kernel<<<grid, threadsPerBlock, sharedMemBytes>>>(
        input, histogram, inputSize, from, to);
}