#include <cuda_runtime.h>

namespace
{
// 32 interleaved copies: copy c of logical bin b is stored at b * 32 + c.
// With 4-byte counters and 32 shared-memory banks, lane c always maps to bank c.
// Using copy = threadIdx.x % 32 therefore makes the hot update path bank-conflict-free.
constexpr int kHistogramCopies = 32;

// A 256-thread block is a good balance on A100/H100-class GPUs:
// enough warps to hide latency, but not so many that same-copy contention
// becomes excessive (each copy is shared by 8 warps at most).
constexpr int kBlockSize = 256;

// Requested tuning knob. 16 bytes/thread gives a 4 KiB tile per block:
// good amortization of loop/control overhead with modest register pressure
// after unrolling on modern data-center GPUs.
constexpr int itemsPerThread = 16;

static_assert(kHistogramCopies == 32, "This implementation assumes 32 shared-memory banks.");
static_assert((kBlockSize % kHistogramCopies) == 0, "Block size must be a multiple of 32.");
static_assert(itemsPerThread > 0, "itemsPerThread must be positive.");

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int v)
{
    // Full-warp tree reduction. All 32 lanes participate in the reduction phase.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ void accumulate_byte(
    const unsigned int value,
    const unsigned int from,
    const unsigned int numBins,
    unsigned int* const localHist)
{
    // Unsigned subtraction folds both bounds checks into one compare:
    //   value in [from, from + numBins - 1]  <=>  (value - from) < numBins
    const unsigned int bin = value - from;
    if (bin < numBins)
    {
        atomicAdd(localHist + bin * kHistogramCopies, 1u);
    }
}

template <int ITEMS_PER_THREAD, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void histogram_range_kernel(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ histogram,
    unsigned int inputSize,
    unsigned int from,
    unsigned int numBins)
{
    extern __shared__ unsigned int s_hist[];

    const unsigned int tid           = threadIdx.x;
    const unsigned int lane          = tid & (kHistogramCopies - 1);  // threadIdx.x % 32
    const unsigned int warpId        = tid / kHistogramCopies;
    const unsigned int warpsPerBlock = BLOCK_SIZE / kHistogramCopies;

    // Per-thread base pointer to "its" shared-memory copy.
    // Copy c of bin b is at localHist[b * 32], where localHist = s_hist + c.
    unsigned int* const localHist = s_hist + lane;

    // Shared storage holds exactly 32 copies of the requested range, not the full 256 bins.
    const unsigned int sharedElems = numBins * kHistogramCopies;
    for (unsigned int i = tid; i < sharedElems; i += BLOCK_SIZE)
    {
        s_hist[i] = 0u;
    }
    __syncthreads();

    // Each block processes tiles of BLOCK_SIZE * ITEMS_PER_THREAD bytes.
    // For a fixed "item" index, threads in a warp read consecutive bytes, so the loads are coalesced.
    const unsigned int tileSize   = BLOCK_SIZE * ITEMS_PER_THREAD;
    const unsigned int gridStride = gridDim.x * tileSize;

    // Fast path: full tiles where every access is in bounds, so the inner loop needs no tail checks.
    unsigned int blockBase = blockIdx.x * tileSize;
    if (inputSize >= tileSize)
    {
        const unsigned int lastFullBlockBase = inputSize - tileSize;
        for (; blockBase <= lastFullBlockBase; blockBase += gridStride)
        {
            const unsigned int threadBase = blockBase + tid;

            #pragma unroll
            for (int item = 0; item < ITEMS_PER_THREAD; ++item)
            {
                const unsigned int value = static_cast<unsigned int>(input[threadBase + item * BLOCK_SIZE]);
                accumulate_byte(value, from, numBins, localHist);
            }
        }
    }

    // Tail path: final partial tile(s).
    // Use offsets relative to blockBase so the bounds checks stay overflow-safe in 32-bit arithmetic.
    for (; blockBase < inputSize; blockBase += gridStride)
    {
        const unsigned int remaining = inputSize - blockBase;

        if (tid < remaining)
        {
            #pragma unroll
            for (int item = 0; item < ITEMS_PER_THREAD; ++item)
            {
                const unsigned int offset = tid + item * BLOCK_SIZE;
                if (offset < remaining)
                {
                    const unsigned int value = static_cast<unsigned int>(input[blockBase + offset]);
                    accumulate_byte(value, from, numBins, localHist);
                }
            }
        }
    }

    __syncthreads();

    // Final merge:
    // Use one warp to reduce one logical bin across the 32 copies. This is intentional:
    // with the b * 32 + c layout, lanes 0..31 reading the same bin access consecutive
    // shared-memory locations and therefore distinct banks. A naive thread-per-bin reduction
    // would instead read addresses with stride 32 and would create worst-case bank conflicts.
    for (unsigned int bin = warpId; bin < numBins; bin += warpsPerBlock)
    {
        unsigned int sum = s_hist[bin * kHistogramCopies + lane];
        sum = warp_reduce_sum(sum);

        if (lane == 0 && sum != 0u)
        {
            atomicAdd(histogram + bin, sum);
        }
    }
}

}  // namespace

void run_histogram(const char* input, unsigned int* histogram, unsigned int inputSize, int from, int to)
{
    // One output counter for each byte value in the inclusive range [from, to].
    // The problem statement guarantees 0 <= from < to <= 255, so numBins <= 256.
    // Max dynamic shared memory per block is therefore:
    //   256 bins * 32 copies * 4 bytes = 32 KiB
    // which fits comfortably within the default per-block dynamic shared-memory limit
    // on modern data-center GPUs such as A100/H100.
    const unsigned int numBins      = static_cast<unsigned int>(to - from + 1);
    const size_t histogramBytes     = static_cast<size_t>(numBins) * sizeof(unsigned int);
    const size_t sharedMemBytes     = static_cast<size_t>(numBins) * kHistogramCopies * sizeof(unsigned int);
    const unsigned int tileSize     = static_cast<unsigned int>(kBlockSize * itemsPerThread);
    const unsigned int blocksForInput = 1u + ((inputSize - 1u) / tileSize);

    // The function is intentionally asynchronous with respect to the host; the caller
    // is responsible for any required synchronization.
    cudaStream_t stream = nullptr;

    // Start from a clean output histogram on the same stream as the kernel launch.
    cudaMemsetAsync(histogram, 0, histogramBytes, stream);

    if (inputSize == 0u)
    {
        return;
    }

    // Launch only about as many blocks as can run concurrently.
    // Each block builds a private shared histogram and then atomically merges it
    // into the global result, so launching many more blocks than can be resident
    // at once only increases merge traffic. The grid-stride loop keeps all input covered.
    int device = 0;
    cudaGetDevice(&device);

    int smCount = 1;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
    if (smCount <= 0)
    {
        smCount = 1;
    }

    int maxBlocksPerSM = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        histogram_range_kernel<itemsPerThread, kBlockSize>,
        kBlockSize,
        sharedMemBytes);
    if (maxBlocksPerSM <= 0)
    {
        maxBlocksPerSM = 1;
    }

    const unsigned int residentGrid = static_cast<unsigned int>(smCount * maxBlocksPerSM);
    const unsigned int gridSize = (blocksForInput < residentGrid) ? blocksForInput : residentGrid;

    // Reinterpret as unsigned bytes so ordinal values 128..255 are handled correctly
    // even if plain "char" is signed on the compilation target.
    histogram_range_kernel<itemsPerThread, kBlockSize>
        <<<gridSize, kBlockSize, sharedMemBytes, stream>>>(
            reinterpret_cast<const unsigned char*>(input),
            histogram,
            inputSize,
            static_cast<unsigned int>(from),
            numBins);
}