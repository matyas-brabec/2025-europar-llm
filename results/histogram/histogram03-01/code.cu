#include <cuda_runtime.h>

namespace {
// Tuned for modern NVIDIA data-center GPUs (A100/H100 class):
// - 256 threads/block gives 8 warps/block
// - one privatized shared-memory histogram per warp
// - up to 8 resident blocks/SM for 256-thread blocks (2048 threads/SM)
constexpr int kBlockThreads = 256;
constexpr unsigned int kWarpSizeConst = 32u;
constexpr unsigned int kBankPadShift = 5u;  // log2(32)
constexpr unsigned int kWarpsPerBlock = static_cast<unsigned int>(kBlockThreads) / kWarpSizeConst;
constexpr unsigned int kMaxBins = 256u;

// One padding slot per 32 bins to reduce shared-memory bank conflicts for indices that
// would otherwise alias every 32 entries. 256 bins -> 256 + 8 = 264 entries per sub-hist.
constexpr unsigned int kPaddedBins = kMaxBins + (kMaxBins >> kBankPadShift);
constexpr unsigned int kSharedHistEntries = kWarpsPerBlock * kPaddedBins;

constexpr unsigned int kVecBytes = 16u;  // sizeof(uint4)
constexpr unsigned int kBytesPerBlockIteration =
    static_cast<unsigned int>(kBlockThreads) * kVecBytes;
constexpr unsigned int kMaxResidentBlocksPerSM = 8u;

static_assert(kWarpSizeConst == 32u, "This kernel assumes 32-thread warps.");
static_assert((kBlockThreads % static_cast<int>(kWarpSizeConst)) == 0,
              "Block size must be a multiple of warp size.");
static_assert(sizeof(uint4) == kVecBytes, "uint4 must be 16 bytes.");
}  // namespace

__device__ __forceinline__ unsigned int padded_bin_index(unsigned int bin) {
    // Insert one hole every 32 bins so bins separated by 32 no longer map to the same bank.
    return bin + (bin >> kBankPadShift);
}

__device__ __forceinline__ void add_byte_to_private_hist(unsigned int byte_value,
                                                         unsigned int* __restrict__ warp_hist,
                                                         unsigned int from_u,
                                                         unsigned int range_size) {
    // Unsigned subtraction makes the range test branch-efficient:
    // if byte_value < from_u, bin underflows to a large unsigned value and fails the test.
    const unsigned int bin = byte_value - from_u;
    if (bin < range_size) {
        atomicAdd(warp_hist + padded_bin_index(bin), 1u);
    }
}

__device__ __forceinline__ void add_word_to_private_hist(unsigned int word,
                                                         unsigned int* __restrict__ warp_hist,
                                                         unsigned int from_u,
                                                         unsigned int range_size) {
    // Manually unpack 4 bytes from a 32-bit lane loaded via uint4.
    add_byte_to_private_hist((word      ) & 0xFFu, warp_hist, from_u, range_size);
    add_byte_to_private_hist((word >>  8) & 0xFFu, warp_hist, from_u, range_size);
    add_byte_to_private_hist((word >> 16) & 0xFFu, warp_hist, from_u, range_size);
    add_byte_to_private_hist((word >> 24) & 0xFFu, warp_hist, from_u, range_size);
}

__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            unsigned int from_u,
                            unsigned int range_size) {
    // Shared-memory privatization:
    // - one padded sub-histogram per warp to reduce inter-warp atomic contention
    // - final merge performs at most one global atomic per non-zero bin per block
    __shared__ unsigned int shared_hist[kSharedHistEntries];

    const unsigned int tid = threadIdx.x;
    const unsigned int global_tid = blockIdx.x * blockDim.x + tid;
    const unsigned int total_threads = gridDim.x * blockDim.x;

    // Scratch size is only 8 * 264 * 4 = 8448 bytes/block, so clearing it fully is cheap.
    for (unsigned int i = tid; i < kSharedHistEntries; i += blockDim.x) {
        shared_hist[i] = 0u;
    }
    __syncthreads();

    unsigned int* const warp_hist = shared_hist + (tid >> 5) * kPaddedBins;

    // The input buffer is assumed to be the cudaMalloc-allocated base pointer, which is
    // sufficiently aligned for 16-byte uint4 vector loads.
    const unsigned char* const uinput = reinterpret_cast<const unsigned char*>(input);
    const uint4* const input4 = reinterpret_cast<const uint4*>(uinput);

    // Main pass: 16 bytes/thread/iteration using vectorized global loads.
    const unsigned int vec_count = inputSize / kVecBytes;
    for (unsigned int i = global_tid; i < vec_count; i += total_threads) {
        const uint4 v = input4[i];
        add_word_to_private_hist(v.x, warp_hist, from_u, range_size);
        add_word_to_private_hist(v.y, warp_hist, from_u, range_size);
        add_word_to_private_hist(v.z, warp_hist, from_u, range_size);
        add_word_to_private_hist(v.w, warp_hist, from_u, range_size);
    }

    // Tail cleanup for the final 0..15 bytes.
    const unsigned int tail_start = vec_count * kVecBytes;
    for (unsigned int i = tail_start + global_tid; i < inputSize; i += total_threads) {
        add_byte_to_private_hist(static_cast<unsigned int>(uinput[i]), warp_hist, from_u, range_size);
    }

    __syncthreads();

    // histogram[bin] corresponds to character value from_u + bin.
    // For a single-block launch we can store directly because the output was pre-zeroed.
    const bool single_block = (gridDim.x == 1u);
    for (unsigned int bin = tid; bin < range_size; bin += blockDim.x) {
        const unsigned int pbin = padded_bin_index(bin);
        unsigned int sum = 0u;

#pragma unroll
        for (unsigned int w = 0; w < kWarpsPerBlock; ++w) {
            sum += shared_hist[w * kPaddedBins + pbin];
        }

        if (sum != 0u) {
            if (single_block) {
                histogram[bin] = sum;
            } else {
                atomicAdd(histogram + bin, sum);
            }
        }
    }
}

void run_histogram(const char* input,
                   unsigned int* histogram,
                   unsigned int inputSize,
                   int from,
                   int to) {
    // Assumes valid [from, to] as specified by the problem statement.
    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int range_size = static_cast<unsigned int>(to - from) + 1u;

    // Enqueue everything on the default stream. The caller explicitly owns synchronization.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(range_size) * sizeof(unsigned int), 0);

    if (inputSize == 0u) {
        return;
    }

    // One block iteration covers 16 bytes/thread = 4096 bytes/block.
    // Cap the grid to 8 blocks/SM so the GPU stays full without creating more block-local
    // histograms than useful for the final merge.
    unsigned int blocks = ((inputSize - 1u) / kBytesPerBlockIteration) + 1u;

    if (blocks > kMaxResidentBlocksPerSM) {
        int device = 0;
        cudaGetDevice(&device);

        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

        const unsigned int max_blocks =
            static_cast<unsigned int>((sm_count > 0) ? sm_count : 1) * kMaxResidentBlocksPerSM;

        if (blocks > max_blocks) {
            blocks = max_blocks;
        }
    }

    histogram_range_kernel<<<blocks, kBlockThreads, 0, 0>>>(
        input, histogram, inputSize, from_u, range_size);
}