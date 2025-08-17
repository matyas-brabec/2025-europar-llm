#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

/*
  CUDA histogram kernel optimized for modern NVIDIA data center GPUs (A100/H100).

  Key optimization points:
  - Shared-memory privatization of the histogram to minimize global-memory atomics.
  - 32 conflict-free copies (one per bank) of the histogram per thread block to avoid bank conflicts.
    Each thread uses copy index = threadIdx.x % 32 (i.e., its lane ID).
    The layout is strided so that bin i of copy c is at offset i * 32 + c. For 32-bit words, this maps
    directly to unique banks across the 32 copies for a given bin, eliminating within-warp bank conflicts.
  - Shared-memory atomicAdd is used to resolve cross-warp collisions (warps share the same 32 copies).
    On Ampere/Hopper, shared-memory atomics are fast and this substantially reduces contention versus
    global-memory atomics on every update.
  - Grid-stride loop with compile-time-controlled itemsPerThread (set to 16 by default) to maximize
    memory throughput on large inputs.
  - Efficient reduction: the first warp sums the 32 copies per bin using warp shuffles and performs a
    single global atomicAdd per bin per block.

  Interface contract:
  - The histogram covers characters in the inclusive range [from, to], where 0 <= from < to <= 255.
  - The 'histogram' output buffer must have size (to - from + 1) and is zeroed by run_histogram before launch.
  - 'input' and 'histogram' are device pointers (cudaMalloc'ed), and inputSize is the number of chars.
  - Synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
*/

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Tunable constant: how many input characters each thread processes per grid-stride iteration.
// 16 is a good default for A100/H100 on large inputs to balance ILP and occupancy.
static constexpr int itemsPerThread = 16;
static constexpr int SHARED_HIST_COPIES = 32; // One per shared memory bank / warp lane (0..31).

// Warp-level reduction helper (sum across a full warp).
__device__ __forceinline__ unsigned int warpReduceSum(unsigned int val) {
    // Full warp mask (assumes warp size = 32)
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    // Tree reduction using shuffle-down; requires SM 3.0+ (available on A100/H100)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histOut,
                                       unsigned int inputSize,
                                       int from,
                                       int to)
{
    // Number of bins in the requested range [from, to] (inclusive).
    const int bins = to - from + 1;

    // Dynamic shared memory layout:
    // We store 32 copies of the histogram (one per bank/lane), with stride 32 across bins:
    // For bin i and copy c in [0,31], index = i * 32 + c.
    extern __shared__ unsigned int sHist[];
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Initialize shared memory histogram copies to zero.
    // Total elements in shared memory = bins * 32.
    for (int idx = threadIdx.x; idx < bins * SHARED_HIST_COPIES; idx += blockDim.x) {
        sHist[idx] = 0u;
    }
    __syncthreads();

    // Global thread parameters.
    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int threadsPerGrid = gridDim.x * blockDim.x;
    const unsigned int stride = threadsPerGrid * itemsPerThread;

    // Process input in a grid-stride loop. Within each outer iteration, each thread processes
    // itemsPerThread items, spaced by blockDim.x to keep loads coalesced.
    for (unsigned int base = t; base < inputSize; base += stride) {
        #pragma unroll
        for (int j = 0; j < itemsPerThread; ++j) {
            unsigned int idx = base + static_cast<unsigned int>(j) * blockDim.x;
            if (idx >= inputSize) break;

            unsigned char ch = input[idx];
            // Check if character is within the requested range [from, to].
            if (ch >= static_cast<unsigned char>(from) && ch <= static_cast<unsigned char>(to)) {
                const int bin = static_cast<int>(ch) - from; // 0..bins-1
                // Each thread updates the shared-memory copy corresponding to its lane ID.
                // Use shared-memory atomics to safely handle cross-warp collisions on the same copy.
                atomicAdd(&sHist[bin * SHARED_HIST_COPIES + lane], 1u);
            }
        }
    }
    __syncthreads();

    // Reduce the 32 copies per bin into a single value per bin using the first warp of the block.
    // Each lane 'lane' loads its copy for bin b, reduces across the warp, and lane 0 atomically
    // adds the result to the global histogram.
    if ((threadIdx.x >> 5) == 0) { // warp 0 only
        for (int b = 0; b < bins; ++b) {
            unsigned int val = sHist[b * SHARED_HIST_COPIES + lane];
            unsigned int sum = warpReduceSum(val);
            if (lane == 0) {
                atomicAdd(&histOut[b], sum);
            }
        }
    }
}

// Host-side launcher.
// - input: device pointer to 'inputSize' chars (cudaMalloc'ed).
// - histogram: device pointer to at least (to - from + 1) unsigned ints (cudaMalloc'ed).
// - inputSize: number of chars in input.
// - from, to: inclusive range [from, to], with 0 <= from < to <= 255.
// The function zeroes the output histogram, computes an efficient launch configuration, and launches the kernel.
// Caller is responsible for synchronization if needed (e.g., cudaDeviceSynchronize).
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Basic parameter checks (defensive; per problem statement inputs are valid).
    if (!input || !histogram || inputSize == 0u) {
        return;
    }
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (from > to) return;

    const int bins = to - from + 1;

    // Zero the output histogram (async on default stream).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int), 0);

    // Choose launch configuration.
    // Fixed block size that provides high occupancy on A100/H100 with modest register pressure.
    const int blockSize = 256;

    // Shared memory per block: bins * 32 copies * sizeof(unsigned int).
    const size_t shmemSize = static_cast<size_t>(bins) * SHARED_HIST_COPIES * sizeof(unsigned int);

    // Determine an appropriate grid size based on occupancy and input size.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, histogram_range_kernel, blockSize, shmemSize);

    // Minimum number of blocks to cover the input with itemsPerThread per thread.
    const unsigned long long itemsPerBlockPerIter = static_cast<unsigned long long>(blockSize) * itemsPerThread;
    int gridForInput = static_cast<int>((static_cast<unsigned long long>(inputSize) + itemsPerBlockPerIter - 1ull) / itemsPerBlockPerIter);
    if (gridForInput <= 0) gridForInput = 1;

    // Use enough blocks to saturate the SMs, but don't be shy about oversubscription; the loop is grid-stride.
    int gridSize = std::max(gridForInput, maxBlocksPerSM * props.multiProcessorCount);
    if (gridSize <= 0) gridSize = props.multiProcessorCount; // Fallback safety.

    // Launch the kernel.
    const unsigned char* inputU = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<gridSize, blockSize, shmemSize>>>(inputU, histogram, inputSize, from, to);
}