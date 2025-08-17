#include <cuda_runtime.h>
#include <stdint.h>

// Optimized histogram-of-bytes kernel for a restricted character range [from, to].
// Key ideas:
// - Use shared-memory privatization to reduce contention on global memory.
// - Create 32 copies of the histogram in shared memory (one per warp lane).
// - Choose a stride so that copy l's base index maps to bank l (modulo 32), and
//   accesses by different lanes to the same bin hit different banks (no conflicts).
// - Each thread processes 'itemsPerThread' contiguous input bytes for coalesced global loads.
// - Accumulate into the lane-private shared histograms with shared-memory atomics to ensure correctness
//   across different warps (multiple warps share the same lane index).
// - Reduce the 32 copies per block and atomically accumulate the per-block totals into the global histogram.

#ifndef __CUDACC_RTC__  // Allow usage with NVRTC if needed
#define INLINE_ATTR __forceinline__
#else
#define INLINE_ATTR
#endif

// Tunable constants for modern NVIDIA data-center GPUs (A100/H100).
// - 256 threads per block provides good occupancy while leaving resources for shared memory.
// - 16 items per thread balances memory bandwidth and shared-memory pressure for large inputs.
static constexpr int threadsPerBlock = 256;
static constexpr int itemsPerThread  = 16;

// Compute a stride >= rangeLen that is odd (coprime with 32) to guarantee that, for any fixed bin,
// addresses accessed by lanes [0..31] map to distinct banks. We also make sure stride >= rangeLen.
INLINE_ATTR __host__ __device__ int compute_stride(int rangeLen) {
    // Round up to multiple of 32, then add 1 to make it odd and to ensure stride > multiple of 32.
    int r32 = (rangeLen + 31) & ~31;
    return r32 + 1;
}

__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ out_hist,
                                       unsigned int n,
                                       int from,
                                       int to,
                                       int stride)
{
    extern __shared__ unsigned int sh[]; // Size: 32 * stride
    const int rangeLen = to - from + 1;

    // Zero out the 32 lane-private histogram copies in shared memory.
    for (int i = threadIdx.x; i < 32 * stride; i += blockDim.x) {
        sh[i] = 0u;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x & 31; // warp lane id in [0,31]
    unsigned int* __restrict__ lane_hist = sh + lane * stride;

    // Each thread processes 'itemsPerThread' contiguous bytes starting at 'start'.
    const unsigned int start = (blockIdx.x * blockDim.x + threadIdx.x) * itemsPerThread;

    // Process assigned input characters and update the lane-private shared histogram.
    // Use atomicAdd in shared memory to avoid cross-warp races (multiple warps share the same 'lane').
    for (int j = 0; j < itemsPerThread; ++j) {
        unsigned int idx = start + j;
        if (idx < n) {
            unsigned char v = static_cast<unsigned char>(input[idx]);
            if (v >= static_cast<unsigned char>(from) && v <= static_cast<unsigned char>(to)) {
                int bin = static_cast<int>(v) - from; // 0..rangeLen-1
                atomicAdd(&lane_hist[bin], 1u);
            }
        }
    }

    __syncthreads();

    // Reduce the 32 lane-private copies into a block-local sum per bin,
    // then atomically accumulate into the global histogram.
    for (int bin = threadIdx.x; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0;
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            sum += sh[l * stride + bin];
        }
        atomicAdd(&out_hist[bin], sum);
    }
}

// Host-side launcher.
// input:     device pointer to chars (cudaMalloc'ed), size = inputSize
// histogram: device pointer to unsigned int array of size (to - from + 1) (cudaMalloc'ed)
// inputSize: number of chars in input
// from, to:  inclusive range of byte values to histogram (0 <= from < to <= 255)
//
// Notes:
// - This function zeroes the output histogram on device before launching the kernel.
// - Caller is responsible for synchronization (e.g., cudaDeviceSynchronize) if needed.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Basic parameter validation (assumed valid per problem statement, but guard against misuse).
    if (!input || !histogram || inputSize == 0 || from < 0 || to > 255 || from > to) {
        // Still zero out histogram if possible when inputSize==0 or invalid range.
        if (histogram && from >= 0 && to <= 255 && from <= to) {
            int rangeLen = to - from + 1;
            cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));
        }
        return;
    }

    const int rangeLen = to - from + 1;
    const int stride   = compute_stride(rangeLen);

    // Clear the device-side histogram (as we use atomicAdd accumulation).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    // Grid size: each block processes threadsPerBlock * itemsPerThread elements.
    unsigned int workPerBlock = threadsPerBlock * itemsPerThread;
    unsigned int blocks = (inputSize + workPerBlock - 1) / workPerBlock;
    if (blocks == 0) blocks = 1;

    // Shared memory: 32 copies (one per lane) times 'stride' bins per copy.
    size_t sharedMemSize = static_cast<size_t>(32) * static_cast<size_t>(stride) * sizeof(unsigned int);

    histogram_range_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, histogram, inputSize, from, to, stride);
}