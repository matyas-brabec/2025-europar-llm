#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

/*
  High-level design:
  - Compute a histogram for a contiguous character range [from, to] within a large text buffer on the GPU.
  - The output histogram has (to - from + 1) bins, counting occurrences of characters with ordinal values in [from, to].
  - Efficient parallelization uses:
      * Grid-stride loops to cover arbitrarily large inputs.
      * Vectorized reads (uint4, 16 bytes per transaction) for high bandwidth on A100/H100.
      * Warp-private shared-memory histograms to reduce shared-memory atomic contention.
      * A per-block reduction of warp histograms into global memory via atomicAdd (one per bin per block).
  - The host function zeroes the output histogram via cudaMemsetAsync and launches the kernel with tuned parameters.
  - No device synchronization is performed here; the caller is responsible for synchronization.
*/

static __device__ __forceinline__ void add_if_in_range(unsigned int* my_hist, unsigned int from, unsigned int bins, unsigned char c) {
    // Compute index relative to 'from' and update only if within range.
    // Using unsigned arithmetic avoids branches for c < from (wrap-around makes idx >= bins).
    unsigned int idx = static_cast<unsigned int>(c) - from;
    if (idx < bins) {
        atomicAdd(&my_hist[idx], 1u);
    }
}

static __device__ __forceinline__ void process_word32(unsigned int* my_hist, unsigned int from, unsigned int bins, unsigned int w) {
    // Unpack 4 bytes from a 32-bit word and update warp-private histogram
    unsigned char b0 = static_cast<unsigned char>(w);
    unsigned char b1 = static_cast<unsigned char>(w >> 8);
    unsigned char b2 = static_cast<unsigned char>(w >> 16);
    unsigned char b3 = static_cast<unsigned char>(w >> 24);
    add_if_in_range(my_hist, from, bins, b0);
    add_if_in_range(my_hist, from, bins, b1);
    add_if_in_range(my_hist, from, bins, b2);
    add_if_in_range(my_hist, from, bins, b3);
}

__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       unsigned int from,
                                       unsigned int to)
{
    // Number of bins for [from, to], inclusive
    const unsigned int bins = to - from + 1;

    // Warp topology
    const int warpsPerBlock = (blockDim.x + 31) >> 5;
    const int warpId = threadIdx.x >> 5;

    // Allocate warp-private histograms in shared memory:
    // Layout: [warp0 bins][warp1 bins]...[warpN bins]
    extern __shared__ unsigned int s_hist[];
    unsigned int* my_hist = s_hist + warpId * bins;

    // Initialize shared memory histograms to zero cooperatively
    for (unsigned int i = threadIdx.x; i < static_cast<unsigned int>(warpsPerBlock) * bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop setup
    const size_t globalThreadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t gridStride = static_cast<size_t>(blockDim.x) * gridDim.x;

    // Vectorized reads: process 16-byte chunks as uint4 for coalesced loads
    const size_t numChunks = static_cast<size_t>(inputSize) >> 4; // inputSize / 16
    const uint4* in4 = reinterpret_cast<const uint4*>(input);

    for (size_t chunk = globalThreadId; chunk < numChunks; chunk += gridStride) {
        // Each uint4 contains 16 bytes as four 32-bit words
        uint4 v = in4[chunk];
        process_word32(my_hist, from, bins, v.x);
        process_word32(my_hist, from, bins, v.y);
        process_word32(my_hist, from, bins, v.z);
        process_word32(my_hist, from, bins, v.w);
    }

    // Process remaining tail bytes (if inputSize not multiple of 16)
    for (size_t i = (numChunks << 4) + globalThreadId; i < static_cast<size_t>(inputSize); i += gridStride) {
        unsigned char c = input[i];
        add_if_in_range(my_hist, from, bins, c);
    }

    __syncthreads();

    // Reduce warp-private histograms into global memory:
    // Each thread handles a subset of bins to maximize parallelism.
    for (unsigned int b = threadIdx.x; b < bins; b += blockDim.x) {
        unsigned int sum = 0;
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * bins + b];
        }
        if (sum) {
            atomicAdd(&histogram[b], sum);
        }
    }
}

/*
  Host-side entry point. Launches the histogram kernel.

  Parameters:
    - input:      device pointer to input chars (cudaMalloc'ed)
    - histogram:  device pointer to output histogram (cudaMalloc'ed).
                  The array must have at least (to - from + 1) elements.
    - inputSize:  number of chars in input
    - from, to:   inclusive range [from, to], 0 <= from < to <= 255

  Notes:
    - This function zeroes the output histogram asynchronously using cudaMemsetAsync.
    - No device synchronization is performed here; caller is responsible for synchronization.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    const unsigned int ufrom = static_cast<unsigned int>(from);
    const unsigned int uto   = static_cast<unsigned int>(to);
    const unsigned int bins  = uto - ufrom + 1;

    // Zero the output histogram (async). Caller handles synchronization.
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    // If there is no input, we're done after zeroing the histogram.
    if (inputSize == 0) return;

    // Choose launch configuration tuned for A100/H100:
    // - Block size: 512 threads (16 warps) provides good occupancy and moderate shared memory usage.
    // - Grid size: up to 4 blocks per SM, but no more than the number of 16-byte chunks to avoid oversubscription.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    const int numSM = props.multiProcessorCount;

    const int blockSize = 512;
    const int warpsPerBlock = (blockSize + 31) >> 5;
    const size_t shmemBytes = static_cast<size_t>(warpsPerBlock) * bins * sizeof(unsigned int);

    // Determine grid size. Use up to 4 blocks per SM to balance latency hiding and resource usage.
    const unsigned int numChunks = inputSize >> 4; // inputSize / 16
    const int maxBlocks = numSM * 4;
    int gridSize = (numChunks > 0) ? std::min(maxBlocks, static_cast<int>(numChunks)) : 1;
    if (gridSize < 1) gridSize = 1;

    // Launch kernel (async). No synchronization here.
    histogram_range_kernel<<<gridSize, blockSize, shmemBytes>>>(
        reinterpret_cast<const unsigned char*>(input),
        histogram,
        inputSize,
        ufrom,
        uto
    );
}