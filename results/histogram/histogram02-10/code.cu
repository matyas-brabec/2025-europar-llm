#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
  CUDA kernel that computes a histogram over a specified inclusive character range [from, to].
  - input: device pointer to the input buffer of bytes (chars)
  - histogram: device pointer to output histogram of length (to - from + 1)
  - inputSize: number of bytes in input
  - from, to: inclusive range of byte values to count
  - rangeLen: to - from + 1
  Optimization strategy:
    - Use shared memory to maintain per-block histograms to reduce global atomic contention.
    - Further reduce contention by allocating warp-private histograms in shared memory.
      Each warp updates its own copy of the histogram with shared-memory atomics.
      At the end, per-block histograms are reduced and merged into global memory with at most
      gridDim.x * rangeLen global atomic updates (one per block per bin).
    - Grid-stride loop to efficiently cover the entire input with arbitrary grid sizes.
  Notes:
    - Input is read as unsigned bytes to interpret all 0..255 values correctly regardless of char signedness.
    - Range check is implemented with a single comparison: (val - from) <= (to - from).
      This avoids two-branch checks and leverages unsigned arithmetic behavior.
*/
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int inputSize,
                                       int from,
                                       int to,
                                       int rangeLen)
{
    extern __shared__ unsigned int s_hist[]; // Layout: [warp0 rangeLen][warp1 rangeLen]...[warpN rangeLen]
    const int tid = threadIdx.x;
    const int warpId = tid >> 5; // warpSize == 32
    const int laneId = tid & 31;
    const int warpsPerBlock = (blockDim.x + 31) >> 5;

    // Zero initialize the shared memory histograms cooperatively
    for (int i = tid; i < warpsPerBlock * rangeLen; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Process input via grid-stride loop, updating this warp's private histogram in shared memory.
    const unsigned int gridStride = blockDim.x * gridDim.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Precompute constants for fast range check
    const unsigned int ufrom = static_cast<unsigned int>(from);
    const unsigned int urangeMax = static_cast<unsigned int>(to - from); // inclusive

    unsigned int* warp_hist = s_hist + warpId * rangeLen;

    while (idx < inputSize) {
        unsigned int v = static_cast<unsigned int>(input[idx]); // 0..255
        unsigned int adj = v - ufrom;
        if (adj <= urangeMax) {
            // adj is in [0, rangeLen-1]
            // Shared memory atomic to this warp's private histogram
            atomicAdd(&warp_hist[static_cast<int>(adj)], 1u);
        }
        idx += gridStride;
    }

    __syncthreads();

    // Reduce per-warp histograms to a per-block histogram and merge into global memory
    for (int bin = tid; bin < rangeLen; bin += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate contributions from each warp
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * rangeLen + bin];
        }
        if (sum) {
            atomicAdd(&histogram[bin], sum);
        }
    }
    // No need for further synchronization
}

/*
  Host entry point that prepares and launches the CUDA kernel.
  Parameters:
    - input: device pointer (cudaMalloc'ed) to the input text data (bytes)
    - histogram: device pointer (cudaMalloc'ed) to the output histogram buffer of length (to - from + 1) unsigned ints
    - inputSize: number of bytes in input
    - from, to: inclusive byte range [from, to] to count (0 <= from <= to <= 255)
  Behavior:
    - Zeros the output histogram on device (async).
    - Launches the kernel with a configuration suitable for modern data center GPUs (A100/H100).
    - Does not perform device synchronization; the caller is responsible for any needed sync.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Handle trivial case early: just zero histogram and return
    const int rangeLen = to - from + 1;
    if (rangeLen <= 0) {
        return; // Invalid range; nothing to do (contract guarantees from <= to, but guard anyway)
    }

    // Zero the output histogram (async on the current stream)
    cudaMemset(histogram, 0, static_cast<size_t>(rangeLen) * sizeof(unsigned int));

    if (inputSize == 0) {
        return;
    }

    // Use a reasonable block size to balance occupancy and shared memory usage.
    // 256 threads per block => 8 warps => shared memory = 8 * rangeLen * 4 bytes <= 8 KB (for rangeLen <= 256)
    const int BLOCK_SIZE = 256;
    const int BLOCK_WARPS = (BLOCK_SIZE + 31) / 32;

    // Choose grid size based on input size and device SM count
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    // Heuristic: aim for multiple resident blocks per SM.
    // Cap the grid by both data amount and a multiplier of SM count to avoid oversubscription of tiny workloads.
    int maxActiveBlocks = prop.multiProcessorCount * 32; // Plenty for saturation on A100/H100
    int dataBlocks = static_cast<int>((inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int gridBlocks = dataBlocks > 0 ? (dataBlocks < maxActiveBlocks ? dataBlocks : maxActiveBlocks) : 1;

    // Dynamic shared memory size: one rangeLen histogram per warp in the block
    size_t shmemBytes = static_cast<size_t>(BLOCK_WARPS) * static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Launch the kernel
    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<gridBlocks, BLOCK_SIZE, shmemBytes>>>(
        d_input,
        histogram,
        inputSize,
        from,
        to,
        rangeLen
    );

    // No cudaDeviceSynchronize() here: caller handles synchronization if needed
}