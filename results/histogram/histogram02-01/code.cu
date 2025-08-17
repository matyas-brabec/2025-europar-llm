#include <cuda_runtime.h>
#include <cstdint>

/*
  CUDA kernel: histogram_range_kernel

  Computes a histogram of the input character buffer restricted to a continuous
  range [from, from + bins - 1]. The global histogram array has length "bins",
  where bin j counts occurrences of character code (from + j).

  Performance strategy:
  - Each block builds a histogram in shared memory using warp-private sub-histograms
    to reduce contention.
  - Threads atomically add to their warp's sub-histogram (fast shared-memory atomics).
  - After processing, the block reduces the warp-private histograms into per-bin sums
    and atomically adds the result to the global histogram.
  - Grid-stride loops allow arbitrary input sizes with a fixed launch configuration.

  Constraints:
  - 0 <= from <= 255, 1 <= bins <= 256 (bins = to - from + 1).
  - input points to device memory (cudaMalloc), as does global_hist.
*/
__global__ void histogram_range_kernel(const unsigned char* __restrict__ input,
                                       unsigned int* __restrict__ global_hist,
                                       unsigned int n,
                                       int from,
                                       int bins)
{
    extern __shared__ unsigned int shmem[]; // Layout: [warp0_hist[bins], warp1_hist[bins], ...]
    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;
    const int warp_id = tid >> 5;   // threadIdx.x / warpSize
    const int num_warps = (block_threads + 31) >> 5;

    // Pointer to this warp's private histogram in shared memory
    unsigned int* warp_hist = shmem + warp_id * bins;

    // Zero-initialize the entire shared memory histogram space cooperatively
    for (int i = tid; i < num_warps * bins; i += block_threads) {
        shmem[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over the input
    const unsigned int stride = block_threads * gridDim.x;
    for (unsigned int idx = blockIdx.x * block_threads + tid; idx < n; idx += stride) {
        // Load character and compute bin index relative to 'from'
        unsigned int c = static_cast<unsigned int>(input[idx]);
        int rel = static_cast<int>(c) - from;

        // Fast inclusive range check without branches: rel in [0, bins-1] iff (unsigned)rel < (unsigned)bins
        if (static_cast<unsigned int>(rel) < static_cast<unsigned int>(bins)) {
            // Atomics to shared memory are fast and localized to this warp's histogram,
            // greatly reducing inter-warp contention.
            atomicAdd(&warp_hist[rel], 1u);
        }
    }
    __syncthreads();

    // Reduce warp-private histograms into per-bin sums and update global histogram
    for (int bin = tid; bin < bins; bin += block_threads) {
        unsigned int sum = 0;
        // Accumulate this bin from all warps' sub-histograms
        for (int w = 0; w < num_warps; ++w) {
            sum += shmem[w * bins + bin];
        }
        if (sum != 0) {
            // Accumulate per-block result into global histogram
            atomicAdd(&global_hist[bin], sum);
        }
    }
}

/*
  Host function: run_histogram

  Launches histogram_range_kernel to compute the histogram of "input" restricted to [from, to].

  Parameters:
  - input:      device pointer (cudaMalloc'd) to a buffer of 'inputSize' chars (plain text).
  - histogram:  device pointer (cudaMalloc'd) to an array of (to - from + 1) unsigned ints.
                This function will zero-initialize it before computing the histogram.
  - inputSize:  number of chars in 'input'.
  - from, to:   inclusive character range [from, to], with 0 <= from <= to <= 255.

  Notes:
  - This function enqueues all operations (memset + kernel launch) into the current stream
    and returns immediately; caller is responsible for synchronization if needed.
  - Uses shared memory per block sized as: (number_of_warps_per_block) * (bins) * sizeof(unsigned int).
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Compute number of bins (inclusive range)
    const int bins = to - from + 1;
    if (bins <= 0 || input == nullptr || histogram == nullptr || inputSize == 0) {
        // Nothing to do; if bins > 0 and inputSize == 0, ensure output is zeroed.
        if (bins > 0 && histogram != nullptr) {
            cudaMemset(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));
        }
        return;
    }

    // Zero-initialize the output histogram
    cudaMemset(histogram, 0, static_cast<size_t>(bins) * sizeof(unsigned int));

    // Launch configuration
    const int block_size = 256;                   // 8 warps per block; good balance for A100/H100
    const int warps_per_block = (block_size + 31) / 32;
    const size_t dynamic_shmem_bytes = static_cast<size_t>(bins) * warps_per_block * sizeof(unsigned int);

    // Choose a grid size based on occupancy for the given dynamic shared memory size
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int blocks_per_sm = 0;
    // Query occupancy; histogram_range_kernel signature is compatible with this API
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        reinterpret_cast<const void*>(histogram_range_kernel),
        block_size,
        dynamic_shmem_bytes
    );

    // Use at least 1 block; typically saturate SMs with occupancy-computed blocks
    int grid_size = blocks_per_sm * prop.multiProcessorCount;
    if (grid_size < 1) grid_size = 1;

    // Launch the kernel
    const unsigned char* d_input_u8 = reinterpret_cast<const unsigned char*>(input);
    histogram_range_kernel<<<grid_size, block_size, dynamic_shmem_bytes>>>(
        d_input_u8,
        histogram,
        inputSize,
        from,
        bins
    );

    // No synchronization here; caller is responsible for stream/device synchronization if needed.
}