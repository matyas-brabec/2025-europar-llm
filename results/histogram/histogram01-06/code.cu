#include <cuda_runtime.h>
#include <stdint.h>

/*
  High-level approach and performance notes:
  - We compute a histogram for a contiguous range [from, to] (0..255) over an input char buffer.
  - The output histogram has (to - from + 1) bins and resides in device memory.
  - To minimize global atomic contention, we:
    1) Use per-warp private histograms in shared memory (size: bins * warpsPerBlock).
    2) Use warp-aggregated updates: for a given warp, all equal values are aggregated so a single lane
       performs the increment, reducing the number of atomic-like operations to shared memory.
    3) At the end, we reduce per-warp histograms to a per-block sum and perform one global atomicAdd per bin per block.

  - The kernel uses grid-stride loops and vectorized loads (uchar4) for memory throughput.
  - The host launcher computes a good grid size via CUDA occupancy API and sets appropriate dynamic shared memory size.
  - The output histogram buffer is memset to zero on device before the kernel launch. Synchronization is left to the caller.
*/

static __device__ __forceinline__ void warp_aggregated_increment(int bin, unsigned int* warp_hist, int bins)
{
    // This function aggregates increments within a warp so that for each unique 'bin' value
    // present among active threads, only one lane performs a single increment equal to the
    // number of lanes requesting that bin.
    //
    // - bin: 0..bins-1 for valid, or -1 to indicate "inactive/no-op"
    // - warp_hist: pointer to the calling warp's private histogram (bins elements)
    // - bins: number of histogram bins
    //
    // It is safe to call this function under warp divergence because we use __activemask()
    // to restrict collectives to the currently active threads.
    unsigned int mask_all = __activemask();
    unsigned int active = __ballot_sync(mask_all, bin >= 0);  // lanes needing an update
    int lane = threadIdx.x & 31;

    while (active)
    {
        int leader = __ffs(active) - 1; // index (0..31) of the first active lane
        int leader_bin = __shfl_sync(mask_all, bin, leader);
        // Collect all lanes whose 'bin' matches leader_bin.
        unsigned int eq = __ballot_sync(mask_all, bin == leader_bin);
        int count = __popc(eq);
        if (lane == leader)
        {
            // Single-lane, race-free update to the warp-private histogram.
            warp_hist[leader_bin] += count;
        }
        active &= ~eq; // remove the lanes we've already handled
    }
}

__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ hist,
                                       unsigned int n,
                                       int from,
                                       int to)
{
    // Compute number of bins for the requested range [from, to].
    const int bins = to - from + 1;

    // Shared memory layout:
    // [ warp private histograms (bins * warpsPerBlock) ]
    extern __shared__ unsigned int shmem[];
    const int warpsPerBlock = (blockDim.x + warpSize - 1) / warpSize;
    unsigned int* warpHists = shmem; // length = bins * warpsPerBlock

    // Zero-initialize all per-warp histograms.
    for (int i = threadIdx.x; i < bins * warpsPerBlock; i += blockDim.x) {
        warpHists[i] = 0;
    }
    __syncthreads();

    // Alias input as bytes for unsigned interpretation and vectorized loads.
    const unsigned char* in = reinterpret_cast<const unsigned char*>(input);
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);

    const unsigned int n4 = n / 4;      // number of uchar4 elements
    const unsigned int tail_start = n4 * 4; // index of first trailing byte

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;

    // Each warp uses its own histogram slice to avoid inter-warp conflicts.
    const int warpId = threadIdx.x / warpSize;
    unsigned int* warp_hist = warpHists + warpId * bins;

    // Process main body with vectorized loads (uchar4).
    for (unsigned int i4 = idx; i4 < n4; i4 += stride) {
        uchar4 v = in4[i4];

        // For each byte, compute its bin or -1 if out of range, then do a warp-aggregated increment.
        int b0 = static_cast<int>(v.x) - from;
        b0 = (b0 >= 0 && b0 < bins) ? b0 : -1;
        warp_aggregated_increment(b0, warp_hist, bins);

        int b1 = static_cast<int>(v.y) - from;
        b1 = (b1 >= 0 && b1 < bins) ? b1 : -1;
        warp_aggregated_increment(b1, warp_hist, bins);

        int b2 = static_cast<int>(v.z) - from;
        b2 = (b2 >= 0 && b2 < bins) ? b2 : -1;
        warp_aggregated_increment(b2, warp_hist, bins);

        int b3 = static_cast<int>(v.w) - from;
        b3 = (b3 >= 0 && b3 < bins) ? b3 : -1;
        warp_aggregated_increment(b3, warp_hist, bins);
    }

    // Handle up to 3 trailing bytes: let the first 'rem' global threads process one byte each,
    // while still calling warp_aggregated_increment from all active lanes.
    const unsigned int rem = n - tail_start;
    int tbin = -1;
    if (idx < rem) {
        unsigned char tb = in[tail_start + idx];
        int tb_bin = static_cast<int>(tb) - from;
        tbin = (tb_bin >= 0 && tb_bin < bins) ? tb_bin : -1;
    }
    // All lanes execute; lanes not assigned a tail byte pass tbin = -1 (no-op).
    warp_aggregated_increment(tbin, warp_hist, bins);

    __syncthreads();

    // Reduce per-warp histograms and update global memory once per bin per block.
    for (int bin = threadIdx.x; bin < bins; bin += blockDim.x) {
        unsigned int total = 0;
        for (int w = 0; w < warpsPerBlock; ++w) {
            total += warpHists[w * bins + bin];
        }
        atomicAdd(&hist[bin], total);
    }
}

void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate range assumptions (as per problem statement they hold):
    // 0 <= from < to <= 255. We handle also the degenerate case from == to.
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (from > to) return;

    const int bins = to - from + 1;

    // Ensure output is clean before accumulation.
    cudaMemset(histogram, 0, bins * sizeof(unsigned int));

    // Choose a performant configuration.
    const int blockSize = 256; // 8 warps, good balance for H100/A100
    const int warpsPerBlock = (blockSize + 31) / 32;
    const size_t sharedBytes = static_cast<size_t>(bins) * warpsPerBlock * sizeof(unsigned int);

    // Use occupancy to determine a grid size that fills the device.
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM,
                                                  histogram_range_kernel,
                                                  blockSize,
                                                  sharedBytes);

    int gridSize = maxBlocksPerSM * prop.multiProcessorCount;
    if (gridSize <= 0) gridSize = prop.multiProcessorCount; // fallback

    // Launch the kernel; caller is responsible for synchronization.
    histogram_range_kernel<<<gridSize, blockSize, sharedBytes>>>(input, histogram, inputSize, from, to);
}