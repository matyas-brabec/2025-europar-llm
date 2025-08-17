#include <cuda_runtime.h>
#include <stdint.h>

// Kernel: Compute histogram for character codes in [from, to] inclusive.
//
// Design notes:
// - Each thread block builds its histogram in shared memory to avoid hammering global memory.
// - The shared histogram is replicated per-warp (warp-private sub-histograms) to remove inter-warp contention.
// - Within each warp, we use warp-aggregated atomics: threads with the same key (character bin) are coalesced
//   so that a single atomicAdd adds the total number of occurrences for that key in that warp iteration.
//   This reduces the number of atomic operations dramatically when many threads see the same key.
// - After processing, per-warp histograms are reduced to a per-block histogram and atomically added to the
//   final global histogram. The global histogram must be zero-initialized by the host function before launch.
//
// Assumptions:
// - 0 <= from < to <= 255 (caller responsibility).
// - input points to device memory with 'inputSize' bytes.
// - histogram points to device memory with (to - from + 1) uint entries, zero-initialized before launch.
// - Caller is responsible for host-device synchronization (e.g., cudaDeviceSynchronize if needed).
__global__ void histogram_char_range_kernel(const unsigned char* __restrict__ input,
                                            unsigned int* __restrict__ histogram,
                                            unsigned int inputSize,
                                            int from, int to)
{
    extern __shared__ unsigned int s_hist[]; // Layout: [warp0 bins][warp1 bins]...[warpN bins]

    const int bins = to - from + 1;
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;     // warp index within block
    const int laneId = tid & 31;     // lane index within warp
    const int warpsPerBlock = (blockDim.x + 31) >> 5;

    // Initialize per-warp shared-memory histograms to zero
    for (int i = tid; i < warpsPerBlock * bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    const unsigned int stride = gridDim.x * blockDim.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Grid-stride loop over input
    while (idx < inputSize) {
        const unsigned char c = input[idx];
        const bool valid = (c >= (unsigned char)from) && (c <= (unsigned char)to);
        const int bin = static_cast<int>(c) - from; // Only valid if 'valid' is true

        // Warp-aggregated atomic update to this warp's private histogram
        const unsigned int active = __activemask();
        unsigned int work = __ballot_sync(active, valid);
        while (work) {
            // Pick the first active lane as the leader for the next key
            const int leader = __ffs(work) - 1;
            // Broadcast the key (bin) from the leader
            const int key = __shfl_sync(active, bin, leader);
            // Compute mask of lanes that have this same key (and are valid)
            const unsigned int eq = __ballot_sync(active, valid && (bin == key));
            if (laneId == leader) {
                // Accumulate the count for this key once per warp
                atomicAdd(&s_hist[warpId * bins + key], __popc(eq));
            }
            // Clear the processed lanes
            work &= ~eq;
        }

        idx += stride;
    }

    __syncthreads();

    // Reduce per-warp histograms into a single per-block histogram and add to global
    for (int i = tid; i < bins; i += blockDim.x) {
        unsigned int sum = 0;
        // Accumulate across warps
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * bins + i];
        }
        // Avoid unnecessary atomics when sum is zero
        if (sum) {
            atomicAdd(&histogram[i], sum);
        }
    }
}


// Host function: launch the histogram kernel for range [from, to] inclusive.
// Parameters:
// - input: device pointer to input text (char array on device).
// - histogram: device pointer to output histogram (size = to - from + 1, unsigned int).
// - inputSize: number of bytes in input.
// - from, to: inclusive character ordinal range [0..255], with from < to.
//
// Notes:
// - This function zero-initializes the device histogram buffer before launching the kernel.
// - It queries the current device to size the grid for good occupancy on A100/H100-class GPUs.
// - All operations are performed on the default stream; caller is responsible for synchronization.
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from, int to)
{
    // Basic parameter checks and clamping to [0, 255]
    if (!input || !histogram) return;
    if (from < 0) from = 0;
    if (to > 255) to = 255;
    if (from > to) return;

    const int bins = to - from + 1;

    // Zero the output histogram on device (as kernel uses atomicAdd to accumulate)
    cudaMemsetAsync(histogram, 0, sizeof(unsigned int) * static_cast<size_t>(bins), 0);

    // Configure launch parameters
    const int threadsPerBlock = 256; // 8 warps per block; good balance between occupancy and shared mem usage

    int device = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Cap the grid to a reasonable multiple of SMs for A100/H100
    int maxBlocks = prop.multiProcessorCount * 8;
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks < 1) blocks = 1;
    if (blocks > maxBlocks) blocks = maxBlocks;

    // Dynamic shared memory size: warpsPerBlock * bins * sizeof(unsigned int)
    const int warpsPerBlock = (threadsPerBlock + 31) / 32;
    const size_t shmemBytes = static_cast<size_t>(warpsPerBlock) * static_cast<size_t>(bins) * sizeof(unsigned int);

    const unsigned char* d_input = reinterpret_cast<const unsigned char*>(input);

    // Launch kernel on default stream (0). Caller handles synchronization if needed.
    histogram_char_range_kernel<<<blocks, threadsPerBlock, shmemBytes>>>(
        d_input, histogram, inputSize, from, to);
}