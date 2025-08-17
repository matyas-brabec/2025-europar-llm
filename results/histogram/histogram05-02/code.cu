#include <cuda_runtime.h>
#include <stdint.h>

// Compile-time tunable: number of items processed by each thread.
// 16 is a good default for Hopper/Ampere for large, bandwidth-bound inputs.
// It provides enough per-thread work to hide latency while keeping register usage reasonable.
static constexpr int itemsPerThread = 16;

// Histogram kernel with shared-memory privatization and per-warp copies to reduce contention.
// - The histogram is restricted to a continuous range [from, to] inclusive.
// - Each thread processes 'itemsPerThread' elements per grid-stride iteration.
// - Each warp maintains a private histogram copy in shared memory (warp-private sub-histograms).
// - The shared histogram is padded/aligned to 32 (number of banks) to minimize bank conflicts.
// - After processing, per-warp histograms are reduced and accumulated into the global histogram.
__global__ void histogram_range_kernel(
    const unsigned char* __restrict__ input, // device pointer to input chars
    unsigned int n,                           // number of input chars
    unsigned int* __restrict__ g_hist,        // device pointer to output histogram (size = range)
    int from,                                 // range start (inclusive)
    int range,                                // number of bins = to - from + 1
    int stride                                // padded stride for one warp's sub-histogram in shared memory
){
    extern __shared__ unsigned int s_hist[]; // size: warpsPerBlock * stride

    const int tid   = threadIdx.x;
    const int bdim  = blockDim.x;
    const int bid   = blockIdx.x;
    const int warpsPerBlock = bdim / warpSize;
    const int warpId = tid / warpSize;

    // Zero the shared-memory histograms (all warp-private copies)
    for (int i = tid; i < warpsPerBlock * stride; i += bdim) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop setup:
    // For coalesced loads, for each j in [0, itemsPerThread), thread t in block reads input at:
    // base + j*blockDim.x + t. Across threads in a warp, addresses are contiguous for a fixed j.
    const size_t blockWork = (size_t)bdim * (size_t)itemsPerThread;
    size_t base = (size_t)bid * blockWork + (size_t)tid;
    const size_t gridStride = (size_t)gridDim.x * blockWork;

    while (base < n) {
        #pragma unroll
        for (int j = 0; j < itemsPerThread; ++j) {
            size_t pos = base + (size_t)j * (size_t)bdim;
            if (pos >= n) break;

            // Treat chars as unsigned to get ordinals 0..255 regardless of signed-char default
            unsigned int c = static_cast<unsigned int>(input[pos]);
            int bin = static_cast<int>(c) - from;
            // Fast in-range check: (unsigned)bin < (unsigned)range is valid for bin in [-inf, +inf]
            if ((unsigned)bin < (unsigned)range) {
                // Atomic add into warp-private histogram to avoid cross-warp bank conflicts/contention.
                // The stride is padded to 32 to keep each warp's sub-hist aligned to 32-bank boundaries.
                atomicAdd(&s_hist[warpId * stride + bin], 1u);
            }
        }
        base += gridStride;
    }
    __syncthreads();

    // Reduce per-warp histograms into a single global histogram using atomics.
    // This step incurs at most 'warpsPerBlock' additions per bin per block, which is small.
    for (int b = tid; b < range; b += bdim) {
        unsigned int sum = 0;
        #pragma unroll
        for (int w = 0; w < warpsPerBlock; ++w) {
            sum += s_hist[w * stride + b];
        }
        if (sum) {
            atomicAdd(&g_hist[b], sum);
        }
    }
}

// Host-side launcher.
// - input: device pointer to input bytes (cudaMalloc'd), size = inputSize
// - histogram: device pointer to output histogram (cudaMalloc'd), size = to - from + 1
// - inputSize: number of bytes in input
// - from, to: inclusive character ordinal range to histogram (0 <= from < to <= 255)
// Synchronization is assumed to be handled by the caller; this function does not call cudaDeviceSynchronize.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    if (inputSize == 0) return;

    const int range = to - from + 1;

    // Pad stride to a multiple of 32 (number of shared memory banks) to minimize bank conflicts
    // and to ensure each warp's sub-histogram starts at a fresh bank boundary.
    const int stride = ((range + 31) / 32) * 32;

    // Execution configuration tuned for modern data center GPUs.
    //  - 256 threads/block is a sweet spot for occupancy and shared memory usage.
    //  - Choose grid size based on SM count to provide enough parallelism. Grid-stride loop handles any size.
    const int blockSize = 256;
    int device = 0;
    cudaGetDevice(&device);
    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);

    // Many small blocks per SM to keep the device busy; excess blocks will be queued by the scheduler.
    // The input is large; grid-stride loop ensures full coverage regardless of grid size.
    int gridSize = smCount > 0 ? smCount * 32 : 1024;
    if (gridSize > 65535) gridSize = 65535;

    // Dynamic shared memory size: one warp-private histogram per warp in the block
    const int warpsPerBlock = blockSize / 32;
    const size_t shmemSize = (size_t)warpsPerBlock * (size_t)stride * sizeof(unsigned int);

    // Initialize output histogram to zero on device
    cudaMemset(histogram, 0, (size_t)range * sizeof(unsigned int));

    // Launch
    histogram_range_kernel<<<gridSize, blockSize, shmemSize>>>(
        reinterpret_cast<const unsigned char*>(input),
        inputSize,
        histogram,
        from,
        range,
        stride
    );
}