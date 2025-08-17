#include <cuda_runtime.h>
#include <stdint.h>

/*
  CUDA kernel to compute a histogram of characters over a specified inclusive range [from, to].

  Key optimizations:
  - Uses shared memory to build per-warp private histograms, which reduces contention compared to directly
    atomically updating global memory.
  - After processing, per-warp histograms are reduced within the block and then atomically added to the
    global histogram (one atomic add per bin per block).
  - Employs a grid-stride loop for scalable coverage of the input and modest ILP (instruction-level parallelism)
    by processing multiple items per thread per loop iteration.

  Parameters:
  - input:     device pointer to the input character buffer
  - histogram: device pointer to the output histogram of size (to - from + 1)
  - N:         number of input characters
  - from, to:  inclusive range of character codes to count (0 <= from <= to <= 255)

  Requirements:
  - 'histogram' must be zero-initialized by the caller (or by the host wrapper before kernel launch).
  - 'histogram' and 'input' are assumed to be device pointers (cudaMalloc'd).
*/
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int* __restrict__ histogram,
                                       unsigned int N,
                                       int from,
                                       int to)
{
    // Assume blockDim.x is a multiple of warpSize (32). Host will configure accordingly.
    const int warpSizeConst = 32;
    const int numWarps = blockDim.x / warpSizeConst;
    const int warpId   = threadIdx.x / warpSizeConst;
    const int lane     = threadIdx.x & (warpSizeConst - 1);

    const int numBins = to - from + 1;

    // Dynamic shared memory layout: numWarps copies of the histogram, each with numBins entries.
    extern __shared__ unsigned int sMem[];
    unsigned int* sWarpHists = sMem; // size: numWarps * numBins

    // Each warp zeroes its private histogram slice
    unsigned int* sMyHist = sWarpHists + warpId * numBins;
    for (int bin = lane; bin < numBins; bin += warpSizeConst) {
        sMyHist[bin] = 0;
    }
    __syncthreads(); // Ensure all shared histograms are initialized before use

    // Grid-stride loop with modest ILP (processing up to ILP items per loop iteration)
    const unsigned int ILP = 4; // Tweakable; balances latency hiding with register pressure
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t start  = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Treat input as unsigned bytes to avoid sign-extension issues with 'char'
    const unsigned char* inBytes = reinterpret_cast<const unsigned char*>(input);

    for (size_t base = start; base < N; base += stride * ILP) {
        #pragma unroll
        for (unsigned int k = 0; k < ILP; ++k) {
            const size_t idx = base + static_cast<size_t>(k) * stride;
            if (idx < N) {
                const unsigned char v = inBytes[idx];
                // Compute bin in a branch-efficient way:
                // bin = v - from; if 0 <= bin < numBins then it's in range.
                const int bin = static_cast<int>(v) - from;
                if (static_cast<unsigned int>(bin) < static_cast<unsigned int>(numBins)) {
                    // Shared-memory atomic add to the warp-private histogram
                    atomicAdd(&sMyHist[bin], 1u);
                }
            }
        }
    }

    __syncthreads(); // Ensure all updates to shared histograms are complete

    // Reduce per-warp histograms into a block aggregate and update global histogram
    for (int bin = threadIdx.x; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum contributions across all warps for this bin
        #pragma unroll
        for (int w = 0; w < numWarps; ++w) {
            sum += sWarpHists[w * numBins + bin];
        }
        if (sum) {
            // One atomicAdd per bin per block into the global histogram
            atomicAdd(&histogram[bin], sum);
        }
    }
    // No need for further synchronization; kernel writes to global memory are complete at exit.
}

/*
  Host function to launch the histogram kernel.

  - input:     device pointer to input chars (cudaMalloc'd). Size: inputSize bytes.
  - histogram: device pointer to output histogram (cudaMalloc'd). Size: (to - from + 1) * sizeof(unsigned int).
  - inputSize: number of chars in 'input'.
  - from, to:  inclusive range [from, to] of character codes to count.

  Notes:
  - This function zero-initializes the device histogram before launching the kernel.
  - No device synchronization is performed here; the caller is responsible for synchronization.
*/
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = (to - from + 1);
    if (numBins <= 0) {
        // Invalid range; nothing to do. Caller promised 0 <= from < to <= 255, but guard anyway.
        return;
    }

    // Ensure output is clean; kernel performs atomicAdd to global memory.
    cudaMemset(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int));

    if (inputSize == 0) {
        // Nothing to process; histogram already zeroed.
        return;
    }

    // Choose a block size that's a multiple of warp size and balances occupancy and shared memory usage.
    // 512 threads/block -> 16 warps/block. Shared memory usage = numBins * 16 * 4 bytes.
    // Max case (numBins=256): 256 * 16 * 4 = 16384 bytes, well within typical per-block shared memory limits.
    const int threadsPerBlock = 512;
    const int numWarps        = threadsPerBlock / 32;

    // Compute the number of blocks to cover the input with modest ILP (must match kernel's ILP).
    const unsigned int ILP = 4;
    unsigned int blocks = (inputSize + (threadsPerBlock * ILP - 1)) / (threadsPerBlock * ILP);
    if (blocks == 0) blocks = 1;

    // Dynamic shared memory size: per-warp histograms
    const size_t sharedMemBytes = static_cast<size_t>(numBins) * numWarps * sizeof(unsigned int);

    // Launch the kernel on the default stream
    histogram_range_kernel<<<blocks, threadsPerBlock, sharedMemBytes>>>(input, histogram, inputSize, from, to);
}