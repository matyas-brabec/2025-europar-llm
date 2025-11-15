#include <cuda_runtime.h>
#include <cassert>

/*
 * CUDA kernel that computes a histogram for a given character range [from, from+numBins-1].
 *
 * Optimization strategy:
 *  - Each block maintains a private histogram in shared memory (shHist).
 *  - Threads atomically update the shared histogram instead of the global one,
 *    drastically reducing contention on global memory.
 *  - After processing their portion of the input, blocks cooperatively merge
 *    their shared histograms into the global histogram using global atomics
 *    (one atomic add per bin per block).
 *
 * Notes:
 *  - The input characters are treated as unsigned bytes (0..255).
 *  - Only characters in the specified range contribute to the histogram.
 *  - numBins is expected to be <= 256 (range of possible byte values).
 */
__global__ void histogram_kernel_range_shared(
    const char * __restrict__ input,   // device pointer to input text
    unsigned int * __restrict__ globalHist, // device pointer to output histogram (numBins entries)
    unsigned int inputSize,            // number of characters in input
    int from,                          // inclusive lower bound of character range
    int numBins                        // number of bins = to - from + 1
)
{
    extern __shared__ unsigned int shHist[]; // shared memory histogram (size numBins)

    // Initialize shared-memory histogram to zero.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        shHist[i] = 0;
    }
    __syncthreads();

    const unsigned int tid    = threadIdx.x;
    const unsigned int tcount = blockDim.x;
    const unsigned int bid    = blockIdx.x;
    const unsigned int bcount = gridDim.x;

    // Grid-stride loop over input.
    // Each thread processes multiple characters, spaced by total number of threads in the grid.
    for (unsigned int idx = bid * tcount + tid; idx < inputSize; idx += tcount * bcount) {
        // Load character as unsigned to avoid sign-extension issues.
        unsigned char c = static_cast<unsigned char>(input[idx]);
        int val = static_cast<int>(c);

        // Check if character lies within the specified range [from, from+numBins-1].
        if (val >= from && val < from + numBins) {
            int bin = val - from;
            // Atomic update in shared memory: much faster and less contended than global atomics.
            atomicAdd(&shHist[bin], 1u);
        }
    }

    __syncthreads();

    // Merge block-local histogram into the global histogram.
    // Each thread is responsible for a subset of bins.
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        unsigned int count = shHist[i];
        if (count > 0) {
            atomicAdd(&globalHist[i], count);
        }
    }
}

/*
 * Host helper to round up division.
 */
static inline int div_up_int(int a, int b)
{
    return (a + b - 1) / b;
}

/*
 * run_histogram
 *
 * Computes the histogram of characters in device array `input` restricted to the
 * continuous range [from, to]. The result is stored in device array `histogram`.
 *
 * Parameters:
 *  - input      : device pointer (cudaMalloc'd) to input characters
 *  - histogram  : device pointer (cudaMalloc'd) to histogram array of size (to - from + 1)
 *  - inputSize  : number of characters in `input`
 *  - from, to   : character range [from, to], with 0 <= from < to <= 255
 *
 * Behavior:
 *  - The function zeroes the histogram array on the device.
 *  - It launches a CUDA kernel that builds per-block histograms in shared memory,
 *    then merges them into the global histogram.
 *  - No device synchronization is performed here; the caller is responsible for it.
 */
void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to)
{
    assert(input  != nullptr);
    assert(histogram != nullptr);
    assert(from >= 0 && from <= 255);
    assert(to   >= 0 && to   <= 255);
    assert(from <= to);

    const int numBins = to - from + 1;

    // Zero the global histogram on device.
    cudaMemset(histogram, 0, numBins * sizeof(unsigned int));

    // If there is no input, we are done (histogram is already zeroed).
    if (inputSize == 0) {
        return;
    }

    // Kernel launch configuration.
    // A reasonable default for modern GPUs is 256 threads per block.
    const int blockSize = 256;

    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Choose grid size based on input size and hardware.
    // Limit to a multiple of SMs to avoid oversubscription of tiny workloads.
    int gridSize = div_up_int(static_cast<int>(inputSize), blockSize);
    int maxBlocks = prop.multiProcessorCount * 16; // up to 16 blocks per SM as a heuristic
    if (gridSize > maxBlocks) {
        gridSize = maxBlocks;
    }

    // Dynamic shared memory size: one unsigned int per bin.
    size_t sharedMemSize = static_cast<size_t>(numBins) * sizeof(unsigned int);

    // Launch the optimized histogram kernel.
    histogram_kernel_range_shared<<<gridSize, blockSize, sharedMemSize>>>(
        input,
        histogram,
        inputSize,
        from,
        numBins
    );

    // No cudaDeviceSynchronize() here; caller handles synchronization and error checking if desired.
}