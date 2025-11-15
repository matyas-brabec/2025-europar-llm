#include <cuda_runtime.h>

/*
 * CUDA histogram kernel with shared-memory privatization and bank-conflict avoidance.
 *
 * - Computes a histogram over characters in [from, to] (inclusive).
 * - Input:  'input'      - device pointer to char buffer of size 'inputSize'.
 * - Output: 'histogram'  - device pointer to unsigned int buffer of size (to - from + 1),
 *                          where histogram[i] counts occurrences of char (i + from).
 *
 * Optimization details:
 * - Each thread processes ITEMS_PER_THREAD input characters.
 * - Per-block histogram is stored in shared memory with 32 copies (one per warp lane).
 *   Layout: sharedHist[bin * WARP_SIZE + lane], where lane = threadIdx.x % 32.
 *   This ensures that for a given bin, all 32 lanes access different banks:
 *     bank_index = (bin * WARP_SIZE + lane) % 32 == lane
 *   so intra-warp bank conflicts are avoided.
 * - Threads update shared memory using atomicAdd (shared-memory atomics are fast on
 *   modern GPUs). Multiple warps in a block can share the same lane-specific copy.
 * - At the end of the kernel, the 32 copies are summed per bin and added to the
 *   global histogram using atomicAdd.
 */

static constexpr int WARP_SIZE        = 32;
/* ITEMS_PER_THREAD controls how many characters each thread processes.
 * A value of 8 gives a good balance between work per thread and occupancy
 * on modern data-center GPUs (e.g., A100, H100) for large inputs. */
static constexpr int ITEMS_PER_THREAD = 8;
/* THREADS_PER_BLOCK should be a multiple of WARP_SIZE. 256 is a good
 * default for high occupancy and efficient shared memory usage. */
static constexpr int THREADS_PER_BLOCK = 256;

__global__ void histogram_kernel(const char * __restrict__ input,
                                 unsigned int * __restrict__ histogram,
                                 unsigned int inputSize,
                                 int from,
                                 int to)
{
    extern __shared__ unsigned int sharedHist[];  // size: numBins * WARP_SIZE

    const int numBins = to - from + 1;
    const int tid     = threadIdx.x;
    const int lane    = tid & (WARP_SIZE - 1);      // thread's lane within its warp
    const int globalThreadId = blockIdx.x * blockDim.x + tid;

    // 1. Zero out shared histogram (all 32 copies).
    // Each thread zeroes multiple elements in a strided fashion.
    const int sharedSize = numBins * WARP_SIZE;
    for (int idx = tid; idx < sharedSize; idx += blockDim.x) {
        sharedHist[idx] = 0;
    }
    __syncthreads();

    // Pointer to this lane's column in the 2D shared histogram:
    // For bin b, this lane's counter is at sharedHist[b * WARP_SIZE + lane].
    unsigned int *threadHistColumn = sharedHist + lane;

    // 2. Process input characters.
    // Each thread processes up to ITEMS_PER_THREAD characters, laid out so that
    // accesses are coalesced across threads in a block.
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        unsigned int idx = static_cast<unsigned int>(globalThreadId * ITEMS_PER_THREAD + i);
        if (idx >= inputSize) break;

        unsigned char c = static_cast<unsigned char>(input[idx]);
        int val = static_cast<int>(c);

        // Map character to bin if within [from, to].
        int bin = val - from;
        if (bin >= 0 && bin < numBins) {
            // Use shared-memory atomicAdd because multiple warps may share the same
            // lane-specific copy. Bank conflicts are avoided by the swizzled layout.
            atomicAdd(&threadHistColumn[bin * WARP_SIZE], 1u);
        }
    }

    __syncthreads();

    // 3. Reduce the 32 lane-specific copies per bin and update global histogram.
    // Each thread handles multiple bins in a strided fashion.
    for (int bin = tid; bin < numBins; bin += blockDim.x) {
        unsigned int sum = 0;
#pragma unroll
        for (int l = 0; l < WARP_SIZE; ++l) {
            sum += sharedHist[bin * WARP_SIZE + l];
        }
        if (sum != 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

/*
 * Host wrapper to launch the histogram kernel.
 *
 * Parameters:
 *   input      - device pointer to input text buffer (cudaMalloc'ed).
 *   histogram  - device pointer to output histogram buffer (cudaMalloc'ed),
 *                with at least (to - from + 1) unsigned int elements.
 *   inputSize  - number of characters in 'input'.
 *   from, to   - character range [from, to] (0 <= from < to <= 255).
 *
 * Notes:
 *   - This function does not perform host-device synchronization; the caller is
 *     responsible for synchronizing (e.g., cudaDeviceSynchronize) before using
 *     the results.
 *   - The histogram buffer is zero-initialized on the device before the kernel
 *     launch using cudaMemsetAsync.
 */
void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    const int numBins = to - from + 1;

    // Zero the output histogram on the device (asynchronously in stream 0).
    cudaMemsetAsync(histogram, 0, static_cast<size_t>(numBins) * sizeof(unsigned int), 0);

    // Determine grid size. Each thread processes ITEMS_PER_THREAD items.
    const unsigned int itemsPerBlock = THREADS_PER_BLOCK * ITEMS_PER_THREAD;
    unsigned int numBlocks = (inputSize + itemsPerBlock - 1) / itemsPerBlock;
    if (numBlocks == 0) {
        numBlocks = 1;  // handle inputSize == 0
    }

    // Shared memory size: numBins * WARP_SIZE counters.
    const size_t sharedMemSize = static_cast<size_t>(numBins) * WARP_SIZE * sizeof(unsigned int);

    histogram_kernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(input,
                                                                      histogram,
                                                                      inputSize,
                                                                      from,
                                                                      to);
}