#include <cuda_runtime.h>
#include <stdint.h>

/*
  Optimized CUDA kernel to compute a histogram over a specified character range [from, to].
  - The input is a device pointer to 'char' (bytes).
  - The output is a device pointer to 'unsigned int' with length (to - from + 1).
  - Range [from, to] is inclusive and 0 <= from < to <= 255.

  Performance strategy:
  - Each block builds its own histogram in shared memory to minimize global atomics.
  - Input is processed with 16-byte coalesced loads (uint4) for bandwidth efficiency.
  - Conditional logic uses a delta check (delta = byte - from; if delta < rangeLen) to test range membership efficiently.
  - The special case [0,255] (full byte range) avoids range checks entirely.
  - After processing, each block atomically accumulates its shared histogram into the global histogram.
*/
__global__ void histogram_range_kernel(const char* __restrict__ input,
                                       unsigned int n,
                                       unsigned int* __restrict__ histogram,
                                       int from,
                                       int to)
{
    extern __shared__ unsigned int s_hist[]; // Dynamic shared memory: (to - from + 1) counters

    const unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);
    const unsigned int from_u   = static_cast<unsigned int>(from);

    // Zero-initialize the per-block shared histogram.
    for (unsigned int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;

    // Fast path if [from, to] == [0, 255]
    const bool fullRange = (from_u == 0u) && (rangeLen == 256u);

    // Vectorized processing: handle as many 16-byte chunks as possible
    const unsigned int nVec       = n >> 4;            // Number of 16-byte units
    const unsigned int tailStart  = nVec << 4;         // Starting byte index of the tail
    const uint4* __restrict__ vIn = reinterpret_cast<const uint4*>(input);

    // Macros to process 4 bytes from a 32-bit word with and without range checks.
    #define PROCESS_WORD_FULL(w_) do {                                     \
        unsigned int _w = (w_);                                            \
        unsigned int _c;                                                   \
        _c = (_w      ) & 0xFFu; atomicAdd(&s_hist[_c], 1u);               \
        _c = (_w >>  8) & 0xFFu; atomicAdd(&s_hist[_c], 1u);               \
        _c = (_w >> 16) & 0xFFu; atomicAdd(&s_hist[_c], 1u);               \
        _c = (_w >> 24) & 0xFFu; atomicAdd(&s_hist[_c], 1u);               \
    } while (0)

    #define PROCESS_WORD_RANGE(w_) do {                                    \
        unsigned int _w = (w_);                                            \
        unsigned int _c, _d;                                               \
        _c = (_w      ) & 0xFFu; _d = _c - from_u; if (_d < rangeLen) atomicAdd(&s_hist[_d], 1u); \
        _c = (_w >>  8) & 0xFFu; _d = _c - from_u; if (_d < rangeLen) atomicAdd(&s_hist[_d], 1u); \
        _c = (_w >> 16) & 0xFFu; _d = _c - from_u; if (_d < rangeLen) atomicAdd(&s_hist[_d], 1u); \
        _c = (_w >> 24) & 0xFFu; _d = _c - from_u; if (_d < rangeLen) atomicAdd(&s_hist[_d], 1u); \
    } while (0)

    // Grid-stride loop over 16-byte (uint4) vectors.
    for (unsigned int i = tid; i < nVec; i += stride) {
        uint4 v = vIn[i];
        if (fullRange) {
            PROCESS_WORD_FULL(v.x);
            PROCESS_WORD_FULL(v.y);
            PROCESS_WORD_FULL(v.z);
            PROCESS_WORD_FULL(v.w);
        } else {
            PROCESS_WORD_RANGE(v.x);
            PROCESS_WORD_RANGE(v.y);
            PROCESS_WORD_RANGE(v.z);
            PROCESS_WORD_RANGE(v.w);
        }
    }

    // Handle any remaining tail bytes (less than 16)
    for (unsigned int i = tailStart + tid; i < n; i += stride) {
        unsigned int c = static_cast<unsigned char>(input[i]); // ensure unsigned value 0..255
        if (fullRange) {
            atomicAdd(&s_hist[c], 1u);
        } else {
            unsigned int d = c - from_u;
            if (d < rangeLen) {
                atomicAdd(&s_hist[d], 1u);
            }
        }
    }

    __syncthreads();

    // Accumulate per-block histogram into the global histogram.
    for (unsigned int i = threadIdx.x; i < rangeLen; i += blockDim.x) {
        unsigned int val = s_hist[i];
        if (val != 0u) {
            atomicAdd(&histogram[i], val);
        }
    }

    #undef PROCESS_WORD_FULL
    #undef PROCESS_WORD_RANGE
}

/*
  Host-side launcher. Assumptions:
  - 'input' and 'histogram' are device pointers allocated via cudaMalloc.
  - 'inputSize' is the number of bytes in 'input'.
  - 'from' and 'to' specify an inclusive range within [0, 255] and from < to.
  - Synchronization is handled by the caller; this function enqueues work on the default stream.

  Behavior:
  - The function zero-initializes the output histogram on the device.
  - It selects a launch configuration suitable for memory-bound workloads on modern data center GPUs.
*/
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Validate parameters minimally (defensive; the problem states valid input).
    if (input == nullptr || histogram == nullptr) return;
    if (from > to) return;

    const unsigned int rangeLen = static_cast<unsigned int>(to - from + 1);

    // Zero the output histogram (asynchronous).
    cudaMemsetAsync(histogram, 0, rangeLen * sizeof(unsigned int));

    if (inputSize == 0u) {
        return; // nothing to do
    }

    // Choose launch parameters.
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const int threadsPerBlock = 256; // good balance for shared-memory atomics and bandwidth

    // Heuristic grid size:
    // - Ensure enough blocks to cover the work assuming each thread handles at least one 16-byte vector.
    // - Also ensure multiple blocks per SM to hide latency.
    unsigned int vecUnits = inputSize >> 4; // number of 16-byte chunks
    unsigned long long blocksFromWork = (vecUnits + threadsPerBlock - 1u) / threadsPerBlock;
    if (blocksFromWork == 0ull) blocksFromWork = 1ull;

    unsigned long long blocks = static_cast<unsigned long long>(prop.multiProcessorCount) * 4ull; // 4x SMs by default
    if (blocks < blocksFromWork) blocks = blocksFromWork;
    if (blocks > static_cast<unsigned long long>(prop.maxGridSize[0])) {
        blocks = static_cast<unsigned long long>(prop.maxGridSize[0]);
    }
    if (blocks == 0ull) blocks = 1ull;

    // Dynamic shared memory size: one counter per bucket.
    size_t shmemSize = static_cast<size_t>(rangeLen) * sizeof(unsigned int);

    // Launch the kernel (asynchronous).
    histogram_range_kernel<<<static_cast<int>(blocks), threadsPerBlock, shmemSize>>>(input, inputSize, histogram, from, to);
}