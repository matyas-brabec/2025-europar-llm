You are an experienced CUDA programmer trying to provide the most optimized code for the given task and the target hardware.
Assume that the target hardware is a modern NVIDIA GPU for data centers such as the H100 or A100, and the code is compiled using the latest CUDA toolkit and the latest compatible host compiler.
The output should contain only the requested source code. The code must not contain any placeholders or unimplemented parts. All high-level overviews and specific explanations should be included in code comments. Do not output any non-source text.

Write a CUDA kernel that will compute a histogram of a text file restricted to a particular char range. The input is plain text stored as an array of chars. The function gets parameters `from` and `to` (0 <= `from` < `to` <= 255) that specify a continuous range `from`--`to` of the histogram that should be computed. The output is an array of `to` - `from` + 1 unsigned integers where position `i` holds the count of occurrences of character with ordinal value `i` + `from`. Besides the CUDA kernel, write a regular C++ function with the following descriptor that invokes the CUDA kernel with appropriate parameters.
```c++
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
```
Assume the `input` and `histogram` arrays are duly allocated by `cudaMalloc` and the `inputSize` is the number of chars in the `input` buffer. Any host-device synchronization is handled by the caller of `run_histogram`.

---

```c++
#include <cuda_runtime.h>

// CUDA kernel to compute a histogram for a restricted range of characters.
// The kernel uses shared memory to accumulate per-block histograms for efficiency.
// Input:
//   input: pointer to the device memory buffer holding the text.
//   inputSize: the number of characters in the input buffer.
//   from: the lower bound (inclusive) of the character range.
//   to: the upper bound (inclusive) of the character range.
//   histogram: pointer to the device memory buffer for the output histogram. Its size must be (to - from + 1).
__global__ void histogramKernel(const char* input, unsigned int inputSize, int from, int to, unsigned int* histogram) {
    // Compute the size of the histogram range.
    int range = to - from + 1;

    // Declare an externally allocated shared memory array for the per-block histogram.
    extern __shared__ unsigned int localHist[];

    // Initialize the shared histogram to zero.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Compute global thread index and grid stride.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple input characters using grid-stride loop.
    while (idx < inputSize) {
        // Cast char to unsigned char and then to unsigned int to handle potential negative values.
        unsigned int c_val = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
        // Only update the histogram if the character is within the specified range.
        if (c_val >= static_cast<unsigned int>(from) && c_val <= static_cast<unsigned int>(to)) {
            // Use atomic operation to update the shared histogram.
            atomicAdd(&localHist[c_val - from], 1);
        }
        idx += stride;
    }
    __syncthreads();

    // Merge the per-block histogram into the global histogram using atomic operations.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        atomicAdd(&histogram[i], localHist[i]);
    }
}

// Host function to configure and launch the CUDA kernel.
// The function assumes that 'input' and 'histogram' have been allocated on the device,
// and that host-device synchronization is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Define the number of threads per block.
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed to cover the input.
    int blocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate the shared memory size required: one unsigned int per histogram bin.
    int range = to - from + 1;
    size_t sharedMemSize = range * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);

    // Note: Any necessary error checking and host-device synchronization
    // (e.g., cudaDeviceSynchronize) is assumed to be performed by the caller.
}
```

---

Add a constant itemsPerThread to the code that controls how many input chars are processed by each thread. Select the appropriate default value for itemsPerThread, assuming the kernel should be optimized for the latest NVIDIA GPUs and the input size is large.

---

```c++
#include <cuda_runtime.h>

// Define the number of items each thread will process in a single iteration.
// This value is chosen to balance memory throughput and computation on modern NVIDIA GPUs.
constexpr int itemsPerThread = 8;

// CUDA kernel to compute a histogram for a restricted range of characters.
// The kernel uses shared memory to accumulate per-block histograms for efficiency.
// Input:
//   input: pointer to the device memory buffer holding the text.
//   inputSize: the number of characters in the input buffer.
//   from: the lower bound (inclusive) of the character range.
//   to: the upper bound (inclusive) of the character range.
//   histogram: pointer to the device memory buffer for the output histogram. Its size must be (to - from + 1).
__global__ void histogramKernel(const char* input, unsigned int inputSize, int from, int to, unsigned int* histogram) {
    // Compute the size of the histogram range.
    int range = to - from + 1;

    // Declare an externally allocated shared memory array for the per-block histogram.
    extern __shared__ unsigned int localHist[];

    // Initialize the shared histogram to zero.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Calculate the global thread id.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread processes 'itemsPerThread' consecutive characters per iteration.
    int base = globalThreadId * itemsPerThread;
    // The stride is scaled by itemsPerThread.
    int stride = blockDim.x * gridDim.x * itemsPerThread;

    // Process multiple items per thread using a grid-stride loop.
    while (base < inputSize) {
        #pragma unroll
        for (int i = 0; i < itemsPerThread; i++) {
            int idx = base + i;
            if (idx < inputSize) {
                // Cast char to unsigned char and then to unsigned int to handle potential negative values.
                unsigned int c_val = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
                // Only update the histogram if the character is within the specified range.
                if (c_val >= static_cast<unsigned int>(from) && c_val <= static_cast<unsigned int>(to)) {
                    // Use atomic operation to update the shared histogram.
                    atomicAdd(&localHist[c_val - from], 1);
                }
            }
        }
        base += stride;
    }
    __syncthreads();

    // Merge the per-block histogram into the global histogram using atomic operations.
    for (int i = threadIdx.x; i < range; i += blockDim.x) {
        atomicAdd(&histogram[i], localHist[i]);
    }
}

// Host function to configure and launch the CUDA kernel.
// The function assumes that 'input' and 'histogram' have been allocated on the device,
// and that host-device synchronization is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Define the number of threads per block.
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed to cover the input.
    int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);

    // Calculate the shared memory size required: one unsigned int per histogram bin.
    int range = to - from + 1;
    size_t sharedMemSize = range * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);

    // Note: Any necessary error checking and host-device synchronization
    // (e.g., cudaDeviceSynchronize) is assumed to be performed by the caller.
}
```

---

Update the previously generated solution to avoid shared memory bank conflicts by using multiple copies of the histogram per thread block. Create 32 copies of the histogram and place each copy in a separate bank (using appropriate stride). Make sure that each copy is accessed by threads from the same warp lanes to avoid intra-warp bank collisions.

---

```c++
#include <cuda_runtime.h>

// Define the number of items each thread will process in a single iteration.
// This value is chosen to balance memory throughput and computation on modern NVIDIA GPUs.
constexpr int itemsPerThread = 8;
// Define the number of copies (banks) of the histogram per block to avoid shared memory bank conflicts.
constexpr int numBanks = 32;

// CUDA kernel to compute a histogram for a restricted range of characters.
// The kernel uses multiple copies of the histogram in shared memory to avoid bank conflicts.
// Each block allocates 'numBanks' copies of the histogram, each of size (to - from + 1).
// Input:
//   input: pointer to the device memory buffer holding the text.
//   inputSize: the number of characters in the input buffer.
//   from: the lower bound (inclusive) of the character range.
//   to: the upper bound (inclusive) of the character range.
//   histogram: pointer to the device memory buffer for the output histogram.
//              Its size must be (to - from + 1). The final result is obtained by summing
//              the corresponding bins from all blocks.
__global__ void histogramKernel(const char* input, unsigned int inputSize, int from, int to, unsigned int* histogram) {
    // Compute the number of histogram bins.
    int range = to - from + 1;

    // Allocate shared memory for multiple copies of the histogram.
    // Total shared memory size per block: numBanks * range integers.
    extern __shared__ unsigned int localHist[];

    // Initialize all copies of the shared histogram to zero.
    // Each thread initializes multiple entries by striding over the entire shared memory array.
    for (int i = threadIdx.x; i < numBanks * range; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Determine which copy (bank) this thread will update.
    // Using threadIdx.x % numBanks ensures that consecutive warp lanes use different banks,
    // thereby avoiding intra-warp bank conflicts.
    int bank = threadIdx.x % numBanks;

    // Compute the global thread ID.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the starting index for this thread.
    int base = globalThreadId * itemsPerThread;
    // Calculate the total stride, scaled by itemsPerThread.
    int stride = blockDim.x * gridDim.x * itemsPerThread;

    // Process input characters in a grid-stride loop, handling 'itemsPerThread' characters per iteration.
    while (base < inputSize) {
        #pragma unroll
        for (int i = 0; i < itemsPerThread; i++) {
            int idx = base + i;
            if (idx < inputSize) {
                // Convert the character to an unsigned value.
                unsigned int c_val = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
                // Update the histogram only if the character is within the specified range.
                if (c_val >= static_cast<unsigned int>(from) && c_val <= static_cast<unsigned int>(to)) {
                    // Update the thread's designated bank for the histogram.
                    atomicAdd(&localHist[bank * range + (c_val - from)], 1);
                }
            }
        }
        base += stride;
    }
    __syncthreads();

    // Each thread aggregates one or more bins from all bank copies and updates the global histogram.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum over all the copies for this histogram bin.
        for (int b = 0; b < numBanks; b++) {
            sum += localHist[b * range + bin];
        }
        // Atomically update the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
}

// Host function to configure and launch the CUDA kernel.
// The function assumes that 'input' and 'histogram' have been allocated on the device,
// and that host-device synchronization is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Define the number of threads per block.
    int threadsPerBlock = 256;
    // Calculate the number of blocks required, accounting for items processed per thread.
    int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);

    // Compute the number of histogram bins.
    int range = to - from + 1;
    // Calculate the shared memory size required: numBanks copies each of size 'range'.
    size_t sharedMemSize = numBanks * range * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);

    // Note: Any necessary error checking and host-device synchronization
    // (e.g., cudaDeviceSynchronize) is assumed to be performed by the caller.
}
```

---

Update the previously generated solution. Store the histogram values in the shared memory so that value `i` of histogram copy `c` (0 <= `c` < 32) is located at offset `i * 32 + c`.

---

```c++
#include <cuda_runtime.h>

// Define the number of items each thread will process in a single iteration.
// This value is chosen to balance memory throughput and computation on modern NVIDIA GPUs.
constexpr int itemsPerThread = 8;
// Define the number of copies (banks) of the histogram per block to avoid shared memory bank conflicts.
constexpr int numBanks = 32;

// CUDA kernel to compute a histogram for a restricted range of characters.
// This version avoids shared memory bank conflicts by using multiple copies of the histogram.
// The shared memory layout is organized so that the value for bin 'i' in histogram copy 'c' is stored
// at offset: i * numBanks + c. This layout ensures that threads from the same warp access different banks.
// Input:
//   input: pointer to the device memory buffer holding the text.
//   inputSize: the number of characters in the input buffer.
//   from: the lower bound (inclusive) of the character range.
//   to: the upper bound (inclusive) of the character range.
//   histogram: pointer to the device memory buffer for the output histogram.
//              Its size must be (to - from + 1). The final result is obtained by summing
//              the corresponding bins from all blocks.
__global__ void histogramKernel(const char* input, unsigned int inputSize, int from, int to, unsigned int* histogram) {
    // Compute the number of histogram bins.
    int range = to - from + 1;

    // Allocate shared memory for multiple copies of the histogram.
    // The shared memory is organized such that for each bin 'i' (0 <= i < range) and bank 'c' (0 <= c < numBanks),
    // the histogram value is stored at offset: i * numBanks + c.
    extern __shared__ unsigned int localHist[];

    // Initialize all copies of the shared histogram to zero.
    for (int i = threadIdx.x; i < range * numBanks; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Determine which copy (bank) this thread will update.
    // Using threadIdx.x % numBanks ensures that consecutive warp lanes use different banks,
    // thereby avoiding intra-warp bank conflicts.
    int bank = threadIdx.x % numBanks;

    // Compute the global thread ID.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the starting index for this thread.
    int base = globalThreadId * itemsPerThread;
    // Calculate the total stride, scaled by itemsPerThread.
    int stride = blockDim.x * gridDim.x * itemsPerThread;

    // Process input characters in a grid-stride loop, handling 'itemsPerThread' characters per iteration.
    while (base < inputSize) {
        #pragma unroll
        for (int i = 0; i < itemsPerThread; i++) {
            int idx = base + i;
            if (idx < inputSize) {
                // Convert the character to an unsigned value.
                unsigned int c_val = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
                // Update the histogram only if the character is within the specified range.
                if (c_val >= static_cast<unsigned int>(from) && c_val <= static_cast<unsigned int>(to)) {
                    // Compute the offset for the given bin and bank.
                    // The value for bin (c_val - from) in bank 'bank' is at offset: (c_val - from) * numBanks + bank.
                    atomicAdd(&localHist[(c_val - from) * numBanks + bank], 1);
                }
            }
        }
        base += stride;
    }
    __syncthreads();

    // Each thread aggregates one or more bins from all bank copies and updates the global histogram.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum the values from all banks for this histogram bin.
        for (int c = 0; c < numBanks; c++) {
            sum += localHist[bin * numBanks + c];
        }
        // Atomically update the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
}

// Host function to configure and launch the CUDA kernel.
// The function assumes that 'input' and 'histogram' have been allocated on the device,
// and that host-device synchronization is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Define the number of threads per block.
    int threadsPerBlock = 256;
    // Calculate the number of blocks required, accounting for items processed per thread.
    int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);

    // Compute the number of histogram bins.
    int range = to - from + 1;
    // Calculate the shared memory size required: numBanks copies each of size 'range'.
    size_t sharedMemSize = numBanks * range * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);

    // Note: Any necessary error checking and host-device synchronization
    // (e.g., cudaDeviceSynchronize) is assumed to be performed by the caller.
}
```

---

Update the previously generated solution. Increase the number of items processed by each thread. Also modify the iteration pattern, so that threads in a warp perform coalesced loads from global memory.

---

```c++
#include <cuda_runtime.h>

// Increase the number of items processed by each thread for higher throughput.
// This value is chosen to maximize memory throughput on modern NVIDIA GPUs.
constexpr int itemsPerThread = 32;
// Define the number of copies (banks) of the histogram per block to avoid shared memory bank conflicts.
constexpr int numBanks = 32;

// CUDA kernel to compute a histogram for a restricted range of characters.
// This version avoids shared memory bank conflicts by using multiple copies of the histogram.
// The shared memory layout is organized so that the value for bin 'i' in histogram copy 'c'
// is stored at offset: i * numBanks + c. The kernel also uses a warp-level iteration pattern
// to ensure that threads in the same warp perform coalesced loads from global memory.
// Input:
//   input: pointer to the device memory buffer holding the text.
//   inputSize: the number of characters in the input buffer.
//   from: the lower bound (inclusive) of the character range.
//   to: the upper bound (inclusive) of the character range.
//   histogram: pointer to the device memory buffer for the output histogram.
//              Its size must be (to - from + 1). The final result is obtained by summing
//              the corresponding bins from all blocks.
__global__ void histogramKernel(const char* input, unsigned int inputSize, int from, int to, unsigned int* histogram) {
    // Calculate the number of histogram bins.
    int range = to - from + 1;

    // Allocate shared memory for multiple copies of the histogram.
    // Shared memory is organized so that for each bin 'i' and bank 'c',
    // the value is stored at offset: i * numBanks + c.
    extern __shared__ unsigned int localHist[];

    // Initialize all copies of the shared histogram to zero.
    for (int i = threadIdx.x; i < range * numBanks; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Determine which copy (bank) this thread will update.
    // Using threadIdx.x % numBanks ensures that consecutive warp lanes use different banks,
    // thereby avoiding intra-warp bank conflicts.
    int bank = threadIdx.x % numBanks;

    // Set up warp-level parameters for coalesced global loads.
    constexpr int warpSize = 32;
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int globalWarpId = globalThreadId / warpSize;     // Unique ID for each warp in the grid.
    int lane = globalThreadId % warpSize;             // Lane index within the warp.
    int warpsPerGrid = (gridDim.x * blockDim.x) / warpSize;

    // Each warp processes contiguous chunks of the input.
    // The chunk size per warp iteration is warpSize * itemsPerThread.
    int warpChunkSize = warpSize * itemsPerThread;
    // Grid-stride loop over warp-level chunks.
    for (int base = globalWarpId * warpChunkSize; base < inputSize; base += warpsPerGrid * warpChunkSize) {
        // Within each warp, threads load items in a coalesced manner.
        // For each iteration, lane 'l' in the warp loads the element at offset: base + i * warpSize + lane.
        #pragma unroll
        for (int i = 0; i < itemsPerThread; i++) {
            int idx = base + i * warpSize + lane;
            if (idx < inputSize) {
                // Convert the character to an unsigned value.
                unsigned int c_val = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
                // Update the histogram only if the character is within the specified range.
                if (c_val >= static_cast<unsigned int>(from) && c_val <= static_cast<unsigned int>(to)) {
                    // Calculate the shared memory offset for the bin corresponding to c_val.
                    // The value for bin (c_val - from) in bank 'bank' is stored at:
                    // (c_val - from) * numBanks + bank.
                    atomicAdd(&localHist[(c_val - from) * numBanks + bank], 1);
                }
            }
        }
    }
    __syncthreads();

    // Aggregate the per-block histogram bins from all banks and update the global histogram.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum the values for this bin over all banks.
        for (int c = 0; c < numBanks; c++) {
            sum += localHist[bin * numBanks + c];
        }
        // Atomically update the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
}

// Host function to configure and launch the CUDA kernel.
// The function assumes that 'input' and 'histogram' have been allocated on the device,
// and that host-device synchronization is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Define the number of threads per block.
    int threadsPerBlock = 256;
    // Each thread processes 'itemsPerThread' characters; hence, each block processes threadsPerBlock * itemsPerThread characters.
    // Calculate the number of blocks required to cover the input.
    int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);

    // Calculate the number of histogram bins.
    int range = to - from + 1;
    // Calculate the shared memory size required: numBanks copies for each histogram bin.
    size_t sharedMemSize = numBanks * range * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);

    // Note: Any necessary error checking and host-device synchronization
    // (e.g., cudaDeviceSynchronize) is assumed to be performed by the caller.
}
```

---

Update the previously generated solution. Increase the number of items processed by each thread further and increase the block size.

---

```c++
#include <cuda_runtime.h>

// Increase the number of items processed by each thread for higher throughput.
// This value is chosen to maximize memory throughput on modern NVIDIA GPUs.
constexpr int itemsPerThread = 64;  // Increased from 32 to 64
// Define the number of copies (banks) of the histogram per block to avoid shared memory bank conflicts.
constexpr int numBanks = 32;

// CUDA kernel to compute a histogram for a restricted range of characters.
// This version avoids shared memory bank conflicts by using multiple copies of the histogram.
// The shared memory layout is organized so that the value for bin 'i' in histogram copy 'c'
// is stored at offset: i * numBanks + c. The kernel also uses a warp-level iteration pattern
// to ensure that threads in the same warp perform coalesced loads from global memory.
// Input:
//   input: pointer to the device memory buffer holding the text.
//   inputSize: the number of characters in the input buffer.
//   from: the lower bound (inclusive) of the character range.
//   to: the upper bound (inclusive) of the character range.
//   histogram: pointer to the device memory buffer for the output histogram.
//              Its size must be (to - from + 1). The final result is obtained by summing
//              the corresponding bins from all blocks.
__global__ void histogramKernel(const char* input, unsigned int inputSize, int from, int to, unsigned int* histogram) {
    // Calculate the number of histogram bins.
    int range = to - from + 1;

    // Allocate shared memory for multiple copies of the histogram.
    // Shared memory is organized so that for each bin 'i' and bank 'c',
    // the value is stored at offset: i * numBanks + c.
    extern __shared__ unsigned int localHist[];

    // Initialize all copies of the shared histogram to zero.
    for (int i = threadIdx.x; i < range * numBanks; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Determine which copy (bank) this thread will update.
    // Using threadIdx.x % numBanks ensures that consecutive warp lanes use different banks,
    // thereby avoiding intra-warp bank conflicts.
    int bank = threadIdx.x % numBanks;

    // Set up warp-level parameters for coalesced global loads.
    constexpr int warpSize = 32;
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int globalWarpId = globalThreadId / warpSize;     // Unique ID for each warp in the grid.
    int lane = globalThreadId % warpSize;             // Lane index within the warp.
    int warpsPerGrid = (gridDim.x * blockDim.x) / warpSize;

    // Each warp processes contiguous chunks of the input.
    // The chunk size per warp iteration is warpSize * itemsPerThread.
    int warpChunkSize = warpSize * itemsPerThread;
    // Grid-stride loop over warp-level chunks.
    for (int base = globalWarpId * warpChunkSize; base < inputSize; base += warpsPerGrid * warpChunkSize) {
        // Within each warp, threads load items in a coalesced manner.
        // For each iteration, lane 'l' in the warp loads the element at offset: base + i * warpSize + lane.
        #pragma unroll
        for (int i = 0; i < itemsPerThread; i++) {
            int idx = base + i * warpSize + lane;
            if (idx < inputSize) {
                // Convert the character to an unsigned value.
                unsigned int c_val = static_cast<unsigned int>(static_cast<unsigned char>(input[idx]));
                // Update the histogram only if the character is within the specified range.
                if (c_val >= static_cast<unsigned int>(from) && c_val <= static_cast<unsigned int>(to)) {
                    // Calculate the shared memory offset for the bin corresponding to c_val.
                    // The value for bin (c_val - from) in bank 'bank' is stored at:
                    // (c_val - from) * numBanks + bank.
                    atomicAdd(&localHist[(c_val - from) * numBanks + bank], 1);
                }
            }
        }
    }
    __syncthreads();

    // Aggregate the per-block histogram bins from all banks and update the global histogram.
    for (int bin = threadIdx.x; bin < range; bin += blockDim.x) {
        unsigned int sum = 0;
        // Sum the values for this bin over all banks.
        for (int c = 0; c < numBanks; c++) {
            sum += localHist[bin * numBanks + c];
        }
        // Atomically update the global histogram.
        atomicAdd(&histogram[bin], sum);
    }
}

// Host function to configure and launch the CUDA kernel.
// The function assumes that 'input' and 'histogram' have been allocated on the device,
// and that host-device synchronization is handled by the caller.
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to) {
    // Increase the block size for higher occupancy on modern GPUs.
    int threadsPerBlock = 512;  // Increased from 256 to 512
    // Each thread processes 'itemsPerThread' characters; hence, each block processes threadsPerBlock * itemsPerThread characters.
    // Calculate the number of blocks required to cover the input.
    int blocks = (inputSize + threadsPerBlock * itemsPerThread - 1) / (threadsPerBlock * itemsPerThread);

    // Calculate the number of histogram bins.
    int range = to - from + 1;
    // Calculate the shared memory size required: numBanks copies for each histogram bin.
    size_t sharedMemSize = numBanks * range * sizeof(unsigned int);

    // Launch the CUDA kernel.
    histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(input, inputSize, from, to, histogram);

    // Note: Any necessary error checking and host-device synchronization
    // (e.g., cudaDeviceSynchronize) is assumed to be performed by the caller.
}
```

