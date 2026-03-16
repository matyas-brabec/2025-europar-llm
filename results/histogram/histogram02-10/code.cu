#include <cuda_runtime.h>

namespace {

// Tuned for recent data-center GPUs (A100/H100 class):
// - 256 threads/block gives 8 warps/block, a good balance of occupancy and per-block work.
// - 8 bytes/thread per outer-loop iteration keeps the loop overhead low without bloating
//   register usage.
constexpr int kBlockThreads      = 256;
constexpr int kWarpSize          = 32;
constexpr int kWarpsPerBlock     = kBlockThreads / kWarpSize;
constexpr int kItemsPerThread    = 8;
constexpr int kPrivateBins       = 256;  // Full 8-bit byte domain.
constexpr int kTileBytesPerBlock = kBlockThreads * kItemsPerThread;
constexpr unsigned int kFullMask = 0xFFFFFFFFu;

static_assert(kBlockThreads % kWarpSize == 0, "Block size must be a multiple of warp size.");
static_assert(kPrivateBins == 256, "This kernel assumes 8-bit input characters.");

// Warp-aggregated update into a warp-private shared-memory histogram.
//
// Each warp owns its own 256-bin histogram in shared memory, so there is no inter-warp sharing.
// __match_any_sync() groups lanes that want to increment the same relative bin, and only the
// leader lane performs the increment with the population count of the whole group.
//
// Because the histogram is warp-private and there is at most one writer per bin per step,
// a regular shared-memory increment is sufficient here; __syncwarp() keeps the warp converged
// between steps on architectures with independent thread scheduling.
__device__ __forceinline__
void warp_aggregated_add(unsigned int rel_bin,
                         bool valid,
                         unsigned int* warp_hist,
                         int lane)
{
    const unsigned int valid_mask = __ballot_sync(kFullMask, valid);
    if (valid) {
        const unsigned int peers = __match_any_sync(valid_mask, rel_bin);
        if (lane == (__ffs(peers) - 1)) {
            warp_hist[rel_bin] += static_cast<unsigned int>(__popc(peers));
        }
    }
    __syncwarp(kFullMask);
}

// Kernel parameters:
// - input: device pointer to the raw text bytes.
// - histogram: device pointer to the output array of length `bins`.
// - inputSize: number of bytes in input.
// - from: first byte value in the requested range.
// - bins: number of output bins, i.e. (to - from + 1).
//
// The output convention is:
//   histogram[i] == count of byte value (from + i)
__global__ __launch_bounds__(kBlockThreads)
void histogram_range_kernel(const char* __restrict__ input,
                            unsigned int* __restrict__ histogram,
                            unsigned int inputSize,
                            int from,
                            int bins)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & (kWarpSize - 1);

    const unsigned int from_u = static_cast<unsigned int>(from);
    const unsigned int bins_u = static_cast<unsigned int>(bins);

    // Reinterpret as unsigned bytes so values 128..255 are handled correctly even when
    // the host/compiler treats char as signed.
    const unsigned char* const text = reinterpret_cast<const unsigned char*>(input);

    // One full 256-bin private histogram per warp:
    // 8 warps * 256 bins * 4 bytes = 8 KiB shared memory per block.
    // This is small on A100/H100 and keeps indexing simple and fast.
    __shared__ unsigned int s_hist[kWarpsPerBlock * kPrivateBins];

    // Zero the whole shared histogram. The fixed 256-bin stride is intentional:
    // it avoids dynamic address arithmetic and still costs very little shared memory.
    #pragma unroll
    for (int i = tid; i < kWarpsPerBlock * kPrivateBins; i += kBlockThreads) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    unsigned int* const warp_hist = s_hist + warp_id * kPrivateBins;

    // Use size_t for indexing so inputs close to 4 GiB cannot wrap a 32-bit grid-stride index.
    const size_t input_size  = static_cast<size_t>(inputSize);
    const size_t tile_bytes  = static_cast<size_t>(kTileBytesPerBlock);
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * tile_bytes;

    size_t tile_start = static_cast<size_t>(blockIdx.x) * tile_bytes;

    while (tile_start < input_size) {
        // Fast path for a full tile: no bounds checks on the loads.
        if (tile_start + tile_bytes <= input_size) {
            size_t idx = tile_start + static_cast<size_t>(tid);

            #pragma unroll
            for (int item = 0; item < kItemsPerThread; ++item, idx += kBlockThreads) {
                const unsigned int rel = static_cast<unsigned int>(text[idx]) - from_u;
                warp_aggregated_add(rel, rel < bins_u, warp_hist, lane);
            }
        } else {
            // Tail tile: some lanes/items may fall outside the input range.
            size_t idx = tile_start + static_cast<size_t>(tid);

            #pragma unroll
            for (int item = 0; item < kItemsPerThread; ++item, idx += kBlockThreads) {
                const bool present = idx < input_size;

                unsigned int rel = 0u;
                if (present) {
                    rel = static_cast<unsigned int>(text[idx]) - from_u;
                }

                warp_aggregated_add(rel, present && (rel < bins_u), warp_hist, lane);
            }
        }

        tile_start += grid_stride;
    }

    __syncthreads();

    // Final block-local reduction:
    // because bins <= 256 by contract and blockDim.x == 256, one thread can own one output bin.
    const bool single_block = (gridDim.x == 1);

    if (tid < bins) {
        unsigned int sum = 0u;

        #pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            sum += s_hist[w * kPrivateBins + tid];
        }

        // If only one block is launched, it overwrites the whole output histogram directly,
        // so no global atomic and no pre-zeroing of the output are needed.
        if (single_block) {
            histogram[tid] = sum;
        } else if (sum != 0u) {
            atomicAdd(&histogram[tid], sum);
        }
    }
}

}  // namespace

void run_histogram(const char *input,
                   unsigned int *histogram,
                   unsigned int inputSize,
                   int from,
                   int to)
{
    // Per the API contract:
    // - input and histogram are device pointers allocated by cudaMalloc.
    // - caller handles synchronization.
    // This function therefore uses the default stream and launches asynchronously.
    cudaStream_t stream = 0;

    const int bins = to - from + 1;
    const size_t hist_bytes = static_cast<size_t>(bins) * sizeof(unsigned int);

    // Empty input still requires a zeroed output histogram.
    if (inputSize == 0u) {
        cudaMemsetAsync(histogram, 0, hist_bytes, stream);
        return;
    }

    const size_t tile_bytes = static_cast<size_t>(kTileBytesPerBlock);
    const int needed_blocks =
        static_cast<int>((static_cast<size_t>(inputSize) + tile_bytes - 1u) / tile_bytes);

    // A single block can directly write the final histogram, so avoid the output memset in that
    // case. For multi-block launches, zero the output once and let blocks accumulate into it.
    int grid_size = 1;

    if (needed_blocks > 1) {
        int device = 0;
        int sm_count = 1;
        int blocks_per_sm = 1;

        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            histogram_range_kernel,
            kBlockThreads,
            0);

        int resident_blocks = sm_count * blocks_per_sm;
        if (resident_blocks < 1) {
            resident_blocks = 1;
        }

        // Grid-stride processing lets one resident wave cover arbitrarily large inputs, and
        // keeping the grid around one resident wave avoids unnecessary final global reductions.
        grid_size = (needed_blocks < resident_blocks) ? needed_blocks : resident_blocks;
    }

    if (grid_size != 1) {
        cudaMemsetAsync(histogram, 0, hist_bytes, stream);
    }

    histogram_range_kernel<<<grid_size, kBlockThreads, 0, stream>>>(
        input,
        histogram,
        inputSize,
        from,
        bins);
}