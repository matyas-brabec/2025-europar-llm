#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// In this implementation we assume that k is a power‐of‐two between 32 and 1024.
// Each query is processed by one warp (32 threads). Each warp maintains
// an “intermediate result” of the best k candidates (with squared Euclidean distances)
// stored distributed among its 32 threads (each holding k/32 candidates in registers).
// A shared‐memory candidate buffer (size k per warp) is used to batch new candidate data,
// and whenever the buffer is full the warp “merges” its candidate buffer with its intermediate result.
// (The merge is done by lane 0 serially gathering all candidate values via warp shuffles,
// performing a simple insertion‐sort and merge of two sorted arrays, and then writing the new
// sorted result back into registers via shared memory.)
//
// We process the data points in “tiles” that are loaded cooperatively by all threads in each block
// into shared memory. A block uses BLOCK_SIZE threads (multiple of 32) and each warp in the block handles one query.
// The shared memory layout is as follows:
//   [Tile Data]               : float2 array of size TILE_SIZE
//   [Warp Candidate Buffer]   : Candidate array of size (warpsPerBlock * k)
//   [Warp Buffer Count]       : int array of size (warpsPerBlock)
//   [Merge Temporary Buffer]  : Candidate array of size (warpsPerBlock * (2*k))
//   [Merge Output Buffer]     : Candidate array of size (warpsPerBlock * k)
//
#define TILE_SIZE 256
#define BLOCK_SIZE 128  // Use 128 threads per block => 4 warps per block

// Structure to hold a candidate (index and squared distance)
struct Candidate {
    int idx;
    float dist;
};

//
// The knn_kernel processes query points (each warp processing one query),
// using data points processed in "tiles" loaded into shared memory
// and candidate buffering + merging done in shared memory and with warp‐shuffle intrinsics.
//
__global__ void knn_kernel(const float2 *query, int query_count,
                           const float2 *data, int data_count,
                           std::pair<int, float> *result, int k)
{
    // Each warp processes one query.
    // Compute global warp id and lane id.
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane = threadIdx.x & 31; // threadIdx.x % 32

    // Return if warp id is beyond query_count.
    if (warp_id >= query_count)
        return;

    // Load the query point for this warp.
    float2 q = query[warp_id];

    // Compute per-thread candidate count: L = k/32.
    int L = k >> 5;   // k/32; note: k is guaranteed to be a power-of-two >= 32

    // Each thread maintains a private sorted list of L best candidates.
    // They are stored in registers in arrays localD and localIdx.
    // Sorted in ascending order (smaller distances come first).
    // Initially, fill with “infinite” distance.
    float localD[32]; // maximum L is 1024/32=32.
    int   localIdx[32];
#pragma unroll
    for (int i = 0; i < L; i++) {
        localD[i]   = FLT_MAX;
        localIdx[i] = -1;
    }

    // -------------------------------------------------------------------------
    // Shared Memory Layout
    // Use dynamically–allocated shared memory.
    // The layout is:
    //   [Tile Data]               : float2 sharedData[TILE_SIZE]
    //   [Warp Candidate Buffer]   : Candidate warpCandBuffer[ warpsPerBlock * k ]
    //   [Warp Buffer Count]       : int warpBuffCount[ warpsPerBlock ]
    //   [Merge Temporary Buffer]  : Candidate mergeBuffer[ warpsPerBlock * (2*k) ]
    //   [Merge Output Buffer]     : Candidate mergeOutput[ warpsPerBlock * k ]
    //
    extern __shared__ char s_mem[];
    float2 *sharedData = (float2*) s_mem;  // tile data: size = TILE_SIZE * sizeof(float2)
    Candidate *candBuffer = (Candidate*)(sharedData + TILE_SIZE); 
    int   *buffCount = (int*)(candBuffer + (blockDim.x/32) * k);
    Candidate *mergeBuffer = (Candidate*)(buffCount + (blockDim.x/32));
    Candidate *mergeOutput = (Candidate*)(mergeBuffer + (blockDim.x/32) * (2 * k));
    // -------------------------------------------------------------------------

    // Identify warp's shared memory region.
    int warp_in_block = threadIdx.x / 32;  // warp id within block.
    Candidate *myCandBuffer = candBuffer + warp_in_block * k;  // candidate buffer for this warp.
    int *myBuffCount = buffCount + warp_in_block;              // candidate count for this warp.
    Candidate *myMergeBuffer = mergeBuffer + warp_in_block * (2 * k); // temporary merge buffer (2*k per warp)
    Candidate *myMergeOutput = mergeOutput + warp_in_block * k;       // merge output buffer (k per warp)

    // Initialize candidate buffer count once per warp.
    if (lane == 0)
        *myBuffCount = 0;
    __syncwarp();

    // -------------------------------------------------------------------------
    // Process all data points in tiles.
    // Each block loads a tile of data into shared memory.
    for (int tileStart = 0; tileStart < data_count; tileStart += TILE_SIZE) {

        // Each thread in block cooperatively loads the tile.
        for (int i = threadIdx.x; i < TILE_SIZE && (tileStart + i) < data_count; i += blockDim.x) {
            sharedData[i] = data[tileStart + i];
        }
        __syncthreads();  // ensure full tile is loaded

        // Each warp processes the tile.
        // Each warp thread loops over candidate points with stride = 32.
        for (int i = lane; i < TILE_SIZE && (tileStart + i) < data_count; i += 32) {
            // Load candidate data point from shared memory.
            float2 p = sharedData[i];
            // Compute squared Euclidean distance.
            float dx = q.x - p.x;
            float dy = q.y - p.y;
            float dist = dx * dx + dy * dy;
            int data_idx = tileStart + i;

            // Determine the current global worst (largest) distance among the warp’s intermediate result.
            // Each thread’s worst candidate is the last element in its local array, so compute a reduction.
            float my_worst = localD[L-1];
            for (int offset = 16; offset > 0; offset /= 2) {
                float tmp = __shfl_down_sync(0xffffffff, my_worst, offset);
                if (tmp > my_worst)
                    my_worst = tmp;
            }
            float globalThreshold = my_worst;

            // If the candidate is better than the worst current candidate, add it to the shared candidate buffer.
            if (dist < globalThreshold) {
                int pos = atomicAdd(myBuffCount, 1);
                if (pos < k) {
                    myCandBuffer[pos].idx = data_idx;
                    myCandBuffer[pos].dist = dist;
                }
            }
        }
        __syncwarp();  // synchronize warp lanes

        // Check if the candidate buffer is full.
        if (*myBuffCount >= k) {
            // WHEN THE BUFFER IS FULL, MERGE IT WITH THE PRIVATE INTERMEDIATE RESULT.
            // All lanes participate, but lane 0 does the heavy merge work.
            if (lane == 0) {
                // Allocate temporary arrays on lane 0 for merging.
                // tempInter will hold the intermediate candidates (gathered from registers)
                // tempBuff will hold the candidate buffer copy from shared memory.
                Candidate tempInter[1024];  // maximum k is 1024
                Candidate tempBuff[1024];   // candidate buffer copy

                // Gather the intermediate result from each lane using warp shuffle.
                // The warp’s k candidates are distributed: candidate r is held by lane (r % 32) at index (r/32).
                for (int r = 0; r < k; r++) {
                    int src_lane = r & 31;     // r % 32
                    int pos = r >> 5;          // r / 32
                    int candIdx = __shfl_sync(0xffffffff, localIdx[pos], src_lane);
                    float candDist = __shfl_sync(0xffffffff, localD[pos], src_lane);
                    tempInter[r].idx = candIdx;
                    tempInter[r].dist = candDist;
                }
                // Copy candidate buffer (from shared memory) to tempBuff.
                for (int r = 0; r < k; r++) {
                    tempBuff[r] = myCandBuffer[r];
                }

                // Sort tempInter array using insertion sort (ascending order by distance).
                for (int i = 1; i < k; i++) {
                    Candidate key = tempInter[i];
                    int j = i - 1;
                    while (j >= 0 && tempInter[j].dist > key.dist) {
                        tempInter[j+1] = tempInter[j];
                        j--;
                    }
                    tempInter[j+1] = key;
                }
                // Sort tempBuff array using insertion sort.
                for (int i = 1; i < k; i++) {
                    Candidate key = tempBuff[i];
                    int j = i - 1;
                    while (j >= 0 && tempBuff[j].dist > key.dist) {
                        tempBuff[j+1] = tempBuff[j];
                        j--;
                    }
                    tempBuff[j+1] = key;
                }
                // Merge the two sorted arrays (each of length k) and select the first k best candidates.
                Candidate merged[1024];
                int i = 0, j = 0;
                for (int r = 0; r < k; r++) {
                    if (i < k && (j >= k || tempInter[i].dist <= tempBuff[j].dist)) {
                        merged[r] = tempInter[i++];
                    } else {
                        merged[r] = tempBuff[j++];
                    }
                }
                // Write the merged result to the merge output buffer in shared memory.
                for (int r = 0; r < k; r++) {
                    myMergeOutput[r] = merged[r];
                }
                // Reset candidate buffer count to 0.
                *myBuffCount = 0;
            }
            __syncwarp();
            // All warp lanes now load the merged intermediate result from shared memory
            // into their registers.
            for (int pos = 0; pos < L; pos++) {
                int r = pos * 32 + lane; // r is the global index among k candidates.
                localIdx[pos] = myMergeOutput[r].idx;
                localD[pos] = myMergeOutput[r].dist;
            }
        }
        __syncthreads(); // ensure all threads see updated shared memory before next tile
    } // end for each tile

    // -------------------------------------------------------------------------
    // After processing all data tiles, if the candidate buffer is not empty, merge it with the intermediate result.
    if (*myBuffCount > 0) {
        if (lane == 0) {
            Candidate tempInter[1024];
            Candidate tempBuff[1024];
            // Gather intermediate result from registers.
            for (int r = 0; r < k; r++) {
                int src_lane = r & 31;
                int pos = r >> 5;
                int candIdx = __shfl_sync(0xffffffff, localIdx[pos], src_lane);
                float candDist = __shfl_sync(0xffffffff, localD[pos], src_lane);
                tempInter[r].idx = candIdx;
                tempInter[r].dist = candDist;
            }
            // Copy candidate buffer (which may be partially filled) into tempBuff and fill remaining with FLT_MAX.
            int count = *myBuffCount;
            for (int r = 0; r < count; r++) {
                tempBuff[r] = myCandBuffer[r];
            }
            for (int r = count; r < k; r++) {
                tempBuff[r].idx = -1;
                tempBuff[r].dist = FLT_MAX;
            }
            // Sort both arrays.
            for (int i = 1; i < k; i++) {
                Candidate key = tempInter[i];
                int j = i - 1;
                while (j >= 0 && tempInter[j].dist > key.dist) {
                    tempInter[j+1] = tempInter[j];
                    j--;
                }
                tempInter[j+1] = key;
            }
            for (int i = 1; i < k; i++) {
                Candidate key = tempBuff[i];
                int j = i - 1;
                while (j >= 0 && tempBuff[j].dist > key.dist) {
                    tempBuff[j+1] = tempBuff[j];
                    j--;
                }
                tempBuff[j+1] = key;
            }
            // Merge the two sorted arrays into merged[ ] (length k).
            Candidate merged[1024];
            int i = 0, j = 0;
            for (int r = 0; r < k; r++) {
                if (i < k && (j >= k || tempInter[i].dist <= tempBuff[j].dist)) {
                    merged[r] = tempInter[i++];
                } else {
                    merged[r] = tempBuff[j++];
                }
            }
            // Write merged result to merge output buffer.
            for (int r = 0; r < k; r++) {
                myMergeOutput[r] = merged[r];
            }
            *myBuffCount = 0;
        }
        __syncwarp();
        // All warp lanes load the merged result into registers.
        for (int pos = 0; pos < L; pos++) {
            int r = pos * 32 + lane;
            localIdx[pos] = myMergeOutput[r].idx;
            localD[pos] = myMergeOutput[r].dist;
        }
    }

    // -------------------------------------------------------------------------
    // Write final intermediate result to global memory.
    // The final k nearest neighbors for query[warp_id] are stored in sorted order (ascending distance).
    // Each warp writes its k candidates to result[warp_id * k ... warp_id * k + k - 1].
    for (int pos = 0; pos < L; pos++) {
        int r = pos * 32 + lane;
        int out_idx = warp_id * k + r;
        result[out_idx].first  = localIdx[pos];
        result[out_idx].second = localD[pos];
    }
}

//
// Host interface function.
// Allocates grid/block dimensions and configures shared memory usage.
// This function assumes that the input arrays 'query', 'data', and 'result' have been allocated on the device (via cudaMalloc).
//
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result, int k)
{
    // Each query is processed by one warp.
    int warpsNeeded = query_count;
    int threadsNeeded = warpsNeeded * 32;
    
    // We choose BLOCK_SIZE threads per block.
    int blockSize = BLOCK_SIZE;
    int gridSize = (threadsNeeded + blockSize - 1) / blockSize;

    // Compute shared memory size:
    // Shared tile data: TILE_SIZE * sizeof(float2)
    size_t shmemTile = TILE_SIZE * sizeof(float2);
    // Warp candidate buffer: (warps per block = (blockSize/32)) * k * sizeof(Candidate)
    size_t shmemCandBuffer = (blockSize / 32) * k * sizeof(Candidate);
    // Warp buffer count: (blockSize/32) * sizeof(int)
    size_t shmemBuffCount = (blockSize / 32) * sizeof(int);
    // Merge temporary buffer: (blockSize/32) * (2*k) * sizeof(Candidate)
    size_t shmemMergeBuffer = (blockSize / 32) * (2 * k) * sizeof(Candidate);
    // Merge output buffer: (blockSize/32) * k * sizeof(Candidate)
    size_t shmemMergeOutput = (blockSize / 32) * k * sizeof(Candidate);

    size_t shmemSize = shmemTile + shmemCandBuffer + shmemBuffCount + shmemMergeBuffer + shmemMergeOutput;

    // Launch the kernel.
    knn_kernel<<<gridSize, blockSize, shmemSize>>>(
        query, query_count, data, data_count, result, k);
        
    // (Error checking omitted for brevity.)
}