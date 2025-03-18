#include <cuda_runtime.h>
#include <cfloat>
#include <utility>

// Define a simple POD struct that has the same memory layout as std::pair<int, float>.
// This struct is used on the device side for storing a neighbor’s index ("first")
// and the corresponding squared distance ("second").
struct ResultPair {
    int first;
    float second;
};

// Tile size for loading data points into shared memory. This value was chosen to
// balance shared‐memory usage (each float2 takes 8 bytes) and coalesced global loads.
#define TILE_SIZE 256

// The CUDA kernel computes the k-nearest neighbors for a set of query points against
// a large dataset of 2D points. Each thread processes a single query point.
// To optimize global memory accesses, the kernel tiles the data points into shared memory.
// Each thread maintains a candidate max‐heap (of size k) in local memory, so that after
// processing all data points the k smallest squared distances (and their indices) are found.
// Finally, each thread sorts its candidate list in ascending order (closest neighbor first)
// and writes the results into the output array.
__global__ void knn_kernel(const float2 * __restrict__ query,
                           int query_count,
                           const float2 * __restrict__ data,
                           int data_count,
                           ResultPair * __restrict__ result,
                           int k)
{
    // Each thread is responsible for one query.
    int qIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qIdx >= query_count)
        return;

    // Load the query point into registers.
    float2 q = query[qIdx];

    // We'll maintain a candidate max-heap (by squared Euclidean distance) in local memory.
    // Since k is guaranteed to be a power of 2 between 32 and 1024, we reserve an array
    // of maximum size 1024. Only the first k elements are used.
    const int maxK = 1024;
    int cand_idx[maxK];
    float cand_dist[maxK];
    int heap_size = 0; // current number of elements in the heap

    // Shared memory tile for data points.
    __shared__ float2 tile[TILE_SIZE];

    // Loop over the dataset in tiles.
    for (int tile_start = 0; tile_start < data_count; tile_start += TILE_SIZE)
    {
        // Cooperative loading of a tile of data points into shared memory.
        // Use a stride of blockDim.x to cover all TILE_SIZE elements.
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x)
        {
            int dataIndex = tile_start + i;
            if (dataIndex < data_count)
                tile[i] = data[dataIndex];
        }
        __syncthreads();

        // Determine how many elements are valid in this tile.
        int tile_elems = TILE_SIZE;
        if (tile_start + TILE_SIZE > data_count)
            tile_elems = data_count - tile_start;

        // For every data point in the tile, compute the squared Euclidean distance
        // and update the candidate heap.
        for (int i = 0; i < tile_elems; i++)
        {
            float2 dpoint = tile[i];
            float dx = q.x - dpoint.x;
            float dy = q.y - dpoint.y;
            float dist = dx * dx + dy * dy;
            int dataIdx = tile_start + i;  // global index of this data point

            // If we haven't yet collected k candidates, add the new candidate.
            if (heap_size < k)
            {
                cand_idx[heap_size] = dataIdx;
                cand_dist[heap_size] = dist;
                heap_size++;
                // Once we reach k elements, build a max-heap from the candidate list.
                if (heap_size == k)
                {
                    // Perform in-place heap construction.
                    // For each non-leaf node (from k/2-1 down to 0), sift down.
                    for (int j = k/2 - 1; j >= 0; j--)
                    {
                        int parent = j;
                        while (true)
                        {
                            int left = 2 * parent + 1;
                            int right = left + 1;
                            int largest = parent;
                            if (left < k && cand_dist[left] > cand_dist[largest])
                                largest = left;
                            if (right < k && cand_dist[right] > cand_dist[largest])
                                largest = right;
                            if (largest == parent)
                                break;
                            // Swap candidate at index 'parent' with that at index 'largest'
                            float tmpd = cand_dist[parent];
                            cand_dist[parent] = cand_dist[largest];
                            cand_dist[largest] = tmpd;
                            int tmpi = cand_idx[parent];
                            cand_idx[parent] = cand_idx[largest];
                            cand_idx[largest] = tmpi;
                            parent = largest;
                        }
                    }
                }
            }
            else
            {
                // If the heap is full, check whether the new candidate is closer than
                // the current farthest candidate (at the root of the max-heap).
                if (dist < cand_dist[0])
                {
                    // Replace the worst candidate with the new candidate...
                    cand_dist[0] = dist;
                    cand_idx[0] = dataIdx;
                    // ...and sift-down to restore the max-heap property.
                    int parent = 0;
                    while (true)
                    {
                        int left = 2 * parent + 1;
                        int right = left + 1;
                        int largest = parent;
                        if (left < k && cand_dist[left] > cand_dist[largest])
                            largest = left;
                        if (right < k && cand_dist[right] > cand_dist[largest])
                            largest = right;
                        if (largest == parent)
                            break;
                        float tmpd = cand_dist[parent];
                        cand_dist[parent] = cand_dist[largest];
                        cand_dist[largest] = tmpd;
                        int tmpi = cand_idx[parent];
                        cand_idx[parent] = cand_idx[largest];
                        cand_idx[largest] = tmpi;
                        parent = largest;
                    }
                }
            }
        } // end loop over tile data points
        __syncthreads();
    } // end tile loop

    // At this point, cand_idx[0..k-1] and cand_dist[0..k-1] hold the k nearest neighbors
    // but in an unsorted max-heap (largest distance at the root). We now sort them in ascending
    // order (nearest neighbor first) using an in-thread heapsort.
    for (int i = k - 1; i > 0; i--)
    {
        // Swap the root with the element at index i.
        float tmpd = cand_dist[0];
        cand_dist[0] = cand_dist[i];
        cand_dist[i] = tmpd;
        int tmpi = cand_idx[0];
        cand_idx[0] = cand_idx[i];
        cand_idx[i] = tmpi;
        // Sift down the new root in the heap of size i.
        int parent = 0;
        while (true)
        {
            int left = 2 * parent + 1;
            int right = left + 1;
            int largest = parent;
            if (left < i && cand_dist[left] > cand_dist[largest])
                largest = left;
            if (right < i && cand_dist[right] > cand_dist[largest])
                largest = right;
            if (largest == parent)
                break;
            float t = cand_dist[parent];
            cand_dist[parent] = cand_dist[largest];
            cand_dist[largest] = t;
            int ti = cand_idx[parent];
            cand_idx[parent] = cand_idx[largest];
            cand_idx[largest] = ti;
            parent = largest;
        }
    }
    // After heapsort the candidate list is sorted in ascending order by distance.

    // Write the sorted candidate list to the result array.
    // For query qIdx, the j-th nearest neighbor is stored at result[qIdx * k + j].
    int base = qIdx * k;
    for (int i = 0; i < k; i++)
    {
        result[base + i].first = cand_idx[i];
        result[base + i].second = cand_dist[i];
    }
}

// Host function that launches the k-NN kernel.
// query: array of query points (each float2 encodes (x,y)) on device
// data: array of data points on device
// result: output array for k-nearest neighbors (each element is std::pair<int, float>)
// k: number of nearest neighbors to find per query (power-of-two between 32 and 1024)
void run_knn(const float2 *query, int query_count,
             const float2 *data, int data_count,
             std::pair<int, float> *result,
             int k)
{
    // Choose a block size that offers good occupancy on modern GPUs.
    int blockSize = 256;  // 256 threads per block
    int gridSize = (query_count + blockSize - 1) / blockSize;

    // Launch the kernel.
    // Note: We reinterpret the std::pair pointer as our ResultPair pointer;
    // since both are standard layout with an int followed by a float, their memory layout is identical.
    knn_kernel<<<gridSize, blockSize>>>(query, query_count, data, data_count,
                                        reinterpret_cast<ResultPair *>(result), k);
    // It is assumed that error-checking and synchronization (cudaDeviceSynchronize)
    // are handled by the caller if necessary.
}