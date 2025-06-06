@include ./histogram01.prompt.md

Optimize the CUDA kernel by utilization of shared memory for histogram privatization. Add a constant `itemsPerThread` to the code that controls how many input chars are processed by each thread. Select the appropriate default value for `itemsPerThread`, assuming the kernel should be optimized for the latest NVIDIA GPUs and the input size is large.

Avoid shared memory bank conflicts by using multiple copies of the histogram per thread block. Create 32 copies of the histogram and place each copy in a separate bank (using appropriate stride). Make sure that each copy is accessed by threads from the same warp lanes to avoid intra-warp bank collisions.
