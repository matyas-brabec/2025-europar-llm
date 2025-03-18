@include ./histogram01.prompt.md

Optimize the CUDA kernel by utilization of shared memory for histogram privatization. Add a constant `itemsPerThread` to the code that controls how many input chars are processed by each thread. Select the appropriate default value for `itemsPerThread`, assuming the kernel should be optimized for the latest NVIDIA GPUs and the input size is large.

Avoid shared memory bank conflicts by using multiple copies of the histogram per thread block. Create 32 copies of the histogram so that each thread accesses a copy number `threadIdx.x % 32`. The histogram copies need to be stored so that each one occupies a single memory bank. This can be achieved by striding histogram values so that value `i` of histogram copy `c` (0 <= `c` < 32) is located at offset `i * 32 + c`. Make sure the run function allocates enough shared memory for the 32 copies of the histogram.
