First prompt: works, 21ms (Hopper), equiv. of base shm generated by static prompts

Skipping milestone 2 (alredy covered)
Second prompt: works, 2.8ms, properly used itemsPerThread==8, seq. iteration

Third prompt: makes 32 copies of the histogram, but the layout is seq. (no perf. improvement due to the bank conflicts)

Fourth prompt: properly divides the copies into banks

EXTENSION:

Fifth prompt correctly reorders the iteration

Sixth prompt: increased the block size to 512 (Still quite conservative) and itemsPerThread to 64 (otpimum is 1024 threads and ~512 IPT)



https://chatgpt.com/c/67d74040-5e48-8001-b2d6-709487c16e81
