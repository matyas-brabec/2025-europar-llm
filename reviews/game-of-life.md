# Game of Life

This file contains an in-depth review of the "one-shot" generated solutions.

## Correctness table

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GoL_01       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| GoL_02       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| GoL_02_tiled | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌   | ❌🛠️ | ❌   |
| GoL_03       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ❌   | ✅   | ✅   |
| GoL_04       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| GoL_05       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |
| GoL_06       | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   | ✅   |

✅ – Correct solution (compiled successfully and returned the correct GoL grid): 59/70 (84%)

❌ – Compiled and ran without a runtime error but returned incorrect results: 11/70 (16%)

❌💥 – Compiled but crashed during execution: 0/70 (0%)

❌⚙️ – Did not compile: 0/70 (0%)

🛠️ –  Indicator denotes the source code a small edit to make it compile (this mark is added alongside one of the above). The erroneous line(s) was/were commented and prefixed with `/// @FIXED` comment.
