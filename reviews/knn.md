# kNN

This file contains a summary of the single-response generated solutions.

## Correctness table

(Evaluated on an Nvidia B40 GPU with NVCC 12.8 and GCC 13.2.0)

**k=1024, n=4'194'304, m=4'096, r=10**

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ✅  | ❌💥 | ❌💥 | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN02     | ✅  | ❌💥 | ❌💥 | ✅  | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ❌💥 |
|    kNN03     | ❌💥 | ✅  | ❌⚙️ | ❌💥 | ❌💥 | ✅  | ✅  | ❌💥 | ✅  | ✅  |
|    kNN04     | ❌💥 | ✅  | ❌💥 | ✅  | ❌⚙️ | ❌💥 | ❌⚙️ | ❌💥 | ❌💥 | ❌💥 |
|    kNN05     | ❌💥 | ❌💥 | ❌💥 | ❌💥 | ❌💥 | ✅  | ❌💥 | ❌💥 | ❌⚙️ | ❌💥 |
|    kNN06     | ❌💥 | ❌  | ❌💥 | ❌💥 | ✅  | ❌💥 | ❌💥 | ❌💥 | ✅  | ❌💥 |
|    kNN07     | ✅  | ✅  | ❌💥 | ❌💥 | ❌  | ✅  | ✅  | ✅  | ❌💥 | ✅  |
|    kNN08     | ❌⚙️ | ✅  | ✅  | ❌⚙️ | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ❌⚙️ |

**k=32, n=4'194'304, m=4'096, r=10**

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ✅  | ✅  | ❌  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN02     | ✅  | ✅  | ✅  | ✅  | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN03     | ✅  | ✅  | ❌⚙️ | ❌💥 | ❌💥 | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN04     | ✅  | ✅  | ✅  | ✅  | ❌⚙️ | ❌💥 | ❌⚙️ | ✅  | ✅  | ✅  |
|    kNN05     | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ❌⚙️ | ✅  |
|    kNN06     | ✅  | ✅  | ✅  | ❌💥 | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN07     | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ❌💥 | ✅  |
|    kNN08     | ❌⚙️ | ✅  | ✅  | ❌⚙️ | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ❌⚙️ |

**Summary for k=1024 and k=32**

✅ – Correct solution (compiled successfully and returned the correct results): 101/160 (63%)

❌ – Compiled and ran without a runtime error but returned incorrect results: 3/160 (2%)

❌💥 – Compiled but crashed during execution (Or timed out): 38/160 (24%)

❌⚙️ – Did not compile: 18/160 (11%)

**Combined across both k choices**

If either measurement does not compile, the cell is `❌⚙️`. Otherwise, if either measurement crashes, the cell is `❌💥`. Otherwise, if either measurement returns incorrect results, the cell is `❌`. The cell is `✅` only when both `k=32` and `k=1024` succeed and verify correctly.

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ✅  | ❌💥 | ❌💥 | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN02     | ✅  | ❌💥 | ❌💥 | ✅  | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ❌💥 |
|    kNN03     | ❌💥 | ✅  | ❌⚙️ | ❌💥 | ❌💥 | ✅  | ✅  | ❌💥 | ✅  | ✅  |
|    kNN04     | ❌💥 | ✅  | ❌💥 | ✅  | ❌⚙️ | ❌💥 | ❌⚙️ | ❌💥 | ❌💥 | ❌💥 |
|    kNN05     | ❌💥 | ❌💥 | ❌💥 | ❌💥 | ❌💥 | ✅  | ❌💥 | ❌💥 | ❌⚙️ | ❌💥 |
|    kNN06     | ❌💥 | ❌  | ❌💥 | ❌💥 | ✅  | ❌💥 | ❌💥 | ❌💥 | ✅  | ❌💥 |
|    kNN07     | ✅  | ✅  | ❌💥 | ❌💥 | ❌  | ✅  | ✅  | ✅  | ❌💥 | ✅  |
|    kNN08     | ❌⚙️ | ✅  | ✅  | ❌⚙️ | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ❌⚙️ |

**Summary for combined across both k choices**

✅ – Correct solution (compiled successfully and returned the correct results): 36/80 (45%)

❌ – Compiled and ran without a runtime error but returned incorrect results: 2/80 (3%)

❌💥 – Compiled but crashed during execution (Or timed out): 33/80 (41%)

❌⚙️ – Did not compile: 9/80 (11%)

🛠️ – Indicator denotes the source code a small edit to make it compile (this mark is added alongside one of the above). The erroneous line(s) was/were commented and prefixed with `/// @FIXED` comment.
