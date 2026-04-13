# kNN

This file contains a summary of the single-response generated solutions.

## Correctness table

(Evaluated on an Nvidia H100 GPU with NVCC 12.8 and GCC 13.2.0)

**k=1024, n=4'194'304, m=4'096, r=10**

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ❌  | ✅  | ❌⚙️ | ❌  | ❌  | ❌💥 | ❌💥 | ✅  | ❌  |
|    kNN02     | ❌💥 | ✅  | ❌💥 | ✅  | ❌💥🛠️ | ❌  | ❌  | ✅  | ✅🛠️ | ❌  |
|    kNN03     | ✅  | ✅  | ❌💥🛠️ | ✅  | ❌🛠️ | ❌  | ✅  | ✅  | ✅  | ❌💥🛠️ |
|    kNN04     | ✅🛠️ | ❌⚙️ | ✅  | ✅  | ❌💥 | ✅  | ❌💥 | ❌💥 | ✅🛠️ | ❌💥 |
|    kNN05     | ❌  | ❌🛠️ | ❌💥 | ❌  | ✅🛠️ | ✅  | ✅  | ❌💥 | ❌🛠️ | ❌💥 |
|    kNN06     | ✅  | ❌💥 | ❌💥 | ❌💥🛠️ | ✅  | ❌⚙️ | ❌💥🛠️ | ❌💥 | ❌💥🛠️ | ❌💥 |
|    kNN07     | ❌💥 | ❌💥 | ❌💥 | ❌💥 | ❌🛠️ | ❌  | ❌  | ❌💥 | ✅  | ❌🛠️ |
|    kNN08     | ❌  | ✅🛠️ | ❌💥🛠️ | ❌  | ✅  | ❌💥 | ✅🛠️ | ❌🛠️ | ❌💥 | ❌💥 |

**k=32, n=4'194'304, m=4'096, r=10**

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ❌  | ✅  | ❌⚙️ | ❌  | ❌  | ✅  | ❌💥 | ✅  | ❌  |
|    kNN02     | ❌💥 | ✅  | ✅  | ✅  | ❌💥🛠️ | ❌  | ❌  | ❌  | ✅🛠️ | ❌  |
|    kNN03     | ✅  | ✅  | ❌💥🛠️ | ✅  | ❌🛠️ | ❌  | ✅  | ✅  | ❌  | ✅🛠️ |
|    kNN04     | ✅🛠️ | ❌⚙️ | ✅  | ✅  | ❌💥 | ✅  | ❌💥 | ❌💥 | ✅🛠️ | ❌  |
|    kNN05     | ❌  | ❌🛠️ | ❌💥 | ✅  | ✅🛠️ | ✅  | ✅  | ❌💥 | ✅🛠️ | ❌💥 |
|    kNN06     | ✅  | ❌💥 | ❌💥 | ❌💥🛠️ | ✅  | ❌⚙️ | ✅🛠️ | ❌💥 | ❌💥🛠️ | ❌  |
|    kNN07     | ❌💥 | ✅  | ❌  | ✅  | ❌💥🛠️ | ❌  | ❌  | ❌💥 | ✅  | ❌🛠️ |
|    kNN08     | ❌  | ✅🛠️ | ❌🛠️ | ❌  | ✅  | ❌  | ✅🛠️ | ❌🛠️ | ✅  | ❌  |

**Summary for k=1024 and k=32**

✅ – Correct solution (compiled successfully and returned the correct results): 61/160 (38%)

❌ – Compiled and ran without a runtime error but returned incorrect results: 45/160 (28%)

❌💥 – Compiled but crashed during execution (Or timed out): 48/160 (30%)

❌⚙️ – Did not compile: 6/160 (4%)

**Combined across both k choices**

If either measurement does not compile, the cell is `❌⚙️`. Otherwise, if either measurement crashes, the cell is `❌💥`. Otherwise, if either measurement returns incorrect results, the cell is `❌`. The cell is `✅` only when both `k=32` and `k=1024` succeed and verify correctly.

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ❌  | ✅  | ❌⚙️ | ❌  | ❌  | ❌💥 | ❌💥 | ✅  | ❌  |
|    kNN02     | ❌💥 | ✅  | ❌💥 | ✅  | ❌💥🛠️ | ❌  | ❌  | ❌  | ✅🛠️ | ❌  |
|    kNN03     | ✅  | ✅  | ❌💥🛠️ | ✅  | ❌🛠️ | ❌  | ✅  | ✅  | ❌  | ❌💥🛠️ |
|    kNN04     | ✅🛠️ | ❌⚙️ | ✅  | ✅  | ❌💥 | ✅  | ❌💥 | ❌💥 | ✅🛠️ | ❌💥 |
|    kNN05     | ❌  | ❌🛠️ | ❌💥 | ❌  | ✅🛠️ | ✅  | ✅  | ❌💥 | ❌🛠️ | ❌💥 |
|    kNN06     | ✅  | ❌💥 | ❌💥 | ❌💥🛠️ | ✅  | ❌⚙️ | ❌💥🛠️ | ❌💥 | ❌💥🛠️ | ❌💥 |
|    kNN07     | ❌💥 | ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️ | ❌  | ❌  | ❌💥 | ✅  | ❌🛠️ |
|    kNN08     | ❌  | ✅🛠️ | ❌💥🛠️ | ❌  | ✅  | ❌💥 | ✅🛠️ | ❌🛠️ | ❌💥 | ❌💥 |

**Summary for combined across both k choices**

✅ – Correct solution (compiled successfully and returned the correct results): 25/80 (31%)

❌ – Compiled and ran without a runtime error but returned incorrect results: 21/80 (26%)

❌💥 – Compiled but crashed during execution (Or timed out): 31/80 (39%)

❌⚙️ – Did not compile: 3/80 (4%)

🛠️ – Indicator denotes the source code a small edit to make it compile (this mark is added alongside one of the above). The erroneous line(s) was/were commented and prefixed with `/// @FIXED` comment.
