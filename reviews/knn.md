# kNN

This file contains a summary of the single-response generated solutions.

## Correctness table

(Evaluated on an Nvidia B40 GPU with NVCC 12.8 and GCC 13.2.0)

**k=1024, n=4'194'304, m=4'096, r=10**

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ✅  | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN02     | ✅  | ✅  | ✅  | ❌💥 | ✅  | ❌⚙️ | ✅  | ✅  | ❌💥 | ✅  |
|    kNN03     | ❌💥 | ❌  | ❌💥 | ✅  | ❌⚙️ | ✅  | ❌⚙️ | ❌💥 | ✅  | ✅  |
|    kNN04     | ✅  | ❌🛠️ | ✅  | ❌  | ❌💥 | ❌💥 | ✅  | ✅  | ✅  | ❌💥 |
|    kNN05     | ❌⚙️ | ❌💥 | ❌💥 | ❌💥 | ✅  | ❌💥 | ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️|
|    kNN06     | ❌💥 | ❌💥 | ❌💥 | ❌🛠️ | ❌💥 | ❌💥 | ✅  | ✅  | ❌💥 | ❌💥 |
|    kNN07     | ❌💥 | ❌🛠️ | ❌💥 | ❌⚙️ | ✅  | ❌⚙️ | ❌⚙️ | ✅  | ❌💥 | ❌💥 |
|    kNN08     | ❌  | ❌  | ❌💥 | ❌  | ❌💥 | ❌⚙️ | ❌💥 | ❌⚙️ | ❌  | ❌⚙️ |

**k=32, n=4'194'304, m=4'096, r=10**

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ✅  | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN02     | ✅  | ✅  | ✅  | ✅  | ✅  | ❌⚙️ | ✅  | ✅  | ❌💥 | ✅  |
|    kNN03     | ❌💥 | ❌  | ❌💥 | ✅  | ❌⚙️ | ✅  | ❌⚙️ | ❌💥 | ✅  | ✅  |
|    kNN04     | ✅  | ❌🛠️ | ✅  | ❌  | ✅  | ❌💥 | ✅  | ✅  | ✅  | ❌💥 |
|    kNN05     | ❌⚙️ | ✅  | ❌  | ✅  | ✅  | ❌💥 | ✅  | ❌💥 | ❌💥 | ✅🛠️ |
|    kNN06     | ✅  | ✅  | ✅  | ❌🛠️ | ✅  | ❌💥 | ✅  | ✅  | ❌  | ✅  |
|    kNN07     | ✅  | ✅🛠️ | ✅  | ❌⚙️ | ✅  | ❌⚙️ | ❌⚙️ | ✅  | ❌💥 | ✅  |
|    kNN08     | ❌  | ❌  | ❌  | ❌💥 | ✅  | ❌⚙️ | ❌  | ❌⚙️ | ❌  | ❌⚙️ |

**Summary for k=1024 and k=32**

✅ – Correct solution (compiled successfully and returned the correct results): 76/160 (48%)

❌ – Compiled and ran without a runtime error but returned incorrect results: 20/160 (13%)

❌💥 – Compiled but crashed during execution (Or timed out): 42/160 (26%)

❌⚙️ – Did not compile: 22/160 (14%)

🛠️ – Indicator denotes the source code a small edit to make it compile (this mark is added alongside one of the above). The erroneous line(s) was/were commented and prefixed with `/// @FIXED` comment.

**Combined across both k choices**

If either measurement does not compile, the cell is `❌⚙️`. Otherwise, if either measurement crashes, the cell is `❌💥`. Otherwise, if either measurement returns incorrect results, the cell is `❌`. The cell is `✅` only when both `k=32` and `k=1024` succeed and verify correctly.

| Test Case    | 01  | 02  | 03  | 04  | 05  | 06  | 07  | 08  | 09  | 10  |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|    kNN01     | ✅  | ✅  | ❌⚙️ | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
|    kNN02     | ✅  | ✅  | ✅  | ❌💥 | ✅  | ❌⚙️ | ✅  | ✅  | ❌💥 | ✅  |
|    kNN03     | ❌💥 | ❌  | ❌💥 | ✅  | ❌⚙️ | ✅  | ❌⚙️ | ❌💥 | ✅  | ✅  |
|    kNN04     | ✅  | ❌🛠️ | ✅  | ❌  | ❌💥 | ❌💥 | ✅  | ✅  | ✅  | ❌💥 |
|    kNN05     | ❌⚙️ | ❌💥 | ❌💥 | ❌💥 | ✅  | ❌💥 | ❌💥 | ❌💥 | ❌💥 | ❌💥🛠️|
|    kNN06     | ❌💥 | ❌💥 | ❌💥 | ❌🛠️ | ❌💥 | ❌💥 | ✅  | ✅  | ❌💥 | ❌💥 |
|    kNN07     | ❌💥 | ❌🛠️ | ❌💥 | ❌⚙️ | ✅  | ❌⚙️ | ❌⚙️ | ✅  | ❌💥 | ❌💥 |
|    kNN08     | ❌  | ❌  | ❌💥 | ❌💥 | ❌💥 | ❌⚙️ | ❌💥 | ❌⚙️ | ❌  | ❌⚙️ |

**Summary for combined across both k choices**

✅ – Correct solution (compiled successfully and returned the correct results): 30/80 (38%)

❌ – Compiled and ran without a runtime error but returned incorrect results: 8/80 (10%)

❌💥 – Compiled but crashed during execution (Or timed out): 31/80 (39%)

❌⚙️ – Did not compile: 11/80 (14%)

🛠️ – Indicator denotes the source code a small edit to make it compile (this mark is added alongside one of the above). The erroneous line(s) was/were commented and prefixed with `/// @FIXED` comment.
