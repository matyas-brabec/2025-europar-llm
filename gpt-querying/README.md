# Script for querying LLMs via OpenAI API

## Installation

(Note that this subdirectory requires different dependencies than `./framework`. The base requirement is Python 3.8 or higher.)

1. (Optional): Create the virtual environment: `python3 -m venv .venv` and activate it: `source .venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`

To use OpenAI (paid API):

* Create an API key: <https://platform.openai.com/api-keys>
* Rename `.env.example` to `.env` and save the API key there

If you want to use a different LLM, you can modify the `MODEL` variable in the `main.py` file. The default is set to `o3-mini-2025-01-31`, which is the latest OpenAI model as of January 2025. Note that requests to other models may incur different costs and require different API keys (or also different API configurations).

## Usage

Run via the `main.py` file. Select the desired assignment by uncommenting the corresponding line in the `main.py` file.

The script can either generate the LLM response locally (via OpenAI API) or create a batch of requests.
To obtain LLM responses for a batch, follow the [OpenAI guide](https://platform.openai.com/docs/guides/batch).

## Code handling notes

Sometimes, it was necessary to fix bugs in LLM-generated code. Each fixed bug (line or a small block) is labeled with a `@FIXED` annotation in doc-comment (prefixed with `///`), and the original is kept in a comment like this:

```c++
// Assumptions:
//   - The input and histogram arrays are allocated on the device (via cudaMalloc).
//   - Any required host-device synchronization (e.g., cudaDeviceSynchronize) is handled by the caller.
/// @FIXED
/// extern "C" void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
void run_histogram(const char *input, unsigned int *histogram, unsigned int inputSize, int from, int to)
{
    // Calculate the number of histogram bins (i.e., the size of the histogram output array).
    int histRange = to - from + 1;

    // Zero-initialize the global histogram array.
    cudaMemset(histogram, 0, histRange * sizeof(unsigned int));
```
