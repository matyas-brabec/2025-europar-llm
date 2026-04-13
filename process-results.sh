#!/usr/bin/env bash

# This script processes the results of the experiments.
# - Use gpt-querying/main.py to prepare batch files with LLM queries (results/gol-request.jsonl, results/knn-request.jsonl, results/histogram-request.jsonl).
# - Run the batch files using the OpenAI web interface for batch processing (https://platform.openai.com/batches/).
# - Download the results (results/gol-response.jsonl, results/knn-response.jsonl, results/histogram-response.jsonl).
# - Run this script to process the results (parsing the responses, renaming files into expected names, doing trivial post-processing, etc.).

set -euo pipefail

mkdir -p results/gol results/knn results/histogram
rm -rf results/gol/*/ results/knn/* results/histogram/*

(
    cd gpt-querying
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt

    python main.py
)

find results/gol -type f -name "code.cu" | while read -r file; do
    mv "$file" "${file%/code.cu}/gol.cu"
done

(cd results/gol && python add-other-func-impl.py .)
(cd results && python remove-extern.py gol)
