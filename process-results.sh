#!/usr/bin/env bash

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
