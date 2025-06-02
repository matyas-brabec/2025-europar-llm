# The Framework

This directory contains the framework for testing and evaluating CUDA implementations of the Game of Life, k-NN, and Histogram algorithms as explained in the root directory: [../README.md](../README.md).

## Configure virtual environment

To set up the virtual environment, run the following command (requires Python 3.8 or higher):

```sh
python3 -m venv .venv
```

Then, activate the virtual environment:

```sh
source .venv/bin/activate
```

### Install Python dependencies

Install the required Python packages using pip:

```sh
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## How the Framework Interacts with the Interface

Given the root folder containing the infrastructure and a list of `*.cu` files (or a directory containing subdirectories with the list of source files), the framework works as follows:

1. It creates a copy of the root directory.
2. It sequentially injects each `.cu` file into the root directory.
3. The framework compiles and runs each collection of source files, following the process described in the Interface section.
4. The results of each run are collected and compiled into a final report.

## Printing Custom Graphs

To produce graphs similar to those in the paper, use the Python script [print-graph.py](./framework/print-graph.py).

```bash
python3 print-graph.py <config_keys> <path_to_experiments> <path_to_references> [tag]
```

## Parameters

1. **Config keys**: Specifies graph styles based on [print-graph.config.json](./framework/print-graph.config.json). Combine multiple keys using `@` (e.g., `hist@with-log-scale`).
2. **Path to experiments**: The CSV file containing experiment results.
3. **Path to references**: The CSV file containing reference results.
4. **Tag (optional)**: A tag for naming the output file.

### Example Usage

```bash
cd framework
python3 print-graph.py gol@with-log-scale ../measured-times/gol-hopper.csv ../measured-times/gol-hopper.csv TESTING_GRAPH_
```

This generates a file `TESTING_GRAPH_gol-hopper.graph.pdf` in the framework directory.

Additionally, **LaTeX is required** for proper graph rendering. If LaTeX is not installed, disable it in `print-graph.py` by setting `"use_tex": false` in the configuration file `print-graph.config.json` (line 88).
