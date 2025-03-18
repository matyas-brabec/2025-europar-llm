# The Framework

## Configure virtual environment

To set up the virtual environment, run the following command:

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

## Interface

The framework consists of three parts, each following the same output format. To execute a part of the framework, navigate to its root directory and run:

```sh
make run
```

### Possible Outcomes

- **Compilation Fails:** No output is generated.
- **Successful Execution:** The program prints to `stdout`, displaying the runtime, standard deviation, and verification result. Example output:

  ```text
  42.42 0.02 OK
  ```

  or

  ```text
  42.42 0.02 FAILED
  ```

  where `OK` indicates correct results and `FAILED` indicates verification failure.
  The program performs 10 iterations, preceded by 3 warm-up runs.

- **Runtime Error:** The program encounters an error during execution.

## How the Framework Interacts with the Interface

Given the root folder containing the infrastructure and a list of `*.cu` files (or a directory containing subdirectories with the list of source files), the framework works as follows:

1. It creates a copy of the root directory.
2. It sequentially injects each `.cu` file into the root directory.
3. The framework compiles and runs each collection of source files, following the process described in the Interface section.
4. The results of each run are collected and compiled into a final report.
