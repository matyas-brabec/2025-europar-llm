#!/usr/bin/env python3

import datetime
import os
import subprocess
import sys
import pathlib

WORK_DIR = pathlib.Path("__work_dir__")
LOG_DIR = pathlib.Path("__log_dir__")
SUCCESS_CODE = 0

time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = LOG_DIR / time

rnd_string = os.urandom(4).hex()
WORK_DIR = WORK_DIR / f"{time}_{rnd_string}"

for dir in [WORK_DIR, LOG_DIR]:
    os.makedirs(dir, exist_ok=True)

def print_usage(file):
    print("Usage: python3 run-experiments.py <infrastructure_directory> <implementations_directory> [arguments passed to make]", file=file)

if "-h" in sys.argv or "--help" in sys.argv:
    print_usage(sys.stdout)
    exit(0)

if len(sys.argv) < 3:
    print_usage(sys.stderr)
    exit(1)

infrastructure_directory = pathlib.Path(sys.argv[1])
implementations_directory = pathlib.Path(sys.argv[2])

def get_log_name(cu_file):
    dir_relative_to_implementations = cu_file.parent.relative_to(implementations_directory)
    dir_relative_to_implementations = dir_relative_to_implementations.as_posix().replace("/", "_")

    log_file = f"{LOG_DIR}/{time}_{dir_relative_to_implementations}.log"
    current_dir = os.getcwd()
    return f"{current_dir}/{log_file}"

os.system(f"rm -rf {WORK_DIR}")
os.system(f"cp -r {infrastructure_directory} {WORK_DIR}")

class RunResult:
    def __init__(self, cu_file):
        self.cu_file = cu_file
        self.compiles = ''
        self.time = ''
        self.standard_deviation = ''
        self.valid_result = ''
        self.runtime_error = ''

    def does_not_compile(self):
        self.compiles = False
        return self

    def with_runtime_error(self, exit_code):
        self.compiles = True
        self.runtime_error = exit_code
        return self

    def set_values(self, stdout : str):
        output = stdout.strip().split(" ")
        if len(output) < 3:
            self.compiles = True
            self.runtime_error = "Invalid output"
            return self

        time, stddev, valid = output[-3:]

        self.compiles = True
        self.time = time
        self.standard_deviation = stddev
        self.valid_result = valid == 'OK'

        return self

    def csv_line(self):
        relative_path = self.cu_file.relative_to(implementations_directory)
        experiment_id = str(relative_path.with_suffix(''))
        compiled = str(self.compiles)

        return f"{experiment_id};{self.time};{self.standard_deviation};{compiled};{self.valid_result};{self.runtime_error};{' '.join(sys.argv[3:])}"

def run_with(cu_file):
    result = RunResult(cu_file)

    os.system(f"cp {cu_file} {WORK_DIR}/{cu_file.name}")
    log_file = get_log_name(cu_file)

    with open(log_file, "w") as log:
        log.write(f"Running {cu_file}\n")

        cmd = f"cd {WORK_DIR} && make {' '.join(sys.argv[3:])}"
        process = subprocess.Popen(cmd, shell=True, stdout=log, stderr=log, text=True)
        _ = process.communicate()

        exit_code = process.returncode

        if exit_code != SUCCESS_CODE:
            return result.does_not_compile()

        cmd = f"cd {WORK_DIR} && make run {' '.join(sys.argv[3:])}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=log, text=True)
        stdout, _ = process.communicate()

        exit_code = process.returncode

        if exit_code != SUCCESS_CODE:
            return result.with_runtime_error(exit_code)

    return result.set_values(stdout)

print("experiment_id;time;std;compiled;verified;runtime_err;extra", flush=True)

for cu_file in pathlib.Path(implementations_directory).rglob("*.cu"):
    result = run_with(cu_file)

    print(result.csv_line())
    sys.stdout.flush()

for cu_file in pathlib.Path(implementations_directory).rglob("*.cuh"):
    result = run_with(cu_file)

    print(result.csv_line())
    sys.stdout.flush()
