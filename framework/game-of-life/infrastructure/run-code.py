#/usr/bin/env python3

import subprocess
import sys

GRID_SIZE = 16384
ITERATIONS = 200
ID = 'run_game_of_life'
EXE = 'bin/game_of_life'

WARMUP_ITERATIONS = 3

CORRECT_CHECK_SUM = '2485608-2490945-2495873-2476437-2482346-2491329-2488768-2475112' # 16384x16384, 200 iterations

CHECK_SUM_IDX = 4
TIME_IDX = 3

if len(sys.argv) != 2:
    print("Usage: run-code.py <cu_file>")
    exit(1)

cu_file = sys.argv[1]

def get_memory_layout_mode():
    with open(cu_file, "r") as f:
        code = f.read().split("\n")

    for line in code:
        if "MEMORY_LAYOUT" in line:
            mode = line.split(" ")[-1].strip()
            return {
                "BOOLS": 0,
                "ROWS": 1,
                "TILES": 2
            }[mode]

    raise Exception("Could not find MEMORY_LAYOUT mode in code")

cmd = f"{EXE} {ID} {get_memory_layout_mode()} {GRID_SIZE} {ITERATIONS}"
print('running: ', cmd, file=sys.stderr)

result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
output = result.stdout
exit_code = result.returncode

if exit_code != 0:
    print("Compilation failed")
    print(result.stderr, file=sys.stderr)
    exit(1)

relevant_lines = output.strip().split("\n")[1 + WARMUP_ITERATIONS:]

all_checksums = [line.split(";")[CHECK_SUM_IDX] for line in relevant_lines]
valid = all(checksum == CORRECT_CHECK_SUM for checksum in all_checksums)

all_times = [float(line.split(";")[TIME_IDX]) for line in relevant_lines]
average_time = sum(all_times) / len(all_times)
std_dev_time = (sum([(time - average_time) ** 2 for time in all_times]) / len(all_times)) ** 0.5

print(f"{average_time} {std_dev_time} {'OK' if valid else 'FAILED'}")
