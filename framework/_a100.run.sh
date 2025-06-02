#!/bin/bash

#SBATCH -p gpu-short
#SBATCH -A kdss
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100
#SBATCH --time=2:00:00
#SBATCH --output=__log_dir__/job-ampere-%j.csv
#SBATCH --error=__log_dir__/job-ampere-%j.err
#SBATCH --exclusive
#SBATCH --nodelist=ampere02

log_dir=__log_dir__
mkdir -p $log_dir

python ./run-experiments.py "$@"
