#!/bin/bash

#SBATCH -p gpu-short
#SBATCH -A kdss
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:V100
#SBATCH --time=2:00:00
#SBATCH --output=__log_dir__/job-volta-%j.csv
#SBATCH --error=__log_dir__/job-volta-%j.err
#SBATCH --exclusive
#SBATCH --nodelist=volta05

log_dir=__log_dir__
mkdir -p $log_dir

python ./run-experiments.py "$@"
