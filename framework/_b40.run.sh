#!/bin/bash

#SBATCH -p gpu-short
#SBATCH -A kdss
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:BW6000
#SBATCH --time=2:00:00
#SBATCH --output=__log_dir__/job-blackwell-%j.csv
#SBATCH --error=__log_dir__/job-blackwell-%j.err
#SBATCH --exclusive
#SBATCH --nodelist=bw01

log_dir=__log_dir__
mkdir -p $log_dir

python ./run-experiments.py "$@"
