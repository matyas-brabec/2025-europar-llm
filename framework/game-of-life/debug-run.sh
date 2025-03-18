#!/bin/bash

cd infrastructure/ &&
srun -p gpu-short -A kdss --cpus-per-task=32 --mem=2GB --time=2:00:00 --gres=gpu:V100 make GOL_IMPL=../"$1"/gol.cu run test $2 $3 $4
