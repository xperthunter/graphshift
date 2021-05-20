#!/usr/bin/env bash

#SBATCH --output test.out
#SBATCH -p production
#SBATCH -w rafter-21
#SBATCH -t 01:00:00
#SBATCH --mem 100
#SBATCH --gres=gpu:1,gpu_mem:100

echo "START"
aklog
hostname
nvidia-smi

module load anaconda3

python3 -m venv gsvenv
echo "END"
