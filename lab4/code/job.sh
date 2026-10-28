#!/bin/bash

# EXAMPLE USAGE:
# See stat-215-a-gsi/computing/psc-instructions.pdf for guidance.
# From lab4/code on a Bridges-2 login node:
# sbatch job.sh configs/default.yaml

#SBATCH --account=mth260030p
#SBATCH --job-name=lab4-autoencoder
#SBATCH --cpus-per-task=5          # V100 nodes have 40 cores / 8 GPUs
#SBATCH --time=04:00:00            # shorten this once you know how long training takes
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1

module load anaconda3/2024.10-1
conda activate env_215a

python run_autoencoder.py "$1"
