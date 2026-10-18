#!/bin/bash
#SBATCH --job-name=lab3-kmeans
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=lab3-%j.out

set -e

cd "$(dirname "$0")"

python parallel_kmeans.py