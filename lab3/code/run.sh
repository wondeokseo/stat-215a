#!/bin/bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stat215a
cd "$(dirname "$0")"

python make_figure.py