#!/bin/bash
set -e
cd "$(dirname "$0")"
conda run -n stat215a python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 lab2.ipynb
