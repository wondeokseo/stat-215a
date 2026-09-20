#!/bin/bash

conda activate stat215a
jupyter nbconvert --to notebook --execute --inplace lab1.ipynb
conda deactivate

# To initialize everything so the notebook will run, run the following in the terminal:
# conda env create -f code/environment.yaml
# cd code
# source run.sh