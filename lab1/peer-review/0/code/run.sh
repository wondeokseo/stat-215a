#!/bin/bash
conda activate stat215a
quarto render lab1.ipynb --to pdf

conda deactivate
