#!/bin/bash
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -p gpu_h100
#SBATCH --gpus 2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

python -W ignore base_model.py
wait
