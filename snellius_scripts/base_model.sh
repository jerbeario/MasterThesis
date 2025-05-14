#!/bin/bash
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

python -W ignore base_model.py
wait
