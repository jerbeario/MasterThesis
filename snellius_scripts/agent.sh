#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

wandb agent jerbeario-university-of-amsterdam/Wildfire_FullCNN_sweep/605xirsd
wait