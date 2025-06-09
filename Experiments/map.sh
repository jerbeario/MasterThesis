#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

python -W ignore HazardMapper/model.py -n map -z multi_hazard -s 1 -a SimpleCNN -p 5 -c 1 --map
wait
sbatch bin_map.sh
