#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1


source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

python HazardMapper/model.py -n baseline -z landslide -s 0.1 -a SpatialAttentionCNN -p 5 -c 1
wait