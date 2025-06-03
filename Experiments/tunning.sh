#!/bin/bash
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1


source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

# Run the model with the specified argument
python -W ignore HazardMapper/model.py -n tunning -z landslide --hyper -s 0.5 -a SpatialAttentionCNN -p 5 -c 1
wait
