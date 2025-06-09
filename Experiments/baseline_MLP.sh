#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1


source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

hazard="flood"
experiment="final_baseline"

python HazardMapper/model.py -n $experiment -z $hazard -s 1 -a MLP -p 1 -c 1 -e 10
python HazardMapper/model.py -n $experiment -z $hazard -s 1 -a SimpleCNN -p 5 -c 1 -e 10
python HazardMapper/model.py -n $experiment -z $hazard -s 1 -a SpatialAttentionCNN -p 5 -c 1 -e 10

wait