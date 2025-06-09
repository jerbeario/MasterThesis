#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -p rome


source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

python HazardMapper/partition.py -z landslide
wait
