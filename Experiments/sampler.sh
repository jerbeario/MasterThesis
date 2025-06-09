#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -p rome

#SBATCH --mem=120G


source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

# Run the model with the specified argument
python -W ignore HazardMapper/sampler.py
wait
