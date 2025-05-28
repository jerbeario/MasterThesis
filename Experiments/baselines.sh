#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -p rome

#SBATCH --mem=60G



source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

python base_model.py 
wait