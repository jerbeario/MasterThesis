#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -n 2
#SBATCH -p rome

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeremy.palmerio@student.uva.nl
#SBATCH --mem=20G



source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

python partition.py 
wait
