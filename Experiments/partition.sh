#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -p rome

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeremy.palmerio@student.uva.nl
#SBATCH --mem=80G



source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

python partition.py wildfire
python partition.py landslides

wait
