#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -p rome

#SBATCH --mem=45G



source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

hazard="flood"
experiment="final_baseline"

python HazardMapper/model.py -n $experiment -z $hazard -s 1 -a LR -p 1 -c 1
python HazardMapper/model.py -n $experiment -z $hazard -s 1 -a RF -p 1 -c 1
# python HazardMapper/model.py -n $experiment -z $hazard -s 1 -a MLPC -p 1 -c 1

wait