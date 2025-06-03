#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1
#SBATCH --array=0-4  # Launch 5 jobs in parallel

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

# Define an array of arguments
args=(1 3 5 7 9)

# Get the argument for this task
arg=${args[$SLURM_ARRAY_TASK_ID]}

# Construct the name dynamically
name="class_ratio_${arg}"

echo "Running with argument: $arg and name: $name"
python -W ignore HazardMapper/model.py -n $name -z landslide -c $arg