#!/bin/bash
#SBATCH -n 1
#SBATCH -t 8:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1
#SBATCH --array=0-5  # Launch 5 jobs in parallel

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

# Define an array of arguments
args=(1 0.5 0.1 0.05 0.01 0.005)

# Get the argument for this task
arg=${args[$SLURM_ARRAY_TASK_ID]}

name="sample_size_${arg}"

echo "Running with argument: $arg"
python -W ignore HazardMapper/model.py -n $name -z wildfire -s $arg
