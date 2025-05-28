#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1
#SBATCH --array=0-4  # Launch 5 jobs in parallel

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

# Define an array of arguments
args=(1 3 5 7 9)

# Get the argument for this task
arg=${args[$SLURM_ARRAY_TASK_ID]}

echo "Running with argument: $arg"
python -W ignore base_model.py $arg