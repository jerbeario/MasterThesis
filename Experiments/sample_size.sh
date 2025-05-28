#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1
#SBATCH --array=0-3  # Launch 4 jobs in parallel

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis

# Define an array of arguments
args=(0.5 0.1 1.0 0.01)

# Get the argument for this task
arg=${args[$SLURM_ARRAY_TASK_ID]}

echo "Running with argument: $arg"
python -W ignore base_model.py $arg