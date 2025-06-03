#!/bin/bash
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH -p gpu_h100
#SBATCH --gpus 1
#SBATCH --array=0-1  # Launch 2 jobs in parallel


source ~/miniconda3/etc/profile.d/conda.sh
conda activate MasterThesis


# Define an array of arguments
args=("custom" "default")

# Get the argument for this task
arg=${args[$SLURM_ARRAY_TASK_ID]}
echo "Running with argument: $arg"

# Run the model with the specified argument
python -W ignore HazardMapper/model.py -n sampler -z landslide --sampler $arg
wait
