#!/bin/bash
#SBATCH --array=0-359
#SBATCH --job-name=apo   
#SBATCH --time=41:00:00
#SBATCH --mem-per-cpu=6000M
#SBATCH --account=def-szepesva
#SBATCH --output=apo%A%a.out
#SBATCH --error=apo%A%a.err

bash $SLURM_ARRAY_TASK_ID.sh