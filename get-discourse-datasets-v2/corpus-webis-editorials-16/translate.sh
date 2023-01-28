#!/usr/bin/env bash
#SBATCH --job-name=argumentation
#SBATCH --account=hkg02
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-06:00:00
#SBATCH --array=0-41%5

module purge
module load python/3

python translate_annotations.py --array_id $SLURM_ARRAY_TASK_ID
