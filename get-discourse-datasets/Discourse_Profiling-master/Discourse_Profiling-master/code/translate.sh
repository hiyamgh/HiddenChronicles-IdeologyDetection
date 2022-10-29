#!/usr/bin/env bash
#SBATCH --job-name=transl
#SBATCH --account=hkg02
#SBATCH --partition=normal

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array 1-15%5

module purge
module load python/3

python translate.py $(head -n $SLURM_ARRAY_TASK_ID languages.txt | tail -n 1)
