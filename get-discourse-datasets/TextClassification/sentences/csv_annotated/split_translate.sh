#!/usr/bin/env bash
#SBATCH --job-name=spltrans
#SBATCH --account=hkg02
#SBATCH --partition=normal

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

module purge
module load python/3

python split_translate.py
