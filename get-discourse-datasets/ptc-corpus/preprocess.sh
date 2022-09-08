#!/usr/bin/env bash
#SBATCH --job-name=jobarr
#SBATCH --account=hkg02
#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module purge
module load python/3

python preprocess_datasets.py