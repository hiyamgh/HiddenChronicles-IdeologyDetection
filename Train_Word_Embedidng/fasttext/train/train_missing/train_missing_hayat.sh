#!/usr/bin/env bash
#SBATCH --job-name=fast-ha
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

python train_missing.py --archive hayat --start_year 1950 --end_year 2000
