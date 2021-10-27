#!/usr/bin/env bash
#SBATCH --job-name=nh75-90
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

python generate_data_fasttext.py --archive nahar --start_year 1975 --end_year 1990