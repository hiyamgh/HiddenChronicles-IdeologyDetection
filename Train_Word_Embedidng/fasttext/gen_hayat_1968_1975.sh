#!/usr/bin/env bash
#SBATCH --job-name=ha68-75
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

python generate_data_fasttext.py --archive hayat --start_year 1968 --end_year 1975