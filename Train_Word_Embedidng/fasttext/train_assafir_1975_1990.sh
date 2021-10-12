#!/usr/bin/env bash
#SBATCH --job-name=fas75-90
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

python train_fasttext.py --archive assafir --wordNgrams 4 --dim 300 --ws 5 --minCount 100 --model skipgram --neg 15 --lr 0.001 --start_year 1975 --end_year 1990