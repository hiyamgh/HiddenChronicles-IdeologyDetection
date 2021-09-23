#!/usr/bin/env bash
#SBATCH --job-name=nahar-w2v
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

python train_word2vec.py --archive nahar --size 300 --window 5 --mincount 100 --sg 1 --negative 15 --learning_rate 0.001