#!/usr/bin/env bash
#SBATCH --job-name=assafir
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu


python seq_procrustes.py F:/newspapers/word2vec/nahar/embeddings/ --start-year 1933 --end-year 2005 --year-inc 1