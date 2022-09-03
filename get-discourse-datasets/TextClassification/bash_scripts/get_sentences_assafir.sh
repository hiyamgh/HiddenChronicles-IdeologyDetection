#!/usr/bin/env bash
#SBATCH --job-name=getsentas
#SBATCH --account=hkg02
#SBATCH --partition=normal

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

python get_sentences.py --archive assafir
