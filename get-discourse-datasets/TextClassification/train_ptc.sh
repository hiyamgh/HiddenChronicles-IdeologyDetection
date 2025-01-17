#!/usr/bin/env bash
#SBATCH --job-name=trainptc
#SBATCH --account=hkg02
#SBATCH --partition=medium

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load java/java8
module load python/3
module load python/pytorch

python train_bert.py

