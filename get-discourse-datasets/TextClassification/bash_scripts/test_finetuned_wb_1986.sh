#!/usr/bin/env bash
#SBATCH --job-name=testft86
#SBATCH --account=hkg02
#SBATCH --partition=normal

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load java/java8
module load python/3
module load python/pytorch

python use_finetuned.py --test_set testing_datasets_discourse/nahar/df_test_1986.xlsx

