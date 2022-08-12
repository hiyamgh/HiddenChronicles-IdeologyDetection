#!/usr/bin/env bash
#SBATCH --job-name=testft07
#SBATCH --account=hkg02
#SBATCH --partition=gpu

#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --time=0-06:00:00

module purge
module load cuda
module load java/java8
module load python/3
module load python/pytorch

python use_finetuned.py --test_set testing_datasets_discourse/nahar/df_test_2007_Mukawama.xlsx

