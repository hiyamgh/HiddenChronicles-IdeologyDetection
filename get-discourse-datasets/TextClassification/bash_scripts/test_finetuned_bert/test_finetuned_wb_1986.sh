#!/usr/bin/env bash
#SBATCH --job-name=testft86
#SBATCH --account=hkg02
#SBATCH --partition=gpu

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --time=0-06:00:00

module purge
module load cuda/10.1
module load java/java8
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load python/3

nvcc --version

echo "----------------------------------"

nvidia-smi

python use_finetuned.py --test_set testing_datasets_discourse/nahar/df_test_1986_Mukawama.xlsx

