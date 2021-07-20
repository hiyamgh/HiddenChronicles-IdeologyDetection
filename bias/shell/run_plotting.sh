#!/usr/bin/env bash
#SBATCH --job-name=bias
#SBATCH --account=hkg02
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

python create_final_plots_all_new.py