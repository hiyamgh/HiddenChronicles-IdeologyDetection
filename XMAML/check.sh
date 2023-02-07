#!/usr/bin/env bash

#SBATCH --job-name=XMAML # Job name
#SBATCH --account=hkg02
#SBATCH --partition=normal # Partition
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks

# Load any necessary modules, in this case OpenMPI with CUDA
module purge
module load python/3
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1

python check_downloaded_models.py
