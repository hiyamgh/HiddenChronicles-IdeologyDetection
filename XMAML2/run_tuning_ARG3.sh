#!/usr/bin/env bash

#SBATCH --job-name=ARG-tun # Job name
#SBATCH --account=hkg02
#SBATCH --partition=arza # Partition
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks
#SBATCH --gres=gpu:k20:1 # Number of GPUs
#SBATCH --time=0-01:00:00
#SBATCH --mem=59000
#SBATCH --array=3-7%1

# Load any necessary modules, in this case OpenMPI with CUDA
module purge
module load python/3
module load cuda/10.1
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load java/java8

nvcc --version

echo "----------------------------------"

nvidia-smi

python xmaml.py $(head -n $SLURM_ARRAY_TASK_ID cross_domain_ar_tuning/argumentation_cross_domain.txt | tail -n 1)
