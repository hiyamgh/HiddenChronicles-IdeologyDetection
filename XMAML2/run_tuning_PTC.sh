#!/usr/bin/env bash

#SBATCH --job-name=PTC-tun # Job name
#SBATCH --account=hkg02
#SBATCH --partition=gpu # Partition
#SBATCH --nodes=2 # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks
#SBATCH --gres=gpu:v100d32q:1 # Number of GPUs
#SBATCH --time=0-06:00:00
#SBATCH --array=1-7%1

# Load any necessary modules, in this case OpenMPI with CUDA
module purge
module load python/3
module load cuda/10.1
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load java/java8

nvcc --version

echo "----------------------------------"

nvidia-smi

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python xmaml.py $(head -n $SLURM_ARRAY_TASK_ID cross_domain_ar_tuning/PTC_cross_domain.txt | tail -n 1)
