#!/usr/bin/env bash
#SBATCH --job-name=alllangs
#SBATCH --account=hkg02
#SBATCH --partition=gpu

#SBATCH --mem=122000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=0-06:00:00
#SBATCH --array=1-55%10

module purge
module load cuda/11.7.1
module load java/java8
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load python/3

nvcc --version

echo "----------------------------------"

nvidia-smi

python train_maml_system.py $(head -n $SLURM_ARRAY_TASK_ID experiments_langs.txt | tail -n 1) --continue_from_epoch latest --experiment_name all_langs-threewayprotomaml/$SLURM_ARRAY_TASK_ID/
