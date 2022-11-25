#!/usr/bin/env bash
#SBATCH --job-name=MAMLP
#SBATCH --account=hkg02
#SBATCH --partition=gpu

#SBATCH --mem=122000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=0-06:00:00

module purge
module load cuda/11.7.1
module load java/java8
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load python/3

nvcc --version

echo "----------------------------------"

nvidia-smi

python train_maml_system.py --name_of_args_json_file experiment_config/v2.json --continue_from_epoch latest