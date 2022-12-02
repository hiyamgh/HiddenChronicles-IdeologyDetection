#!/usr/bin/env bash
#SBATCH --job-name=vexp3
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

#firstjob=1
#
#if [ $SLURM_ARRAY_TASK_ID -eq $firstjob ];
#    then
#        echo "first job"
#        python train_maml_system.py --name_of_args_json_file experiment_config/vexp3.json
#    else
#        echo "not first job"
#        python train_maml_system.py --name_of_args_json_file experiment_config/vexp3.json --continue_from_epoch latest
#fi


python train_maml_system.py --continue_from_epoch latest --total_epochs 100 --total_iter_per_epoch 100 --total_epochs_before_pause 100 --init_inner_loop_learning_rate 4e-5 --meta_learning_rate 3e-5 --meta_inner_optimizer_learning_rate 6e-5 --experiment_name threewayprotomaml_exp211