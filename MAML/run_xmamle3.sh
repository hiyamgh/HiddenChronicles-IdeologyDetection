#!/usr/bin/env bash
#SBATCH --job-name=XMAMLe3
#SBATCH --account=hkg02
#SBATCH --partition=gpu

#SBATCH --mem=120000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --time=0-06:00:00

module purge
module load cuda/10.1
module load java/java8
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load python/3

nvcc --version

echo "----------------------------------"

nvidia-smi

#label_col=Label_general
#labels=main_contents,context_informing_contents,additional_supportive_contents
#python xmaml.py --dev_datasets_ids 3,4 --test_dataset_eval 5 --label_col $label_col --labels $labels --per_gpu_train_batch_size 16 --meta_learn_iter 300 --inner_train_steps 5 --output_dir_meta x_mamle3/maml_trained/ --eval_task_dir x_mamle3/maml_result/

#label_col=Label_general
#labels=main_contents,context_informing_contents,additional_supportive_contents
#python xmaml.py --dev_datasets_ids 3,4 --dev_dataset_finetune 4 --test_dataset_eval 5 --label_col $label_col --labels $labels --per_gpu_train_batch_size 16 --meta_learn_iter 300 --inner_train_steps 5 --output_dir_meta x_mamle3/maml_trained/ --eval_task_dir x_mamle3/maml_result/

label_col=Label_general
labels=main_contents,context_informing_contents,additional_supportive_contents
python xmaml.py --dev_datasets_ids 3,4 --dev_dataset_finetune 4 --test_dataset_eval 5 --label_col $label_col --labels $labels --per_gpu_train_batch_size 16 --meta_learn_iter 600 --inner_train_steps 5 --output_dir_meta x_mamle3/maml_trained/ --eval_task_dir x_mamle3/maml_result/
