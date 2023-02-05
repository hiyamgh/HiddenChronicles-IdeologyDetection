#!/usr/bin/env bash

#SBATCH --job-name=XMAML # Job name
#SBATCH --account=hkg02
#SBATCH --partition=gpu # Partition
#SBATCH --nodes=2 # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks
#SBATCH --gres=gpu:v100d32q:1 # Number of GPUs
#SBATCH --time=0-06:00:00

# Load any necessary modules, in this case OpenMPI with CUDA
module purge
module load python/3
module load cuda/10.1
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1

nvcc --version

echo "----------------------------------"

nvidia-smi

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


#srun python xmaml.py --bert_model aubmindlab/bert-base-arabertv2 --model_type bert --dev_datasets_ids ARG_ar,corp_PRST_ar_ARG --dev_dataset_finetune corp_PRST_ar_ARG --test_dataset_eval corp_SSM_ar_ARG --do_validation 1 --do_finetuning 0 --do_evaluation 0  --meta_learn_iter 20 --n 16 --inner_train_steps 3 --inner_lr 0.0001 --labels anecdote,assumption,other,common-ground,testimony,statistics --output_dir_meta results_tuning/ARG_ARABERT/
srun python xmaml.py --bert_model aubmindlab/bert-large-arabertv02 --model_type bert --dev_datasets_ids ARG_ar,corp_PRST_ar_ARG --dev_dataset_finetune corp_PRST_ar_ARG --test_dataset_eval corp_SSM_ar_ARG --do_validation 1 --do_finetuning 0 --do_evaluation 0  --meta_learn_iter 20 --n 16 --inner_train_steps 3 --inner_lr 0.0001 --labels anecdote,assumption,other,common-ground,testimony,statistics --output_dir_meta results_tuning/ARG_ARABERT/