#!/usr/bin/env bash
#SBATCH --job-name=emb-da2
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

bert_model=aubmindlab/bert-base-arabertv2
path_to_model=/scratch/7613491_hkg02/political_discourse_mining_hiyam/get-discourse-datasets/checkpoint-99900/pytorch_model.bin
embeddings_path=embeddings/nahar-1982-1986-da.pickle

python get_embeddings_after_domain_adaptation.py --bert_model $bert_model --path_to_model $path_to_model --embeddings_path $embeddings_path
