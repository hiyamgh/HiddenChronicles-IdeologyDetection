#!/usr/bin/env bash
#SBATCH --job-name=viznaas
#SBATCH --account=hkg02
#SBATCH --partition=gpu
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/

python words_are_malleable.py --path1 $EMB_DIR1 --path2=$EMB_DIR2
