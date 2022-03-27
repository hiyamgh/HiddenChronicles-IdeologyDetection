#!/usr/bin/env bash
#SBATCH --job-name=ev-d-ha
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/hayat/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
keywords_file=input/all_keywords3.txt

python evaluate_stability.py --models_path1 $EMB_DIR1 --models_path2=$EMB_DIR1 --keywords_path ${keywords_file} --mode d-hayat

