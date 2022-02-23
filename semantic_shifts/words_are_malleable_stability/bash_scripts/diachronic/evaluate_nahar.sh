#!/usr/bin/env bash
#SBATCH --job-name=eval-na
#SBATCH --account=hkg02
#SBATCH --partition=medium
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
arch=nahar
keywords_file=input/all_keywords2.txt
dirmats=/scratch/7613491_hkg02/political_discourse_mining_hiyam/semantic_shifts_modified/results_diachronic_new/
y_start=1983
y_end=2009

python evaluate_stability2.py --path1 ${EMB_DIR1} --path2 ${EMB_DIR2} --start_year ${y_start} --end_year ${y_end} --archive ${arch} --words_file ${keywords_file} --dir_name_matrices ${dirmats}

