#!/usr/bin/env bash
#SBATCH --job-name=nnnaha
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

STD_DIR1=../Train_Word_Embedidng/fasttext/data/nahar/start_end/
STD_DIR2=../Train_Word_Embedidng/fasttext/data/hayat/start_end/
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/1968_1975/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/hayat/1968_1975/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
RES_DIR=/results/1968_1975/

val1=nahar_1968_1975
val2=hayat_1968_1975
python test_fasttext.py --data_a $STD_DIR1/1968_1975.txt --data_b $STD_DIR2/1968_1975.txt --embed_a $EMB_DIR1/1968_1975.bin --embed_b $EMB_DIR2/1968_1975.bin --name_split_a ${val1} --name_split_b ${val2} --out_topk $RES_DIR/"detect_${val1}_${val2}/" --k 200

