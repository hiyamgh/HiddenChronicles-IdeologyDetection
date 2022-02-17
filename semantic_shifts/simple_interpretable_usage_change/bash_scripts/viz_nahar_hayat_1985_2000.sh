#!/usr/bin/env bash
#SBATCH --job-name=viznaha
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

code=1985_2000

STD_DIR1=../Train_Word_Embedidng/fasttext/data/nahar/start_end/
STD_DIR2=../Train_Word_Embedidng/fasttext/data/hayat/start_end/
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/${code}/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/hayat/${code}/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
RES_DIR=/results/${code}/

val1=nahar_${code}
val2=hayat_${code}
python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "words_${code}.txt" --out_topk $RES_DIR/"detect_${val1}_${val2}/" --k 200

