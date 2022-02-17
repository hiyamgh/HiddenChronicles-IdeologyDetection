#!/usr/bin/env bash
#SBATCH --job-name=viz82
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

code=1982

STD_DIR1=../Train_Word_Embedidng/fasttext/data/nahar/
STD_DIR2=../Train_Word_Embedidng/fasttext/data/assafir/
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
RES_DIR1=results2/${code}/100/
RES_DIR2=results2/${code}/200/
RES_DIR3=results2/${code}/300/

val1=nahar_${code}
val2=assafir_${code}

python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/political_parties.pkl" --out_topk $RES_DIR1 --k 100
python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/politicians.pkl" --out_topk $RES_DIR1 --k 100
python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/ideologies.pkl" --out_topk $RES_DIR1 --k 100

python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/political_parties.pkl" --out_topk $RES_DIR2 --k 200
python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/politicians.pkl" --out_topk $RES_DIR2 --k 200
python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/ideologies.pkl" --out_topk $RES_DIR2 --k 200

python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/political_parties.pkl" --out_topk $RES_DIR3 --k 300
python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/politicians.pkl" --out_topk $RES_DIR3 --k 300
python visualize_neighbours.py --data_a $STD_DIR1/${code}.txt --data_b $STD_DIR2/${code}.txt --embed_a $EMB_DIR1/${code}.bin --embed_b $EMB_DIR2/${code}.bin --name_split_a ${val1} --name_split_b ${val2} --words "wikipedia/keywords/ideologies.pkl" --out_topk $RES_DIR3 --k 300

