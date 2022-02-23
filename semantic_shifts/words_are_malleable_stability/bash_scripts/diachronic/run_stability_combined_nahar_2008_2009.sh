#!/usr/bin/env bash
#SBATCH --job-name=comb-na
#SBATCH --account=hkg02
#SBATCH --partition=medium
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

iterations=(1)
neighs=(100)
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
arch1=nahar
arch2=nahar
keywords_file=input/all_keywords2.txt
savedir=/scratch/7613491_hkg02/political_discourse_mining_hiyam/semantic_shifts_modified/results_diachronic_new/nahar/
USCOUNTER=1
kval=100
tval=1
yprev=2008
ycurr=2009

python words_are_malleable2.py --path1 $EMB_DIR1 --path2=$EMB_DIR1 --model1 ${yprev}.bin --model2 ${ycurr}.bin --model1_name ${arch1}_${yprev} --model2_name ${arch2}_${ycurr} --k ${kval} --t ${tval} --words_file ${keywords_file} --method=combined --save_dir ${savedir}

