#!/usr/bin/env bash
#SBATCH --job-name=stb-lin
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-3%3

module load python/3
years=(2000 2001 2002)
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
arch1=nahar
arch2=assafir
keywords_file=input/keywords.txt
USCOUNTER=1

for y in ${years[@]}; do
    if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
        python words_are_malleable.py --path1 $EMB_DIR1 --path2=$EMB_DIR2 --model1 ${y}.bin --model2 ${y}.bin --model1_name ${arch1}_${y} --model2_name ${arch2}_${y} --words_file ${keywords_file} --method=linear
    fi
    USCOUNTER=$(expr $USCOUNTER + 1)
done