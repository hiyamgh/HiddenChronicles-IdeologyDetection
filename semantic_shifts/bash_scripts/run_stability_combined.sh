#!/usr/bin/env bash
#SBATCH --job-name=stab
#SBATCH --account=hkg02
#SBATCH --partition=gpu
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-9%3

module load python/3

iterations=(5)
neighs=(100 200 300)
years=(1982 2006 2007)
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
arch1=nahar
arch2=assafir
keywords_file=/input/keywords.txt
USCOUNTER=1

for t in ${iterations[@]}; do
    for k in ${neighs[@]}; do
        for y in ${years[@]}; do
            if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
                python words_are_malleable.py --path1 $EMB_DIR1 --path2=$EMB_DIR2 --model1 ${y}.bin --model2 ${y}.bin --model1_name ${arch1}_${y} --model2_name ${arch2}_${y} --k ${k} --t ${t} --method=combined
            fi
            USCOUNTER=$(expr $USCOUNTER + 1)
        done
    done
done

