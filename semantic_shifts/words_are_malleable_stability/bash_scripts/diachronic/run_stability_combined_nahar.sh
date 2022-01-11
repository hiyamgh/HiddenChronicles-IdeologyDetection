#!/usr/bin/env bash
#SBATCH --job-name=stb-comb
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-26%5

module load python/3

iterations=(1)
neighs=(100)
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
arch1=nahar
arch2=nahar
keywords_file=input/all_keywords.txt
USCOUNTER=1

for t in ${iterations[@]}; do
    for k in ${neighs[@]}; do
        for y in {1983..2009}; do
            ycurr=$((y))
            yprev=$((y - 1))
            echo "$yprev vs $ycurr in $arch1 and $arch2"
            if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
                python words_are_malleable.py --path1 $EMB_DIR1 --path2=$EMB_DIR1 --model1 ${yprev}.bin --model2 ${ycurr}.bin --model1_name ${arch1}_${yprev} --model2_name ${arch2}_${ycurr} --k ${k} --t ${t} --words_file ${keywords_file} --method=combined
            fi
            USCOUNTER=$(expr $USCOUNTER + 1)
        done
    done
done

