#!/usr/bin/env bash
#SBATCH --job-name=sync
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-12%6

module load python/3

neighs=(100)
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR3=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/hayat/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/

arch1=nahar
arch2=assafir
arch3=hayat
keywords_file=input/all_keywords3.txt
savedir=/scratch/7613491_hkg02/political_discourse_mining_hiyam/semantic_shifts_modified/stability_synchronic/
USCOUNTER=1

for k in ${neighs[@]}; do
    for y in {1988..2000}; do
        echo "synchronic: year $y in: $arch1 vs. $arch2 vs. $arch3"
        if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
            python words_are_malleable3.py --path1 $EMB_DIR1 --path2 $EMB_DIR2 --path3 $EMB_DIR3 --model1 ${y}.bin --model2 ${y}.bin --model3 ${y}.bin --model1_name ${arch1}_${y} --model2_name ${arch2}_${y} --model3_name ${arch3}_${y} --k ${k} --words_file ${keywords_file} --method=combined --save_dir ${savedir}
        fi
        USCOUNTER=$(expr $USCOUNTER + 1)
    done
done