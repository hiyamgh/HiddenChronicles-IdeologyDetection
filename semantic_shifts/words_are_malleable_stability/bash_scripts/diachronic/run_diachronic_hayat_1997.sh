#!/usr/bin/env bash
#SBATCH --job-name=diach_ha
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

k=100
EMB_DIR1=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/hayat/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
arch1=hayat
arch2=hayat
keywords_file=input/all_keywords3.txt
savedir=/scratch/7613491_hkg02/political_discourse_mining_hiyam/semantic_shifts_modified/stability_diachronic/hayat/
USCOUNTER=1
ycurr=1997
yprev=1996

echo "$yprev vs $ycurr in $arch1 and $arch2"
python words_are_malleable4.py --path1 $EMB_DIR1 --path2=$EMB_DIR1 --model1 ${yprev}.bin --model2 ${ycurr}.bin --model1_name ${arch1}_${yprev} --model2_name ${arch2}_${ycurr} --k ${k} --words_file ${keywords_file} --method=combined --save_dir ${savedir}


