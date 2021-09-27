#!/usr/bin/env bash
#SBATCH --job-name=nn
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-10%10

STD_DIR1=../Train_Word_Embedidng/fasttext/data/nahar/
STD_DIR2=../Train_Word_Embedidng/fasttext/data/assafir/
EMB_DIR1=../Train_Word_Embedidng/fasttext/nahar/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
EMB_DIR2=../Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/
RES_DIR=/results/

START=2000
END=2009
USCOUNTER=1
for ((i=START;i<=END;i++)); do
   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
    echo $i
    val1=nahar_${i}
    val2=assafir_${i}
    python test_fasttext.py --data_a $STD_DIR1/${i}.txt --data_b $STD_DIR2/${i}.txt --embed_a $EMB_DIR1/${i}.bin --embed_b $EMB_DIR2/${i}.bin --name_split_a nahar_${i} --name_split_b assafir_${i} --out_topk $RES_DIR/"detect_${val1}_${val2}/" --k 500
   fi
   USCOUNTER=$(expr $USCOUNTER + 1)
done

