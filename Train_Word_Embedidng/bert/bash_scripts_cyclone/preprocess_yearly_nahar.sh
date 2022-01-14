#!/usr/bin/env bash
#SBATCH --job-name=prep-na
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-28%15

module load Java/1.8.0_45
module load Python/3.6.8-GCCcore-8.2.0
DIR=/onyx/data/p078/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/data/nahar/
RES_DIR=/onyx/data/p078/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/data/nahar-preprocessed/

USCOUNTER=1
for year in {1982..2009}; do
   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
    echo $year
    python preprocess_yearly.py --dir $DIR --out_dir $RES_DIR --file ${year}.txt
   fi
   USCOUNTER=$(expr $USCOUNTER + 1)
done

