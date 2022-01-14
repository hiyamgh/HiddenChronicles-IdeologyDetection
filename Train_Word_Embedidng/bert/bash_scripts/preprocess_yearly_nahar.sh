#!/usr/bin/env bash
#SBATCH --job-name=prep-na
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-28%2

module load java/java8
module load python/3

DIR=/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/data/nahar/
RES_DIR=/scratch/7613491_hkg02/political_discourse_mining_hiyam/arabert/nahar-preprocessed/

USCOUNTER=1
for year in {1982..2009}; do
   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
    echo $year
    python preprocess_yearly.py --dir $DIR --out_dir $RES_DIR --file ${year}.txt
   fi
   USCOUNTER=$(expr $USCOUNTER + 1)
done

