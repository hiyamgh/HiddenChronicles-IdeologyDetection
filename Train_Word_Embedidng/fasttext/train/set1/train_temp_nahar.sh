#!/usr/bin/env bash
#SBATCH --job-name=fast-nahar
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-10%5

module load python/3


# START=1933
# END=2009
# USCOUNTER=1
# for i in {1933..2009}; do
#    if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
#     echo $i
#     python train_fasttext_temp.py --archive nahar --year $i
#    fi
#    USCOUNTER=$(expr $USCOUNTER + 1)
# done

USCOUNTER=1
for i in {2000..2009}; do
   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
    echo $i
    python train_fasttext_set1.py --archive nahar --year $i
   fi
   USCOUNTER=$(expr $USCOUNTER + 1)
done