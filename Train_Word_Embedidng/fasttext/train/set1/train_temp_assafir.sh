#!/usr/bin/env bash
#SBATCH --job-name=fast-as
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-32%5

module load python/3


# START=1974
# END=2011
# USCOUNTER=1
# for i in {1974..2011}; do
#    if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
#     echo $i
#     python train_fasttext_temp.py --archive assafir --year $i
#    fi
#    USCOUNTER=$(expr $USCOUNTER + 1)
# done
START=1974
END=2011
USCOUNTER=1
for i in {1980..2011}; do
   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
    echo $i
    python train_fasttext_set1.py --archive assafir --year $i
   fi
   USCOUNTER=$(expr $USCOUNTER + 1)
done