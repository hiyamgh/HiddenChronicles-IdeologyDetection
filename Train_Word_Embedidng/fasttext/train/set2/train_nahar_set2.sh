#!/usr/bin/env bash
#SBATCH --job-name=fast-na
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-77%5

module load python/3

START=1933
END=2009
USCOUNTER=1
for i in {1933..2009}; do
   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
    echo $i
    python train_fasttext_set2.py --archive nahar --year $i
   fi
   USCOUNTER=$(expr $USCOUNTER + 1)
done