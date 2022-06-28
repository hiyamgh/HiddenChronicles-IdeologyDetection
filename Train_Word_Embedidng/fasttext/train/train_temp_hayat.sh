#!/usr/bin/env bash
#SBATCH --job-name=fast-ha
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-51%2

module load python/3


START=1950
END=2000
USCOUNTER=1
for i in {1950..2000}; do
   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
    echo $i
    python train_fasttext_temp.py --archive hayat --year $i
   fi
   USCOUNTER=$(expr $USCOUNTER + 1)
done