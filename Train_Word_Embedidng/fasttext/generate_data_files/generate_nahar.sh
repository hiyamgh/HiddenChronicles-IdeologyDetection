#!/usr/bin/env bash
#SBATCH --job-name=gen-nahar
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-28%10

module load python/3

python generate_data_fasttext.py --archive nahar --start_year 1982 --end_year 2009


USCOUNTER=1
for year in {1982..2009}; do
   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
    echo $year
    generate_data_fasttext.py --archive nahar --year $year
   fi
   USCOUNTER=$(expr $USCOUNTER + 1)
done