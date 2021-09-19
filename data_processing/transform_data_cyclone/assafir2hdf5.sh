#!/usr/bin/env bash
#SBATCH --job-name=assafir
#SBATCH --account=p078
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load Python/3.6.8-GCCcore-8.2.0
python data2hdf5.py assafir > assafir2hdf5_$SLURM_JOBID.txt