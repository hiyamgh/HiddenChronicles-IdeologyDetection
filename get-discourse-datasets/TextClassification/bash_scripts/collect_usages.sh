#!/usr/bin/env bash
#SBATCH --job-name=collusg
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load java/java8
module load python/3
module load python/pytorch

python usage_collector.py