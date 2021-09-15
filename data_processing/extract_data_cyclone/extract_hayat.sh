#!/usr/bin/env bash
#SBATCH --job-name=hayat
#SBATCH --account=p078
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

tar -xvf hayat.tar.gz