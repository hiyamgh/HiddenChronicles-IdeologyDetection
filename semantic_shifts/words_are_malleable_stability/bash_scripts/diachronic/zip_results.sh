#!/usr/bin/env bash
#SBATCH --job-name=cmprs
#SBATCH --account=hkg02
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

zip -r results_diachronic_new.zip results_diachronic_new
