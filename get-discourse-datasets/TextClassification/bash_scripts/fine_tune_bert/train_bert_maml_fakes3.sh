#!/usr/bin/env bash
#SBATCH --job-name=ft-FN3
#SBATCH --account=hkg02
#SBATCH --partition=gpu

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --time=0-06:00:00

module purge
module load cuda/10.1
module load java/java8
module load torch/1.7.1-v100-gcc-7.2.0-cuda-10.1-openmpi-4.0.1
module load python/3

nvcc --version

echo "----------------------------------"

nvidia-smi

df_train=input/FAKES/feature_extraction_train_updated_updated.csv
df_dev=input/FAKES/feature_extraction_dev_updated_updated.csv
df_test=input/FAKES/feature_extraction_test_updated.csv

python run_classifier_maml.py --train_set $df_train  --dev_set $df_dev --test_set $df_test --N_shot 15 --output_dir bert_output4/FOMAML/FAKES3/

