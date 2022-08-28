#!/usr/bin/env bash
#SBATCH --job-name=ft-FN1
#SBATCH --account=hkg02
#SBATCH --partition=medium

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu


module load java/java8
module load gcc/10.1.0
module load python/3

df_train=input/FAKES/feature_extraction_train_updated_updated.csv
df_dev=input/FAKES/feature_extraction_dev_updated_updated.csv
df_test=input/FAKES/feature_extraction_test_updated.csv

python run_classifier_maml.py --train_set $df_train  --dev_set $df_dev --test_set $df_test --N_shot 5 --output_dir bert_output4/FOMAML/FAKES1/

