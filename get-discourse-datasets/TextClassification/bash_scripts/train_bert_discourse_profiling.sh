#!/usr/bin/env bash
#SBATCH --job-name=traindp
#SBATCH --account=hkg02
#SBATCH --partition=large

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load java/java8
module load python/3
module load python/pytorch

python train_bert.py --train_set input/Discourse_Profiling/df_train_cleaned.xlsx --dev_set input/Discourse_Profiling/df_dev_cleaned.xlsx --test_set input/Discourse_Profiling/df_test_cleaned.xlsx --text_column Sentence_ar --label_column Label --output_dir bert_output/Discourse_Profiling/

