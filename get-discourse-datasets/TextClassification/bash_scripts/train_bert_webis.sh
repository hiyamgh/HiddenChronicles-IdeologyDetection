#!/usr/bin/env bash
#SBATCH --job-name=trainwb
#SBATCH --account=hkg02
#SBATCH --partition=normal

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=onode07
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load java/java8
module load python/3
module load python/pytorch

python train_bert.py --train_set input/corpus-webis-editorials-16/df_train.xlsx --dev_set input/corpus-webis-editorials-16/df_dev.xlsx --test_set input/corpus-webis-editorials-16/df_test.xlsx --text_column Sentence_ar --label_column Label --output_dir bert_output/corpus-webis-editorials-16/

