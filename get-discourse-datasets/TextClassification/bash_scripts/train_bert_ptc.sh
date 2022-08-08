#!/usr/bin/env bash
#SBATCH --job-name=trainptc
#SBATCH --account=hkg02
#SBATCH --partition=medium

#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load java/java8
module load python/3
module load python/pytorch

python train_bert.py --train_set input/ptc_corpus/df_train_single.xlsx --dev_set input/ptc_corpus/df_dev_single.xlsx --test_set input/ptc_corpus/df_dev_single.xlsx --text_column context_ar --label_column label --output_dir bert_output/ptc_corpus/

