#!/usr/bin/env bash
#SBATCH --job-name=testgp
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

python train_bert.py --train_set input/corpus-webis-editorials-16/df_train.xlsx --dev_set input/corpus-webis-editorials-16/df_dev.xlsx --test_set input/corpus-webis-editorials-16/df_test.xlsx --text_column Sentence_ar --label_column Label --output_dir bert_output4/corpus-webis-editorials-16/

