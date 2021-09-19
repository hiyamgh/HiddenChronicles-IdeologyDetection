#!/usr/bin/env bash
#SBATCH --job-name=train-hayat
#SBATCH --account=p078
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-576%10

module load Python/3.6.8-GCCcore-8.2.0

# 576 parameters
learning_rates=(0.001 0.01) # 2
sizes=(100 200 300) # 3
windows=(3 5 7 10) # 4
min_counts=(10 100 500) # 3
sgs=(0 1) # 2
negative_samplings=(5 10 15 20) # 4
archive=hayat

for lr in ${learning_rates[@]}; do
    for sz in ${sizes[@]}; do
        for w in ${windows[@]}; do
            for mc in ${min_counts[@]}; do
                for sg in ${sgs[@]}; do
                    for ns in ${negative_samplings[@]}; do
                        echo "train_word2vec.py --archive ${archive} --size ${sz} --window ${w} --mincount ${mc} --sg ${sg} --negative ${ns} --learning_rate ${lr}"
                        python train_word2vec.py --archive ${archive} --size ${sz} --window ${w} --mincount ${mc} --sg ${sg} --negative ${ns} --learning_rate ${lr}
                    done
                done
            done
        done
    done
done
