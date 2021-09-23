#!/usr/bin/env bash
#SBATCH --job-name=train-hayat
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-900%5

module load python/3

# 5,600 hyper-parameter
learning_rates=(0.001 0.01 0.03 0.1 0.3) # 5
sizes=(100 150 200 250 300 350 400) # 7
windows=(3 5 7 10) # 4
min_counts=(10 100 200 300 400) # 5
sgs=(0 1) # 2
negative_samplings=(5 10 15 20) # 4
archive=hayat

# Variables for tracking hyperparameters
#NUM_CALLS=5600/900 = 6.2; 900*6 = 5400, so one more
NUM_CALLS=$1 # initially 1

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

echo "checking if the array task ID is equal to 900, if yes, then execute ..."
if [ $SLURM_ARRAY_TASK_ID -le 900 ] && [ $NUMCALLS -lt 7 ]; then
    echo "I am the job 900"
    sleep 2m
    NUMCALLS=$((NUMCALLS + 1))
    sbatch train_hayat.sh $NUMCALLS
fi