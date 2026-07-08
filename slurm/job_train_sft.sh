#!/bin/bash
#SBATCH --job-name=train_sft
#SBATCH --output=logs/train_sft_%j.out
#SBATCH --error=logs/train_sft_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10

cd $WORK/intent-schema-project

module purge
module load pytorch-gpu/py3/2.4.0
source $WORK/env_intent/bin/activate

cd $WORK/intent-schema-project/src
python train_sft.py
