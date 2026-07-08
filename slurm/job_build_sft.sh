#!/bin/bash
#SBATCH --job-name=build_sft
#SBATCH --output=logs/build_sft_%j.out
#SBATCH --error=logs/build_sft_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4

cd $WORK/intent-schema-project

module purge
module load pytorch-gpu/py3/2.4.0
source $WORK/env_intent/bin/activate

cd $WORK/intent-schema-project/src
python create_training_examples.py
python build_sft_dataset.py

