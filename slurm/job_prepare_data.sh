#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

#SBATCH -A ruk@v100
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

cd $WORK/intent-schema-project

module purge
module load pytorch-gpu/py3/2.4.0
source $WORK/env_intent/bin/activate

export PYTHONPATH=$WORK/intent-schema-project/src:$PYTHONPATH

python src/prepare_data.py

