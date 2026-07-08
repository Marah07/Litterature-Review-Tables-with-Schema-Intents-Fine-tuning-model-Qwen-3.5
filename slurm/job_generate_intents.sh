#!/bin/bash
#SBATCH --job-name=generate_intents
#SBATCH --output=logs/gen%j.out
#SBATCH --error=logs/gen%j.err

#SBATCH -A ruk@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

#SBATCH --time=10:00:00

source ~/.bashrc

cd $WORK/intent-schema-project

module purge
module load pytorch-gpu/py3/2.4.0
source $WORK/env_intent/bin/activate

export PYTHONPATH=$WORK/intent-schema-project/src:$PYTHONPATH
export HF_HOME=$WORK/hf_cache
export TRITON_CACHE_DIR=$WORK/triton_cache
export XDG_CACHE_HOME=$WORK/xdg_cache

python src/generate_intents.py

